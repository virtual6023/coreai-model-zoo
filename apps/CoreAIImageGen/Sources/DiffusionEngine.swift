// DiffusionEngine — downloads / loads a Core AI diffusion bundle and generates
// images using Apple's official CoreAIDiffusionPipeline runtime. The pipeline type
// (FLUX.2 / SD3 / SD) is auto-detected from metadata.json, mirroring the zoo's
// `diffusion-runner` reference tool, so any `coreai.diffusion.export` bundle drops in.
//
// The hosted catalog is macOS-only: FLUX.2 klein 4B's peak footprint exceeds the iOS
// per-process memory limit (measured ~0.4 GB over on a 12 GB iPhone 17 Pro — the text
// encoder is not released before the transformer runs). The iOS app still loads smaller
// diffusion bundles (e.g. Stable Diffusion) via "Local…".
//
// Model delivery uses the shared AppShared/ModelDownloader (range-chunked parallel
// download with cross-launch resume and atomic bundle placement) for the `.aimodel`
// directory bundles + the tokenizer, plus a couple of direct GETs for the handful of
// tiny root files (metadata.json, vae_bn_*.npy) that the HF tree API can't enumerate.

import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

/// Thread-safe cancel flag — readable from the pipeline's (possibly off-main)
/// progress callback without touching @MainActor state.
final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
}

@MainActor
final class DiffusionEngine: ObservableObject {

    /// A Core AI-converted diffusion bundle published on the Hugging Face Hub.
    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String           // "org/name" on the Hub
        let bundleDirName: String    // local folder name under Documents/
        let title: String
        let defaultSteps: Int
        let defaultGuidance: Float
    }

    // Hosted catalog — macOS only. FLUX.2 klein 4B overruns the iOS memory limit, so the
    // iOS app ships with an empty catalog and loads smaller bundles via "Local…".
    static let catalog: [ModelOption] = {
        #if os(macOS)
        return [
            ModelOption(
                repoId: "mlboydaisuke/FLUX.2-klein-4B-CoreAI",
                bundleDirName: "FLUX.2-klein-4B",
                title: "FLUX.2 klein 4B",
                defaultSteps: 4, defaultGuidance: 1.0)
        ]
        #else
        return []
        #endif
    }()

    enum Status: Equatable {
        case idle
        case downloading
        case loading
        case ready
        case generating(step: Int, total: Int)
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading model…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .generating(let s, let t): return "Generating… step \(s)/\(t)"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .generating: return true
            default: return false
            }
        }
    }

    @Published var status: Status = .idle
    @Published var image: CGImage?
    @Published var exportURL: URL?
    @Published var modelTitle: String = "—"
    @Published var loadSeconds: Double?
    @Published var generateSeconds: Double?
    @Published var imageSize: String = ""

    /// Shared range-chunked downloader (atomic placement + cross-launch resume).
    let downloader = ModelDownloader()

    private var pipeline: (any DiffusionPipeline)?
    private var descriptor: PipelineDescriptor?
    private var work: Task<Void, Never>?
    private var cancelToken = CancellationToken()

    var canGenerate: Bool { if case .ready = status { return true }; return false }

    // Platform target: iOS runs the lighter 512 / half-VAE components; macOS runs
    // the full 1024 components. The HF bundle is universal — we fetch only the subset
    // this platform needs. The transformer / VAE bundles are resolved by NAME at load
    // (Transformer / Transformer_512 …), so downloading only the half set is enough.
    #if os(iOS)
    private static let fluxMode: DecodeResolution = .half
    private static let decodeResolution: DecodeResolution = .half
    private static let directoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer_512.aimodel", local: "Transformer_512.aimodel"),
        .init(remote: "TextEncoder.aimodel", local: "TextEncoder.aimodel"),
        .init(remote: "VAEDecoder_half.aimodel", local: "VAEDecoder_half.aimodel"),
        .init(remote: "VAEEncoder_half.aimodel", local: "VAEEncoder_half.aimodel"),
        .init(remote: "tokenizer", local: "tokenizer"),
    ]
    #else
    private static let fluxMode: DecodeResolution = .auto
    private static let decodeResolution: DecodeResolution = .full
    private static let directoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer.aimodel", local: "Transformer.aimodel"),
        .init(remote: "TextEncoder.aimodel", local: "TextEncoder.aimodel"),
        .init(remote: "VAEDecoder.aimodel", local: "VAEDecoder.aimodel"),
        .init(remote: "VAEEncoder.aimodel", local: "VAEEncoder.aimodel"),
        .init(remote: "tokenizer", local: "tokenizer"),
    ]
    #endif

    // Tiny root-level files the pipeline needs alongside the bundles. The HF tree API
    // only enumerates directories, so these are fetched with a plain resolve GET.
    private static let rootFiles = ["metadata.json", "vae_bn_mean.npy", "vae_bn_var.npy"]

    // MARK: - Loading

    /// Download a converted bundle from the Hugging Face Hub (cached after the first
    /// run, resumable across launches) and load it.
    func loadFromHub(_ option: ModelOption) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil
        modelTitle = option.title
        status = .downloading
        // Keep the screen awake for the multi-GB download: if the device auto-locks the app
        // gets suspended and the (foreground) URLSession transfer stalls. Re-enabled when the
        // whole flow finishes (see the defer below).
        Self.setIdleTimerDisabled(true)

        // The fine-grained download progress (fraction / byte detail) is read straight off
        // `downloader` by the view (it's an ObservableObject); this Task only sequences the
        // phases and surfaces a terminal error.
        work = Task {
            defer { Self.setIdleTimerDisabled(false) }
            do {
                let dest = try Self.bundleDestination(for: option)
                await downloader.fetch(
                    repo: "https://huggingface.co/\(option.repoId)",
                    items: Self.directoryItems, into: dest)
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                try Task.checkCancellation()
                try await Self.fetchRootFiles(repoId: option.repoId, into: dest)
                try await self.loadPipeline(at: dest)
            } catch is CancellationError {
                // user started another action — leave state to that action
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    private static func setIdleTimerDisabled(_ disabled: Bool) {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = disabled
        #endif
    }

    /// Load a bundle already exported to a local folder.
    func loadLocal(_ url: URL) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil
        modelTitle = url.lastPathComponent
        status = .loading
        work = Task {
            do { try await self.loadPipeline(at: url) }
            catch { self.status = .error("\(error)") }
        }
    }

    private func loadPipeline(at url: URL) async throws {
        status = .loading
        let start = ContinuousClock.now
        let desc = try PipelineDescriptor.resolve(at: url, config: .auto)

        let built: any DiffusionPipeline
        switch desc.type {
        case .some(.flux2):
            built = try await Flux2Pipeline(from: url, config: .auto, mode: Self.fluxMode)
        case .some(.stableDiffusion3):
            built = try await SD3Pipeline(from: url, config: .auto)
        default:
            built = try await StableDiffusionPipeline.load(from: url, config: .auto)
        }

        self.descriptor = desc
        self.pipeline = built
        self.loadSeconds = Self.seconds(since: start)
        let size = built.defaultImageSize
        self.imageSize = "\(size.width)×\(size.height)"
        self.status = .ready
    }

    /// Documents/<bundleDirName> — a clean, dedicated folder that holds the placed
    /// bundles + root files. (Kept separate from any HF cache so a partial transfer
    /// can never leave a half-bundle the runtime would choke on.)
    private static func bundleDestination(for option: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(option.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Fetch the tiny root files with a plain resolve GET (skipping any already present).
    private static func fetchRootFiles(repoId: String, into dest: URL) async throws {
        let fm = FileManager.default
        for name in rootFiles {
            let target = dest.appendingPathComponent(name)
            if fm.fileExists(atPath: target.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(name)") else {
                throw err("bad file url for \(name)")
            }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                throw err("\(name): HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            try data.write(to: target, options: .atomic)
        }
    }

    // MARK: - Generation

    func generate(prompt: String, negativePrompt: String, steps: Int, guidance: Float, seed: UInt32) {
        guard let pipeline, let desc = descriptor, canGenerate else { return }
        work?.cancel()
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)

        let scheduler: SchedulerType =
            (desc.type == .flux2 || desc.type == .stableDiffusion3)
            ? .discreteFlow : .dpmSolverMultistep

        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: steps,
            guidanceScale: guidance,
            schedulerType: scheduler,
            encoderScaleFactor: desc.encoderScaleFactor ?? 0.18215,
            decoderScaleFactor: desc.decoderScaleFactor ?? 0.18215,
            decoderShiftFactor: desc.decoderShiftFactor ?? 0.0,
            decodeResolution: Self.decodeResolution,
            lazyModelLoading: true
        )

        work = Task {
            do {
                let start = ContinuousClock.now
                let result = try await pipeline.generateImages(configuration: config) { @Sendable progress in
                    let s = progress.step, t = progress.totalSteps
                    Task { @MainActor in self.status = .generating(step: s, total: t) }
                    return !token.isCancelled
                }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                let cg = result.images.first
                self.image = cg
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    func cancel() {
        cancelToken.cancel()
        work?.cancel()
        if case .generating = status { status = .ready }
    }

    // MARK: - Helpers

    /// Write a CGImage to a temp PNG for sharing/saving via SwiftUI ShareLink.
    private static func writeTempPNG(_ cg: CGImage?) -> URL? {
        guard let cg else { return nil }
        let name = "coreai-image-\(UInt32.random(in: 0 ... .max)).png"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? url : nil
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "DiffusionEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
