// Gemma4ChatEngine — the RELEASE UI engine: user-selectable GPU / ANE compute modes over the two
// device-verified gemma4 E2B paths, one chat surface.
//
//   GPU  -> Gemma4MonolithBackend (metal-kernel host-cache monolith + head-argmax kernel, 8/8)
//   ANE  -> Gemma4ChunkBackend    (6-chunk host-cache + on-ANE argmax head, 8/8)
//
// Mode switch frees the current backend's models before loading the other (the two sets together
// approach the jetsam ceiling). Each generate() resets the host-owned KV and prefills the current
// user message only (single-turn semantics, same as the verified harnesses).
//
// Headless hooks (devicectl process launch --console --environment-variables):
//   GEMMA_ENGINE = gpu (default) | ane     initial mode
//   GEMMA_VERIFY = 1                        drive the HF-oracle ids through the SELECTED release
//                                           engine, print match n/8 (the release-path 8/8 check)
//   GEMMA_PROMPT / GEMMA_N                  generate + print output, tok/s, per-token profile

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// Common surface of the two release backends. Both are @MainActor classes that own their models
// and host-side KV; `step` runs one token at absolute position `pos` and returns the argmax token
// (or -1 when `needToken` is false and the head dispatch was skipped — prefill positions whose
// logits nobody reads).
@MainActor
protocol Gemma4Backend: AnyObject {
    var modeLabel: String { get }       // "GPU" | "ANE"
    var ctx: Int { get }                // fixed cache capacity (bucket) — prompt+gen must fit
    func load() async throws
    func reset()                        // zero KV + profile counters (new conversation turn)
    func step(_ tok: Int32, _ pos: Int, needToken: Bool) async throws -> Int
    func profileSummary() -> String     // per-token ms breakdown of the last generation
}

enum GemmaMode: String, CaseIterable, Identifiable {
    case gpu = "GPU", ane = "ANE"
    var id: String { rawValue }
}

@MainActor
final class Gemma4ChatEngine: ObservableObject {
    @Published var mode: GemmaMode
    @Published var status = "starting…"
    @Published var ready = false
    @Published var loading = false
    @Published var busy = false
    @Published var output = ""
    @Published var stats = ""

    private var backend: Gemma4Backend?
    private var loadedMode: GemmaMode?  // mode the current backend was loaded for
    private var tokenizer: Tokenizer!
    private let EOT = 106               // gemma <end_of_turn>
    private var eosId = 106

    init() {
        mode = (ProcessInfo.processInfo.environment["GEMMA_ENGINE"] == "ane") ? .ane : .gpu
    }

    private func docs() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }

    // MARK: model delivery

    // Where the published artifact set for this app lives (GEMMA_REPO / the UI field override it).
    static let defaultRepo = "https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI"

    // Download targets (repo subpath -> name under Documents/models) the CURRENT mode still
    // needs. Empty == everything is on disk. The tokenizer ships in the app bundle, and the env
    // overrides (GEMMA_MONO_MODEL/HEAD, GEMMA_QUANT) are dev paths not covered here.
    func missingDownloads() -> [ModelDownloader.Item] {
        var paths: [(remote: String, local: String)] = [("ios-frontend/gemma4_gather_raw", "gemma4_gather_raw")]
        if mode == .gpu {
            paths += [("ios-gpu/gemma4_e2b_metal_int4km_L35.aimodel", "gemma4_e2b_metal_int4km_L35.aimodel"),
                      ("ios-gpu/gemma4_e2b_head_argmax_int4km.aimodel", "gemma4_e2b_head_argmax_int4km.aimodel")]
        } else {
            paths += (1...6).map { ("ios-ane/gemma4_e2b_hostcache_chunk\($0)_int8.aimodel",
                                    "gemma4_e2b_hostcache_chunk\($0)_int8.aimodel") }
            paths += [("ios-ane/gemma4_e2b_head_argmax_int8.aimodel", "gemma4_e2b_head_argmax_int8.aimodel")]
        }
        let models = docs().appendingPathComponent("models")
        return paths
            .filter { !FileManager.default.fileExists(atPath: models.appendingPathComponent($0.local).path) }
            .map { ModelDownloader.Item(remote: $0.remote, local: $0.local) }
    }

    // Load (or re-load after a mode switch) the backend for the current mode.
    // A concurrent call (UI onChange + headless autoTest both react to a mode change) waits for the
    // in-flight load; after the wait it re-checks the target mode, so a rapid GPU→ANE→GPU flip still
    // converges on the last selected mode.
    func load() async {
        while loading { try? await Task.sleep(nanoseconds: 100_000_000) }
        if loadedMode == mode, ready { return }
        loading = true; ready = false
        status = "loading \(mode.rawValue) engine…"
        backend = nil; loadedMode = nil   // free the previous mode's models BEFORE loading the next set
        let target = mode
        let be: Gemma4Backend = (target == .gpu) ? Gemma4MonolithBackend() : Gemma4ChunkBackend()
        do {
            let tLoad = Date()
            try await be.load()
            print(String(format: "[chat] %@ engine load %.1fs", target.rawValue, -tLoad.timeIntervalSinceNow))
            if tokenizer == nil {
                // Bundled tokenizer first; a sideloaded Documents/tokenizer (devicectl) wins if present.
                let sideloaded = docs().appendingPathComponent("tokenizer")
                let folder = FileManager.default.fileExists(atPath: sideloaded.path)
                    ? sideloaded
                    : (Bundle.main.url(forResource: "tokenizer", withExtension: nil) ?? sideloaded)
                tokenizer = try await AutoTokenizer.from(modelFolder: folder, strict: false)
                eosId = tokenizer.eosTokenId ?? EOT
            }
            backend = be
            loadedMode = target
            ready = true
            status = "\(target.rawValue) ready · ctx \(be.ctx)"
        } catch {
            status = "load error: \(error.localizedDescription)"
            print("[chat] \(target.rawValue) load error: \(error)")
        }
        loading = false
        // The selection moved while we were loading (rapid flip) — converge on the latest mode.
        if mode != target { await load() }
    }

    // q1 prefill (head only at the last prompt position) + greedy decode, streamed to `output`.
    func generate(_ prompt: String, maxNew: Int = 64) async {
        guard let be = backend, ready, !busy else { return }
        busy = true; output = ""; stats = ""
        defer { busy = false }
        do {
            let ids = (try tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]])).map { Int32($0) }
            guard ids.count < be.ctx else {
                output = "prompt (\(ids.count) tok) does not fit ctx \(be.ctx)"; return
            }
            be.reset()
            let tPre = Date(); var last = 0
            for (pos, t) in ids.enumerated() {
                last = try await be.step(t, pos, needToken: pos == ids.count - 1)
            }
            let preSec = -tPre.timeIntervalSinceNow

            var gen: [Int] = []
            var pos = ids.count
            let tDec = Date()
            if last != eosId && last != EOT {
                gen.append(last)
                output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)
            }
            // Positions 0..ctx-1 are valid cache columns — stop before overrunning the bucket.
            while gen.count < maxNew, last != eosId, last != EOT, pos < be.ctx {
                last = try await be.step(Int32(last), pos, needToken: true)
                pos += 1
                if last == eosId || last == EOT { break }
                gen.append(last)
                output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)  // live stream
            }
            let decSec = -tDec.timeIntervalSinceNow
            let full = pos >= be.ctx ? " · ctx full" : ""
            stats = String(format: "%@ · prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s%@ | %@",
                           be.modeLabel,
                           ids.count, Double(ids.count) / max(preSec, 1e-6),
                           gen.count, Double(gen.count) / max(decSec, 1e-6),
                           full, be.profileSummary())
        } catch {
            output = "generation error: \(error)"
        }
    }

    // HF-greedy oracle ("Count from one to ten") through the CURRENT release backend -> match n/8.
    private func verifyOracle() async {
        guard let be = backend, ready else { print("[chat] verify skipped (not ready)"); return }
        let prompt: [Int32] = [2, 105, 2364, 107, 4377, 699, 886, 531, 3595, 236764, 15914, 684, 162760, 236761, 106, 107, 105, 4368, 107]
        let ref = [4906, 236764, 1156, 236764, 1806, 236764, 2390, 236764]
        do {
            be.reset()
            var last = 0
            for (pos, t) in prompt.enumerated() {
                last = try await be.step(t, pos, needToken: pos == prompt.count - 1)
            }
            var gen = [last]
            for i in 0..<(ref.count - 1) {
                last = try await be.step(Int32(last), prompt.count + i, needToken: true)
                gen.append(last)
            }
            let n = zip(gen, ref).filter { $0 == $1 }.count
            print("[chat] mode=\(mode.rawValue) decode = \(gen)")
            print("[chat] mode=\(mode.rawValue) ref    = \(ref)")
            print("[chat] mode=\(mode.rawValue) verify match = \(n)/\(ref.count) -> \(n == ref.count ? "PASS ✅ EXACT" : "DEGRADED ⚠️")")
        } catch { print("[chat] mode=\(mode.rawValue) verify error: \(error)") }
    }

    // Headless self-test driver (see header). VERIFY runs the HF-greedy oracle ids through the
    // selected RELEASE engine — the device 8/8 evidence for the shipping path, not just the harness.
    // GEMMA_VERIFY_BOTH=1 additionally flips the mode IN-PROCESS (the real picker scenario: free the
    // current models, load the other set) and verifies again — proves the switch doesn't jetsam.
    func autoTestIfRequested() async {
        let env = ProcessInfo.processInfo.environment
        if env["GEMMA_VERIFY"] == "1" || env["GEMMA_VERIFY_BOTH"] == "1" {
            await verifyOracle()
            if env["GEMMA_VERIFY_BOTH"] == "1" {
                let other: GemmaMode = mode == .gpu ? .ane : .gpu
                print("[chat] switching mode \(mode.rawValue) -> \(other.rawValue) (in-process, frees current models)")
                mode = other
                await load()
                print("[chat] after switch: \(status)")
                await verifyOracle()
            }
        }
        if let p = env["GEMMA_PROMPT"], ready {
            print("[chat] mode=\(mode.rawValue) autotest prompt: \(p)")
            await generate(p, maxNew: Int(env["GEMMA_N"] ?? "24") ?? 24)
            print("[chat] OUT >>> \(output)")
            print("[chat] \(stats)")
        }
    }
}
