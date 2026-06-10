// PipelinedBackend — decode-only loop-free int8lin LanguageBundles on Apple's
// coreai-pipelined GPU engine: async non-blocking encode, on-GPU argmax
// sampling, on-device KV growth, zero custom kernels. One class drives every
// pipelined chat model; `Spec` picks the bundle (downloaded from HF into
// Documents/models/):
//   * Qwen3.5-0.8B int8lin — 50.3–51.5 tok/s decode on iPhone 17 Pro
//     (vs 42.5–45.4 for the fused-kernel static monolith)
//   * LFM2.5-1.2B int8lin  — 38.0–39.6 tok/s decode (~87% of the naive BW
//     ceiling; conv+attention hybrid, first non-Qwen rider)
// Requires the engine extra-states patch on the local coreai-models package
// (the fixed-shape extra states: qwen's SSM conv/rec, lfm2's conv columns).
//
// Engine contract for these bundles (input_ids is STATIC [1,1]):
//   * COREAI_CHUNK_THRESHOLD=1 before engine creation — prefill must run as
//     pipelined S=1 steps (prompt tok/s ≈ decode tok/s).
//   * never call engine.warmup() — it warms query length 256, which the S=1
//     graph rejects; a 1-token generate after load is the warmup.

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class PipelinedBackend {
    struct Spec: Sendable {
        let bundleName: String     // dir under Documents/models/
        let hfRemotePath: String   // subpath inside the mode's HF repo
        let label: String          // stats / status line prefix
        let warmupToken: Int32     // any valid id for the 1-token warmup
    }

    // nonisolated: referenced from the (nonisolated) GemmaMode enum.
    nonisolated static let qwen = Spec(
        bundleName: "qwen3_5_0_8b_decode_int8lin",
        hfRemotePath: "gpu-pipelined/qwen3_5_0_8b_decode_int8lin",
        label: "Qwen ⚡pipelined",
        warmupToken: 9707)
    nonisolated static let lfm2 = Spec(
        bundleName: "lfm2_5_1_2b_instruct_decode_int8lin",
        hfRemotePath: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8lin",
        label: "LFM2.5 ⚡pipelined",
        warmupToken: 1098)

    let spec: Spec
    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    private(set) var ctx = 4096

    init(spec: Spec) { self.spec = spec }

    var loaded: Bool { engine != nil }

    struct GenStats {
        var label = ""
        var prefillTok = 0
        var prefillSec = 0.0
        var decodeTok = 0
        var decodeSec = 0.0
        var summary: String {
            String(format: "%@ · prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s",
                   label,
                   prefillTok, Double(prefillTok) / max(prefillSec, 1e-6),
                   decodeTok, Double(max(0, decodeTok - 1)) / max(decodeSec, 1e-6))
        }
    }

    func load() async throws {
        // S=1 prefill; set before the engine reads ModelConfig.chunkThreshold.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("models").appendingPathComponent(spec.bundleName)
        let bundle = try LanguageBundle(at: dir)
        ctx = bundle.maxContextLength

        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        )
        self.engine = engine
        self.tokenizer = try await bundle.loadTokenizer()

        // queryLength=1 warmup: one throwaway token through the real graph.
        _ = try await run(ids: [spec.warmupToken], maxTokens: 1, eos: nil, onText: { _ in })
    }

    func unload() {
        engine = nil
        tokenizer = nil
    }

    // Chat-templated greedy generation, streaming decoded text via onUpdate.
    func generate(_ prompt: String, maxNew: Int, onUpdate: (String) -> Void) async throws -> GenStats {
        guard let tokenizer else { throw Self.err("backend not loaded") }
        let ids = try templatedIds(prompt, tokenizer: tokenizer)
        guard ids.count < ctx - 1 else { throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(ctx)") }
        let budget = min(maxNew, ctx - ids.count - 1)
        return try await run(ids: ids, maxTokens: budget, eos: tokenizer.eosTokenId) { gen in
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
    }

    private func run(
        ids: [Int32], maxTokens: Int, eos: Int?, onText: ([Int]) -> Void
    ) async throws -> GenStats {
        guard let engine else { throw Self.err("backend not loaded") }
        try await engine.reset()
        let stream = try engine.generate(
            with: ids,
            samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens)
        )
        var stats = GenStats(label: spec.label, prefillTok: ids.count)
        var gen: [Int] = []
        let t0 = SuspendingClock.now
        var tGen = t0
        for try await step in stream {
            if stats.prefillSec == 0 {
                stats.prefillSec = Self.seconds(since: t0)
                tGen = SuspendingClock.now
            }
            let tok = Int(step.tokenId)
            stats.decodeTok += 1
            if let eos, tok == eos { break }
            gen.append(tok)
            onText(gen)
        }
        stats.decodeSec = Self.seconds(since: tGen)
        return stats
    }

    // Chat template via the bundle tokenizer; manual ChatML fallback if the
    // template file isn't picked up by swift-transformers (both families speak
    // ChatML: qwen natively, lfm2 via its <|im_start|>/<|im_end|> markers —
    // lfm2's bos rides on the tokenizer's own post-processor).
    private func templatedIds(_ prompt: String, tokenizer: Tokenizer) throws -> [Int32] {
        if let ids = try? tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]]) {
            return ids.map { Int32($0) }
        }
        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: text).map { Int32($0) }
    }

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let d = SuspendingClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "PipelinedBackend", code: 1,
                userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
