// Gemma 4 E2B on-device 3-stage engine (iOS 27 / Core AI) — FIXED-SHAPE decode (releasable).
//
//   input_ids --gather_embeds--> inputs_embeds --gather_per_layer--> per_layer_inputs
//             --STATIC dual-KV core--> hidden --(fp16->fp32)--> head --> logits --> argmax
//
// The decode core is the iOS STATIC-shape graph (ondevice/export_gemma4_ios.py): every input shape is
// CONSTANT across steps, so the runtime never recompiles -> flat memory. The dynamic static-q1 core it
// replaces was numerically correct (HF 8/8) but grew ~200 MB/step (a new shape specialization per step)
// -> jetsam OOM. Per step we feed `in_step` (the write column) + `position_ids[1,1]` (RoPE) + two
// additive causal masks (full + sliding) instead of a growing `position_ids`; the dual KV caches are
// fixed-capacity Core AI states updated in place (slice_update at in_step). See
// ondevice/CoreAIChat/IOS_FIXED_SHAPE_HANDOFF.md.
//
// Models + tokenizer load from the app's Documents dir (pushed via `devicectl device copy`):
//   Documents/models/gemma4_gather_raw/                 (mmap front-end tables)
//   Documents/models/gemma4_e2b_ios_static_int8.aimodel (this static core)
//   Documents/models/gemma4_e2b_int8_head.aimodel
//   Documents/tokenizer/{tokenizer.json, tokenizer_config.json, chat_template.jinja}

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class Gemma4Engine: ObservableObject {
    @Published var status = "starting…"
    @Published var ready = false
    @Published var busy = false
    @Published var output = ""
    @Published var stats = ""

    // Architecture constants (Gemma 4 E2B).
    private let HID = 1536, PLE_L = 35, PLE_D = 256, VOCAB = 262144
    private let WIN = 512            // sliding-window width W
    private let EOT = 106            // gemma <end_of_turn>
    private let MASK_NEG: Float16 = -1e4   // additive-mask "−inf" (matches export_gemma4_ios.py)
    private var ctx = 2048          // fixed cache capacity — derived from the model's mask input at load

    // The 2.6 GB front-end gather is NOT a Core AI bundle — it's mmap'd (Gemma4Gather) so it stays
    // off dirty-resident memory (iOS OOM fix).
    private var gather: Gemma4Gather!
    private var coreD: InferenceFunctionDescriptor!
    private var headD: InferenceFunctionDescriptor!
    private var coreFn: InferenceFunction!
    private var headFn: InferenceFunction!
    private var stateNames: [String] = []
    // The 4 dual-KV states are NOT stored properties: borrowing `&self.state` into the non-escapable
    // MutableViews escapes self ("lifetime-dependent variable escapes"). They live as function-local
    // vars in `generate` and are passed `inout` to `step`, so the borrow is scoped to the call.

    private var tokenizer: Tokenizer!
    private var eosId = 106
    private var profiled = false  // one-time per-phase memory breakdown on the first step

    private func docs() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }

    // Bytes remaining before iOS jetsam-kills us (the real per-process headroom).
    private func availMB() -> Int { Int(os_proc_available_memory()) / (1024 * 1024) }

    // ---- model loading ----
    private func unit(_ s: String?) -> ComputeUnitKind {
        switch s { case "cpu": return .cpu; case "ane": return .neuralEngine; default: return .gpu }
    }

    func load() async {
        do {
            let env = ProcessInfo.processInfo.environment
            let coreCU = unit(env["GEMMA_CU"] ?? "gpu")       // compute-bound decode core on GPU (or ANE)
            let headCU = unit(env["GEMMA_HEAD_CU"] ?? "gpu")
            let models = docs().appendingPathComponent("models")
            gather = try Gemma4Gather(dir: models.appendingPathComponent("gemma4_gather_raw"))
            let core = try await loadModel(models.appendingPathComponent("gemma4_e2b_ios_static_int8.aimodel"), coreCU)
            let head = try await loadModel(models.appendingPathComponent("gemma4_e2b_int8_head.aimodel"), headCU)
            print("[CoreAIChat] compute units: gather=mmap(cpu) core=\(coreCU) head=\(headCU)")
            guard let coreD = core.functionDescriptor(for: "main"),
                  let headD = head.functionDescriptor(for: "main"),
                  let coreFn = try core.loadFunction(named: "main"),
                  let headFn = try head.loadFunction(named: "main")
            else { status = "failed to load functions"; return }
            self.coreD = coreD; self.headD = headD
            self.coreFn = coreFn; self.headFn = headFn
            self.stateNames = coreD.stateNames
            // Fixed cache capacity = the width of the causal_mask_full input ([1,1,1,ctx]).
            if case .ndArray(let nd)? = coreD.inputDescriptor(of: "causal_mask_full") {
                self.ctx = nd.shape.last.map { $0 < 0 ? 2048 : $0 } ?? 2048
            }
            print("[CoreAIChat] ctx=\(ctx) states=\(stateNames)")

            let tokDir = docs().appendingPathComponent("tokenizer")
            tokenizer = try await AutoTokenizer.from(modelFolder: tokDir, strict: false)
            eosId = tokenizer.eosTokenId ?? EOT

            ready = true
            status = "ready"
            print("[CoreAIChat] loaded — \(stateNames.count) states; ready")
            print("[mem] after load: \(availMB()) MB headroom before jetsam")
        } catch {
            status = "load error: \(error)"
            print("[CoreAIChat] \(status)")
        }
    }

    // Headless self-test: if GEMMA_PROMPT is set, generate it and print result + tok/s to console.
    func autoTestIfRequested() async {
        guard ready, let p = ProcessInfo.processInfo.environment["GEMMA_PROMPT"] else { return }
        print("[CoreAIChat] autotest prompt: \(p)")
        await generate(p, maxNew: 48)
        print("[CoreAIChat] OUT >>> \(output)")
        print("[CoreAIChat] \(stats)")
    }

    private func loadModel(_ url: URL, _ cu: ComputeUnitKind) async throws -> AIModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "Gemma4", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "missing \(url.lastPathComponent) in Documents/models"])
        }
        var o = SpecializationOptions(preferredComputeUnitKind: cu)
        // FIXED-shape graph: shapes are constant every step, so DON'T hint frequent reshapes (that
        // was for the old dynamic core) — it forces a slower general execution path on the static graph.
        o.expectFrequentReshapes = false
        return try await AIModel(contentsOf: url, options: o)
    }

    // ---- NDArray helpers (mirror CoreAIShared usage in the CLI) ----
    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], kind: String) -> NDArray {
        let io = kind == "input" ? d.inputDescriptor(of: name)
            : kind == "state" ? d.stateDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
    }

    // in_step is a 0-D int32 scalar — alloc to the descriptor's own (static) shape.
    private func allocInStep() -> NDArray {
        guard case .ndArray(let nd)? = coreD.inputDescriptor(of: "in_step") else { fatalError("in_step") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(nd.shape.map { $0 < 0 ? 1 : $0 }))
    }

    private func zeroF16(_ arr: inout NDArray) {
        let n = arr.shape.reduce(1, *)
        fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0, count: n))
    }

    private func stateShape(_ name: String) -> [Int] {
        guard case .ndArray(let nd)? = coreD.stateDescriptor(of: name) else { fatalError("state \(name)") }
        return nd.shape.map { $0 < 0 ? ctx : $0 }
    }

    // Additive full causal mask [1,1,1,ctx]: 0 for cache columns [0, pos], MASK_NEG for unwritten/future.
    private func maskFull(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: MASK_NEG, count: ctx)
        let p = min(max(pos, 0), ctx - 1)
        for j in 0...p { m[j] = 0 }
        return m
    }

    // Additive sliding mask [1,1,1,ctx] for the LINEAR sliding cache: 0 for (pos-W, pos], MASK_NEG else.
    private func maskSliding(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: MASK_NEG, count: ctx)
        let p = min(max(pos, 0), ctx - 1)
        let lo = max(0, p - WIN + 1)
        for j in lo...p { m[j] = 0 }
        return m
    }

    private func argmaxLast(_ flat: [Float], rows: Int, width: Int) -> Int {
        let base = (rows - 1) * width
        var best = 0; var bv = flat[base]
        for j in 1..<width { let v = flat[base + j]; if v > bv { bv = v; best = j } }
        return best
    }

    // One forward (gather -> static core -> head) for a single token at absolute position `pos`.
    // The 4 KV states are inout (owned by the caller) so the MutableViews borrow stays in-scope.
    private func step(_ tok: Int32, pos: Int,
                      _ s0: inout NDArray, _ s1: inout NDArray,
                      _ s2: inout NDArray, _ s3: inout NDArray) async throws -> Int {
        // mmap front-end gather -> inputs_embeds / per_layer_inputs (fed straight into the core).
        let g = gather.gather([tok])
        if !profiled { print("[mem] after gather: \(availMB()) MB") }
        var ie = alloc(coreD, "inputs_embeds", [1, 1, HID], kind: "input")
        fillNDArray(&ie, as: Float16.self, with: g.ie.map { Float16($0) })
        var ple = alloc(coreD, "per_layer_inputs", [1, 1, PLE_L, PLE_D], kind: "input")
        fillNDArray(&ple, as: Float16.self, with: g.pli.map { Float16($0) })

        var posA = alloc(coreD, "position_ids", [1, 1], kind: "input")
        fillNDArray(&posA, as: Int32.self, with: [Int32(pos)])
        var inStep = allocInStep()
        fillNDArray(&inStep, as: Int32.self, with: [Int32(pos)])
        var mFull = alloc(coreD, "causal_mask_full", [1, 1, 1, ctx], kind: "input")
        fillNDArray(&mFull, as: Float16.self, with: maskFull(pos))
        var mSlide = alloc(coreD, "causal_mask_sliding", [1, 1, 1, ctx], kind: "input")
        fillNDArray(&mSlide, as: Float16.self, with: maskSliding(pos))

        var hid = alloc(coreD, "hidden", [1, 1, HID], kind: "output")
        var st = InferenceFunction.MutableViews()
        st.insert(&s0, for: stateNames[0])
        st.insert(&s1, for: stateNames[1])
        st.insert(&s2, for: stateNames[2])
        st.insert(&s3, for: stateNames[3])
        var ho = InferenceFunction.MutableViews(); ho.insert(&hid, for: "hidden")
        _ = try await coreFn.run(
            inputs: ["inputs_embeds": ie, "per_layer_inputs": ple, "position_ids": posA,
                     "in_step": inStep, "causal_mask_full": mFull, "causal_mask_sliding": mSlide],
            states: consume st, outputViews: consume ho)
        if !profiled { print("[mem] after core: \(availMB()) MB") }

        let hflat = flattenAsFloat(hid)
        var hin = alloc(headD, "hidden", [1, 1, HID], kind: "input"); fillNDArray(&hin, as: Float.self, with: hflat)
        var lg = alloc(headD, "logits", [1, 1, VOCAB], kind: "output")
        var lo = InferenceFunction.MutableViews(); lo.insert(&lg, for: "logits")
        _ = try await headFn.run(inputs: ["hidden": hin], states: InferenceFunction.MutableViews(), outputViews: consume lo)
        if !profiled { print("[mem] after head: \(availMB()) MB"); profiled = true }
        // Per-step headroom — proves FIXED-shape memory is FLAT across steps (no per-step growth,
        // the whole point of the static port; the dynamic core grew ~200 MB/step -> jetsam).
        print("[mem] step pos=\(pos): \(availMB()) MB free")
        return argmaxLast(flattenAsFloat(lg), rows: 1, width: VOCAB)
    }

    // ---- generation: chat template -> fixed-shape q1 prefill + decode -> streamed text ----
    func generate(_ prompt: String, maxNew: Int = 128) async {
        guard ready, !busy else { return }
        busy = true; output = ""; stats = ""
        defer { busy = false }
        do {
            let messages: [Message] = [["role": "user", "content": prompt]]
            let promptIds = (try tokenizer.applyChatTemplate(messages: messages)).map { Int32($0) }

            // 4 dual-KV states, function-local + zeroed (inout to step keeps the borrow in-scope).
            var s0 = alloc(coreD, stateNames[0], stateShape(stateNames[0]), kind: "state"); zeroF16(&s0)
            var s1 = alloc(coreD, stateNames[1], stateShape(stateNames[1]), kind: "state"); zeroF16(&s1)
            var s2 = alloc(coreD, stateNames[2], stateShape(stateNames[2]), kind: "state"); zeroF16(&s2)
            var s3 = alloc(coreD, stateNames[3], stateShape(stateNames[3]), kind: "state"); zeroF16(&s3)

            let tPre = Date()
            var last = 0
            // Prefill: feed the prompt one token at a time at absolute positions 0..M-1 (in_step=pos).
            for (pos, t) in promptIds.enumerated() {
                last = try await step(t, pos: pos, &s0, &s1, &s2, &s3)
            }
            let prefillSec = -tPre.timeIntervalSinceNow

            var gen: [Int] = []
            var pos = promptIds.count
            let tDec = Date()
            output = tokenizer.decode(tokens: [last], skipSpecialTokens: true)
            if last != eosId && last != EOT { gen.append(last) }
            while gen.count < maxNew, last != eosId, last != EOT {
                last = try await step(Int32(last), pos: pos, &s0, &s1, &s2, &s3)
                pos += 1
                if last == eosId || last == EOT { break }
                gen.append(last)
                output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)  // live stream
            }
            let decodeSec = -tDec.timeIntervalSinceNow
            stats = String(format: "prefill %d tok · %.1f tok/s   |   decode %d tok · %.1f tok/s",
                           promptIds.count, Double(promptIds.count) / max(prefillSec, 1e-6),
                           gen.count, Double(gen.count) / max(decodeSec, 1e-6))
        } catch {
            output = "generation error: \(error)"
        }
    }
}
