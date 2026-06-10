// Gemma 4 E2B on-device GPU MONOLITH engine — the fast device-GPU path (fused-int8 Metal-kernel FFN).
//
// The ANE chunked path is capped by the Core AI beta (the in-graph KV-write SIGSEGV forces host-cache +
// per-token host round-trips; ~6 tok/s). The GPU path has no such limit: the full 35-layer HOST-CACHE
// MONOLITH runs as ONE graph (1 dispatch, no chunk overhead), and Session B's custom **fused-int8 Metal
// kernel FFN** (`coreai_torch.TorchMetalKernel`, MSL embedded in the .aimodel — 100% Core AI) lowers + runs
// on the iPhone GPU at ~2.9× the plain int8 MPSGraph monolith (56 ms vs 163 ms/step core).
//
//   input_ids --mmap gather--> inputs_embeds + per_layer_inputs
//     --metal-kernel host-cache monolith (1 dispatch)--> hidden + the 12 sliding / 3 full *_cur columns
//     --argmax head--> token ; the HOST writes the *_cur columns into the host-owned KV caches at `pos`.
//
// Numerically identical to `Gemma4DecodeHostCache` (Session B, 8/8 on Mac GPU). GPU-only.
//
// `Gemma4MonolithBackend` is the reusable engine (the UI's GPU mode — see Gemma4ChatEngine).
// `Gemma4MonolithEngine.run()` is the headless harness, triggered by GEMMA_MONO_TEST=1:
//   GEMMA_MONO_MODEL   = core under Documents/models (default gemma4_e2b_metal_int4km_L35.aimodel,
//                        the published release core; int8 = dev override)
//   GEMMA_MONO_HEAD    = head under Documents/models (default gemma4_e2b_head_argmax_int4km.aimodel;
//                        same partials contract — e.g. the int8 kernel head)
//   GEMMA_MONO_CU      = gpu (default) | cpu  (the kernel is GPU-only; do NOT use ane)
//   GEMMA_MONO_VERIFY  = 1 -> drive _gen_ref prompt_ids, compare to ref_ids (device 8/8)
//   GEMMA_PROMPT / GEMMA_MONO_N = generate + tok/s
//   GEMMA_MONO_CLEARCACHE = 1 -> delete Library/Caches/coreai-cache BEFORE loading (true-cold load
//     measurement; also the RECOVERY for a poisoned specialization cache — a load attempt against a
//     partially-pushed .aimodel poisons the content-keyed cache entry and later loads ENOENT)

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class Gemma4MonolithBackend: Gemma4Backend {
    static let HID = 1536, PLE_L = 35, PLE_D = 256, VOCAB = 262144
    static let HS = 256, HF = 512, NS = 12, NF = 3, WIN = 512
    static let MASK_NEG: Float16 = -1e4

    let modeLabel = "GPU"
    private(set) var ctx = 512
    private(set) var headKind = "?"
    let coreName: String

    private var gather: Gemma4Gather!
    private var coreModel: AIModel!            // keep the models alive alongside their functions
    private var headModel: AIModel?
    private var coreD: InferenceFunctionDescriptor!
    private var coreFn: InferenceFunction!
    private var headD: InferenceFunctionDescriptor?
    private var headFn: InferenceFunction?
    private var fusedHead = false, kernelHead = false, useArgmax = false

    // Host-owned KV caches: sliding [12,1,1,ctx,256], full [3,1,1,ctx,512].
    private var slidingK = [Float16](), slidingV = [Float16]()
    private var fullK = [Float16](), fullV = [Float16]()
    private(set) var tCore = 0.0, tHead = 0.0, nP = 0

    init() {
        // Default = the published release core (int4-k-means custom-kernel monolith, the zoo's
        // 17.7 tok/s config — the set hosted on the Hugging Face repo). int8 stays the dev override.
        coreName = ProcessInfo.processInfo.environment["GEMMA_MONO_MODEL"] ?? "gemma4_e2b_metal_int4km_L35.aimodel"
    }

    private func docs() -> URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first! }
    // "ane" is for the AOT un-chunk experiment (plain-MPSGraph hostcache cores ONLY — the
    // metal-kernel cores are GPU-only).
    private static func unit(_ s: String?) -> ComputeUnitKind {
        switch s { case "cpu": return .cpu; case "ane": return .neuralEngine; default: return .gpu }
    }

    func load() async throws {
        let env = ProcessInfo.processInfo.environment
        let cu = Self.unit(env["GEMMA_MONO_CU"] ?? "gpu")
        let models = docs().appendingPathComponent("models")
        gather = try Gemma4Gather(dir: models.appendingPathComponent("gemma4_gather_raw"))
        var o = SpecializationOptions(preferredComputeUnitKind: cu); o.expectFrequentReshapes = false
        let tLoad = Date()
        let core = try await AIModel(contentsOf: models.appendingPathComponent(coreName), options: o)
        print(String(format: "[mono] core load+specialize %.1fs (%@)", -tLoad.timeIntervalSinceNow, coreName))
        guard let coreD = core.functionDescriptor(for: "main"), let coreFn = try core.loadFunction(named: "main")
        else { throw NSError(domain: "Gemma4Mono", code: 1, userInfo: [NSLocalizedDescriptionKey: "no core fn in \(coreName)"]) }
        self.coreModel = core; self.coreD = coreD; self.coreFn = coreFn
        if case .ndArray(let nd)? = coreD.inputDescriptor(of: "causal_mask_full") {
            ctx = (nd.shape.last.map { $0 < 0 ? 512 : $0 } ?? 512) - 1
        }
        // FUSED core: lm_head+argmax are IN the core graph (outputs token_id) → NO separate head dispatch.
        fusedHead = coreD.outputNames.contains("token_id")
        // GPU head priority (loaded only if the core is NOT fused):
        //   (1) fused-int8 head+ARGMAX KERNEL: hidden -> (value,index) partials, host reduces to the token.
        //       NO 262K-logit readback, NO MPSGraph argmax (both prior dead-ends) -> kills the ~66ms head.
        //   (2) MPSGraph argmax head (token_id) ; (3) plain logits head (262K readback + host argmax).
        if !fusedHead {
            let kernelURL = models.appendingPathComponent(
                env["GEMMA_MONO_HEAD"] ?? "gemma4_e2b_head_argmax_int4km.aimodel")
            let argURL = models.appendingPathComponent("gemma4_e2b_head_argmax_int8.aimodel")
            kernelHead = FileManager.default.fileExists(atPath: kernelURL.path)
            useArgmax = !kernelHead && FileManager.default.fileExists(atPath: argURL.path)
            let headURL = kernelHead ? kernelURL : (useArgmax ? argURL : models.appendingPathComponent("gemma4_e2b_int8_head.aimodel"))
            var ho = SpecializationOptions(preferredComputeUnitKind: .gpu); ho.expectFrequentReshapes = false
            let tHeadLoad = Date()
            let head = try await AIModel(contentsOf: headURL, options: ho)
            print(String(format: "[mono] head load+specialize %.1fs (%@)", -tHeadLoad.timeIntervalSinceNow, headURL.lastPathComponent))
            guard let headD = head.functionDescriptor(for: "main"), let headFn = try head.loadFunction(named: "main")
            else { throw NSError(domain: "Gemma4Mono", code: 2, userInfo: [NSLocalizedDescriptionKey: "no head fn"]) }
            self.headModel = head; self.headD = headD; self.headFn = headFn
        }
        headKind = fusedHead ? "fused" : (kernelHead ? "kernel+argmax" : (useArgmax ? "mpsgraph-argmax" : "logits"))
        reset()
    }

    func reset() {
        slidingK = [Float16](repeating: 0, count: Self.NS * ctx * Self.HS)
        slidingV = [Float16](repeating: 0, count: Self.NS * ctx * Self.HS)
        fullK = [Float16](repeating: 0, count: Self.NF * ctx * Self.HF)
        fullV = [Float16](repeating: 0, count: Self.NF * ctx * Self.HF)
        tCore = 0; tHead = 0; nP = 0
    }

    func profileSummary() -> String {
        guard nP > 0 else { return "no steps" }
        return String(format: "core=%.0fms head=%.0fms", tCore / Double(nP) * 1000, tHead / Double(nP) * 1000)
    }

    // One step: gather -> monolith core (1 dispatch) -> head -> token; host KV write-back at `pos`.
    // needToken=false skips the separate head dispatch (prefill positions whose logits nobody reads);
    // the fused-head core emits token_id in-graph, so there is nothing to skip there.
    func step(_ tok: Int32, _ pos: Int, needToken: Bool = true) async throws -> Int {
        let g = gather.gather([tok])
        let inputs: [String: NDArray] = [
            "inputs_embeds": fill(coreD, "inputs_embeds", [1, 1, Self.HID], g.ie.map { Float16($0) }),
            "per_layer_inputs": fill(coreD, "per_layer_inputs", [1, 1, Self.PLE_L, Self.PLE_D], g.pli.map { Float16($0) }),
            "position_ids": int32(coreD, "position_ids", pos),
            "causal_mask_full": fill(coreD, "causal_mask_full", [1, 1, 1, ctx + 1], maskFull(pos)),
            "causal_mask_sliding": fill(coreD, "causal_mask_sliding", [1, 1, 1, ctx + 1], maskSliding(pos)),
            "sliding_k": fill(coreD, "sliding_k", [Self.NS, 1, 1, ctx, Self.HS], slidingK),
            "sliding_v": fill(coreD, "sliding_v", [Self.NS, 1, 1, ctx, Self.HS], slidingV),
            "full_k": fill(coreD, "full_k", [Self.NF, 1, 1, ctx, Self.HF], fullK),
            "full_v": fill(coreD, "full_v", [Self.NF, 1, 1, ctx, Self.HF], fullV),
        ]
        let tc = Date()
        var out = try await coreFn.run(inputs: inputs)
        tCore += -tc.timeIntervalSinceNow
        var next = -1
        var hidden = [Float]()
        if fusedHead {  // token_id comes straight out of the core graph (no separate head dispatch)
            if let tv = out.remove("token_id"), let tnd = tv.ndArray { next = Int(readInt32(tnd)) }
        } else if needToken {
            guard let hv = out.remove("hidden"), let hnd = hv.ndArray else {
                throw NSError(domain: "Gemma4Mono", code: 3, userInfo: [NSLocalizedDescriptionKey: "no hidden output"])
            }
            hidden = flattenAsFloat(hnd)
        }
        // host write-back: cur columns -> caches at `pos`.
        if let v = out.remove("sliding_k_cur"), let nd = v.ndArray { writeCur(&slidingK, flattenAsFloat(nd), Self.NS, Self.HS, pos) }
        if let v = out.remove("sliding_v_cur"), let nd = v.ndArray { writeCur(&slidingV, flattenAsFloat(nd), Self.NS, Self.HS, pos) }
        if let v = out.remove("full_k_cur"), let nd = v.ndArray { writeCur(&fullK, flattenAsFloat(nd), Self.NF, Self.HF, pos) }
        if let v = out.remove("full_v_cur"), let nd = v.ndArray { writeCur(&fullV, flattenAsFloat(nd), Self.NF, Self.HF, pos) }

        if !fusedHead, needToken, let headD, let headFn {  // separate head dispatch
            let thd = Date()
            if kernelHead {  // fused-int8 head+argmax kernel: hidden(fp16) -> (values,indices) partials
                let hin = fill(headD, "hidden", [1, 1, Self.HID], hidden.map { Float16($0) })
                var ho2 = try await headFn.run(inputs: ["hidden": hin])
                if let pvv = ho2.remove("partial_values"), let pvnd = pvv.ndArray,
                   let piv = ho2.remove("partial_indices"), let pind = piv.ndArray {
                    next = reducePartials(pvnd, pind)  // token = indices[argmax(values)]
                }
            } else if useArgmax {
                let hin = fill(headD, "hidden", [1, 1, Self.HID], hidden.map { Float16($0) }, asFloat: true)
                var ho2 = try await headFn.run(inputs: ["hidden": hin])
                if let tv = ho2.remove("token_id"), let tnd = tv.ndArray { next = Int(readInt32(tnd)) }
            } else {
                let hin = fill(headD, "hidden", [1, 1, Self.HID], hidden.map { Float16($0) }, asFloat: true)
                var lg = alloc(headD, "logits", [1, 1, Self.VOCAB], "output")
                var lo = InferenceFunction.MutableViews(); lo.insert(&lg, for: "logits")
                _ = try await headFn.run(inputs: ["hidden": hin], states: InferenceFunction.MutableViews(), outputViews: consume lo)
                next = argmaxLast(flattenAsFloat(lg), width: Self.VOCAB)
            }
            tHead += -thd.timeIntervalSinceNow
        }
        nP += 1
        return next
    }

    // ---- helpers ----
    private func writeCur(_ cache: inout [Float16], _ cur: [Float], _ nSlots: Int, _ hd: Int, _ pos: Int) {
        // cur: [nSlots,1,1,1,hd] flat = nSlots*hd ; write slot s's column to cache[s*ctx*hd + pos*hd ..]
        for s in 0..<nSlots { let dst = s * ctx * hd + pos * hd, src = s * hd
            for d in 0..<hd { cache[dst + d] = Float16(cur[src + d]) } }
    }
    private func maskFull(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: Self.MASK_NEG, count: ctx + 1); let p = min(max(pos, 0), ctx - 1)
        for j in 0..<p { m[j] = 0 }; m[ctx] = 0; return m
    }
    private func maskSliding(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: Self.MASK_NEG, count: ctx + 1); let p = min(max(pos, 0), ctx - 1)
        let lo = max(0, p - Self.WIN + 1); if lo < p { for j in lo..<p { m[j] = 0 } }; m[ctx] = 0; return m
    }
    private func readInt32(_ a: NDArray) -> Int32 { var v: Int32 = 0; a.view(as: Int32.self).withUnsafePointer { ptr, _, _ in v = ptr[0] }; return v }
    // GPU head+argmax kernel returns per-threadgroup partials; host picks token = indices[argmax(values)].
    // values are fp32 (exact to the matvec); reading ~32k floats+ints is <0.1ms vs a 262K-logit readback.
    private func reducePartials(_ pv: NDArray, _ pi: NDArray) -> Int {
        let vals = flattenAsFloat(pv)
        guard !vals.isEmpty else { return -1 }
        var idx = [Int32](repeating: 0, count: vals.count)
        pi.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            for j in 0..<idx.count { idx[j] = ptr[j] }  // partials are contiguous 1-D [num_tg]
        }
        var best = 0, bv = vals[0]
        for j in 1..<vals.count where vals[j] > bv { bv = vals[j]; best = j }
        return Int(idx[best])
    }
    private func argmaxLast(_ flat: [Float], width: Int) -> Int {
        let base = flat.count - width; var best = 0, bv = flat[base]
        for j in 1..<width { let v = flat[base + j]; if v > bv { bv = v; best = j } }; return best
    }
    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ kind: String) -> NDArray {
        let io = kind == "input" ? d.inputDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
    }
    private func fill(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ data: [Float16], asFloat: Bool = false) -> NDArray {
        var a = alloc(d, name, shape, "input")
        if asFloat { fillNDArray(&a, as: Float.self, with: data.map { Float($0) }) } else { fillNDArray(&a, as: Float16.self, with: data) }
        return a
    }
    private func int32(_ d: InferenceFunctionDescriptor, _ name: String, _ v: Int) -> NDArray {
        var a = alloc(d, name, [1, 1], "input"); fillNDArray(&a, as: Int32.self, with: [Int32(v)]); return a
    }
}

// Headless harness (GEMMA_MONO_TEST=1) — same backend the UI's GPU mode uses.
@MainActor
enum Gemma4MonolithEngine {
    private static func docs() -> URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first! }
    private static func availMB() -> Int { Int(os_proc_available_memory()) / (1024 * 1024) }

    static func run() async {
        let env = ProcessInfo.processInfo.environment
        if env["GEMMA_MONO_CLEARCACHE"] == "1" {  // true-cold loads / poisoned-cache recovery
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            let cc = caches.appendingPathComponent("coreai-cache")
            do { try FileManager.default.removeItem(at: cc); print("[mono] CLEARED \(cc.lastPathComponent)") }
            catch { print("[mono] clear-cache: \(error.localizedDescription)") }
        }
        let be = Gemma4MonolithBackend()
        print("[mono] ====== gemma4 GPU MONOLITH engine (core=\(be.coreName)) ======  avail \(availMB()) MB")
        do {
            try await be.load()
            print("[mono] fusedHead=\(be.headKind == "fused") (separate head dispatch \(be.headKind == "fused" ? "ELIMINATED" : "used"))")
            print("[mono] loaded core + head=\(be.headKind); ctx=\(be.ctx); avail \(availMB()) MB")

            if env["GEMMA_MONO_VERIFY"] == "1" {
                let prompt: [Int32] = [2, 105, 2364, 107, 4377, 699, 886, 531, 3595, 236764, 15914, 684, 162760, 236761, 106, 107, 105, 4368, 107]
                let ref = [4906, 236764, 1156, 236764, 1806, 236764, 2390, 236764]
                var last = 0
                for (pos, t) in prompt.enumerated() { last = try await be.step(t, pos) }
                var gen = [last]
                for i in 0..<(ref.count - 1) { last = try await be.step(Int32(last), prompt.count + i); gen.append(last) }
                let n = zip(gen, ref).filter { $0 == $1 }.count
                print("[mono] decode = \(gen)\n[mono] ref    = \(ref)\n[mono] match = \(n)/\(ref.count) -> \(n == ref.count ? "PASS ✅ EXACT on GPU" : "DEGRADED ⚠️")")
                return
            }

            let tok = try await AutoTokenizer.from(modelFolder: docs().appendingPathComponent("tokenizer"), strict: false)
            let eos = tok.eosTokenId ?? 106
            let ids = (try tok.applyChatTemplate(messages: [["role": "user", "content": env["GEMMA_PROMPT"] ?? "What is the capital of France?"]])).map { Int32($0) }
            let maxNew = Int(env["GEMMA_MONO_N"] ?? "24") ?? 24
            let tPre = Date(); var last = 0
            for (pos, t) in ids.enumerated() { last = try await be.step(t, pos) }
            let preSec = -tPre.timeIntervalSinceNow
            var gen = [last]; var pos = ids.count; let tDec = Date()
            while gen.count < maxNew, last != eos, last != 106, pos < be.ctx {
                last = try await be.step(Int32(last), pos); pos += 1
                if last == eos || last == 106 { break }; gen.append(last)
            }
            let decSec = -tDec.timeIntervalSinceNow
            print("[mono] OUT >>> \(tok.decode(tokens: gen, skipSpecialTokens: true))")
            print(String(format: "[mono] prefill %d %.1f tok/s | decode %d %.1f tok/s | %d MB free", ids.count, Double(ids.count) / max(preSec, 1e-6), gen.count, Double(gen.count) / max(decSec, 1e-6), availMB()))
            print("[profile] per-token: \(be.profileSummary())")
        } catch { print("[mono] ERROR: \(error)") }
    }
}
