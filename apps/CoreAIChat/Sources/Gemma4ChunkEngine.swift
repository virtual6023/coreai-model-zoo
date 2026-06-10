// Gemma 4 E2B on-device CHUNKED HOST-CACHE engine — the ANE decode path (Session A).
//
// The de-risk + chunk probe proved all 35 layers run on the iPhone ANE in 6 host-cache chunks
// (no Core AI state, no in-graph indexed write). This engine chains them per step, exactly mirroring the
// proven Python `run_step` / `host_write` (ondevice/export_gemma4_hostcache_chunks.py, 8/8 EXACT on Mac GPU):
//
//   input_ids --mmap gather--> inputs_embeds + per_layer_inputs
//     --chunk1..6 (hidden flows; producer L13/L14 K/V flow chunk2 -> chunks 3-6)--> hidden
//     --head--> logits --> argmax ; HOST writes each chunk's *_cur columns into the host-owned KV caches
//
// The KV caches are HOST-OWNED `[Float16]` (NOT Core AI state): sliding [12,1,1,ctx,256], full [3,1,1,ctx,512].
// Each non-shared layer owns a (contiguous) slot range; the producers (L13 sliding slot 11 / L14 full slot 2,
// both in chunk2) feed their current K/V to the stateless consumer chunks 3-6. Topology mirrors
// `gemma4_chunks_plan.json` (kept hardcoded here for a dependency-free device self-test).
//
// `Gemma4ChunkBackend` is the reusable engine (the UI's ANE mode — see Gemma4ChatEngine).
// `Gemma4ChunkEngine.run()` is the headless harness, triggered by GEMMA_CHUNK_TEST=1:
//   GEMMA_CHUNK_CU       = ane (default) | gpu | cpu   (the 6 chunk graphs)
//   GEMMA_CHUNK_HEAD_CU  = ane (default) | gpu | cpu
//   GEMMA_CHUNK_VERIFY   = 1 -> drive the _gen_ref.json prompt_ids, greedy-decode, compare to ref_ids (8/8)
//   GEMMA_PROMPT         = free-text prompt (else the verify/default prompt); GEMMA_CHUNK_N = max new tokens
// Chunk artifacts: Documents/models/gemma4_e2b_hostcache_chunkN_int8.aimodel (push via devicectl copy).

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class Gemma4ChunkBackend: Gemma4Backend {
    // ---- architecture / topology constants (mirror gemma4_chunks_plan.json, E2B --max-layers 8) ----
    static let HID = 1536, PLE_L = 35, PLE_D = 256, VOCAB = 262144
    static let HS = 256, HF = 512        // sliding / full head_dim (nkv = 1)
    static let NS = 12, NF = 3           // sliding / full cache slots
    static let WIN = 512                 // sliding window
    static let MASK_NEG: Float16 = -1e4
    static let PROD_S_SLOT = 11, PROD_F_SLOT = 2          // producer slots in the global caches
    static let PROD_CHUNK = 1                              // chunk2 owns both producers
    static let PROD_S_LIDX = 4, PROD_F_LIDX = 1           // producer column index within chunk2's *_cur

    struct ChunkSpec {
        let name: String, start: Int, end: Int
        let ownSliding: [Int], ownFull: [Int]             // GLOBAL slots (contiguous), in layer order
        // KV-shared consumer: the host pre-concatenates [producer slot history ++ producer cur] into one
        // prod_<type>_kv input (NO in-graph cat — the device-ANE fix; MPSGraph miscompiles cat of 2 inputs).
        let extSlidingKv: Bool, extFullKv: Bool
    }
    // E2B 6-chunk split [(0,8),(8,15),(15,20),(20,25),(25,30),(30,35)].
    static let CHUNKS: [ChunkSpec] = [
        .init(name: "chunk1", start: 0, end: 8, ownSliding: [0, 1, 2, 3, 4, 5, 6], ownFull: [0],
              extSlidingKv: false, extFullKv: false),
        .init(name: "chunk2", start: 8, end: 15, ownSliding: [7, 8, 9, 10, 11], ownFull: [1, 2],
              extSlidingKv: false, extFullKv: false),
        .init(name: "chunk3", start: 15, end: 20, ownSliding: [], ownFull: [], extSlidingKv: true, extFullKv: true),
        .init(name: "chunk4", start: 20, end: 25, ownSliding: [], ownFull: [], extSlidingKv: true, extFullKv: true),
        .init(name: "chunk5", start: 25, end: 30, ownSliding: [], ownFull: [], extSlidingKv: true, extFullKv: true),
        .init(name: "chunk6", start: 30, end: 35, ownSliding: [], ownFull: [], extSlidingKv: true, extFullKv: true),
    ]

    let modeLabel = "ANE"
    private(set) var ctx = 64           // bucket — read from chunk1's causal_mask_full width (= bucket + 1)
    private(set) var headKind = "?"

    private var gather: Gemma4Gather!
    private var chunks: [(AIModel, InferenceFunction, InferenceFunctionDescriptor)] = []
    private var headModel: AIModel!
    private var headD: InferenceFunctionDescriptor!
    private var headFn: InferenceFunction!
    private var useArgmax = false
    private let dump = ProcessInfo.processInfo.environment["GEMMA_CHUNK_DUMP"] == "1"

    // Host-owned KV caches (zeroed). nkv = 1, so [slots, ctx, head_dim] flattened.
    private var slidingK = [Float16](), slidingV = [Float16]()
    private var fullK = [Float16](), fullV = [Float16]()

    // Per-phase timing (ms), accumulated over steps for the breakdown.
    private var tChunk = [Double](repeating: 0, count: 16)
    private(set) var tHead = 0.0, nProf = 0

    private func docs() -> URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first! }
    private static func unit(_ s: String?) -> ComputeUnitKind {
        switch s { case "cpu": return .cpu; case "gpu": return .gpu; default: return .neuralEngine }
    }

    func load() async throws {
        let env = ProcessInfo.processInfo.environment
        let cu = Self.unit(env["GEMMA_CHUNK_CU"] ?? "ane")
        let headCU = Self.unit(env["GEMMA_CHUNK_HEAD_CU"] ?? "ane")   // ANE head avoids the per-token GPU↔ANE switch
        let quant = env["GEMMA_CHUNK_QUANT"] ?? "int8"   // int8 | fp16 (for the ANE numeric A/B)
        let models = docs().appendingPathComponent("models")
        gather = try Gemma4Gather(dir: models.appendingPathComponent("gemma4_gather_raw"))
        chunks = []
        let tLoad = Date()
        for c in Self.CHUNKS {
            let url = models.appendingPathComponent("gemma4_e2b_hostcache_\(c.name)_\(quant).aimodel")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw NSError(domain: "Gemma4Chunk", code: 1, userInfo: [NSLocalizedDescriptionKey: "MISSING \(url.lastPathComponent)"])
            }
            var o = SpecializationOptions(preferredComputeUnitKind: cu); o.expectFrequentReshapes = false
            let m = try await AIModel(contentsOf: url, options: o)
            guard let d = m.functionDescriptor(for: "main"), let fn = try m.loadFunction(named: "main")
            else { throw NSError(domain: "Gemma4Chunk", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(c.name): no main fn"]) }
            chunks.append((m, fn, d))
        }
        print(String(format: "[chunkeng] 6 chunks load+specialize %.1fs (%@)", -tLoad.timeIntervalSinceNow, quant))
        // ctx = bucket from chunk1's causal_mask_full input width (bucket + 1).
        if case .ndArray(let nd)? = chunks[0].2.inputDescriptor(of: "causal_mask_full") {
            ctx = (nd.shape.last.map { $0 < 0 ? 65 : $0 } ?? 65) - 1
        }
        // In-graph argmax head (token_id int32 out) — avoids the 262144-logit readback + CPU argmax
        // (profiled ~109 ms/token). Falls back to the logits head if the argmax artifact is absent.
        // GEMMA_CHUNK_ARGMAX=0 forces the logits head (host argmax).
        let argmaxURL = models.appendingPathComponent("gemma4_e2b_head_argmax_int8.aimodel")
        let logitsURL = models.appendingPathComponent("gemma4_e2b_int8_head.aimodel")
        useArgmax = FileManager.default.fileExists(atPath: argmaxURL.path) && env["GEMMA_CHUNK_ARGMAX"] != "0"
        let headURL = useArgmax ? argmaxURL : logitsURL
        var ho = SpecializationOptions(preferredComputeUnitKind: headCU); ho.expectFrequentReshapes = false
        let head = try await AIModel(contentsOf: headURL, options: ho)
        guard let headD = head.functionDescriptor(for: "main"), let headFn = try head.loadFunction(named: "main")
        else { throw NSError(domain: "Gemma4Chunk", code: 3, userInfo: [NSLocalizedDescriptionKey: "no head fn"]) }
        self.headModel = head; self.headD = headD; self.headFn = headFn
        headKind = useArgmax ? "in-graph-argmax" : "logits"
        reset()
    }

    func reset() {
        slidingK = [Float16](repeating: 0, count: Self.NS * ctx * Self.HS)
        slidingV = [Float16](repeating: 0, count: Self.NS * ctx * Self.HS)
        fullK = [Float16](repeating: 0, count: Self.NF * ctx * Self.HF)
        fullV = [Float16](repeating: 0, count: Self.NF * ctx * Self.HF)
        tChunk = [Double](repeating: 0, count: 16); tHead = 0; nProf = 0
    }

    func profileSummary() -> String {
        guard nProf > 0 else { return "no steps" }
        let n = Double(nProf)
        let per = (0..<Self.CHUNKS.count).map { String(format: "%.0f", tChunk[$0] / n * 1000) }.joined(separator: "/")
        let sum = (0..<Self.CHUNKS.count).reduce(0.0) { $0 + tChunk[$1] } / n * 1000
        return String(format: "chunks[%@]=%.0fms head=%.0fms", per, sum, tHead / n * 1000)
    }

    // One decode step: gather -> chunk1..6 -> head -> argmax ; then host-write *_cur at `pos`.
    // needToken=false skips the head dispatch (prefill positions whose logits nobody reads).
    func step(_ tok: Int32, _ pos: Int, needToken: Bool = true) async throws -> Int {
        let g = gather.gather([tok])
        let ie = g.ie.map { Float16($0) }                 // [1,1,HID]
        let pli = g.pli.map { Float16($0) }               // [1,1,PLE_L,PLE_D]
        let mFull = maskFull(pos), mSlide = maskSliding(pos)

        var hidden = ie
        var prodSK = [Float16](), prodSV = [Float16](), prodFK = [Float16](), prodFV = [Float16]()
        // captured own-slot *_cur for the post-step host-write: (chunkIdx) -> (sk,sv,fk,fv)
        var pending: [(ChunkSpec, [Float16], [Float16], [Float16], [Float16])] = []

        for (ci, c) in Self.CHUNKS.enumerated() {
            let (_, fn, d) = chunks[ci]
            var inputs: [String: NDArray] = [
                "hidden_in": fill(d, "hidden_in", [1, 1, Self.HID], hidden),
                "per_layer_inputs": fill(d, "per_layer_inputs", [1, 1, c.end - c.start, Self.PLE_D],
                                         Array(pli[(c.start * Self.PLE_D)..<(c.end * Self.PLE_D)])),
                "position_ids": int32(d, "position_ids", pos),
            ]
            if d.inputNames.contains("causal_mask_full") { inputs["causal_mask_full"] = fill(d, "causal_mask_full", [1, 1, 1, ctx + 1], mFull) }
            if d.inputNames.contains("causal_mask_sliding") { inputs["causal_mask_sliding"] = fill(d, "causal_mask_sliding", [1, 1, 1, ctx + 1], mSlide) }
            if !c.ownSliding.isEmpty {
                let lo = c.ownSliding[0] * ctx * Self.HS
                inputs["sliding_k"] = fill(d, "sliding_k", [c.ownSliding.count, 1, 1, ctx, Self.HS], Array(slidingK[lo..<(lo + c.ownSliding.count * ctx * Self.HS)]))
                inputs["sliding_v"] = fill(d, "sliding_v", [c.ownSliding.count, 1, 1, ctx, Self.HS], Array(slidingV[lo..<(lo + c.ownSliding.count * ctx * Self.HS)]))
            }
            if !c.ownFull.isEmpty {
                let lo = c.ownFull[0] * ctx * Self.HF
                inputs["full_k"] = fill(d, "full_k", [c.ownFull.count, 1, 1, ctx, Self.HF], Array(fullK[lo..<(lo + c.ownFull.count * ctx * Self.HF)]))
                inputs["full_v"] = fill(d, "full_v", [c.ownFull.count, 1, 1, ctx, Self.HF], Array(fullV[lo..<(lo + c.ownFull.count * ctx * Self.HF)]))
            }
            // KV-shared consumer: host pre-cat [producer slot history (B) ++ producer cur (1)] = B+1,
            // fed as ONE input so the chunk does NO in-graph cat (the device-ANE fix). The slot still
            // holds positions 0..p-1 (cur is host-written into it only after the step); the mask marks
            // 0..p-1 + the appended cur (index B) valid — identical to the in-graph-cat semantics.
            if c.extSlidingKv {
                let lo = Self.PROD_S_SLOT * ctx * Self.HS
                var kk = Array(slidingK[lo..<(lo + ctx * Self.HS)]); kk.append(contentsOf: prodSK)  // [1,1,ctx+1,HS]
                var vv = Array(slidingV[lo..<(lo + ctx * Self.HS)]); vv.append(contentsOf: prodSV)
                inputs["prod_sliding_kv_k"] = fill(d, "prod_sliding_kv_k", [1, 1, ctx + 1, Self.HS], kk)
                inputs["prod_sliding_kv_v"] = fill(d, "prod_sliding_kv_v", [1, 1, ctx + 1, Self.HS], vv)
            }
            if c.extFullKv {
                let lo = Self.PROD_F_SLOT * ctx * Self.HF
                var kk = Array(fullK[lo..<(lo + ctx * Self.HF)]); kk.append(contentsOf: prodFK)  // [1,1,ctx+1,HF]
                var vv = Array(fullV[lo..<(lo + ctx * Self.HF)]); vv.append(contentsOf: prodFV)
                inputs["prod_full_kv_k"] = fill(d, "prod_full_kv_k", [1, 1, ctx + 1, Self.HF], kk)
                inputs["prod_full_kv_v"] = fill(d, "prod_full_kv_v", [1, 1, ctx + 1, Self.HF], vv)
            }

            let tc = Date()
            var out = try await fn.run(inputs: inputs)   // auto-allocate outputs
            tChunk[ci] += -tc.timeIntervalSinceNow
            guard let hv = out.remove("hidden"), let hnd = hv.ndArray else {
                throw NSError(domain: "Gemma4Chunk", code: 4, userInfo: [NSLocalizedDescriptionKey: "\(c.name) no hidden"])
            }
            let hflat = flattenAsFloat(hnd)
            hidden = hflat.map { Float16($0) }
            // Per-chunk hidden checksum on the FIRST token (REAL inputs) — run on ANE and GPU and
            // diff to localise the first chunk whose hidden diverges (GEMMA_CHUNK_DUMP=1).
            if dump && pos == 0 {
                let mean = hflat.reduce(0, +) / Float(max(hflat.count, 1))
                let amax = hflat.map { abs($0) }.max() ?? 0
                print("[chunkdump] \(c.name) hidden mean=\(String(format: "%.5f", mean)) absmax=\(String(format: "%.4f", amax))")
            }

            if !c.ownSliding.isEmpty || !c.ownFull.isEmpty {
                var sk = [Float16](), sv = [Float16](), fk = [Float16](), fv = [Float16]()
                if !c.ownSliding.isEmpty {
                    if let v = out.remove("sliding_k_cur"), let nd = v.ndArray { sk = flattenAsFloat(nd).map { Float16($0) } }
                    if let v = out.remove("sliding_v_cur"), let nd = v.ndArray { sv = flattenAsFloat(nd).map { Float16($0) } }
                }
                if !c.ownFull.isEmpty {
                    if let v = out.remove("full_k_cur"), let nd = v.ndArray { fk = flattenAsFloat(nd).map { Float16($0) } }
                    if let v = out.remove("full_v_cur"), let nd = v.ndArray { fv = flattenAsFloat(nd).map { Float16($0) } }
                }
                if ci == Self.PROD_CHUNK {  // chunk2 owns both producers -> capture their current columns
                    prodSK = Array(sk[(Self.PROD_S_LIDX * Self.HS)..<((Self.PROD_S_LIDX + 1) * Self.HS)])
                    prodSV = Array(sv[(Self.PROD_S_LIDX * Self.HS)..<((Self.PROD_S_LIDX + 1) * Self.HS)])
                    prodFK = Array(fk[(Self.PROD_F_LIDX * Self.HF)..<((Self.PROD_F_LIDX + 1) * Self.HF)])
                    prodFV = Array(fv[(Self.PROD_F_LIDX * Self.HF)..<((Self.PROD_F_LIDX + 1) * Self.HF)])
                }
                pending.append((c, sk, sv, fk, fv))
            }
        }

        // Head -> token. In-graph argmax head returns token_id (1 int, no 262K readback); the
        // logits-head fallback returns 262144 logits + host argmax.
        var next = -1
        if needToken {
            let thd = Date()
            let hin = fill(headD, "hidden", [1, 1, Self.HID], hidden, asFloat: true)
            if useArgmax {
                var out = try await headFn.run(inputs: ["hidden": hin])
                if let tv = out.remove("token_id"), let tnd = tv.ndArray { next = Int(readInt32(tnd)) }
            } else {
                var lg = alloc(headD, "logits", [1, 1, Self.VOCAB], "output")
                var lo = InferenceFunction.MutableViews(); lo.insert(&lg, for: "logits")
                _ = try await headFn.run(inputs: ["hidden": hin], states: InferenceFunction.MutableViews(), outputViews: consume lo)
                next = argmaxLast(flattenAsFloat(lg), width: Self.VOCAB)
            }
            tHead += -thd.timeIntervalSinceNow
        }
        nProf += 1

        // HOST write-back: each owned slot's current column -> global cache at position `pos`.
        for (c, sk, sv, fk, fv) in pending {
            for (j, gslot) in c.ownSliding.enumerated() {
                let dst = gslot * ctx * Self.HS + pos * Self.HS, src = j * Self.HS
                for d in 0..<Self.HS { slidingK[dst + d] = sk[src + d]; slidingV[dst + d] = sv[src + d] }
            }
            for (j, gslot) in c.ownFull.enumerated() {
                let dst = gslot * ctx * Self.HF + pos * Self.HF, src = j * Self.HF
                for d in 0..<Self.HF { fullK[dst + d] = fk[src + d]; fullV[dst + d] = fv[src + d] }
            }
        }
        return next
    }

    // ---- helpers ----
    private func maskFull(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: Self.MASK_NEG, count: ctx + 1)
        let p = min(max(pos, 0), ctx - 1)
        for j in 0..<p { m[j] = 0 }   // past 0..p-1
        m[ctx] = 0                    // current token
        return m
    }
    private func maskSliding(_ pos: Int) -> [Float16] {
        var m = [Float16](repeating: Self.MASK_NEG, count: ctx + 1)
        let p = min(max(pos, 0), ctx - 1)
        let loIdx = max(0, p - Self.WIN + 1)
        if loIdx < p { for j in loIdx..<p { m[j] = 0 } }
        m[ctx] = 0
        return m
    }
    private func readInt32(_ a: NDArray) -> Int32 {
        var v: Int32 = 0
        a.view(as: Int32.self).withUnsafePointer { ptr, _, _ in v = ptr[0] }
        return v
    }
    private func argmaxLast(_ flat: [Float], width: Int) -> Int {
        let base = flat.count - width
        var best = 0, bv = flat[base]
        for j in 1..<width { let v = flat[base + j]; if v > bv { bv = v; best = j } }
        return best
    }
    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ kind: String) -> NDArray {
        let io = kind == "input" ? d.inputDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
    }
    private func fill(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ data: [Float16], asFloat: Bool = false) -> NDArray {
        var a = alloc(d, name, shape, "input")
        if asFloat { fillNDArray(&a, as: Float.self, with: data.map { Float($0) }) }
        else { fillNDArray(&a, as: Float16.self, with: data) }
        return a
    }
    private func int32(_ d: InferenceFunctionDescriptor, _ name: String, _ v: Int) -> NDArray {
        var a = alloc(d, name, [1, 1], "input"); fillNDArray(&a, as: Int32.self, with: [Int32(v)]); return a
    }
}

// Headless harness (GEMMA_CHUNK_TEST=1) — same backend the UI's ANE mode uses.
@MainActor
enum Gemma4ChunkEngine {
    private static func docs() -> URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first! }
    private static func availMB() -> Int { Int(os_proc_available_memory()) / (1024 * 1024) }

    static func run() async {
        let env = ProcessInfo.processInfo.environment
        print("[chunkeng] ====== gemma4 CHUNKED host-cache engine (cu=\(env["GEMMA_CHUNK_CU"] ?? "ane")) ======")
        print("[chunkeng] avail before load: \(availMB()) MB")
        do {
            let be = Gemma4ChunkBackend()
            try await be.load()
            print("[chunkeng] loaded 6 chunks + head(\(be.headKind)); ctx=\(be.ctx); avail: \(availMB()) MB")

            // ---- VERIFY mode: drive _gen_ref prompt_ids, greedy decode, compare to ref_ids (device 8/8) ----
            if env["GEMMA_CHUNK_VERIFY"] == "1" {
                let prompt: [Int32] = [2, 105, 2364, 107, 4377, 699, 886, 531, 3595, 236764, 15914, 684, 162760, 236761, 106, 107, 105, 4368, 107]
                let ref = [4906, 236764, 1156, 236764, 1806, 236764, 2390, 236764]
                var last = 0
                for (pos, t) in prompt.enumerated() { last = try await be.step(t, pos) }
                var gen = [last]
                for i in 0..<(ref.count - 1) { last = try await be.step(Int32(last), prompt.count + i); gen.append(last) }
                let n = zip(gen, ref).filter { $0 == $1 }.count
                print("[chunkeng] chunked decode = \(gen)")
                print("[chunkeng] HF greedy ref  = \(ref)")
                print("[chunkeng] match = \(n)/\(ref.count) -> \(n == ref.count ? "PASS ✅ chunked decode EXACT on ANE" : "DEGRADED ⚠️")")
                return
            }

            // ---- generate mode: tokenize a prompt, decode, report tok/s ----
            let tok = try await AutoTokenizer.from(modelFolder: docs().appendingPathComponent("tokenizer"), strict: false)
            let eos = tok.eosTokenId ?? 106
            let promptText = env["GEMMA_PROMPT"] ?? "What is the capital of France?"
            let maxNew = Int(env["GEMMA_CHUNK_N"] ?? "32") ?? 32
            let ids = (try tok.applyChatTemplate(messages: [["role": "user", "content": promptText]])).map { Int32($0) }
            let tPre = Date(); var last = 0
            for (pos, t) in ids.enumerated() { last = try await be.step(t, pos) }
            let preSec = -tPre.timeIntervalSinceNow
            var gen = [last]; var pos = ids.count
            let tDec = Date()
            while gen.count < maxNew, last != eos, last != 106, pos < be.ctx {
                last = try await be.step(Int32(last), pos); pos += 1
                if last == eos || last == 106 { break }
                gen.append(last)
            }
            let decSec = -tDec.timeIntervalSinceNow
            print("[chunkeng] OUT >>> \(tok.decode(tokens: gen, skipSpecialTokens: true))")
            print(String(format: "[chunkeng] prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s | %d MB free",
                         ids.count, Double(ids.count) / max(preSec, 1e-6), gen.count, Double(gen.count) / max(decSec, 1e-6), availMB()))
            print("[profile] per-token: \(be.profileSummary()) (steps=\(be.nProf))")
        } catch {
            print("[chunkeng] ERROR: \(error)")
        }
    }
}
