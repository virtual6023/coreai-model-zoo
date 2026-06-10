// mmap-based Gemma 4 front-end gather — replaces the 2.6 GB `frontend_int8.aimodel` whose
// dirty-resident int8 tables OOM the app on iPhone. The two giant tables (embed_tokens 0.4 GB,
// embed_tokens_per_layer 2.35 GB) are memory-mapped (Data .mappedIfSafe -> file-backed, NOT
// dirty), and we read only the gathered rows on demand (the standard on-device approach:
// CoreML-LLM, litert-lm `MemoryMappedFile`). Resident drops from 2.6 GB to ~60 MB.
//
// Mirrors `GatherEmbeds`/`GatherPerLayer.forward` exactly (verified numpy==torch by
// ondevice/export_gemma4_gather_raw.py):
//   inputs_embeds = q_e[id]·scale_e[id]·√1536
//   tokens        = (q_p[id]·scale_p[id]·√256).reshape(s,35,256)
//   proj          = RMSNorm256( (inputs_embeds @ projᵀ)·1536^-.5 )
//   per_layer_inputs = (proj + tokens)·2^-.5

import Accelerate
import Foundation

final class Gemma4Gather {
    struct Meta: Decodable {
        let V: Int, D: Int, PLD: Int, L: Int, ld: Int
        let embed_scale_main: Float, embed_scale_pl: Float
        let proj_scale: Float, input_scale: Float, rms_eps: Float
    }
    let m: Meta
    // Raw POSIX mmap (PROT_READ, MAP_PRIVATE) — guarantees the giant int8 tables stay file-backed
    // (clean, paged on demand), NOT counted as the app's dirty footprint. `Data(.mappedIfSafe)` is
    // only a hint and was observed to fall back to a full read (footprint 6.53 GB -> jetsam).
    private let qEmbed: UnsafePointer<Int8>   // [V, D] int8
    private let qPL: UnsafePointer<Int8>      // [V, PLD] int8
    private let sEmbed: [Float]   // [V]
    private let sPL: [Float]      // [V]
    private let projW: [Float]    // [PLD, D] row-major
    private let normW: [Float]    // [ld]

    private static func mmapInt8(_ url: URL, expect: Int) throws -> UnsafePointer<Int8> {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw NSError(domain: "Gemma4Gather", code: 1, userInfo: [NSLocalizedDescriptionKey: "open \(url.lastPathComponent)"]) }
        defer { close(fd) }  // the mapping outlives the fd
        var st = Darwin.stat()
        guard fstat(fd, &st) == 0, Int(st.st_size) == expect else {
            throw NSError(domain: "Gemma4Gather", code: 2, userInfo: [NSLocalizedDescriptionKey: "size \(url.lastPathComponent) != \(expect)"])
        }
        guard let p = mmap(nil, expect, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw NSError(domain: "Gemma4Gather", code: 3, userInfo: [NSLocalizedDescriptionKey: "mmap \(url.lastPathComponent)"])
        }
        madvise(p, expect, MADV_RANDOM)  // gather = random row access; avoid read-ahead
        return UnsafeRawPointer(p).assumingMemoryBound(to: Int8.self)
    }

    init(dir: URL) throws {
        func floats(_ n: String) throws -> [Float] {
            let d = try Data(contentsOf: dir.appendingPathComponent(n))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        m = try JSONDecoder().decode(Meta.self, from: try Data(contentsOf: dir.appendingPathComponent("meta.json")))
        qEmbed = try Self.mmapInt8(dir.appendingPathComponent("embed_tokens.i8"), expect: m.V * m.D)
        qPL = try Self.mmapInt8(dir.appendingPathComponent("embed_per_layer.i8"), expect: m.V * m.PLD)
        sEmbed = try floats("embed_tokens.scale.f32")
        sPL = try floats("embed_per_layer.scale.f32")
        projW = try floats("proj.f32")
        normW = try floats("proj_norm.f32")
        precondition(projW.count == m.PLD * m.D && normW.count == m.ld, "gather file shape mismatch")
    }

    /// Returns (inputs_embeds [s*D], per_layer_inputs [s*PLD]) as Float.
    func gather(_ ids: [Int32]) -> (ie: [Float], pli: [Float]) {
        let s = ids.count, D = m.D, PLD = m.PLD, L = m.L, ld = m.ld
        var ie = [Float](repeating: 0, count: s * D)
        var tk = [Float](repeating: 0, count: s * PLD)
        for (j, id) in ids.enumerated() {
            let r = Int(id)
            let se = sEmbed[r] * m.embed_scale_main
            let be = r * D, oe = j * D
            for k in 0..<D { ie[oe + k] = Float(qEmbed[be + k]) * se }
            let sp = sPL[r] * m.embed_scale_pl
            let bp = r * PLD, op = j * PLD
            for k in 0..<PLD { tk[op + k] = Float(qPL[bp + k]) * sp }
        }
        // proj[s,PLD] = (ie[s,D] @ projWᵀ[D,PLD]) · proj_scale   (projW is [PLD,D] row-major)
        var proj = [Float](repeating: 0, count: s * PLD)
        ie.withUnsafeBufferPointer { iep in
            projW.withUnsafeBufferPointer { wp in
                proj.withUnsafeMutableBufferPointer { pp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(s), Int32(PLD), Int32(D), m.proj_scale,
                                iep.baseAddress, Int32(D), wp.baseAddress, Int32(D),
                                0.0, pp.baseAddress, Int32(PLD))
                }
            }
        }
        // RMSNorm over each [ld] group + combine: (RMSNorm(proj) + tokens) · input_scale
        var pli = [Float](repeating: 0, count: s * PLD)
        for j in 0..<s {
            for l in 0..<L {
                let off = j * PLD + l * ld
                var ss: Float = 0
                for k in 0..<ld { let v = proj[off + k]; ss += v * v }
                let inv = 1.0 / (ss / Float(ld) + m.rms_eps).squareRoot()
                for k in 0..<ld {
                    pli[off + k] = (proj[off + k] * inv * normW[k] + tk[off + k]) * m.input_scale
                }
            }
        }
        return (ie, pli)
    }
}
