// De-risk: does Session B's HOST-CACHE gemma4 core (the macOS-GPU win — no Core AI state, no in-graph
// indexed KV write, just masked SDPA over plain inputs + `cat`) also LOWER + RUN on the iPhone?
//
// Session B proved it runs on the Mac GPU via Core AI (8/8 EXACT, 35-layer int8) where the in-graph
// `mutable_slice_update` core SIGTRAP'd. The open question for the iOS port: does removing the indexed
// write (the device-GPU SIGSEGV / device-ANE "MLIR pass manager failed" trigger) unblock the DEVICE?
//
// ISOLATED from Gemma4Engine. Triggered by GEMMA_HOSTCACHE_TEST=1. The host-cache core has NO states:
// the 4 KV caches are plain INPUTS and the new columns come back as 4 extra OUTPUTS (hidden + *_cur).
// For a lowering/run probe we don't need the host write-back (correctness) — we just run a few steps
// with zero caches and confirm the graph loads + executes (GPU) / compiles (ANE).
//   GEMMA_HOSTCACHE_CU    = ane (default) | gpu | cpu
//   GEMMA_HOSTCACHE_MODEL = filename under Documents/models (default gemma4_e2b_hostcache_L35_int8.aimodel)
//   GEMMA_HOSTCACHE_STEPS = steps to run (default 4)

import CoreAI
import CoreAIShared
import Foundation

enum HostCacheTest {
    private static func availMB() -> Int { Int(os_proc_available_memory()) / (1024 * 1024) }

    private static func unit(_ s: String) -> (ComputeUnitKind, String) {
        switch s {
        case "gpu": return (.gpu, "gpu")
        case "cpu": return (.cpu, "cpu")
        default: return (.neuralEngine, "ane")
        }
    }

    // Allocate an input NDArray from its descriptor and fill it (values irrelevant to the lowering
    // probe): position_ids -> int32; *mask* -> 0 (attend all); everything else (embeds/per_layer/caches)
    // -> small 0.05 fp16.
    private static func makeInput(_ d: InferenceFunctionDescriptor, _ name: String, pos: Int) -> NDArray? {
        guard case .ndArray(let nd)? = d.inputDescriptor(of: name) else { return nil }
        let shape = nd.shape.map { $0 < 0 ? 1 : $0 }
        var arr = NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
        let n = shape.reduce(1, *)
        if name == "position_ids" {
            fillNDArray(&arr, as: Int32.self, with: [Int32](repeating: Int32(pos), count: n))
        } else if name.contains("mask") {
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0, count: n))
        } else {
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0.05, count: n))
        }
        return arr
    }

    static func run() async {
        let env = ProcessInfo.processInfo.environment
        let (cu, cuName) = unit(env["GEMMA_HOSTCACHE_CU"] ?? "ane")
        let fileName = env["GEMMA_HOSTCACHE_MODEL"] ?? "gemma4_e2b_hostcache_L35_int8.aimodel"
        let steps = Int(env["GEMMA_HOSTCACHE_STEPS"] ?? "4") ?? 4
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = docs.appendingPathComponent("models").appendingPathComponent(fileName)

        print("[hostcache] ====== gemma4 HOST-CACHE core device de-risk ======")
        print("[hostcache] cu=\(cuName) model=\(fileName) steps=\(steps)")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("[hostcache] MISSING model at \(url.path)"); return
        }
        print("[hostcache] avail before load: \(availMB()) MB")
        do {
            var o = SpecializationOptions(preferredComputeUnitKind: cu)
            o.expectFrequentReshapes = false
            let model = try await AIModel(contentsOf: url, options: o)
            print("[hostcache] LOADED on \(cuName) ✅  functions=\(model.functionNames)")
            print("[hostcache] avail after load: \(availMB()) MB")

            let fname = model.functionNames.first { $0.hasPrefix("main") } ?? "main"
            guard let d = model.functionDescriptor(for: fname),
                  let fn = try model.loadFunction(named: fname) else {
                print("[hostcache] FAILED to load function \(fname)"); return
            }
            print("[hostcache] fn=\(fname)")
            print("[hostcache]   inputs=\(d.inputNames)")
            print("[hostcache]   states=\(d.stateNames)")
            print("[hostcache]   outputs=\(d.outputNames)")

            var runMs = [Double]()
            for step in 0..<steps {
                var inputs: [String: NDArray] = [:]
                for name in d.inputNames {
                    if let a = makeInput(d, name, pos: step) { inputs[name] = a }
                }
                // Auto-allocate the outputs (run() defaults outputViews) — works for ANY chunk's output
                // set: the monolith / chunk1 / chunk2 emit hidden + the 4 *_cur columns; the stateless
                // chunks 3/4 emit hidden only. We only inspect `hidden` for the lowering/run probe.
                let t0 = Date()
                var outputs = try await fn.run(inputs: inputs)
                let ms = -t0.timeIntervalSinceNow * 1000
                runMs.append(ms)
                guard let hv = outputs.remove("hidden"), let hidden = hv.ndArray else {
                    print("[hostcache] step \(step): RAN but produced no 'hidden' output"); continue
                }
                let flat = flattenAsFloat(hidden)
                let finite = flat.allSatisfy { $0.isFinite }
                let mean = flat.reduce(0, +) / Float(max(flat.count, 1))
                print("[hostcache] step \(step): RAN ✅ run=\(String(format: "%.1f", ms))ms finite=\(finite) mean=\(String(format: "%.4f", mean)) | \(availMB()) MB free")
            }
            if runMs.count > 1 {
                let steady = runMs.dropFirst().reduce(0, +) / Double(runMs.count - 1)
                print(String(format: "[hostcache] run() first %.0fms -> steady %.1fms (%.1f core tok/s)", runMs[0], steady, 1000.0 / steady))
            }
            print("[hostcache] DONE ✅ — gemma4 HOST-CACHE core LOWERS + RUNS on \(cuName).")
        } catch {
            print("[hostcache] ERROR on \(cuName): \(error)")
        }
    }
}
