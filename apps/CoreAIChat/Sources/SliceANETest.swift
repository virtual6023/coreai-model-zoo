// De-risk self-test: does the gemma4 DUAL-head-dim (256/512) KV slice LOWER + RUN on the iPhone ANE?
//
// This is ISOLATED from Gemma4Engine (the Phase-1 GPU core) — it loads a standalone slice .aimodel
// (ondevice/_iso_gemma4_ane_slice.py output) with the requested compute unit and runs a few decode
// steps with random/zero inputs. The de-risk question is purely "does the ANE accept + execute the
// dual-KV graph" (numerics are validated separately in PyTorch), so the input VALUES don't matter.
//
// Triggered headless by CoreAIChatApp when GEMMA_SLICE_TEST=1; reuses the app's build/deploy/signing.
//   GEMMA_SLICE_CU    = ane (default) | gpu | cpu
//   GEMMA_SLICE_MODEL = filename under Documents/models  (default _iso_gemma4_ane_slice_fp16.aimodel)
//   GEMMA_SLICE_STEPS = how many decode steps to run (default 4)

import CoreAI
import CoreAIShared
import Foundation

enum SliceANETest {
    private static func availMB() -> Int { Int(os_proc_available_memory()) / (1024 * 1024) }

    private static func unit(_ s: String) -> (ComputeUnitKind, String) {
        switch s {
        case "gpu": return (.gpu, "gpu")
        case "cpu": return (.cpu, "cpu")
        default: return (.neuralEngine, "ane")
        }
    }

    // Allocate an input NDArray from its descriptor and fill it (values are irrelevant to the
    // lowering de-risk): in_step -> int32 0; *cos* -> 1; *mask* -> 0; everything else -> small 0.05.
    private static func makeInput(_ d: InferenceFunctionDescriptor, _ name: String) -> NDArray? {
        guard case .ndArray(let nd)? = d.inputDescriptor(of: name) else { return nil }
        let shape = nd.shape.map { $0 < 0 ? 1 : $0 }
        var arr = NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
        let n = shape.reduce(1, *)
        if name == "in_step" {
            fillNDArray(&arr, as: Int32.self, with: [Int32](repeating: 0, count: n))
        } else if name.contains("cos") {
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 1, count: n))
        } else if name.contains("mask") {
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0, count: n))
        } else {
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0.05, count: n))
        }
        return arr
    }

    private static func makeState(_ d: InferenceFunctionDescriptor, _ name: String) -> NDArray {
        guard case .ndArray(let nd)? = d.stateDescriptor(of: name) else { fatalError("state \(name)") }
        let shape = nd.shape.map { $0 < 0 ? 1 : $0 }
        var a = NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
        fillNDArray(&a, as: Float16.self, with: [Float16](repeating: 0, count: shape.reduce(1, *)))
        return a
    }

    static func run() async {
        let env = ProcessInfo.processInfo.environment
        let (cu, cuName) = unit(env["GEMMA_SLICE_CU"] ?? "ane")
        let fileName = env["GEMMA_SLICE_MODEL"] ?? "_iso_gemma4_ane_slice_fp16.aimodel"
        let steps = Int(env["GEMMA_SLICE_STEPS"] ?? "4") ?? 4
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = docs.appendingPathComponent("models").appendingPathComponent(fileName)

        print("[sliceANE] ====== gemma4 dual-KV slice ANE de-risk ======")
        print("[sliceANE] cu=\(cuName) model=\(fileName) steps=\(steps)")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("[sliceANE] MISSING model at \(url.path)"); return
        }
        print("[sliceANE] avail before load: \(availMB()) MB")
        do {
            var o = SpecializationOptions(preferredComputeUnitKind: cu)
            o.expectFrequentReshapes = false
            let model = try await AIModel(contentsOf: url, options: o)
            print("[sliceANE] LOADED on \(cuName) ✅  functions=\(model.functionNames)")
            print("[sliceANE] avail after load: \(availMB()) MB")

            let fname = model.functionNames.first { $0.hasPrefix("main") } ?? "main_256_1"
            guard let d = model.functionDescriptor(for: fname),
                  let fn = try model.loadFunction(named: fname) else {
                print("[sliceANE] FAILED to load function \(fname)"); return
            }
            print("[sliceANE] fn=\(fname)")
            print("[sliceANE]   inputs=\(d.inputNames)")
            print("[sliceANE]   states=\(d.stateNames)")
            print("[sliceANE]   outputs=\(d.outputNames)")

            let stateNames = d.stateNames
            guard stateNames.count == 4 else {
                print("[sliceANE] expected 4 dual-KV states, got \(stateNames.count)"); return
            }
            // 4 dual-KV states as function-local vars (the MutableViews borrow must stay in scope).
            var s0 = makeState(d, stateNames[0]); var s1 = makeState(d, stateNames[1])
            var s2 = makeState(d, stateNames[2]); var s3 = makeState(d, stateNames[3])

            guard let outName = d.outputNames.first,
                  case .ndArray(let outND)? = d.outputDescriptor(of: outName) else {
                print("[sliceANE] no ndarray output"); return
            }
            let outShape = outND.shape.map { $0 < 0 ? 1 : $0 }

            for step in 0..<steps {
                var inputs: [String: NDArray] = [:]
                for name in d.inputNames {
                    guard var a = makeInput(d, name) else { continue }
                    if name == "in_step" { fillNDArray(&a, as: Int32.self, with: [Int32(step)]) }
                    inputs[name] = a
                }
                var hid = NDArray(descriptor: outND.resolvingDynamicDimensions(outShape))

                var sv = InferenceFunction.MutableViews()
                sv.insert(&s0, for: stateNames[0]); sv.insert(&s1, for: stateNames[1])
                sv.insert(&s2, for: stateNames[2]); sv.insert(&s3, for: stateNames[3])
                var ov = InferenceFunction.MutableViews(); ov.insert(&hid, for: outName)

                _ = try await fn.run(inputs: inputs, states: consume sv, outputViews: consume ov)
                let flat = flattenAsFloat(hid)
                let finite = flat.allSatisfy { $0.isFinite }
                let mean = flat.reduce(0, +) / Float(max(flat.count, 1))
                print("[sliceANE] step \(step): RAN ✅ \(outName)\(outShape) finite=\(finite) mean=\(String(format: "%.4f", mean)) | \(availMB()) MB free")
            }
            print("[sliceANE] DONE ✅ — gemma4 DUAL-KV slice LOWERS + RUNS on \(cuName).")
        } catch {
            print("[sliceANE] ERROR on \(cuName): \(error)")
        }
    }
}
