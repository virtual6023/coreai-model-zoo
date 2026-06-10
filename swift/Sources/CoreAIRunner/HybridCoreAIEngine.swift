// Copyright 2026.
//
// ⚠️ DRAFT — written on macOS 26.6 against the EXACT Core AI Swift API used by Apple's
// `CoreAISequentialEngine.swift`, but NOT yet compiled (the `coreai-models/swift` package
// is `.macOS("27.0")`). Compile + fix on macOS 27 (fast loop), then drive the banked
// qwen3.5 bundle. See ondevice/README.md "Swift runner" for the build/verify runbook.
//
// Generic N-STATE Core AI inference engine. Apple's CoreAISequentialEngine hard-codes 2
// states (KV); Qwen3.5 is hybrid with 4 states (keyCache, valueCache, convState, recState).
// This generalizes the same pattern to any number of states read from the function
// descriptor, allocated once at full capacity (no dynamic KV growth in v1 — simplest correct
// version; add 2× growth later like CoreAISequentialEngine if memory matters).
//
// Model contract (qwen3.5 all-in-one bundle, ondevice/artifacts/qwen3_5_*_int8_stateful.aimodel):
//   inputs : input_ids (Int32 [1,S]), position_ids (Int32 [1,S])
//   output : logits (Float16 [1, S_or_1, vocab]) — bundle is last_token_only -> [1,1,vocab]
//   states : keyCache, valueCache, convState, recState (in-place, persist across calls)
// position_ids carries FULL positions [0..total); offset = total - S. Same convention as
// CoreAISequentialEngine (it builds [0..processed+batch)).
//
// INTEGRATION: this reuses module-internal helpers (`fillNDArray`, `readNDArray`) from
// CoreAILanguageModels, so the cleanest path is to drop this file INTO
// coreai-models/swift/Sources/CoreAILanguageModels/InferenceEngines/ (same module). If kept
// in a separate package instead, reimplement those two helpers against NDArray's public
// `mutableView(as:)`/`view(as:)` + `withUnsafe[Mutable]Pointer` (see copyCache in the template).

import CoreAI
import Foundation

public final class HybridCoreAIEngine {
    public struct Spec {
        public var functionName: String = "main"
        public var maxContextLength: Int = 4096
        public var vocabSize: Int
        public init(vocabSize: Int) { self.vocabSize = vocabSize }
    }

    private let function: InferenceFunction
    private let descriptor: InferenceFunctionDescriptor
    private let spec: Spec

    private let inputIdsName: String
    private let positionIdsName: String
    private let logitsName: String
    private let stateNames: [String]

    private let inputIdsDesc: NDArrayDescriptor
    private let positionIdsDesc: NDArrayDescriptor
    private let logitsDesc: NDArrayDescriptor

    // One NDArray per state, allocated at full capacity, mutated in place across steps.
    private var states: [String: NDArray] = [:]
    private var processedTokenCount: Int = 0

    public init(modelURL: URL, spec: Spec) async throws {
        self.spec = spec
        let prepared = try await PreparedModel.prepare(at: modelURL)
        let model = prepared.model

        guard let descriptor = model.functionDescriptor(for: spec.functionName) else {
            throw EngineError.message("function '\(spec.functionName)' not found")
        }
        self.descriptor = descriptor

        // 2 inputs (input_ids, position_ids), >=1 output, N states.
        guard descriptor.inputNames.count == 2 else {
            throw EngineError.message("expected 2 inputs, got \(descriptor.inputNames)")
        }
        self.inputIdsName = descriptor.inputNames[0]
        self.positionIdsName = descriptor.inputNames[1]
        self.logitsName = descriptor.outputNames[0]
        self.stateNames = descriptor.stateNames

        guard case .ndArray(let iid) = descriptor.inputDescriptor(of: inputIdsName),
              case .ndArray(let pid) = descriptor.inputDescriptor(of: positionIdsName),
              case .ndArray(let lid) = descriptor.outputDescriptor(of: logitsName) else {
            throw EngineError.message("could not read input/output descriptors")
        }
        self.inputIdsDesc = iid
        self.positionIdsDesc = pid
        self.logitsDesc = lid

        // Allocate every state at full capacity (dynamic dims -> maxContextLength), zero-init.
        // KV caches have a dynamic seq dim -> resolved to maxContextLength; conv/rec states are
        // already static -> resolvingDynamicDimensions is a no-op on them.
        for name in stateNames {
            guard case .ndArray(let sdesc) = descriptor.stateDescriptor(of: name) else {
                throw EngineError.message("state '\(name)' is not an NDArray")
            }
            let resolved = sdesc.resolvingDynamicDimensions(
                sdesc.shape.map { $0 < 0 ? spec.maxContextLength : $0 })
            var arr = NDArray(descriptor: resolved)
            zeroFillFloat16(&arr)   // states are fp16 (qwen3.5 bundle is fp16)
            states[name] = arr
        }

        guard let fn = try model.loadFunction(named: spec.functionName) else {
            throw EngineError.message("could not load function '\(spec.functionName)'")
        }
        self.function = fn
    }

    /// Run one forward pass over `tokens` (prefill: many; decode: one). Returns the logits of
    /// the LAST position (the bundle already emits last-token-only, so logits is [1,1,vocab]).
    private func forward(_ tokens: [Int32]) async throws -> [LogitsScalarType] {
        let batch = tokens.count
        let total = processedTokenCount + batch

        var inputIds = NDArray(descriptor: inputIdsDesc.resolvingDynamicDimensions([1, batch]))
        fillNDArray(&inputIds, as: Int32.self, with: tokens[...])

        var positionIds = NDArray(descriptor: positionIdsDesc.resolvingDynamicDimensions([1, total]))
        fillNDArray(&positionIds, as: Int32.self, count: total) { Int32($0) }

        var logits = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, 1, spec.vocabSize]))

        var stateViews = InferenceFunction.MutableViews()
        for name in stateNames { stateViews.insert(&states[name]!, for: name) }

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logits, for: logitsName)

        _ = try await function.run(
            inputs: [inputIdsName: inputIds, positionIdsName: positionIds],
            states: consume stateViews,
            outputViews: consume outputViews)

        processedTokenCount = total
        return readNDArray(logits, as: LogitsScalarType.self, count: spec.vocabSize)
    }

    /// Greedy generation: prefill `promptTokens`, then sample argmax `maxNewTokens` times.
    /// Returns the generated token ids (excluding the prompt).
    public func generateGreedy(promptTokens: [Int32], maxNewTokens: Int) async throws -> [Int32] {
        var out: [Int32] = []
        var logits = try await forward(promptTokens)            // prefill
        for _ in 0..<maxNewTokens {
            let next = argmax(logits)
            out.append(next)
            logits = try await forward([next])                  // decode one
        }
        return out
    }

    private func argmax(_ logits: [LogitsScalarType]) -> Int32 {
        var best = 0
        var bestVal = logits[0]
        for i in 1..<logits.count where logits[i] > bestVal { bestVal = logits[i]; best = i }
        return Int32(best)
    }

    private func zeroFillFloat16(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        var view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, _, _ in for i in 0..<count { ptr[i] = 0 } }
    }

    enum EngineError: Error { case message(String) }
}
