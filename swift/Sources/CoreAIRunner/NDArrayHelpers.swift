// Copyright 2026.
// ⚠️ DRAFT (compile on macOS 27). Self-contained reimplementations of the NDArray fill/read
// helpers that are module-internal in Apple's CoreAILanguageModels, so CoreAIRunner needs no
// dependency on that package. Built against NDArray's public view API as used in
// CoreAISequentialEngine (`mutableView(as:)` / `view(as:)` + `withUnsafe[Mutable]Pointer`).

import CoreAI

/// Core AI emits fp16 logits (matches Apple's `LogitsScalarType` in CoreAILanguageModels).
public typealias LogitsScalarType = Float16

/// Fill an NDArray of scalar type `T` from a sequence (row-major, count = product of shape).
@inlinable
func fillNDArray<T, S: Sequence>(_ array: inout NDArray, as type: T.Type, with values: S)
where S.Element == T {
    var view = array.mutableView(as: T.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        var i = 0
        for v in values { ptr[i] = v; i += 1 }
    }
}

/// Fill `count` elements via a closure of the flat index.
@inlinable
func fillNDArray<T>(_ array: inout NDArray, as type: T.Type, count: Int, _ make: (Int) -> T) {
    var view = array.mutableView(as: T.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<count { ptr[i] = make(i) }
    }
}

/// Read `count` elements out of an NDArray into a Swift array.
@inlinable
func readNDArray<T>(_ array: NDArray, as type: T.Type, count: Int) -> [T] {
    var out = [T]()
    out.reserveCapacity(count)
    array.view(as: T.self).withUnsafePointer { ptr, _, _ in
        for i in 0..<count { out.append(ptr[i]) }
    }
    return out
}
