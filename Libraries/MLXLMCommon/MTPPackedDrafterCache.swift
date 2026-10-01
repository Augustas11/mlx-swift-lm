// Copyright © 2026 Apple Inc.

import MLX

/// Ragged packed view of one drafter attention layer over row-owned caches.
///
/// Each row appends `inputCounts[row]` leading columns of the packed
/// `[rows, heads, width, headDim]` update at its own offset. Past and appended
/// K/V stay row-local: the packed keys are each row's own prefix followed by
/// its appended columns and zero right padding, and the mask hides both that
/// padding and every right-padded input column. RoPE uses each row's own
/// offset through ``batchOffset``.
///
/// The row caches are read, never written. ``rowStates()`` returns each row's
/// complete post-update K/V for the caller to publish.
///
/// This conforms to ``KVCache`` directly rather than subclassing
/// ``BaseKVCache``: `BaseKVCache` binds the `ropeOffset` requirement to the
/// scalar `offset` default, which a subclass cannot replace, and that would
/// position every row's RoPE from zero.
package final class MTPPackedDrafterLayerCache: BatchPositionedKVCache {
    private let rowStatesBefore: [[MLXArray]]
    private let initialOffsets: [Int]
    private let inputCounts: [Int]
    private var packedKeys: MLXArray?
    private var packedValues: MLXArray?

    /// - Parameters:
    ///   - rowStates: each row's live `[keys, values]`, both `[1, heads,
    ///     offset, headDim]`, or `[]` for an empty row.
    ///   - inputCounts: valid leading input columns per row, at least one.
    package init(rowStates: [[MLXArray]], inputCounts: [Int]) {
        precondition(rowStates.count == inputCounts.count && !rowStates.isEmpty)
        self.rowStatesBefore = rowStates
        self.initialOffsets = rowStates.map { $0.count == 2 ? $0[0].dim(2) : 0 }
        self.inputCounts = inputCounts
    }

    package private(set) var offset = 0
    package var maxSize: Int? { nil }
    package var isTrimmable: Bool { false }
    package func trim(_: Int) -> Int { 0 }

    package var state: [MLXArray] {
        get { [packedKeys, packedValues].compactMap { $0 } }
        set {}
    }

    package var metaState: [String] {
        get { [] }
        set {}
    }

    package func innerState() -> [MLXArray] {
        [packedKeys, packedValues].compactMap { $0 }
    }

    private var finalOffsets: [Int] {
        zip(initialOffsets, inputCounts).map(+)
    }

    public var batchOffset: MLXArray {
        MLXArray(initialOffsets.map(Int32.init))
    }

    package func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let capacity = finalOffsets.max() ?? 0
        func pack(_ incoming: MLXArray, slot: Int) -> MLXArray {
            let rows = rowStatesBefore.indices.map { row -> MLXArray in
                var parts: [MLXArray] = []
                if initialOffsets[row] > 0 {
                    parts.append(rowStatesBefore[row][slot])
                }
                parts.append(incoming[row ..< row + 1, 0..., 0 ..< inputCounts[row], 0...])
                let pad = capacity - finalOffsets[row]
                if pad > 0 {
                    parts.append(
                        MLXArray.zeros(
                            [1, incoming.dim(1), pad, incoming.dim(3)], dtype: incoming.dtype))
                }
                return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
            }
            return rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
        }
        let packedKeys = pack(keys, slot: 0)
        let packedValues = pack(values, slot: 1)
        self.packedKeys = packedKeys
        self.packedValues = packedValues
        offset = capacity
        return (packedKeys, packedValues)
    }

    package func makeMask(
        n: Int, windowSize _: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let batch = initialOffsets.count
        let capacity = finalOffsets.max() ?? 0
        let queries =
            MLXArray(initialOffsets.map(Int32.init)).reshaped([batch, 1, 1, 1])
            + MLXArray(Int32(0) ..< Int32(n)).reshaped([1, 1, n, 1])
        let keys = MLXArray(Int32(0) ..< Int32(capacity)).reshaped([1, 1, 1, capacity])
        let live = MLXArray(finalOffsets.map(Int32.init)).reshaped([batch, 1, 1, 1])
        return .array((keys .<= queries) & (keys .< live))
    }

    package func copy() -> any KVCache {
        MTPPackedDrafterLayerCache(rowStates: rowStatesBefore, inputCounts: inputCounts)
    }

    /// Each row's complete post-update `[keys, values]`, without padding.
    /// `nil` until the packed forward has called ``update(keys:values:)``.
    package func rowStates() -> [[MLXArray]]? {
        guard let packedKeys, let packedValues else { return nil }
        return finalOffsets.enumerated().map { row, live in
            [
                packedKeys[row ..< row + 1, 0..., 0 ..< live, 0...],
                packedValues[row ..< row + 1, 0..., 0 ..< live, 0...],
            ]
        }
    }
}
