//
//  DeepseekV4PoolCache.swift
//  mlx-swift-lm
//
//  The compressor state of one DeepSeek-V4 compressed-attention branch: the
//  raw kv/gate rows of a window still filling up (fewer than `ratio`) and the
//  growing pool of compressed entries, one per `ratio` tokens.
//
//  Port of mlx-vlm 0.7.10 `PoolingCache` (`mlx_vlm/models/cache.py:2693`),
//  without its speculation transaction (MTP rewind comes with the DSpark
//  slice).
//

import Foundation
import MLX

public final class DeepseekV4PoolCache: BaseKVCache {
    public let ratio: Int

    /// `[B, ratio, D]` raw rows; the first `remainder` are live.
    var bufferKV: MLXArray?
    var bufferGate: MLXArray?
    public private(set) var remainder = 0

    /// `[B, P, D]` compressed entries in token order, or `nil` before the first window.
    public private(set) var pooled: MLXArray?

    public init(ratio: Int) {
        precondition(ratio > 0)
        self.ratio = ratio
        super.init()
    }

    /// Compressed entries held (the reference's `offset`).
    public var pooledCount: Int { pooled?.dim(1) ?? 0 }

    public override func innerState() -> [MLXArray] {
        [bufferKV, bufferGate, pooled].compactMap { $0 }
    }

    /// Rows of every window this call completes, `[B, W * ratio, D]` (empty
    /// when none), and the token position of the first such row.
    ///
    /// `kv`/`gate` are `[B, L, D]` projections of the tokens at `offset ..<
    /// offset + L`. A prompt chunk (`L > 1`) completes as many windows as the
    /// buffered remainder plus the chunk fill; a decode step completes at most
    /// one.
    public func accumulateWindows(kv: MLXArray, gate: MLXArray, offset: Int)
        -> (kv: MLXArray, gate: MLXArray, base: Int)
    {
        let (B, L) = (kv.dim(0), kv.dim(1))
        let (kvWidth, gateWidth) = (kv.dim(2), gate.dim(2))
        if bufferKV == nil {
            bufferKV = MLXArray.zeros([B, ratio, kvWidth], dtype: kv.dtype)
            bufferGate = MLXArray.zeros([B, ratio, gateWidth], dtype: gate.dtype)
        }
        let emptyKV = MLXArray.zeros([B, 0, kvWidth], dtype: kv.dtype)
        let emptyGate = MLXArray.zeros([B, 0, gateWidth], dtype: gate.dtype)

        if L > 1 {
            let total = L + remainder
            let usable = (total / ratio) * ratio
            let newRemainder = total % ratio
            var ready = (emptyKV, emptyGate, 0)
            if usable > 0 {
                let take = usable - remainder
                ready = (
                    concatenated([bufferKV![0..., ..<remainder], kv[0..., ..<take]], axis: 1),
                    concatenated([bufferGate![0..., ..<remainder], gate[0..., ..<take]], axis: 1),
                    offset - remainder
                )
                remainder = 0
            }
            if newRemainder > 0 {
                bufferKV![0..., remainder ..< newRemainder] = kv[0..., (L - newRemainder)...]
                bufferGate![0..., remainder ..< newRemainder] = gate[0..., (L - newRemainder)...]
            }
            remainder = newRemainder
            return ready
        }

        bufferKV![0..., remainder ..< (remainder + 1)] = kv
        bufferGate![0..., remainder ..< (remainder + 1)] = gate
        remainder = (remainder + 1) % ratio
        if remainder == 0 {
            // A copy: the next decode step writes into the buffer again.
            return (bufferKV![.ellipsis], bufferGate![.ellipsis], offset - ratio + 1)
        }
        return (emptyKV, emptyGate, 0)
    }

    /// Append `[B, W, D]` new entries and return the whole pool (`[B, 0, D]` while empty).
    public func appendPooled(_ entries: MLXArray) -> MLXArray {
        if entries.dim(1) > 0 {
            pooled = pooled.map { concatenated([$0, entries], axis: 1) } ?? entries
        }
        return pooled ?? entries
    }

    /// `[L, P]` visibility of pooled entries to the queries at `offset ..< offset
    /// + L`: entry `i` is visible to the query at `t` iff `i < (t + 1) / ratio`.
    /// `nil` for one query or an empty pool (every entry visible).
    public func poolMask(queries L: Int, offset: Int) -> MLXArray? {
        guard let pooled, L > 1 else { return nil }
        let poolIndex = MLXArray.arange(pooled.dim(1), dtype: .int32)
        let queryIndex = MLXArray.arange(offset + 1, offset + L + 1, dtype: .int32)
        return poolIndex .< expandedDimensions(queryIndex.floorDivide(Int32(ratio)), axis: 1)
    }

    /// `[]`, `[bufferKV, bufferGate]` or `[bufferKV, bufferGate, pooled]`: the
    /// buffers exist from the first call on, before any entry is pooled.
    public override var state: [MLXArray] {
        get { innerState() }
        set {
            guard [0, 2, 3].contains(newValue.count) else {
                fatalError("DeepseekV4PoolCache state must have 0, 2 or 3 arrays")
            }
            bufferKV = newValue.count >= 2 ? newValue[0] : nil
            bufferGate = newValue.count >= 2 ? newValue[1] : nil
            pooled = newValue.count == 3 ? newValue[2] : nil
        }
    }

    public override var metaState: [String] {
        get { [String(ratio), String(remainder), String(bufferKV != nil), String(pooled != nil)] }
        set {
            guard newValue.count == 4, Int(newValue[0]) == ratio, let remainder = Int(newValue[1])
            else { fatalError("DeepseekV4PoolCache metaState mismatch: \(newValue)") }
            self.remainder = remainder
        }
    }

    public override func copy() -> any KVCache {
        let new = DeepseekV4PoolCache(ratio: ratio)
        new.bufferKV = bufferKV?[.ellipsis]
        new.bufferGate = bufferGate?[.ellipsis]
        new.pooled = pooled?[.ellipsis]
        new.remainder = remainder
        return new
    }
}
