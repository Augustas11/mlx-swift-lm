//
//  DeepseekV4PoolCache.swift
//  mlx-swift-lm
//
//  The compressor state of one DeepSeek-V4 compressed-attention branch: the
//  raw kv/gate rows of a window still filling up (fewer than `ratio`), the
//  growing pool of compressed entries, one per `ratio` tokens, and for the
//  overlapping ratio-4 compressor the first half-channels of the last
//  completed window, which the next window pools as well.
//
//  Port of mlx-vlm 0.7.10 `PoolingCache` (`mlx_vlm/models/cache.py:2693`),
//  without its speculation transaction (MTP rewind comes with the DSpark
//  slice).
//

import Foundation
import MLX

public final class DeepseekV4PoolCache: BaseKVCache {
    public let ratio: Int

    /// `[B, ratio, D]` raw rows (float32 projections); the first `remainder` are live.
    var bufferKV: MLXArray?
    var bufferGate: MLXArray?
    public private(set) var remainder = 0

    /// Ratio-4 overlap state (`kv_state[:, :ratio]` / `score_state[:, :ratio]`,
    /// `inference/model.py:309-310, 336-338, 353-360`): the last completed
    /// window's first half-channels of kv and of gate (without `ape`),
    /// `[B, ratio, D / 2]` float32, or `nil` before the first window. It is
    /// carried across calls, so a window pools its predecessor whether the
    /// two arrived in one call or in two.
    public var overlapKV: MLXArray?
    public var overlapGate: MLXArray?

    /// `[B, P, D]` compressed entries in token order, or `nil` before the first window.
    private var pooledStorage: MLXArray?
    private var count = 0

    public var pooled: MLXArray? {
        guard let pooledStorage else { return nil }
        return pooledStorage[0..., ..<count]
    }

    public init(ratio: Int) {
        precondition(ratio > 0)
        self.ratio = ratio
        super.init()
    }

    /// Batch rows of the window buffers, `nil` before the first call.
    public var bufferRows: Int? { bufferKV?.dim(0) }

    /// Compressed entries held (the reference's `offset`).
    public var pooledCount: Int { count }

    public override func innerState() -> [MLXArray] {
        [bufferKV, bufferGate, pooledStorage, overlapKV, overlapGate].compactMap { $0 }
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
                // With no completed window the existing remainder stays in
                // place; only the new chunk belongs after it.
                let take = newRemainder - remainder
                bufferKV![0..., remainder ..< newRemainder] = kv[0..., (L - take)...]
                bufferGate![0..., remainder ..< newRemainder] = gate[0..., (L - take)...]
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
        let added = entries.dim(1)
        if added > 0 {
            let capacity = pooledStorage?.dim(1) ?? 0
            if count + added > capacity {
                let newCapacity = max(count + added, max(16, capacity * 2))
                let reserve = MLXArray.zeros(
                    [entries.dim(0), newCapacity - capacity, entries.dim(2)], dtype: entries.dtype)
                pooledStorage = pooledStorage.map { concatenated([$0, reserve], axis: 1) } ?? reserve
            }
            pooledStorage![0..., count ..< (count + added)] = entries
            count += added
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

    /// `[]`, `[bufferKV, bufferGate]`, `[bufferKV, bufferGate, pooled]` or,
    /// once an overlapping window completed, `[bufferKV, bufferGate, pooled,
    /// overlapKV, overlapGate]`: the buffers exist from the first call on,
    /// before any entry is pooled, and the overlap state with the first entry.
    public override var state: [MLXArray] {
        get { [bufferKV, bufferGate, pooled, overlapKV, overlapGate].compactMap { $0 } }
        set {
            guard [0, 2, 3, 5].contains(newValue.count) else {
                fatalError("DeepseekV4PoolCache state must have 0, 2, 3 or 5 arrays")
            }
            bufferKV = newValue.count >= 2 ? newValue[0] : nil
            bufferGate = newValue.count >= 2 ? newValue[1] : nil
            pooledStorage = newValue.count >= 3 ? newValue[2] : nil
            count = pooledStorage?.dim(1) ?? 0
            overlapKV = newValue.count == 5 ? newValue[3] : nil
            overlapGate = newValue.count == 5 ? newValue[4] : nil
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(ratio), String(remainder), String(bufferKV != nil), String(pooled != nil),
                String(overlapKV != nil),
            ]
        }
        set {
            guard newValue.count == 5, Int(newValue[0]) == ratio, let remainder = Int(newValue[1])
            else { fatalError("DeepseekV4PoolCache metaState mismatch: \(newValue)") }
            self.remainder = remainder
        }
    }

    public override func copy() -> any KVCache {
        let new = DeepseekV4PoolCache(ratio: ratio)
        new.bufferKV = bufferKV?[.ellipsis]
        new.bufferGate = bufferGate?[.ellipsis]
        new.pooledStorage = pooledStorage?[.ellipsis]
        new.overlapKV = overlapKV?[.ellipsis]
        new.overlapGate = overlapGate?[.ellipsis]
        new.count = count
        new.remainder = remainder
        return new
    }
}
