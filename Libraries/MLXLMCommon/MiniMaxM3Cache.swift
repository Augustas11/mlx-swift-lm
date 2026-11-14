//
//  MiniMaxM3Cache.swift
//  mlx-swift-lm
//
//  The cache of one MiniMax M3 sparse-attention layer: the attention keys and
//  values, plus the lightning indexer's single-head keys that pick the key
//  blocks each query attends.
//
//  Port of mlx-vlm 0.7.11 `MiniMaxM3KVCache`
//  (`mlx_vlm/models/minimax_m3_vl/language.py`), single row.
//

import Foundation
import MLX

/// The cache a MiniMax M3 sparse-attention layer reads and writes for one
/// sequence: the attention keys and values (`update(keys:values:)`) and the
/// indexer keys of the same tokens (`updateIndex`).
public protocol MiniMaxM3SparseCache: KVCache {
    /// Append the indexer keys (`[1, 1, L, D]`) of the tokens the last
    /// `update(keys:values:)` added; return every cached indexer key
    /// (`[1, 1, offset, D]`).
    func updateIndex(_ keys: MLXArray) -> MLXArray
}

/// A sparse-attention layer cache for several sequences decoded in one
/// forward (continuous batching). The model projects every row together and
/// runs each row's norms, RoPE, block selection and attention against its own
/// `rowCaches` entry, as a serial forward of that row does.
public protocol MiniMaxM3BatchedSparseCache: KVCache {
    /// One cache per batch row, in row order.
    var rowCaches: [any MiniMaxM3SparseCache] { get }

    /// Valid leading columns per row of the current forward (right-padded
    /// rows), or `nil` when every column of every row is valid. A padded
    /// column attends nothing and writes no cache.
    var rowLengths: [Int]? { get }

    /// Speculative verification: each row's valid columns attend as
    /// consecutive one-token steps (cache update, block selection and
    /// attention per column), exactly as ordinary decode of those tokens.
    /// Projections, norms and experts still run on every column at once.
    var attendsColumnwise: Bool { get }
}

extension MiniMaxM3BatchedSparseCache {
    public var attendsColumnwise: Bool { false }
}

/// `MiniMaxM3BatchedSparseCache` over existing per-sequence caches. The rows
/// carry the state; `offset` is unused.
public final class MiniMaxM3CacheBatch: BaseKVCache, MiniMaxM3BatchedSparseCache {
    public let rowCaches: [any MiniMaxM3SparseCache]
    public private(set) var rowLengths: [Int]?

    public init(rowCaches: [any MiniMaxM3SparseCache]) {
        precondition(!rowCaches.isEmpty, "MiniMaxM3CacheBatch needs at least one row")
        self.rowCaches = rowCaches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        rowCaches.flatMap { $0.innerState() }
    }

    public override func copy() -> any KVCache {
        MiniMaxM3CacheBatch(rowCaches: rowCaches.map { $0.copy() as! any MiniMaxM3SparseCache })
    }

    /// Ragged packed forward: valid leading columns per row.
    public override func prepare(lengths: [Int]?) {
        rowLengths = lengths.flatMap { Set($0).count <= 1 ? nil : $0 }
    }

    public override func finalize() {
        rowLengths = nil
    }
}

public final class MiniMaxM3Cache: BaseKVCache, MiniMaxM3SparseCache, CustomDebugStringConvertible {
    public var step = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    private var indexKeys: MLXArray?

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [keys, values, indexKeys].compactMap { $0 }
    }

    /// Grow `buffer` (`[B, H, capacity, D]`) to hold `incoming` more rows after
    /// `previous`, in `step`-row increments.
    private func reserve(_ buffer: MLXArray?, like new: MLXArray, previous: Int) -> MLXArray {
        if let buffer, previous + new.dim(2) <= buffer.dim(2) {
            return buffer
        }
        let nSteps = (step + new.dim(2) - 1) / step
        let grown = MLXArray.zeros(
            [new.dim(0), new.dim(1), nSteps * step, new.dim(3)], dtype: new.dtype)
        guard var buffer else { return grown }
        if previous % step != 0 {
            buffer = buffer[.ellipsis, ..<previous, 0...]
        }
        return concatenated([buffer, grown], axis: 2)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        self.keys = reserve(self.keys, like: keys, previous: previous)
        self.values = reserve(self.values, like: values, previous: previous)
        offset += keys.dim(2)
        self.keys![.ellipsis, previous ..< offset, 0...] = keys
        self.values![.ellipsis, previous ..< offset, 0...] = values
        return (self.keys![.ellipsis, ..<offset, 0...], self.values![.ellipsis, ..<offset, 0...])
    }

    /// Append the indexer keys (`[B, 1, L, D]`) of the tokens the last
    /// ``update(keys:values:)`` added; returns every cached indexer key.
    ///
    /// Call after `update`: the indexer keys of the tokens at
    /// `offset - L ..< offset` are written there.
    public func updateIndex(_ keys: MLXArray) -> MLXArray {
        let end = offset
        let previous = end - keys.dim(2)
        precondition(previous >= 0, "MiniMaxM3Cache: index keys past the attention keys")
        indexKeys = reserve(indexKeys, like: keys, previous: previous)
        indexKeys![.ellipsis, previous ..< end, 0...] = keys
        return indexKeys![.ellipsis, ..<end, 0...]
    }

    public override var isTrimmable: Bool { true }

    /// Both buffers keep their rows; the next update overwrites the trimmed ones.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, max(n, 0))
        offset -= trimmed
        return trimmed
    }

    /// `[keys, values, indexKeys]` up to `offset`; empty while nothing is cached.
    public override var state: [MLXArray] {
        get {
            guard let keys, let values, let indexKeys, offset > 0 else { return [] }
            return [
                keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...],
                indexKeys[.ellipsis, ..<offset, 0...],
            ]
        }
        set {
            guard !newValue.isEmpty else {
                keys = nil
                values = nil
                indexKeys = nil
                offset = 0
                return
            }
            precondition(newValue.count == 3, "MiniMaxM3Cache state has 3 arrays")
            precondition(
                newValue[0].dim(2) == newValue[2].dim(2),
                "MiniMaxM3Cache index keys must cover every cached token")
            keys = newValue[0]
            values = newValue[1]
            indexKeys = newValue[2]
            offset = newValue[0].dim(2)
        }
    }

    public override func copy() -> any KVCache {
        let new = MiniMaxM3Cache()
        new.step = step
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), keys: \(keys?.shape.description ?? "-"), index: \(indexKeys?.shape.description ?? "-")"
    }
}
