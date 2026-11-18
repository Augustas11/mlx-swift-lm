import Foundation
import MLX

/// The cache a Qwen4-Exp full-attention layer reads and writes for one
/// sequence: the attention keys and values (`update(keys:values:)`) and the
/// QSA indexer's raw keys of the same tokens (`updateIndexer`).
public protocol Qwen4ExpSparseCache: KVCache {
    /// Tokens the indexer has seen; equal to `offset` between forwards.
    var indexerOffset: Int { get }

    /// Append indexer keys `[1, L, D]` and return every cached one `[1, T, D]`.
    func updateIndexer(keys: MLXArray) -> MLXArray
}

/// A Qwen4-Exp full-attention layer cache for several sequences decoded in
/// one forward (continuous batching). The model projects every row together
/// and runs each row's norms, RoPE, indexer block selection and attention
/// against its own `rowCaches` entry, as a serial forward of that row does.
public protocol Qwen4ExpBatchedSparseCache: KVCache {
    /// One cache per batch row, in row order.
    var rowCaches: [any Qwen4ExpSparseCache] { get }

    /// Valid leading columns per row of the current forward (right-padded
    /// rows), or `nil` when every column of every row is valid. A padded
    /// column attends nothing and writes no cache.
    var rowLengths: [Int]? { get }
}

/// `Qwen4ExpBatchedSparseCache` over existing per-sequence caches. The rows
/// carry the state; `offset` is unused.
public final class Qwen4ExpAttentionCacheBatch: BaseKVCache, Qwen4ExpBatchedSparseCache {
    public let rowCaches: [any Qwen4ExpSparseCache]
    public private(set) var rowLengths: [Int]?

    public init(rowCaches: [any Qwen4ExpSparseCache]) {
        precondition(!rowCaches.isEmpty, "Qwen4ExpAttentionCacheBatch needs at least one row")
        self.rowCaches = rowCaches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        rowCaches.flatMap { $0.innerState() }
    }

    public override var isTrimmable: Bool { false }

    public override func copy() -> any KVCache {
        Qwen4ExpAttentionCacheBatch(
            rowCaches: rowCaches.map { $0.copy() as! any Qwen4ExpSparseCache })
    }

    /// Ragged packed forward: valid leading columns per row.
    public override func prepare(lengths: [Int]?) {
        rowLengths = lengths
    }

    public override func finalize() {
        rowLengths = nil
    }
}

/// The cache of one Qwen4-Exp (Qwen3.8-Flash-Next) full-attention layer: the
/// attention keys and values, and the QSA indexer's raw keys.
///
/// Mirrors mlx-lm's `CacheList(KVCache(), KVCache())` for `qwen4_exp`: the
/// indexer keys grow with the attention keys, and nothing reads indexer
/// values, so they are stored one wide.
///
/// It is not trimmable: the model's linear-attention layers keep recurrent
/// state that cannot be rewound, so trimming one layer alone would leave the
/// model inconsistent.
public final class Qwen4ExpAttentionCache: BaseKVCache, Qwen4ExpSparseCache {
    private var kv = KVCacheSimple()
    private var indexer = KVCacheSimple()

    public override init() {
        super.init()
    }

    /// Tokens the indexer has seen; equal to `offset` between forwards.
    public var indexerOffset: Int { indexer.offset }

    public override func innerState() -> [MLXArray] {
        kv.innerState() + indexer.innerState()
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = kv.update(keys: keys, values: values)
        offset = kv.offset
        return result
    }

    /// Append indexer keys `[B, L, D]` and return every cached one `[B, T, D]`.
    public func updateIndexer(keys: MLXArray) -> MLXArray {
        let k = expandedDimensions(keys, axis: 1)
        let unused = MLXArray.zeros([keys.dim(0), 1, keys.dim(1), 1], dtype: keys.dtype)
        let (all, _) = indexer.update(keys: k, values: unused)
        return all.squeezed(axis: 1)
    }

    public override var state: [MLXArray] {
        get {
            let attention = kv.state
            let index = indexer.state
            guard !attention.isEmpty, !index.isEmpty else { return [] }
            return attention + index
        }
        set {
            guard newValue.count == 4 else {
                fatalError(
                    "Qwen4ExpAttentionCache state must have 4 arrays (keys, values, indexer keys, indexer values)"
                )
            }
            kv.state = Array(newValue[0 ..< 2])
            indexer.state = Array(newValue[2 ..< 4])
            precondition(
                kv.offset == indexer.offset,
                "Qwen4ExpAttentionCache: attention and indexer lengths differ")
            offset = kv.offset
        }
    }

    public override var isTrimmable: Bool { false }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpAttentionCache()
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }
}
