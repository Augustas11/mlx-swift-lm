import Foundation
import MLX

/// The cache a GLM-5-Next sparse-attention layer reads and writes for one
/// sequence: the MLA latent and the indexer's pooled keys.
public protocol GLM5NextLatentCache: KVCache {
    /// Indexer keys per pooled key.
    var kpool: Int { get }

    /// Pooled indexer keys `[1, P, D]`, `nil` before the first complete window.
    var pooledKeys: MLXArray? { get }

    /// Append `[1, 1, L, R]` and return the valid latent `[1, 1, offset, R]`.
    func appendLatent(_ values: MLXArray) -> MLXArray

    /// Add indexer keys and gate scores `[1, L, D]`; return the newly completed
    /// pool windows `[1, n * kpool, D]`, if any. The rest waits for later tokens.
    func takeCompleteWindows(keys: MLXArray, gates: MLXArray) -> (MLXArray, MLXArray)?

    /// Append pooled keys `[1, n, D]` for the windows `takeCompleteWindows` returned.
    func appendPooled(_ pooled: MLXArray)
}

/// A sparse-attention layer cache for several sequences decoded in one
/// forward (continuous batching). The model projects every row together and
/// runs each row's attention against its own `rowCaches` entry, as a serial
/// decode of that row does.
public protocol GLM5NextBatchedLatentCache: KVCache {
    /// One cache per batch row, in row order.
    var rowCaches: [any GLM5NextLatentCache] { get }
}

/// `GLM5NextBatchedLatentCache` over existing per-sequence caches. The rows
/// carry the state; `offset` is unused.
public final class GLM5NextLatentCacheBatch: BaseKVCache, GLM5NextBatchedLatentCache {
    public let rowCaches: [any GLM5NextLatentCache]

    public init(rowCaches: [any GLM5NextLatentCache]) {
        precondition(!rowCaches.isEmpty, "GLM5NextLatentCacheBatch needs at least one row")
        self.rowCaches = rowCaches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        rowCaches.flatMap { $0.innerState() }
    }

    public override func copy() -> any KVCache {
        GLM5NextLatentCacheBatch(rowCaches: rowCaches.map { $0.copy() as! any GLM5NextLatentCache })
    }
}

/// `GLM5NextLatentCache` for one sequence: the latent `[B, 1, T, kv_lora_rank]`,
/// the pooled keys, the not-yet-complete pool window, and the raw keys and
/// gates of the last `kpool - 1 + maxTrim` tokens already folded into pools,
/// so `trim` of up to `maxTrim` tokens rebuilds the pool state exactly.
public final class GLM5NextAttentionCache: BaseKVCache, GLM5NextLatentCache {
    public let kpool: Int
    /// Tokens `trim` can always remove: the decode window plus the MTP
    /// proposal depth.
    public let maxTrim: Int
    public var step = 256

    public private(set) var latent: MLXArray?
    public private(set) var pooledKeys: MLXArray?
    private var pendingKeys: MLXArray?
    private var pendingGates: MLXArray?
    private var foldedKeys: MLXArray?
    private var foldedGates: MLXArray?

    public init(kpool: Int, maxTrim: Int = 16) {
        precondition(kpool >= 1 && maxTrim >= 0)
        self.kpool = kpool
        self.maxTrim = maxTrim
        super.init()
    }

    /// Folded raw keys kept for `trim`.
    private var foldedCapacity: Int { kpool - 1 + maxTrim }

    public override func innerState() -> [MLXArray] {
        [latent, pooledKeys, pendingKeys, pendingGates, foldedKeys, foldedGates].compactMap { $0 }
    }

    public func appendLatent(_ values: MLXArray) -> MLXArray {
        let previous = offset
        let length = values.dim(2)
        if latent == nil || previous + length > latent!.dim(2) {
            let steps = (step + length - 1) / step
            let grown = MLXArray.zeros(
                [values.dim(0), values.dim(1), steps * step, values.dim(3)], dtype: values.dtype)
            if var current = latent {
                if previous % step != 0 {
                    current = current[.ellipsis, ..<previous, 0...]
                }
                latent = concatenated([current, grown], axis: 2)
            } else {
                latent = grown
            }
        }
        offset += length
        latent![.ellipsis, previous ..< offset, 0...] = values
        return latent![.ellipsis, ..<offset, 0...]
    }

    public func takeCompleteWindows(keys: MLXArray, gates: MLXArray) -> (MLXArray, MLXArray)? {
        let allKeys = pendingKeys.map { concatenated([$0, keys], axis: 1) } ?? keys
        let allGates = pendingGates.map { concatenated([$0, gates], axis: 1) } ?? gates
        let total = allKeys.dim(1)
        let usable = (total / kpool) * kpool
        if usable < total {
            pendingKeys = contiguous(allKeys[0..., usable...])
            pendingGates = contiguous(allGates[0..., usable...])
        } else {
            pendingKeys = nil
            pendingGates = nil
        }
        guard usable > 0 else { return nil }
        let readyKeys = allKeys[0..., ..<usable]
        let readyGates = allGates[0..., ..<usable]
        foldedKeys = keepTail(foldedKeys, readyKeys)
        foldedGates = keepTail(foldedGates, readyGates)
        return (readyKeys, readyGates)
    }

    private func keepTail(_ kept: MLXArray?, _ added: MLXArray) -> MLXArray? {
        guard foldedCapacity > 0 else { return nil }
        let all = kept.map { concatenated([$0, added], axis: 1) } ?? added
        return all.dim(1) > foldedCapacity
            ? contiguous(all[0..., (all.dim(1) - foldedCapacity)...]) : all
    }

    public func appendPooled(_ pooled: MLXArray) {
        pooledKeys = pooledKeys.map { concatenated([$0, pooled], axis: 1) } ?? pooled
    }

    // MARK: - Trim

    /// Raw keys of the last tokens still on hand (folded, then pending).
    private var rawCount: Int { (foldedKeys?.dim(1) ?? 0) + (pendingKeys?.dim(1) ?? 0) }

    /// Whether this layer feeds an indexer (its own-indexer layers do; a
    /// shared-indexer layer keeps only the latent).
    private var indexed: Bool { pooledKeys != nil || rawCount > 0 }

    /// The largest `m <= n` whose trim leaves the raw keys of the new partial
    /// pool window on hand.
    private func exactTrimCount(_ n: Int) -> Int {
        var m = min(max(n, 0), offset)
        guard indexed else { return m }
        while m > 0 {
            let partial = (offset - m) % kpool
            if partial == 0 || rawCount - m >= partial { break }
            m -= 1
        }
        return m
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        !indexed || positions <= maxTrim
    }

    /// Remove the last `n` tokens (fewer when the kept raw keys cannot rebuild
    /// the partial window; up to `maxTrim` always succeed). The pools, the
    /// partial window and the latent are then those of the shorter sequence,
    /// bit for bit.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let m = exactTrimCount(n)
        guard m > 0 else { return 0 }
        let newOffset = offset - m
        if indexed, rawCount > 0 {
            let parts = [foldedKeys, pendingKeys].compactMap { $0 }
            let gateParts = [foldedGates, pendingGates].compactMap { $0 }
            let raw = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
            let rawGates = gateParts.count == 1 ? gateParts[0] : concatenated(gateParts, axis: 1)
            let kept = max(raw.dim(1) - m, 0)
            let pending = newOffset % kpool
            let folded = kept - pending
            pendingKeys = pending > 0 ? contiguous(raw[0..., folded ..< kept]) : nil
            pendingGates = pending > 0 ? contiguous(rawGates[0..., folded ..< kept]) : nil
            foldedKeys = folded > 0 ? contiguous(raw[0..., ..<folded]) : nil
            foldedGates = folded > 0 ? contiguous(rawGates[0..., ..<folded]) : nil
        }
        if let pooled = pooledKeys {
            let pools = newOffset / kpool
            pooledKeys = pools > 0 ? pooled[0..., ..<pools] : nil
        }
        offset = newOffset
        return m
    }

    // MARK: - Serialization

    /// `[latent, pooledKeys, pendingKeys, pendingGates, foldedKeys, foldedGates]`;
    /// absent parts are empty arrays. A 4-array state (no folded keys) loads too.
    public override var state: [MLXArray] {
        get {
            let empty = MLXArray.zeros([0])
            return [
                offset > 0 ? latent.map { $0[.ellipsis, ..<offset, 0...] } ?? empty : empty,
                pooledKeys ?? empty, pendingKeys ?? empty, pendingGates ?? empty,
                foldedKeys ?? empty, foldedGates ?? empty,
            ]
        }
        set {
            precondition(
                newValue.count == 4 || newValue.count == 6,
                "GLM5NextAttentionCache state has 4 or 6 arrays")
            func present(_ array: MLXArray) -> MLXArray? { array.size == 0 ? nil : array }
            latent = present(newValue[0])
            offset = latent?.dim(2) ?? 0
            pooledKeys = present(newValue[1])
            pendingKeys = present(newValue[2])
            pendingGates = present(newValue[3])
            foldedKeys = newValue.count == 6 ? present(newValue[4]) : nil
            foldedGates = newValue.count == 6 ? present(newValue[5]) : nil
        }
    }

    public override var metaState: [String] {
        get { ["\(kpool)"] }
        set {
            precondition(
                newValue == ["\(kpool)"], "GLM5NextAttentionCache pool width mismatch")
        }
    }

    public override func copy() -> any KVCache {
        let new = GLM5NextAttentionCache(kpool: kpool, maxTrim: maxTrim)
        new.step = step
        new.offset = offset
        new.latent = latent?[.ellipsis]
        new.pooledKeys = pooledKeys?[.ellipsis]
        new.pendingKeys = pendingKeys?[.ellipsis]
        new.pendingGates = pendingGates?[.ellipsis]
        new.foldedKeys = foldedKeys?[.ellipsis]
        new.foldedGates = foldedGates?[.ellipsis]
        return new
    }
}
