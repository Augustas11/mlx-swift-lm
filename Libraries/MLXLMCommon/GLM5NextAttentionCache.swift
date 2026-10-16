import Foundation
import MLX

/// Cache for one sparse-attention layer: the MLA latent (`[B, 1, T, kv_lora_rank]`)
/// and the indexer's pooled keys plus the not-yet-complete pool window.
public final class GLM5NextAttentionCache: BaseKVCache {
    public let kpool: Int
    public var step = 256

    public private(set) var latent: MLXArray?
    public private(set) var pooledKeys: MLXArray?
    private var pendingKeys: MLXArray?
    private var pendingGates: MLXArray?

    public init(kpool: Int) {
        self.kpool = kpool
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [latent, pooledKeys, pendingKeys, pendingGates].compactMap { $0 }
    }

    /// Append `[B, 1, L, R]` and return the valid latent `[B, 1, offset, R]`.
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

    /// Add indexer keys and gate scores `[B, L, D]`; return the newly completed
    /// pool windows `[B, n * kpool, D]`, if any. The rest waits for later tokens.
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
        return (allKeys[0..., ..<usable], allGates[0..., ..<usable])
    }

    public func appendPooled(_ pooled: MLXArray) {
        pooledKeys = pooledKeys.map { concatenated([$0, pooled], axis: 1) } ?? pooled
    }

    /// `[latent, pooledKeys, pendingKeys, pendingGates]`; absent parts are empty arrays.
    public override var state: [MLXArray] {
        get {
            let empty = MLXArray.zeros([0])
            return [
                latent.map { $0[.ellipsis, ..<offset, 0...] } ?? empty,
                pooledKeys ?? empty, pendingKeys ?? empty, pendingGates ?? empty,
            ]
        }
        set {
            precondition(newValue.count == 4, "GLM5NextAttentionCache state has 4 arrays")
            func present(_ array: MLXArray) -> MLXArray? { array.size == 0 ? nil : array }
            latent = present(newValue[0])
            offset = latent?.dim(2) ?? 0
            pooledKeys = present(newValue[1])
            pendingKeys = present(newValue[2])
            pendingGates = present(newValue[3])
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
        let new = GLM5NextAttentionCache(kpool: kpool)
        new.step = step
        new.offset = offset
        new.latent = latent?[.ellipsis]
        new.pooledKeys = pooledKeys?[.ellipsis]
        new.pendingKeys = pendingKeys?[.ellipsis]
        new.pendingGates = pendingGates?[.ellipsis]
        return new
    }
}
