// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Compiled single-token decode for ``TokenIterator``'s generic serial path
/// (issue #406, ported from ml-explore/mlx-swift-lm#550).
///
/// The caller keeps owning ordinary ``KVCacheSimple`` layers. This decoder
/// adopts their prefilled buffers into ``FixedCapacityKVCache`` instances,
/// runs the compiled step against those, and after every step points each
/// ``KVCacheSimple`` back at the fixed-capacity buffers with the host-known
/// offset. The caller's layers therefore stay valid ``KVCacheSimple`` caches
/// at every token boundary: conversation-cache reuse, trims, the cold tier and
/// prompt-cache serialization see the same class, layout and `state` they see
/// after eager decode.
///
/// Capacity grows in ``capacityStep`` blocks. Growing re-adopts the layers
/// into larger caches and builds a new compiled step; the old step is released,
/// which erases its traces, so a retrace never replays a stale graph.
final class SerialCompiledDecoder {

    /// Rows added per capacity block, matching ``KVCacheSimple/step``. Attention
    /// spans the block, so a larger step trades masked attention work for
    /// fewer retraces.
    static let capacityStep = 256

    private let model: any FixedCapacityKVCacheProviding
    private let layers: [KVCacheSimple]
    private var caches: [FixedCapacityKVCache]
    private var decode: (MLXArray) -> MLXArray

    /// Tokens represented in every layer. Host-side, so stepping never
    /// synchronizes the graph-valued cache positions.
    private(set) var processedTokenCount: Int

    /// Number of compiled steps built so far (one per capacity block).
    private(set) var compiledStepCount = 1

    var capacity: Int { caches[0].maxTokens }

    private init(
        model: any FixedCapacityKVCacheProviding, layers: [KVCacheSimple],
        processedTokenCount: Int, capacity: Int
    ) {
        self.model = model
        self.layers = layers
        self.processedTokenCount = processedTokenCount
        let caches = layers.map { FixedCapacityKVCache(adopting: $0, maxTokens: capacity) }
        self.caches = caches
        self.decode = makeFixedCapacityDecodeStep(model: model, caches: caches)
    }

    /// A decoder for `cache`, or `nil` when the model or cache cannot use the
    /// compiled path. Only an audited ``FixedCapacityKVCacheProviding`` model
    /// whose every layer is a prefilled, batch-1 ``KVCacheSimple`` (exactly that
    /// class, not a subclass) at one shared offset qualifies.
    ///
    /// - Parameter remainingTokens: tokens still to be fed, when known; bounds
    ///   the first capacity block.
    static func make(
        model: any LanguageModel, cache: [KVCache], remainingTokens: Int?
    ) -> SerialCompiledDecoder? {
        guard let provider = model as? any FixedCapacityKVCacheProviding, !cache.isEmpty
        else { return nil }
        var layers: [KVCacheSimple] = []
        layers.reserveCapacity(cache.count)
        for layer in cache {
            guard type(of: layer) == KVCacheSimple.self, let simple = layer as? KVCacheSimple,
                let keys = simple.keys, simple.values != nil, keys.ndim == 4, keys.dim(0) == 1
            else { return nil }
            layers.append(simple)
        }
        let offset = layers[0].offset
        guard offset > 0, layers.allSatisfy({ $0.offset == offset }) else { return nil }

        // The cache must have the topology the model's own fixed-capacity
        // factory builds (one full-attention layer per entry). Construction
        // allocates nothing.
        guard let probe = try? provider.newFixedCapacityCache(maxTokens: 1),
            probe.count == layers.count
        else { return nil }

        let capacity = Self.capacity(after: offset, remainingTokens: remainingTokens)
        guard capacity <= Int(Int32.max) else { return nil }
        return SerialCompiledDecoder(
            model: provider, layers: layers, processedTokenCount: offset, capacity: capacity)
    }

    /// Next capacity block: always above `offset`, at most one
    /// ``capacityStep`` block past it, and no larger than the tokens left.
    static func capacity(after offset: Int, remainingTokens: Int?) -> Int {
        let block = (offset / capacityStep + 1) * capacityStep
        guard let remainingTokens, remainingTokens > 0 else { return block }
        return min(block, offset + remainingTokens)
    }

    /// Whether `cache` is still exactly the layers this decoder writes back
    /// into, at the offset it last wrote. A replaced layer or an external trim
    /// means the decoder must be rebuilt (or decode falls back to eager).
    func owns(_ cache: [KVCache]) -> Bool {
        guard cache.count == layers.count else { return false }
        for (entry, layer) in zip(cache, layers) {
            guard entry === layer, layer.offset == processedTokenCount else { return false }
        }
        return true
    }

    /// Feed one `[1, 1]` token and return `[1, 1, vocab]` logits.
    ///
    /// - Parameter remainingTokens: tokens still to be fed including this one,
    ///   when known; bounds a capacity block this step has to open.
    func step(_ token: MLXArray, remainingTokens: Int?) -> MLXArray {
        precondition(token.shape == [1, 1], "compiled serial decode feeds one [1, 1] token")
        if processedTokenCount >= capacity {
            grow(remainingTokens: remainingTokens)
        }
        let logits = decode(token)
        processedTokenCount += 1
        for (cache, layer) in zip(caches, layers) {
            cache.writeBack(into: layer, offset: processedTokenCount)
        }
        return logits
    }

    private func grow(remainingTokens: Int?) {
        let next = Self.capacity(after: processedTokenCount, remainingTokens: remainingTokens)
        precondition(next > processedTokenCount && next <= Int(Int32.max))
        // The layers already hold every written row (write-back after each
        // step), so the new block re-adopts them. Assigning `decode` releases
        // the previous compiled function and erases its traces.
        let caches = layers.map { FixedCapacityKVCache(adopting: $0, maxTokens: next) }
        self.caches = caches
        self.decode = makeFixedCapacityDecodeStep(model: model, caches: caches)
        compiledStepCount += 1
    }
}
