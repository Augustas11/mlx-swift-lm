// Copyright © 2026 Apple Inc.

import MLX

/// Minimal cache classification used by the public MTP transaction facade.
///
/// Kept independent of the later typed cache-configuration subsystem so this
/// backport preserves the 3.31.4 dependency and runtime graph.
package struct KVCacheLeaf {
    package enum Kind {
        case recurrent
        case rotating(RotatingKVCache)
        case attention
    }

    package let path: [Int]
    package let cache: KVCache

    package var kind: Kind {
        switch cache {
        case is MambaCache, is ArraysCache:
            .recurrent
        case let rotating as RotatingKVCache:
            .rotating(rotating)
        default:
            .attention
        }
    }

    package var isAttentionCache: Bool {
        if case .recurrent = kind { return false }
        return true
    }
}

package enum KVCacheTree {
    package static func leaves(in cache: [KVCache]) -> [KVCacheLeaf] {
        var leaves: [KVCacheLeaf] = []
        for (index, entry) in cache.enumerated() {
            appendLeaves(from: entry, path: [index], to: &leaves)
        }
        return leaves
    }

    private static func appendLeaves(
        from cache: KVCache,
        path: [Int],
        to leaves: inout [KVCacheLeaf]
    ) {
        if let list = cache as? CacheList {
            for (index, child) in list.children.enumerated() {
                appendLeaves(from: child, path: path + [index], to: &leaves)
            }
        } else {
            leaves.append(KVCacheLeaf(path: path, cache: cache))
        }
    }
}

package protocol KVCacheRoundStrategy: AnyObject {
    var slot: Int { get }
    var presented: KVCache { get }
    var writtenPositions: Int { get }
    var liveOffset: Int { get }
    var attentionBound: Int? { get }
    func commit(retaining: Int)
    func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)?
}

extension KVCacheRoundStrategy {
    package func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)? { nil }

    package var emittedLength: Int {
        Swift.min(liveOffset, attentionBound ?? .max)
    }
}

package protocol KVCacheLeafRestorePoint {
    func canRestore(into leaf: KVCache) -> Bool
    @discardableResult
    func restore(into leaf: KVCache, retaining: Int) -> Bool
}

private final class AppendOnlyMTPRoundStrategy: KVCacheRoundStrategy {
    let slot: Int
    let live: KVCache
    private let startOffset: Int

    init(slot: Int, live: KVCache) {
        self.slot = slot
        self.live = live
        self.startOffset = live.offset
    }

    var presented: KVCache { live }
    var writtenPositions: Int { live.offset - startOffset }
    var liveOffset: Int { live.offset }
    var attentionBound: Int? { live.maxSize }

    func commit(retaining: Int) {
        let written = writtenPositions
        let keep = Swift.min(Swift.max(retaining, 0), written)
        if written > keep {
            precondition(
                live.trim(written - keep) == written - keep,
                "MTP transaction cache failed exact trim")
        }
    }
}

private final class RotatingMTPRoundStrategy: KVCacheRoundStrategy {
    let slot: Int
    let live: RotatingKVCache
    let staged: RotatingStagedKVCache

    init(slot: Int, live: RotatingKVCache) {
        self.slot = slot
        self.live = live
        self.staged = RotatingStagedKVCache(live: live)
    }

    var presented: KVCache { staged }
    var writtenPositions: Int { staged.stagedCount }
    var liveOffset: Int { live.offset }
    var attentionBound: Int? { live.maxSize }

    func commit(retaining: Int) {
        staged.commit(retaining: retaining)
    }

    func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)? {
        guard !live.isTrimmable(after: retaining) else { return nil }
        let state = live.state
        guard state.count == 2 else { return nil }
        return RotatingMTPRestorePoint(
            state: state.map { $0[.ellipsis] },
            metaState: live.metaState,
            replay: staged.stagedArrays)
    }
}

private struct RotatingMTPRestorePoint: KVCacheLeafRestorePoint {
    let state: [MLXArray]
    let metaState: [String]
    let replay: (MLXArray, MLXArray)?

    func canRestore(into leaf: KVCache) -> Bool { leaf is RotatingKVCache }

    @discardableResult
    func restore(into leaf: KVCache, retaining: Int) -> Bool {
        guard let live = leaf as? RotatingKVCache else { return false }
        live.state = state
        live.metaState = metaState
        guard retaining > 0, let replay else { return true }
        _ = live.update(
            keys: replay.0[.ellipsis, ..<retaining, 0...],
            values: replay.1[.ellipsis, ..<retaining, 0...])
        return true
    }
}

package final class KVCacheRound {
    let strategies: [any KVCacheRoundStrategy]
    package let caches: [KVCache]
    package let maximumPositions: Int

    init(strategies: [any KVCacheRoundStrategy], maximumPositions: Int) {
        self.strategies = strategies
        self.caches = strategies.map(\.presented)
        self.maximumPositions = maximumPositions
    }

    package var writtenPositions: Int {
        guard let first = strategies.first else { return 0 }
        let written = first.writtenPositions
        guard strategies.allSatisfy({ $0.writtenPositions == written }) else { return 0 }
        return written
    }
}

package struct KVCacheRoundCommit {
    package let committedPositions: Int
    package let discardedPositions: Int
    package let emittedLengths: [Int]
}

package final class KVCacheStorage {
    package var cache: [KVCache]
    private var openRound: KVCacheRound?

    private struct LeafRewind {
        let slot: Int
        let point: (any KVCacheLeafRestorePoint)?
    }

    private struct CompletedRound {
        let committedPositions: Int
        let leaves: [LeafRewind]
    }

    private var lastRound: CompletedRound?
    package private(set) var processedTokenCount: Int
    package var roundIsOpen: Bool { openRound != nil }

    package init(_ cache: [KVCache], processedTokenCount: Int? = nil) {
        self.cache = cache
        self.processedTokenCount =
            processedTokenCount ?? Self.inferProcessedTokenCount(from: cache)
    }

    package func commitProcessedTokens(_ count: Int) {
        precondition(count >= 0, "processed token count cannot move backwards")
        let (updated, overflow) = processedTokenCount.addingReportingOverflow(count)
        precondition(!overflow, "processed token count overflow")
        processedTokenCount = updated
        lastRound = nil
    }

    @discardableResult
    package func trim(_ count: Int) -> Int {
        precondition(count >= 0 && !roundIsOpen)
        let trimmed = trimPromptCache(cache, numTokens: count)
        processedTokenCount -= Swift.min(trimmed, processedTokenCount)
        lastRound = nil
        return trimmed
    }

    @discardableResult
    package func rewindSpeculative(_ count: Int) -> Int {
        precondition(count >= 0 && !roundIsOpen)
        let rewound = rewindSpeculativePromptCache(cache, numTokens: count)
        processedTokenCount -= Swift.min(rewound, processedTokenCount)
        return rewound
    }

    package func beginRound(maximumPositions: Int) -> KVCacheRound? {
        precondition(!roundIsOpen)
        let leaves = KVCacheTree.leaves(in: cache)
        guard leaves.count == cache.count,
            leaves.allSatisfy({ $0.path.count == 1 && $0.isAttentionCache })
        else { return nil }

        var strategies: [any KVCacheRoundStrategy] = []
        for leaf in leaves {
            let strategy: (any KVCacheRoundStrategy)?
            switch leaf.kind {
            case .rotating(let rotating):
                strategy = RotatingMTPRoundStrategy(slot: leaf.path[0], live: rotating)
            case .attention:
                strategy = leaf.cache.isTrimmable(after: maximumPositions)
                    ? AppendOnlyMTPRoundStrategy(slot: leaf.path[0], live: leaf.cache)
                    : nil
            case .recurrent:
                strategy = nil
            }
            guard let strategy else { return nil }
            strategies.append(strategy)
        }

        lastRound = nil
        let round = KVCacheRound(strategies: strategies, maximumPositions: maximumPositions)
        openRound = round
        return round
    }

    package func commit(_ round: KVCacheRound, retaining: Int) -> KVCacheRoundCommit {
        precondition(openRound === round)
        let written = round.writtenPositions
        precondition((0 ... written).contains(retaining))
        var rewinds: [LeafRewind] = []
        for strategy in round.strategies {
            rewinds.append(
                LeafRewind(
                    slot: strategy.slot,
                    point: strategy.makeRestorePoint(retaining: retaining)))
            strategy.commit(retaining: retaining)
        }
        processedTokenCount += retaining
        openRound = nil
        lastRound = rewinds.contains(where: { $0.point != nil })
            ? CompletedRound(committedPositions: retaining, leaves: rewinds)
            : nil
        return KVCacheRoundCommit(
            committedPositions: retaining,
            discardedPositions: written - retaining,
            emittedLengths: round.strategies.map(\.emittedLength))
    }

    package func rollback(_ round: KVCacheRound) -> KVCacheRoundCommit {
        commit(round, retaining: 0)
    }

    package func rewindLastRound(_ count: Int) -> Int {
        precondition(count >= 0 && !roundIsOpen)
        guard count > 0 else { return 0 }
        guard let lastRound else { return trim(count) }
        let rewound = Swift.min(count, lastRound.committedPositions)
        guard rewound > 0 else { return 0 }

        let resolved = lastRound.leaves.map { record -> (LeafRewind, KVCache)? in
            guard cache.indices.contains(record.slot) else { return nil }
            let leaf = cache[record.slot]
            if let point = record.point {
                return point.canRestore(into: leaf) ? (record, leaf) : nil
            }
            return leaf.isTrimmable ? (record, leaf) : nil
        }
        guard resolved.allSatisfy({ $0 != nil }) else { return trim(count) }
        for (record, leaf) in resolved.compactMap({ $0 }) {
            if let point = record.point {
                _ = point.restore(
                    into: leaf,
                    retaining: lastRound.committedPositions - rewound)
            } else {
                precondition(leaf.trim(rewound) == rewound)
            }
        }
        processedTokenCount -= rewound
        self.lastRound = nil
        return rewound
    }

    package func emittedLength(forLeaf index: Int) -> Int {
        guard cache.indices.contains(index) else { return processedTokenCount }
        return Swift.min(processedTokenCount, cache[index].maxSize ?? .max)
    }

    package func copy() -> KVCacheStorage {
        precondition(!roundIsOpen)
        return KVCacheStorage(
            cache.map { $0.copy() }, processedTokenCount: processedTokenCount)
    }

    private static func inferProcessedTokenCount(from cache: [KVCache]) -> Int {
        let leaves = KVCacheTree.leaves(in: cache)
        let attentionOffsets = leaves.filter(\.isAttentionCache).map { $0.cache.offset }
        return attentionOffsets.min() ?? leaves.map { $0.cache.offset }.min() ?? 0
    }
}
