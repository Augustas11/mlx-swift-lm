// Copyright © 2026 Apple Inc.

/// Row and query-position metadata for a speculative MTP cache transaction.
///
/// External schedulers can use this value to associate an opaque transaction
/// handle with one request row. The metadata does not mutate cache position.
public struct MTPKVCacheTransactionPosition: Sendable, Hashable {
    /// Scheduler-owned row index associated with the transaction.
    public let rowIndex: Int

    /// Absolute query position at which this transaction begins.
    public let queryOffset: Int

    /// Creates row metadata without changing cache state.
    public init(rowIndex: Int = 0, queryOffset: Int) {
        precondition(rowIndex >= 0, "rowIndex must be nonnegative")
        precondition(queryOffset >= 0, "queryOffset must be nonnegative")
        self.rowIndex = rowIndex
        self.queryOffset = queryOffset
    }
}

/// Cache strategy selected for one speculative MTP transaction.
public enum MTPKVCacheTransactionMode: Sendable, Hashable {
    /// Writes are isolated in an opaque staged cache round.
    case staged

    /// A hybrid attention/recurrent cache writes in place and restores a
    /// target-provided recurrent checkpoint when its speculative tail loses.
    case nativeRewind
}

/// Result of resolving an MTP cache transaction.
public struct MTPKVCacheTransactionCommit: Sendable, Hashable {
    /// Positions kept by the cache. The storage timeline advanced by this amount.
    public let committedPositions: Int

    /// Positions written during the round and then discarded.
    public let discardedPositions: Int

    /// Per cache leaf, how much of the emitted sequence remains addressable.
    public let emittedLengths: [Int]

    package init(_ commit: KVCacheRoundCommit) {
        self.init(
            committedPositions: commit.committedPositions,
            discardedPositions: commit.discardedPositions,
            emittedLengths: commit.emittedLengths)
    }

    package init(
        committedPositions: Int,
        discardedPositions: Int,
        emittedLengths: [Int]
    ) {
        self.committedPositions = committedPositions
        self.discardedPositions = discardedPositions
        self.emittedLengths = emittedLengths
    }

    /// Reconcile target-emitted shared K/V against this commit's cache lengths.
    ///
    /// Returns `false` when a snapshot cannot be mapped to its source cache
    /// leaves. Callers must stop speculating instead of guessing in that case.
    @discardableResult
    public func reconcileSharedKVState(
        _ state: inout LMOutput.State?
    ) -> Bool {
        if let sharedKV = state?[mtpSharedKVStatesKey] {
            guard let sources = state?[mtpSharedKVSourceIndicesKey],
                sharedKV.keys.allSatisfy({ key in
                    guard let source = sources[key] else { return false }
                    return emittedLengths.indices.contains(source)
                })
            else { return false }
        }
        return reconcileMTPSharedKVState(
            &state, discarding: discardedPositions,
            emittedLength: { emittedLengths[$0] })
    }
}

/// Errors raised while resolving an MTP cache transaction.
public enum MTPKVCacheTransactionError: Error, Equatable {
    /// The transaction was already committed or rolled back.
    case alreadyResolved

    /// The requested retained prefix falls outside the transaction's written range.
    case invalidRetainedPositions(retaining: Int, written: Int)

    /// A target-native recurrent checkpoint could not restore the requested tail.
    case nativeRewindFailed(requested: Int, actual: Int)

    /// The caller did not configure target state through the transaction before
    /// attempting to commit a native-rewind write.
    case nativeWriteNotConfigured

    /// The target write did not match the width and checkpoint contract of the
    /// native-rewind transaction.
    case invalidNativeWrite(
        expectedPositions: Int,
        attentionPositionDeltas: [Int],
        recurrentCheckpoints: Int,
        recurrentLeaves: Int
    )
}

private struct NativeRewindTransactionState {
    let maximumPositions: Int
    let attentionSlots: [Int]
    let attentionStartOffsets: [Int]
    let recurrentSnapshots: [Int: MambaCache]
}

/// Public owner for cache storage that supports speculative MTP transactions.
///
/// The owner and its transaction handles are intentionally not `Sendable`.
/// An external scheduler must serialize access by row. The concrete staging
/// strategies, cache plan, and round implementation remain package-scoped.
public final class MTPKVCacheStorage {
    package let storage: KVCacheStorage
    private var publicTransactionIsOpen = false

    /// Creates a transaction owner around a target model cache.
    ///
    /// The optional generation parameters are accepted for source compatibility
    /// with newer releases. The 3.31.4 cache graph has no typed cache plan, so
    /// the already-realized cache is adopted unchanged.
    public init(cache: [KVCache], parameters: GenerateParameters? = nil) throws {
        _ = parameters
        self.storage = KVCacheStorage(cache)
    }

    package init(storage: KVCacheStorage) {
        self.storage = storage
    }

    /// The realized cache array owned by this storage.
    public var cache: [KVCache] { storage.cache }

    /// Authoritative logical position represented by the cache.
    public var processedTokenCount: Int { storage.processedTokenCount }

    /// Whether a public transaction is currently open on this storage.
    public var transactionIsOpen: Bool { publicTransactionIsOpen }

    /// Begin one speculative cache transaction.
    ///
    /// Attention-only cache topologies use an isolated staged round. A hybrid
    /// attention/recurrent topology may use target-native rewind only when the
    /// caller supplies a sufficient model rewind depth and a nonempty prefix
    /// that is unconditionally retained. Current hybrid caches support one
    /// speculative position beyond that prefix.
    ///
    /// Returns `nil` for invalid widths, nested transactions, or cache/model
    /// combinations that cannot provide exact transaction semantics. Refusal
    /// leaves the cache unchanged.
    public func beginTransaction(
        maximumPositions: Int,
        nativeRewindDepth: Int = 0,
        unconditionallyRetainedPositions: Int = 0,
        position: MTPKVCacheTransactionPosition? = nil
    ) -> MTPKVCacheTransaction? {
        guard maximumPositions > 0, !publicTransactionIsOpen, !storage.roundIsOpen else {
            return nil
        }

        let position =
            position
            ?? MTPKVCacheTransactionPosition(queryOffset: storage.processedTokenCount)

        if let round = storage.beginRound(maximumPositions: maximumPositions) {
            publicTransactionIsOpen = true
            return MTPKVCacheTransaction(
                owner: self, implementation: .staged(round), position: position,
                unconditionallyRetainedPositions: 0)
        }

        guard unconditionallyRetainedPositions > 0,
            unconditionallyRetainedPositions <= maximumPositions
        else { return nil }
        let speculativePositions = maximumPositions - unconditionallyRetainedPositions
        let cache = storage.cache
        let leaves = KVCacheTree.leaves(in: cache)
        let attentionLeaves = leaves.filter(\.isAttentionCache)
        let recurrentLeaves = leaves.filter { !$0.isAttentionCache }
        guard
            speculativePositions == 1,
            nativeRewindDepth >= speculativePositions,
            leaves.count == cache.count,
            leaves.allSatisfy({ $0.path.count == 1 }),
            !attentionLeaves.isEmpty,
            !recurrentLeaves.isEmpty,
            attentionLeaves.allSatisfy({
                $0.cache.isTrimmable(after: maximumPositions)
            }),
            recurrentLeaves.allSatisfy({ $0.cache is MambaCache })
        else { return nil }

        // An all-accepted prior round intentionally retains its checkpoint so
        // generation can rewind committed lookahead during early finalization.
        // Opening the next round proves that lookahead was consumed; prevent
        // the old checkpoint from making the new handle look already written.
        discardSpeculativePromptCacheCheckpoints(cache)
        let nativeState = NativeRewindTransactionState(
            maximumPositions: maximumPositions,
            attentionSlots: attentionLeaves.map { $0.path[0] },
            attentionStartOffsets: attentionLeaves.map { $0.cache.offset },
            recurrentSnapshots: Dictionary(
                uniqueKeysWithValues: recurrentLeaves.map { leaf in
                    let live = leaf.cache as! MambaCache
                    return (leaf.path[0], live.copy() as! MambaCache)
                }))
        publicTransactionIsOpen = true
        return MTPKVCacheTransaction(
            owner: self, implementation: .nativeRewind(nativeState),
            position: position,
            unconditionallyRetainedPositions: unconditionallyRetainedPositions)
    }

    /// Record a successful non-transactional model write on the shared timeline.
    ///
    /// Use this for prefill or passthrough calls. Transaction commits update the
    /// timeline themselves and must not be recorded a second time.
    public func recordProcessedTokens(_ count: Int) {
        precondition(!publicTransactionIsOpen, "cannot record tokens during a transaction")
        storage.commitProcessedTokens(count)
    }

    /// Rewind positions retained by the most recent staged transaction.
    @discardableResult
    public func rewindLastTransaction(_ count: Int) -> Int {
        precondition(!publicTransactionIsOpen, "cannot rewind during a transaction")
        return storage.rewindLastRound(count)
    }

    /// Rewind committed-but-unemitted lookahead after generation stops early.
    ///
    /// Ordinary attention caches trim directly. Hybrid recurrent caches fall
    /// back to their target-native checkpoint.
    @discardableResult
    public func rewindCommittedLookahead(_ count: Int) -> Int {
        precondition(!publicTransactionIsOpen, "cannot rewind during a transaction")
        let trimmed = storage.trim(count)
        return trimmed > 0 ? trimmed : storage.rewindSpeculative(count)
    }

    /// Release any target-native recurrent checkpoint once no lookahead remains.
    public func discardNativeRewindCheckpoint() {
        precondition(!publicTransactionIsOpen, "cannot discard during a transaction")
        discardSpeculativePromptCacheCheckpoints(storage.cache)
    }

    /// Reconcile target-emitted shared K/V against the storage's current lengths.
    @discardableResult
    public func reconcileSharedKVState(
        _ state: inout LMOutput.State?, discarding: Int
    ) -> Bool {
        if let sharedKV = state?[mtpSharedKVStatesKey] {
            guard let sources = state?[mtpSharedKVSourceIndicesKey],
                sharedKV.keys.allSatisfy({ key in
                    guard let source = sources[key] else { return false }
                    return storage.cache.indices.contains(source)
                })
            else { return false }
        }
        return reconcileMTPSharedKVState(
            &state, discarding: discarding,
            emittedLength: storage.emittedLength(forLeaf:))
    }

    /// Create an independent cache-storage snapshot.
    public func copy() -> MTPKVCacheStorage {
        precondition(!publicTransactionIsOpen, "cannot copy during a transaction")
        return MTPKVCacheStorage(storage: storage.copy())
    }

    fileprivate func transactionDidResolve() {
        precondition(publicTransactionIsOpen, "resolving a transaction that is not open")
        publicTransactionIsOpen = false
    }
}

/// An open speculative MTP transaction over one row-owned cache storage.
///
/// Dropping an unresolved handle is safe: both strategies restore the cache to
/// its pre-transaction position. The handle and any cache references obtained
/// from it must not cross concurrent row owners.
public final class MTPKVCacheTransaction {
    fileprivate enum Implementation {
        case staged(KVCacheRound)
        case nativeRewind(NativeRewindTransactionState)
    }

    private let owner: MTPKVCacheStorage
    private let implementation: Implementation
    private let unconditionallyRetainedPositions: Int
    private var resolved = false
    private var nativeWriteWasConfigured = false

    /// Scheduler metadata supplied when the transaction began.
    public let position: MTPKVCacheTransactionPosition

    fileprivate init(
        owner: MTPKVCacheStorage,
        implementation: Implementation,
        position: MTPKVCacheTransactionPosition,
        unconditionallyRetainedPositions: Int
    ) {
        self.owner = owner
        self.implementation = implementation
        self.position = position
        self.unconditionallyRetainedPositions = unconditionallyRetainedPositions
    }

    deinit {
        guard !resolved else { return }
        switch implementation {
        case .staged(let round):
            _ = owner.storage.rollback(round)
        case .nativeRewind(let state):
            restoreNativeStart(state)
        }
        resolved = true
        owner.transactionDidResolve()
    }

    /// Cache array to pass to the target model for this transaction.
    public var cache: [KVCache] {
        switch implementation {
        case .staged(let round): round.caches
        case .nativeRewind: owner.storage.cache
        }
    }

    /// Configure target state for the model write owned by this transaction.
    ///
    /// The native-rewind strategy requests a recurrent checkpoint immediately
    /// after its unconditional prefix. The staged strategy clears that request.
    /// Call this before invoking the target model with ``cache``.
    public func configureTargetStateForWrite(_ state: inout LMOutput.State) {
        precondition(!resolved, "cannot configure a resolved transaction")
        switch implementation {
        case .staged:
            state[mtpCacheCheckpointIndexKey] = nil
        case .nativeRewind:
            state[mtpCacheCheckpointIndexKey] = unconditionallyRetainedPositions
            nativeWriteWasConfigured = true
        }
    }

    /// Strategy selected for this transaction.
    public var mode: MTPKVCacheTransactionMode {
        switch implementation {
        case .staged: .staged
        case .nativeRewind: .nativeRewind
        }
    }

    /// The widest write this transaction was opened to accept.
    public var maximumPositions: Int {
        switch implementation {
        case .staged(let round): round.maximumPositions
        case .nativeRewind(let state): state.maximumPositions
        }
    }

    /// Positions written through the transaction cache so far.
    public var writtenPositions: Int {
        switch implementation {
        case .staged(let round): return round.writtenPositions
        case .nativeRewind(let state):
            let deltas = nativeAttentionPositionDeltas(state)
            guard let first = deltas.first,
                deltas.allSatisfy({ $0 == first })
            else { return 0 }
            return first
        }
    }

    /// Keep the accepted prefix and discard the rejected tail.
    @discardableResult
    public func commit(retaining: Int) throws -> MTPKVCacheTransactionCommit {
        guard !resolved else { throw MTPKVCacheTransactionError.alreadyResolved }

        let result: MTPKVCacheTransactionCommit
        switch implementation {
        case .staged(let round):
            let written = round.writtenPositions
            guard retaining >= unconditionallyRetainedPositions, retaining <= written else {
                throw MTPKVCacheTransactionError.invalidRetainedPositions(
                    retaining: retaining, written: written)
            }
            result = MTPKVCacheTransactionCommit(
                owner.storage.commit(round, retaining: retaining))
        case .nativeRewind(let state):
            guard nativeWriteWasConfigured else {
                throw MTPKVCacheTransactionError.nativeWriteNotConfigured
            }
            let deltas = nativeAttentionPositionDeltas(state)
            let checkpointCount = nativeRecurrentCheckpointCount(state)
            guard deltas.allSatisfy({ $0 == state.maximumPositions }),
                checkpointCount == state.recurrentSnapshots.count
            else {
                throw MTPKVCacheTransactionError.invalidNativeWrite(
                    expectedPositions: state.maximumPositions,
                    attentionPositionDeltas: deltas,
                    recurrentCheckpoints: checkpointCount,
                    recurrentLeaves: state.recurrentSnapshots.count)
            }
            let written = state.maximumPositions
            guard retaining >= unconditionallyRetainedPositions, retaining <= written else {
                throw MTPKVCacheTransactionError.invalidRetainedPositions(
                    retaining: retaining, written: written)
            }
            let discarding = written - retaining
            if discarding > 0 {
                let rewound = rewindSpeculativePromptCache(
                    owner.storage.cache, numTokens: discarding)
                guard rewound == discarding else {
                    throw MTPKVCacheTransactionError.nativeRewindFailed(
                        requested: discarding, actual: rewound)
                }
            }
            owner.storage.commitProcessedTokens(retaining)
            result = MTPKVCacheTransactionCommit(
                committedPositions: retaining,
                discardedPositions: discarding,
                emittedLengths: owner.storage.cache.indices.map {
                    owner.storage.emittedLength(forLeaf: $0)
                })
        }

        resolved = true
        owner.transactionDidResolve()
        return result
    }

    /// Restore the cache to its pre-transaction state without advancing time.
    @discardableResult
    public func rollback() throws -> MTPKVCacheTransactionCommit {
        guard !resolved else { throw MTPKVCacheTransactionError.alreadyResolved }
        let discarded: Int
        switch implementation {
        case .staged(let round):
            discarded = round.writtenPositions
            _ = owner.storage.rollback(round)
        case .nativeRewind(let state):
            discarded = nativeAttentionPositionDeltas(state).max() ?? 0
            restoreNativeStart(state)
        }
        resolved = true
        owner.transactionDidResolve()
        return MTPKVCacheTransactionCommit(
            committedPositions: 0,
            discardedPositions: Swift.max(0, discarded),
            emittedLengths: owner.storage.cache.indices.map {
                owner.storage.emittedLength(forLeaf: $0)
            })
    }

    private func nativeAttentionPositionDeltas(
        _ state: NativeRewindTransactionState
    ) -> [Int] {
        zip(state.attentionSlots, state.attentionStartOffsets).map { slot, start in
            owner.storage.cache[slot].offset - start
        }
    }

    private func nativeRecurrentCheckpointCount(
        _ state: NativeRewindTransactionState
    ) -> Int {
        state.recurrentSnapshots.keys.reduce(into: 0) { result, slot in
            if (owner.storage.cache[slot] as? MambaCache)?.hasSpeculativeCheckpoint == true {
                result += 1
            }
        }
    }

    private func restoreNativeStart(_ state: NativeRewindTransactionState) {
        for (slot, start) in zip(state.attentionSlots, state.attentionStartOffsets) {
            let live = owner.storage.cache[slot]
            let appended = live.offset - start
            if appended > 0 {
                _ = live.trim(appended)
            }
        }
        for (slot, snapshot) in state.recurrentSnapshots {
            guard let live = owner.storage.cache[slot] as? MambaCache else { continue }
            snapshot.copyContents(to: live)
            live.discardSpeculativeCheckpoint()
        }
    }
}

/// Bring target-emitted MTP shared K/V into line with what the cache retained.
///
/// The rejected tail is removed before each entry is clamped to the amount its
/// source cache leaf can still describe. Returns `false` when source metadata is
/// missing, so the caller can stop speculation rather than reconcile a guess.
@discardableResult
public func reconcileMTPSharedKVState(
    _ state: inout LMOutput.State?,
    discarding: Int,
    emittedLength: (Int) -> Int
) -> Bool {
    precondition(discarding >= 0, "discarding must be nonnegative")
    guard let sharedKV = state?[mtpSharedKVStatesKey] else { return true }
    guard let sources = state?[mtpSharedKVSourceIndicesKey],
        sharedKV.keys.allSatisfy({ sources[$0] != nil }),
        sharedKV.values.allSatisfy({ $0.0.dim(-2) == $0.1.dim(-2) })
    else { return false }

    state?[mtpSharedKVStatesKey] = sharedKV.reduce(into: [:]) { result, entry in
        let (key, kv) = entry
        var length = kv.0.dim(-2)
        if discarding > 0 {
            length = Swift.max(0, length - discarding)
        }
        let bound = Swift.max(0, emittedLength(sources[key]!))
        let start = Swift.max(0, length - bound)
        result[key] = (
            kv.0[.ellipsis, start ..< length, 0...],
            kv.1[.ellipsis, start ..< length, 0...]
        )
    }
    if discarding > 0, let offsets = state?[mtpSharedKVOffsetsKey] {
        state?[mtpSharedKVOffsetsKey] = offsets.mapValues {
            Swift.max(0, $0 - discarding)
        }
    }
    return true
}
