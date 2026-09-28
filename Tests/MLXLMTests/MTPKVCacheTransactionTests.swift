import MLX
import MLXLMCommon
import Testing

private func transactionKV(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
    let values = positions.flatMap { Array(repeating: Float($0), count: 2) }
    let keys = MLXArray(values, [1, 1, positions.count, 2])
    let vals = MLXArray(values.map { -$0 }, [1, 1, positions.count, 2])
    return (keys, vals)
}
private func transactionSharedKVState(span: Int, includeSources: Bool = true) -> LMOutput.State {
    let keys = MLXArray(Array(0 ..< span).map(Float.init), [1, 1, span, 1])
    let values = -keys
    var state = LMOutput.State()
    state[mtpSharedKVStatesKey] = [
        "full_attention": (keys, values),
        "sliding_attention": (keys, values),
    ]
    if includeSources {
        state[mtpSharedKVSourceIndicesKey] = [
            "full_attention": 0,
            "sliding_attention": 1,
        ]
    }
    state[mtpSharedKVOffsetsKey] = ["full_attention": span]
    return state
}

@Suite("MTP KV-cache transaction facade", .serialized)
struct MTPKVCacheTransactionTests {
    @Test func beginCommitKeepsAcceptedPrefix() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 3))

        let (keys, values) = transactionKV(0 ..< 3)
        _ = transaction.cache[0].update(keys: keys, values: values)

        #expect(transaction.writtenPositions == 3)
        let commit = try transaction.commit(retaining: 2)

        #expect(commit.committedPositions == 2)
        #expect(commit.discardedPositions == 1)
        #expect(commit.emittedLengths == [2])
        #expect(storage.processedTokenCount == 2)
        #expect(leaf.offset == 2)
        #expect(!storage.transactionIsOpen)
    }

    @Test func rollbackDiscardsWrittenRows() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 2))

        let (keys, values) = transactionKV(0 ..< 2)
        _ = transaction.cache[0].update(keys: keys, values: values)

        let commit = try transaction.rollback()

        #expect(commit.committedPositions == 0)
        #expect(commit.discardedPositions == 2)
        #expect(storage.processedTokenCount == 0)
        #expect(leaf.offset == 0)
        #expect(!storage.transactionIsOpen)
    }

    @Test func rewindLastTransactionTakesBackCommittedRows() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 3))

        let (keys, values) = transactionKV(0 ..< 3)
        _ = transaction.cache[0].update(keys: keys, values: values)
        _ = try transaction.commit(retaining: 3)

        let rewound = storage.rewindLastTransaction(2)

        #expect(rewound == 2)
        #expect(storage.processedTokenCount == 1)
        #expect(leaf.offset == 1)
    }

    @Test func unsupportedCacheCannotBeginTransaction() throws {
        let storage = try MTPKVCacheStorage(cache: [MambaCache()])

        #expect(storage.beginTransaction(maximumPositions: 1) == nil)
        #expect(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1) == nil)
        #expect(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: Int.min) == nil)
        #expect(!storage.transactionIsOpen)
        #expect(storage.processedTokenCount == 0)
    }

    @Test func rowAndPositionMetadataSurviveResolution() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let position = MTPKVCacheTransactionPosition(rowIndex: 7, queryOffset: 42)
        let transaction = try #require(
            storage.beginTransaction(maximumPositions: 1, position: position))

        #expect(transaction.position == position)
        #expect(transaction.position.rowIndex == 7)
        #expect(transaction.position.queryOffset == 42)

        let (keys, values) = transactionKV(0 ..< 1)
        _ = transaction.cache[0].update(keys: keys, values: values)
        _ = try transaction.commit(retaining: 1)

        #expect(transaction.position == position)
        #expect(storage.processedTokenCount == 1)
    }

    @Test func defaultPositionUsesCurrentCacheTimeline() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let first = try #require(storage.beginTransaction(maximumPositions: 2))
        let (firstKeys, firstValues) = transactionKV(0 ..< 2)
        _ = first.cache[0].update(keys: firstKeys, values: firstValues)
        _ = try first.commit(retaining: 2)

        let second = try #require(storage.beginTransaction(maximumPositions: 1))

        #expect(second.position.rowIndex == 0)
        #expect(second.position.queryOffset == 2)
        _ = try second.rollback()
    }

    @Test func resolvingTransactionTwiceThrows() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 1))

        _ = try transaction.rollback()

        #expect(throws: MTPKVCacheTransactionError.alreadyResolved) {
            try transaction.rollback()
        }
    }

    @Test func invalidOrNestedTransactionRefusesWithoutMutation() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])

        #expect(storage.beginTransaction(maximumPositions: 0) == nil)
        let first = try #require(storage.beginTransaction(maximumPositions: 1))
        #expect(storage.transactionIsOpen)
        #expect(storage.beginTransaction(maximumPositions: 1) == nil)
        #expect(leaf.offset == 0)

        _ = try first.rollback()
        #expect(!storage.transactionIsOpen)
    }

    @Test func invalidRetainingLeavesTransactionResolvable() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 2))
        let (keys, values) = transactionKV(0 ..< 2)
        _ = transaction.cache[0].update(keys: keys, values: values)

        #expect(
            throws: MTPKVCacheTransactionError.invalidRetainedPositions(
                retaining: 3, written: 2)
        ) {
            try transaction.commit(retaining: 3)
        }
        #expect(storage.transactionIsOpen)

        _ = try transaction.commit(retaining: 1)
        #expect(storage.processedTokenCount == 1)
        #expect(!storage.transactionIsOpen)
    }

    @Test func droppingStagedTransactionRollsBackAndUnlocksStorage() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])

        do {
            let transaction = try #require(storage.beginTransaction(maximumPositions: 2))
            let (keys, values) = transactionKV(0 ..< 2)
            _ = transaction.cache[0].update(keys: keys, values: values)
            #expect(leaf.offset == 2)
        }

        #expect(!storage.transactionIsOpen)
        #expect(storage.processedTokenCount == 0)
        #expect(leaf.offset == 0)
        _ = try #require(storage.beginTransaction(maximumPositions: 1)).rollback()
    }

    @Test func rotatingCachesReportPerLeafLengthsAndRewindPastWrap() throws {
        let narrow = RotatingKVCache(maxSize: 4, keep: 0)
        let wide = RotatingKVCache(maxSize: 8, keep: 0)
        let storage = try MTPKVCacheStorage(cache: [narrow, wide])
        storage.recordProcessedTokens(6)
        for position in 0 ..< 6 {
            let (keys, values) = transactionKV(position ..< (position + 1))
            _ = narrow.update(keys: keys, values: values)
            _ = wide.update(keys: keys, values: values)
        }

        let transaction = try #require(storage.beginTransaction(maximumPositions: 2))
        let (keys, values) = transactionKV(6 ..< 8)
        for cache in transaction.cache {
            _ = cache.update(keys: keys, values: values)
        }
        let commit = try transaction.commit(retaining: 2)

        #expect(commit.emittedLengths == [4, 8])
        #expect(storage.processedTokenCount == 8)
        #expect(storage.rewindLastTransaction(1) == 1)
        #expect(storage.processedTokenCount == 7)
        #expect(narrow.offset == 7)
        #expect(wide.offset == 7)
    }

    @Test func hybridCapabilityIsExplicitAndRollbackBeforeWriteIsSafe() throws {
        let storage = try MTPKVCacheStorage(cache: [KVCacheSimple(), MambaCache()])

        #expect(storage.beginTransaction(maximumPositions: 2) == nil)
        #expect(
            storage.beginTransaction(
                maximumPositions: 3, nativeRewindDepth: 2,
                unconditionallyRetainedPositions: 1) == nil,
            "the current native hybrid contract is deliberately MTP-1")

        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))
        #expect(transaction.mode == .nativeRewind)
        #expect(transaction.writtenPositions == 0)
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)
        #expect(state[mtpCacheCheckpointIndexKey] == 1)
        _ = try transaction.rollback()
        #expect(!storage.transactionIsOpen)
        #expect(storage.processedTokenCount == 0)
    }

    @Test func hybridCapabilityRefusesAttentionCacheThatWouldWrap() throws {
        let attention = RotatingKVCache(maxSize: 4, keep: 0)
        let recurrent = MambaCache()
        let (keys, values) = transactionKV(0 ..< 3)
        _ = attention.update(keys: keys, values: values)
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])

        #expect(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1) == nil)
        #expect(attention.offset == 3)
        #expect(!storage.transactionIsOpen)
    }

    @Test func publicSharedKVReconciliationDropsTailThenClampsHead() throws {
        var state: LMOutput.State? = transactionSharedKVState(span: 10)
        #expect(
            reconcileMTPSharedKVState(
                &state, discarding: 3,
                emittedLength: { $0 == 0 ? Int.max : 4 }))

        let sharedKV = try #require(state?[mtpSharedKVStatesKey])
        #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 7)
        #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 4)
        #expect(state?[mtpSharedKVOffsetsKey]?["full_attention"] == 7)
    }

    @Test func publicSharedKVReconciliationRefusesMissingSources() throws {
        var state: LMOutput.State? = transactionSharedKVState(
            span: 6, includeSources: false)

        #expect(
            !reconcileMTPSharedKVState(
                &state, discarding: 1, emittedLength: { _ in Int.max }))
        let sharedKV = try #require(state?[mtpSharedKVStatesKey])
        #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 6)
    }

    @Test func commitReconciliationRefusesOutOfRangeSourceLeaf() throws {
        let leaf = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [leaf])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 1))
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)
        #expect(state[mtpCacheCheckpointIndexKey] == nil)

        let (keys, values) = transactionKV(0 ..< 1)
        _ = transaction.cache[0].update(keys: keys, values: values)
        let commit = try transaction.commit(retaining: 1)

        state = transactionSharedKVState(span: 1)
        state[mtpSharedKVSourceIndicesKey]?["sliding_attention"] = 9
        var optionalState: LMOutput.State? = state
        #expect(!commit.reconcileSharedKVState(&optionalState))
        let sharedKV = try #require(optionalState?[mtpSharedKVStatesKey])
        #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 1)
    }

    @Test func commitReconciliationUsesItsOwnDiscardedTail() throws {
        let first = KVCacheSimple()
        let second = KVCacheSimple()
        let storage = try MTPKVCacheStorage(cache: [first, second])
        let transaction = try #require(storage.beginTransaction(maximumPositions: 3))
        let (keys, values) = transactionKV(0 ..< 3)
        for cache in transaction.cache {
            _ = cache.update(keys: keys, values: values)
        }
        let commit = try transaction.commit(retaining: 2)
        var state: LMOutput.State? = transactionSharedKVState(span: 3)

        #expect(commit.reconcileSharedKVState(&state))
        let sharedKV = try #require(state?[mtpSharedKVStatesKey])
        #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 2)
        #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 2)
        #expect(state?[mtpSharedKVOffsetsKey]?["full_attention"] == 2)
    }

    @Test func storageReconciliationRefusesOutOfRangeSourceLeaf() throws {
        let storage = try MTPKVCacheStorage(cache: [KVCacheSimple()])
        var state: LMOutput.State? = transactionSharedKVState(span: 2)

        #expect(!storage.reconcileSharedKVState(&state, discarding: 1))
        let sharedKV = try #require(state?[mtpSharedKVStatesKey])
        #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 2)
        #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 2)
    }
}
