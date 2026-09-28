import MLX
import Testing

@testable import MLXLMCommon

@Suite("MTP KV-cache native-rewind facade", .serialized)
struct MTPKVCacheTransactionInternalTests {
    @Test func nativeTransactionClearsStaleCheckpointBeforeWrite() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        recurrent.saveSpeculativeCheckpoint(
            convState: MLXArray.ones([1, 1, 4]),
            recurrentState: MLXArray.ones([1, 2, 2, 2]),
            advancedBy: 1)
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])

        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))

        #expect(transaction.writtenPositions == 0)
        #expect(!recurrent.hasSpeculativeCheckpoint)
        _ = try transaction.rollback()
    }

    @Test func nativeRewindCommitRestoresRejectedTail() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])
        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)

        let keys = MLXArray.zeros([1, 1, 2, 2])
        _ = transaction.cache[0].update(keys: keys, values: keys)
        recurrent.saveSpeculativeCheckpoint(
            convState: MLXArray.ones([1, 1, 4]),
            recurrentState: MLXArray.ones([1, 2, 2, 2]),
            advancedBy: 1)
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        #expect(transaction.writtenPositions == 2)
        let commit = try transaction.commit(retaining: 1)

        #expect(commit.committedPositions == 1)
        #expect(commit.discardedPositions == 1)
        #expect(commit.emittedLengths == [1, 1])
        #expect(attention.offset == 1)
        #expect(storage.processedTokenCount == 1)
        #expect(!recurrent.hasSpeculativeCheckpoint)
    }

    @Test func missingTargetStateConfigurationRefusesCommitAndRollsBackExactly() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        recurrent[0] = MLXArray.ones([1, 1, 4])
        recurrent[1] = MLXArray.ones([1, 2, 2, 2])
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])
        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))

        let keys = MLXArray.zeros([1, 1, 2, 2])
        _ = transaction.cache[0].update(keys: keys, values: keys)
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        #expect(
            throws: MTPKVCacheTransactionError.nativeWriteNotConfigured
        ) {
            try transaction.commit(retaining: 1)
        }
        let rollback = try transaction.rollback()

        #expect(rollback.discardedPositions == 2)
        #expect(attention.offset == 0)
        #expect(allClose(recurrent[0]!, MLXArray.ones([1, 1, 4])).item(Bool.self))
        #expect(allClose(recurrent[1]!, MLXArray.ones([1, 2, 2, 2])).item(Bool.self))
        #expect(!recurrent.hasSpeculativeCheckpoint)
        #expect(storage.processedTokenCount == 0)
    }

    @Test func missingNativeCheckpointRefusesCommitAndRemainsResolvable() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])
        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)

        let keys = MLXArray.zeros([1, 1, 2, 2])
        _ = transaction.cache[0].update(keys: keys, values: keys)

        #expect(
            throws: MTPKVCacheTransactionError.invalidNativeWrite(
                expectedPositions: 2,
                attentionPositionDeltas: [2],
                recurrentCheckpoints: 0,
                recurrentLeaves: 1)
        ) {
            try transaction.commit(retaining: 1)
        }
        _ = try transaction.rollback()

        #expect(attention.offset == 0)
        #expect(!storage.transactionIsOpen)
    }

    @Test func wrongNativeWriteWidthRefusesCommitAndRemainsResolvable() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])
        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)

        let keys = MLXArray.zeros([1, 1, 1, 2])
        _ = transaction.cache[0].update(keys: keys, values: keys)
        recurrent.saveSpeculativeCheckpoint(
            convState: MLXArray.ones([1, 1, 4]),
            recurrentState: MLXArray.ones([1, 2, 2, 2]),
            advancedBy: 1)

        #expect(
            throws: MTPKVCacheTransactionError.invalidNativeWrite(
                expectedPositions: 2,
                attentionPositionDeltas: [1],
                recurrentCheckpoints: 1,
                recurrentLeaves: 1)
        ) {
            try transaction.commit(retaining: 1)
        }
        _ = try transaction.rollback()

        #expect(attention.offset == 0)
        #expect(!recurrent.hasSpeculativeCheckpoint)
        #expect(!storage.transactionIsOpen)
    }

    @Test func partialNativeCheckpointsRefuseCommitAndRestoreAllRecurrentLeaves() throws {
        let attention = KVCacheSimple()
        let first = MambaCache()
        let second = MambaCache()
        first[0] = MLXArray.ones([1, 1, 4])
        first[1] = MLXArray.ones([1, 2, 2, 2])
        second[0] = MLXArray.ones([1, 1, 4]) * 2
        second[1] = MLXArray.ones([1, 2, 2, 2]) * 2
        let storage = try MTPKVCacheStorage(cache: [attention, first, second])
        let transaction = try #require(
            storage.beginTransaction(
                maximumPositions: 2, nativeRewindDepth: 1,
                unconditionallyRetainedPositions: 1))
        var state = LMOutput.State()
        transaction.configureTargetStateForWrite(&state)

        let keys = MLXArray.zeros([1, 1, 2, 2])
        _ = transaction.cache[0].update(keys: keys, values: keys)
        first.saveSpeculativeCheckpoint(
            convState: MLXArray.ones([1, 1, 4]),
            recurrentState: MLXArray.ones([1, 2, 2, 2]),
            advancedBy: 1)
        first[0] = MLXArray.zeros([1, 1, 4])
        second[0] = MLXArray.zeros([1, 1, 4])

        #expect(
            throws: MTPKVCacheTransactionError.invalidNativeWrite(
                expectedPositions: 2,
                attentionPositionDeltas: [2],
                recurrentCheckpoints: 1,
                recurrentLeaves: 2)
        ) {
            try transaction.commit(retaining: 1)
        }
        _ = try transaction.rollback()

        #expect(attention.offset == 0)
        #expect(allClose(first[0]!, MLXArray.ones([1, 1, 4])).item(Bool.self))
        #expect(allClose(second[0]!, MLXArray.ones([1, 1, 4]) * 2).item(Bool.self))
        #expect(!first.hasSpeculativeCheckpoint)
        #expect(!second.hasSpeculativeCheckpoint)
    }

    @Test func droppingNativeRewindTransactionRestoresPreTransactionState() throws {
        let attention = KVCacheSimple()
        let recurrent = MambaCache()
        recurrent[0] = MLXArray.ones([1, 1, 4])
        recurrent[1] = MLXArray.ones([1, 2, 2, 2])
        let storage = try MTPKVCacheStorage(cache: [attention, recurrent])

        do {
            let transaction = try #require(
                storage.beginTransaction(
                    maximumPositions: 2, nativeRewindDepth: 1,
                    unconditionallyRetainedPositions: 1))
            let keys = MLXArray.zeros([1, 1, 2, 2])
            _ = transaction.cache[0].update(keys: keys, values: keys)
            recurrent.saveSpeculativeCheckpoint(
                convState: MLXArray.ones([1, 1, 4]),
                recurrentState: MLXArray.ones([1, 2, 2, 2]),
                advancedBy: 1)
            recurrent[0] = MLXArray.zeros([1, 1, 4])
            recurrent[1] = MLXArray.zeros([1, 2, 2, 2])
            withExtendedLifetime(transaction) {}
        }

        #expect(attention.offset == 0)
        #expect(storage.processedTokenCount == 0)
        #expect(!storage.transactionIsOpen)
        #expect(!recurrent.hasSpeculativeCheckpoint)
        #expect(allClose(recurrent[0]!, MLXArray.ones([1, 1, 4])).item(Bool.self))
        #expect(allClose(recurrent[1]!, MLXArray.ones([1, 2, 2, 2])).item(Bool.self))
    }
}
