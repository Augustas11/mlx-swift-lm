// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private let packedOutputStateKey = LMOutput.Key<Int>("tests.mtp.packed.output")

private final class PackedVerificationCache: MTPPackedVerificationCache {
    private(set) var preparedRowMaps: [MTPPackedVerificationRowMap]?
    private(set) var activeRowMaps: [MTPPackedVerificationRowMap]?
    private(set) var baseTokenColumns: [Int] = []
    private(set) var proposalColumnRanges: [Range<Int>] = []
    private(set) var stagedCommitCandidatePositions: [[Int]] = []
    private(set) var packedPrepareCallCount = 0
    private(set) var preparedLengths: [Int]?
    private(set) var activeLengths: [Int]?
    private(set) var prepareCallCount = 0
    private(set) var finalizeCallCount = 0
    var batchOffset: MLXArray
    let maxSize: Int?
    let mtpPackedCheckpointIndex: Int?

    init(offsets: [Int], maxSize: Int? = nil, checkpointIndex: Int? = nil) {
        self.batchOffset = MLXArray(int64: offsets)
        self.maxSize = maxSize
        self.mtpPackedCheckpointIndex = checkpointIndex
    }

    var offset: Int { batchOffset.asArray(Int.self).max() ?? 0 }

    var ropeOffset: RoPEOffset { .batch(batchOffset) }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    var isTrimmable: Bool { false }

    func trim(_: Int) -> Int { 0 }

    func makeMask(
        n _: Int, windowSize _: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }

    func innerState() -> [MLXArray] { [] }

    func prepareMTPPackedVerification(rowMaps: [MTPPackedVerificationRowMap]) throws {
        packedPrepareCallCount += 1
        preparedRowMaps = rowMaps
        activeRowMaps = rowMaps
        baseTokenColumns = rowMaps.map { _ in 0 }
        proposalColumnRanges = rowMaps.map { 1 ..< $0.inputCount }
        stagedCommitCandidatePositions = rowMaps.map { map in
            (1 ..< map.inputCount).map { map.queryOffset + $0 }
        }
    }

    func prepare(lengths: [Int]?) {
        prepareCallCount += 1
        preparedLengths = lengths
        activeLengths = lengths
    }

    func finalize() {
        finalizeCallCount += 1
        activeLengths = nil
        activeRowMaps = nil
    }

    func copy() -> any KVCache {
        PackedVerificationCache(
            offsets: batchOffset.asArray(Int.self),
            maxSize: maxSize,
            checkpointIndex: mtpPackedCheckpointIndex)
    }

    func finishPackedForward(wrongOffsetRow: Int? = nil) {
        guard let activeRowMaps else { return }
        let offsetsWithOverflow = activeRowMaps.map {
            $0.queryOffset.addingReportingOverflow($0.inputCount)
        }
        guard offsetsWithOverflow.allSatisfy({ !$0.overflow }) else { return }
        var offsets = offsetsWithOverflow.map(\.partialValue)
        if let wrongOffsetRow {
            offsets[wrongOffsetRow] += 1
        }
        batchOffset = MLXArray(int64: offsets)
    }
}

private final class PackedVerificationModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    private(set) var callCount = 0
    private(set) var receivedTokens: [Int] = []
    private(set) var receivedMask: [Int] = []
    private(set) var receivedEmitFlag: Bool?
    private(set) var receivedCheckpointIndex: Int?
    private(set) var receivedOpaqueState: Int?
    private(set) var observedActiveLengths: [Int]?
    private(set) var observedActiveRowMaps: [MTPPackedVerificationRowMap]?
    var returnsInvalidLogitsShape = false
    var omitsLastHidden = false
    var omitsSharedKV = false
    var omitsSharedKVSources = false
    var addsUnexpectedSharedKVSource = false
    var returnsMalformedSharedKV = false
    var returnsMalformedPositionDeltas = false
    var returnsScalarPositionDelta = false
    var sharedKVWidth: Int?
    var wrongPostForwardOffsetRow: Int?

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 3])
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        receivedTokens = input.tokens.asArray(Int.self)
        receivedMask = input.mask?.asArray(Int.self) ?? []
        receivedEmitFlag = state?[mtpEmitFlagKey]
        receivedCheckpointIndex = state?[mtpCacheCheckpointIndexKey]
        receivedOpaqueState = state?[packedOutputStateKey]
        observedActiveLengths = (cache?.first as? PackedVerificationCache)?.activeLengths
        observedActiveRowMaps = (cache?.first as? PackedVerificationCache)?.activeRowMaps

        let batchSize = input.tokens.dim(0)
        let width = input.tokens.dim(1)
        let vocabularySize = 3
        var values = [Float]()
        values.reserveCapacity(batchSize * width * vocabularySize)
        for row in 0 ..< batchSize {
            for column in 0 ..< width {
                for vocabularyIndex in 0 ..< vocabularySize {
                    values.append(Float(row * 100 + column * 10 + vocabularyIndex))
                }
            }
        }

        var outputState = LMOutput.State()
        outputState[packedOutputStateKey] = 42
        let hiddenValues = (0 ..< (batchSize * width * 2)).map { flatIndex in
            let row = flatIndex / (width * 2)
            let remainder = flatIndex % (width * 2)
            let column = remainder / 2
            let feature = remainder % 2
            return Float(row * 100 + column * 10 + feature)
        }
        if !omitsLastHidden {
            outputState[mtpLastHiddenStatesKey] = MLXArray(hiddenValues, [batchSize, width, 2])
        }

        let packedCache = cache?.first as? PackedVerificationCache
        packedCache?.finishPackedForward(wrongOffsetRow: wrongPostForwardOffsetRow)
        if !omitsSharedKV {
            let offsets = packedCache?.batchOffset.asArray(Int.self) ?? []
            let sharedWidth = sharedKVWidth ?? offsets.max() ?? 0
            let sharedValues = (0 ..< (batchSize * sharedWidth)).map { flatIndex in
                let row = flatIndex / sharedWidth
                let position = flatIndex % sharedWidth
                return Float(row * 1_000 + position)
            }
            let sharedShape =
                returnsMalformedSharedKV
                ? [1, 1, batchSize * sharedWidth, 1]
                : [batchSize, 1, sharedWidth, 1]
            let keys = MLXArray(sharedValues, sharedShape)
            let values = MLXArray(sharedValues.map { $0 + 10_000 }, sharedShape)
            outputState[mtpSharedKVStatesKey] = ["full_attention": (keys, values)]
            if !omitsSharedKVSources {
                var sources = ["full_attention": 0]
                if addsUnexpectedSharedKVSource {
                    sources["sliding_attention"] = 0
                }
                outputState[mtpSharedKVSourceIndicesKey] = sources
            }
        }
        if returnsScalarPositionDelta {
            outputState[mtpPositionDeltasKey] = MLXArray(Int32(9))
        } else {
            let deltaCount = returnsMalformedPositionDeltas ? batchSize + 1 : batchSize
            outputState[mtpPositionDeltasKey] = MLXArray(0 ..< deltaCount).reshaped(deltaCount, 1)
        }
        let logits = MLXArray(values, [batchSize, width, vocabularySize])
        return LMOutput(
            logits: returnsInvalidLogitsShape ? logits[0..., 0 ..< (width - 1), 0...] : logits,
            state: outputState)
    }
}

@Suite("MTP packed target verification", .serialized)
struct MTPPackedTargetVerificationTests {
    @Test func raggedRowsUseOneTargetCallAndIgnorePadding() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [3, 11])
        let tokens = MLXArray([
            10, 11, 12, 13, 999,
            20, 21, 22, 23, 24,
        ]).reshaped(2, 5)
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 8, queryOffset: 3, inputCount: 4, proposalCount: 3),
            MTPPackedVerificationRowMap(
                rowIndex: 2, queryOffset: 11, inputCount: 5, proposalCount: 4),
        ]

        let output = try verifyMTPPackedTargets(
            model: model,
            tokens: tokens,
            rowMaps: maps,
            cache: [cache])

        #expect(model.callCount == 1)
        #expect(model.receivedTokens == [10, 11, 12, 13, 999, 20, 21, 22, 23, 24])
        #expect(model.receivedMask == [1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        #expect(model.receivedEmitFlag == true)
        #expect(model.receivedOpaqueState == nil)
        #expect(model.observedActiveLengths == [4, 5])
        #expect(model.observedActiveRowMaps == maps)
        #expect(cache.packedPrepareCallCount == 1)
        #expect(cache.preparedRowMaps == maps)
        #expect(cache.baseTokenColumns == [0, 0])
        #expect(cache.proposalColumnRanges == [1 ..< 4, 1 ..< 5])
        #expect(cache.stagedCommitCandidatePositions == [[4, 5, 6], [12, 13, 14, 15]])
        #expect(cache.prepareCallCount == 1)
        #expect(cache.preparedLengths == [4, 5])
        #expect(cache.finalizeCallCount == 1)
        #expect(cache.activeLengths == nil)

        #expect(output.rows.map(\.map.rowIndex) == [8, 2])
        #expect(output.rows[0].proposalLogits.shape == [3, 3])
        #expect(
            output.rows[0].proposalLogits.asArray(Float.self)
                == [0, 1, 2, 10, 11, 12, 20, 21, 22])
        #expect(output.rows[0].bonusLogits.asArray(Float.self) == [30, 31, 32])
        #expect(output.rows[0].lastHidden?.shape == [4, 2])
        #expect(
            output.rows[0].lastHidden?.asArray(Float.self)
                == [0, 1, 10, 11, 20, 21, 30, 31])
        #expect(output.rows[1].proposalLogits.shape == [4, 3])
        #expect(
            output.rows[1].proposalLogits.asArray(Float.self)
                == [100, 101, 102, 110, 111, 112, 120, 121, 122, 130, 131, 132])
        #expect(output.rows[1].bonusLogits.asArray(Float.self) == [140, 141, 142])
        #expect(output.rows[1].lastHidden?.shape == [5, 2])
        #expect(output.rows.allSatisfy { $0.continuationState == nil })
    }

    @Test func continuationStateIsRowLocalAcrossRaggedReorderedRows() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [1, 4])
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 91, queryOffset: 1, inputCount: 2, proposalCount: 1),
            MTPPackedVerificationRowMap(
                rowIndex: 4, queryOffset: 4, inputCount: 3, proposalCount: 2),
        ]

        let output = try verifyMTPPackedTargets(
            model: model,
            tokens: MLXArray([
                10, 11, 999,
                20, 21, 22,
            ]).reshaped(2, 3),
            rowMaps: maps,
            cache: [cache],
            requireContinuationState: true)

        #expect(output.rows.map(\.map.rowIndex) == [91, 4])
        let first = try #require(output.rows[0].continuationState)
        let second = try #require(output.rows[1].continuationState)

        #expect(first.lastHidden.shape == [1, 2, 2])
        #expect(first.lastHidden.asArray(Float.self) == [0, 1, 10, 11])
        #expect(second.lastHidden.shape == [1, 3, 2])
        #expect(
            second.lastHidden.asArray(Float.self)
                == [100, 101, 110, 111, 120, 121])

        #expect(first.sharedKVSourceIndices == ["full_attention": 0])
        #expect(first.sharedKVOffsets == ["full_attention": 3])
        #expect(second.sharedKVOffsets == ["full_attention": 7])
        #expect(first.queryOffset == 3)
        #expect(second.queryOffset == 7)

        let firstKeys = try #require(first.sharedKV["full_attention"]?.0)
        let firstValues = try #require(first.sharedKV["full_attention"]?.1)
        let secondKeys = try #require(second.sharedKV["full_attention"]?.0)
        #expect(firstKeys.shape == [1, 1, 3, 1])
        #expect(firstKeys.asArray(Float.self) == [0, 1, 2])
        #expect(firstValues.asArray(Float.self) == [10_000, 10_001, 10_002])
        #expect(secondKeys.shape == [1, 1, 7, 1])
        #expect(secondKeys.asArray(Float.self) == [1_000, 1_001, 1_002, 1_003, 1_004, 1_005, 1_006])
        #expect(firstKeys.asArray(Float.self).allSatisfy { $0 < 1_000 })
        #expect(secondKeys.asArray(Float.self).allSatisfy { $0 >= 1_000 })

        #expect(first.positionDeltas?.shape == [1, 1])
        #expect(first.positionDeltas?.asArray(Int.self) == [0])
        #expect(second.positionDeltas?.asArray(Int.self) == [1])
        #expect(cache.finalizeCallCount == 1)
        #expect(cache.activeLengths == nil)
        #expect(cache.activeRowMaps == nil)
    }

    @Test func recurrentCacheRequestsUnconditionalPrefixCheckpoint() throws {
        let model = PackedVerificationModel()
        let attention = PackedVerificationCache(offsets: [3, 7])
        let recurrent = PackedVerificationCache(
            offsets: [3, 7], checkpointIndex: 1)
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 0, queryOffset: 3, inputCount: 2, proposalCount: 1),
            MTPPackedVerificationRowMap(
                rowIndex: 1, queryOffset: 7, inputCount: 2, proposalCount: 1),
        ]

        _ = try verifyMTPPackedTargets(
            model: model,
            tokens: MLXArray([10, 11, 20, 21]).reshaped(2, 2),
            rowMaps: maps,
            cache: [attention, recurrent])

        #expect(model.receivedCheckpointIndex == 1)
        #expect(attention.finalizeCallCount == 1)
        #expect(recurrent.finalizeCallCount == 1)
    }

    @Test func recurrentCheckpointRequiresAtLeastOneSpeculativeColumn() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [0], checkpointIndex: 1)

        #expect(throws: MTPPackedVerificationError.invalidCheckpointIndex(1)) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: MLXArray([7]).reshaped(1, 1),
                rowMaps: [
                    .init(rowIndex: 0, queryOffset: 0, inputCount: 1, proposalCount: 0)
                ],
                cache: [cache])
        }

        #expect(model.callCount == 0)
    }

    @Test func packedMambaRowsResolveAcceptedAndRejectedProposalsIndependently() throws {
        let first = MambaCache()
        first.state = [
            MLXArray([Float(1)]).reshaped(1, 1),
            MLXArray([Float(2)]).reshaped(1, 1),
        ]
        let second = MambaCache()
        second.state = [
            MLXArray([Float(3)]).reshaped(1, 1),
            MLXArray([Float(4)]).reshaped(1, 1),
        ]
        let batch = try MTPPackedMambaBatchCache(rowCaches: [first, second])
        try batch.prepareMTPPackedVerification(rowMaps: [
            .init(rowIndex: 0, queryOffset: 5, inputCount: 2, proposalCount: 1),
            .init(rowIndex: 1, queryOffset: 9, inputCount: 2, proposalCount: 1),
        ])
        batch.saveSpeculativeCheckpoint(
            convState: MLXArray([Float(10), 30]).reshaped(2, 1),
            recurrentState: MLXArray([Float(20), 40]).reshaped(2, 1),
            advancedBy: 1)
        batch.state = [
            MLXArray([Float(11), 31]).reshaped(2, 1),
            MLXArray([Float(21), 41]).reshaped(2, 1),
        ]

        let transactions = try batch.rowTransactions()
        try transactions[0].commit(retaining: 1)
        try transactions[1].commit(retaining: 2)

        #expect(first.state[0].asArray(Float.self) == [10])
        #expect(first.state[1].asArray(Float.self) == [20])
        #expect(second.state[0].asArray(Float.self) == [31])
        #expect(second.state[1].asArray(Float.self) == [41])
    }

    /// A zero-proposal row packed beside a one-proposal row is right-padded to
    /// width 2. Committing its only valid column must restore the checkpoint
    /// (post-column-1) state, not the complete state that absorbed the pad.
    @Test func paddedZeroProposalRowCommitsCheckpointNotPaddedState() throws {
        let proposing = MambaCache()
        proposing.state = [
            MLXArray([Float(1)]).reshaped(1, 1),
            MLXArray([Float(2)]).reshaped(1, 1),
        ]
        let padded = MambaCache()
        padded.state = [
            MLXArray([Float(3)]).reshaped(1, 1),
            MLXArray([Float(4)]).reshaped(1, 1),
        ]
        let batch = try MTPPackedMambaBatchCache(rowCaches: [proposing, padded])
        try batch.prepareMTPPackedVerification(rowMaps: [
            .init(rowIndex: 0, queryOffset: 5, inputCount: 2, proposalCount: 1),
            .init(rowIndex: 1, queryOffset: 9, inputCount: 1, proposalCount: 0),
        ])
        batch.saveSpeculativeCheckpoint(
            convState: MLXArray([Float(10), 30]).reshaped(2, 1),
            recurrentState: MLXArray([Float(20), 40]).reshaped(2, 1),
            advancedBy: 1)
        // Complete-width state: row 1's values here include the pad column.
        batch.state = [
            MLXArray([Float(11), 31]).reshaped(2, 1),
            MLXArray([Float(21), 41]).reshaped(2, 1),
        ]

        let transactions = try batch.rowTransactions()
        try transactions[0].commit(retaining: 2)
        try transactions[1].commit(retaining: 1)

        #expect(proposing.state[0].asArray(Float.self) == [11])
        #expect(proposing.state[1].asArray(Float.self) == [21])
        #expect(padded.state[0].asArray(Float.self) == [30])
        #expect(padded.state[1].asArray(Float.self) == [40])
    }

    @Test func ordinaryRowMaySharePackedForwardWithoutProposals() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [0])

        let output = try verifyMTPPackedTargets(
            model: model,
            tokens: MLXArray([7, 999, 999]).reshaped(1, 3),
            rowMaps: [
                .init(rowIndex: 5, queryOffset: 0, inputCount: 1, proposalCount: 0)
            ],
            cache: [cache])

        #expect(model.callCount == 1)
        #expect(output.rows[0].proposalLogits.shape == [0, 3])
        #expect(output.rows[0].proposalLogits.size == 0)
        #expect(output.rows[0].bonusLogits.asArray(Float.self) == [0, 1, 2])
        #expect(cache.proposalColumnRanges == [1 ..< 1])
        #expect(cache.stagedCommitCandidatePositions == [[]])
    }

    @Test func mixedOrdinaryAndProposalRowsMatchSerialIndexingAfterReorder() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [12, 1])
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 91, queryOffset: 12, inputCount: 1, proposalCount: 0),
            MTPPackedVerificationRowMap(
                rowIndex: 4, queryOffset: 1, inputCount: 3, proposalCount: 2),
        ]

        let output = try verifyMTPPackedTargets(
            model: model,
            tokens: MLXArray([
                30, 31, 32, 999, 999,
                40, 41, 42, 43, 44,
            ]).reshaped(2, 5),
            rowMaps: maps,
            cache: [cache])

        #expect(model.callCount == 1)
        #expect(output.rows.map(\.map.rowIndex) == [91, 4])
        for (packedRow, map) in maps.enumerated() {
            let proposalColumns = 0 ..< map.proposalCount
            let serialProposalOracle = proposalColumns.flatMap { column in
                (0 ..< 3).map { vocabularyIndex in
                    Float(packedRow * 100 + column * 10 + vocabularyIndex)
                }
            }
            let serialBonusOracle = (0 ..< 3).map { vocabularyIndex in
                Float(packedRow * 100 + (map.inputCount - 1) * 10 + vocabularyIndex)
            }
            if map.proposalCount == 0 {
                #expect(output.rows[packedRow].proposalLogits.shape == [0, 3])
                #expect(output.rows[packedRow].proposalLogits.size == 0)
            } else {
                #expect(
                    output.rows[packedRow].proposalLogits.asArray(Float.self)
                        == serialProposalOracle)
            }
            #expect(output.rows[packedRow].bonusLogits.asArray(Float.self) == serialBonusOracle)
            #expect(output.rows[packedRow].lastHidden?.dim(0) == map.inputCount)
        }
        #expect(output.rows[0].proposalLogits.shape == [0, 3])
        #expect(output.rows[0].lastHidden?.asArray(Float.self) == [0, 1])
        #expect(
            output.rows[1].lastHidden?.asArray(Float.self)
                == [100, 101, 110, 111, 120, 121])
        #expect(cache.proposalColumnRanges == [1 ..< 1, 1 ..< 3])
        #expect(cache.stagedCommitCandidatePositions == [[], [2, 3]])
    }

    @Test func plainCacheIsRejectedBeforePrepareOrTargetCall() throws {
        let model = PackedVerificationModel()
        let cache = KVCacheSimple()

        #expect(
            throws: MTPPackedVerificationError.unsupportedCache(cacheIndex: 0)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: MLXArray([1, 2]).reshaped(1, 2),
                rowMaps: [
                    .init(rowIndex: 0, queryOffset: 0, inputCount: 2, proposalCount: 1)
                ],
                cache: [cache])
        }

        #expect(model.callCount == 0)
        #expect(cache.offset == 0)
    }

    @Test func emptyCacheIsRejectedBeforeTargetCall() throws {
        let model = PackedVerificationModel()

        #expect(throws: MTPPackedVerificationError.emptyCache) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: MLXArray([1, 2]).reshaped(1, 2),
                rowMaps: [
                    .init(rowIndex: 0, queryOffset: 0, inputCount: 2, proposalCount: 1)
                ],
                cache: [])
        }

        #expect(model.callCount == 0)
    }

    @Test func invalidRowMapsFailBeforeTargetCall() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [3, 11])
        let tokens = MLXArray.zeros([2, 4], dtype: .int32)

        #expect(
            throws: MTPPackedVerificationError.invalidProposalCount(
                rowIndex: 4, proposalCount: 2, inputCount: 2)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 2, proposalCount: 2),
                    .init(rowIndex: 9, queryOffset: 11, inputCount: 4, proposalCount: 1),
                ],
                cache: [cache])
        }

        #expect(
            throws: MTPPackedVerificationError.duplicateRowIndex(4)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 2, proposalCount: 1),
                    .init(rowIndex: 4, queryOffset: 11, inputCount: 2, proposalCount: 1),
                ],
                cache: [cache])
        }

        #expect(
            throws: MTPPackedVerificationError.invalidVerificationInputCount(
                rowIndex: 4, inputCount: 3, expectedForProposalCount: 2)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 3, proposalCount: 1),
                    .init(rowIndex: 9, queryOffset: 11, inputCount: 2, proposalCount: 1),
                ],
                cache: [cache])
        }

        #expect(
            throws: MTPPackedVerificationError.rowCountMismatch(expected: 2, actual: 1)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 2, proposalCount: 1)
                ],
                cache: [cache])
        }

        #expect(model.callCount == 0)
        #expect(cache.preparedLengths == nil)
        #expect(cache.finalizeCallCount == 0)
    }

    @Test func postForwardOffsetOverflowFailsBeforeCacheOrTargetMutation() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [Int.max])

        #expect(
            throws: MTPPackedVerificationError.postForwardCacheOffsetOverflow(
                rowIndex: 17,
                queryOffset: Int.max,
                inputCount: 1)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: MLXArray([1]).reshaped(1, 1),
                rowMaps: [
                    .init(rowIndex: 17, queryOffset: Int.max, inputCount: 1, proposalCount: 0)
                ],
                cache: [cache],
                requireContinuationState: true)
        }

        #expect(model.callCount == 0)
        #expect(cache.packedPrepareCallCount == 0)
        #expect(cache.prepareCallCount == 0)
        #expect(cache.finalizeCallCount == 0)
    }

    @Test func mismatchedCachePositionsFailClosedAndFinalize() throws {
        let model = PackedVerificationModel()
        let cache = PackedVerificationCache(offsets: [0, 0])
        let tokens = MLXArray.zeros([2, 3], dtype: .int32)

        #expect(
            throws: MTPPackedVerificationError.cacheOffsetMismatch(
                cacheIndex: 0, expected: [3, 11], actual: [0, 0])
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 3, proposalCount: 2),
                    .init(rowIndex: 9, queryOffset: 11, inputCount: 2, proposalCount: 1),
                ],
                cache: [cache])
        }

        #expect(model.callCount == 0)
        #expect(cache.preparedLengths == [3, 2])
        #expect(cache.packedPrepareCallCount == 1)
        #expect(cache.preparedRowMaps?.map(\.rowIndex) == [4, 9])
        #expect(cache.finalizeCallCount == 1)
        #expect(cache.activeLengths == nil)
        #expect(cache.activeRowMaps == nil)
    }

    @Test func invalidTargetLogitsStillFinalizeCache() throws {
        let model = PackedVerificationModel()
        model.returnsInvalidLogitsShape = true
        let cache = PackedVerificationCache(offsets: [3, 11])
        let tokens = MLXArray.zeros([2, 3], dtype: .int32)

        #expect(
            throws: MTPPackedVerificationError.invalidLogitsShape(
                expectedBatch: 2, expectedWidth: 3, actualShape: [2, 2, 3])
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: [
                    .init(rowIndex: 4, queryOffset: 3, inputCount: 3, proposalCount: 2),
                    .init(rowIndex: 9, queryOffset: 11, inputCount: 1, proposalCount: 0),
                ],
                cache: [cache])
        }

        #expect(model.callCount == 1)
        #expect(cache.preparedLengths == [3, 1])
        #expect(cache.packedPrepareCallCount == 1)
        #expect(cache.finalizeCallCount == 1)
        #expect(cache.activeLengths == nil)
        #expect(cache.activeRowMaps == nil)
    }

    @Test func requiredContinuationStateFailsClosedWhenAbsent() throws {
        let tokens = MLXArray([1, 2]).reshaped(1, 2)
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 7, queryOffset: 3, inputCount: 2, proposalCount: 1)
        ]

        do {
            let model = PackedVerificationModel()
            model.omitsLastHidden = true
            let cache = PackedVerificationCache(offsets: [3])
            #expect(throws: MTPPackedVerificationError.missingLastHidden) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }

        do {
            let model = PackedVerificationModel()
            model.omitsSharedKV = true
            let cache = PackedVerificationCache(offsets: [3])
            #expect(throws: MTPPackedVerificationError.missingSharedKV) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }

        do {
            let model = PackedVerificationModel()
            model.omitsSharedKVSources = true
            let cache = PackedVerificationCache(offsets: [3])
            #expect(throws: MTPPackedVerificationError.missingSharedKVSourceIndices) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }
    }

    @Test func malformedContinuationStateFailsClosed() throws {
        let tokens = MLXArray([
            1, 2,
            3, 4,
        ]).reshaped(2, 2)
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 7, queryOffset: 1, inputCount: 2, proposalCount: 1),
            MTPPackedVerificationRowMap(
                rowIndex: 3, queryOffset: 4, inputCount: 2, proposalCount: 1),
        ]

        do {
            let model = PackedVerificationModel()
            model.addsUnexpectedSharedKVSource = true
            let cache = PackedVerificationCache(offsets: [1, 4])
            #expect(
                throws: MTPPackedVerificationError.sharedKVSourceKeyMismatch(
                    expected: ["full_attention"],
                    actual: ["full_attention", "sliding_attention"])
            ) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }

        do {
            let model = PackedVerificationModel()
            model.returnsMalformedSharedKV = true
            let cache = PackedVerificationCache(offsets: [1, 4])
            #expect(
                throws: MTPPackedVerificationError.invalidSharedKVBatch(
                    layerType: "full_attention", expectedBatch: 2, actualBatch: 1)
            ) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }

        do {
            let model = PackedVerificationModel()
            model.returnsMalformedPositionDeltas = true
            let cache = PackedVerificationCache(offsets: [1, 4])
            #expect(
                throws: MTPPackedVerificationError.invalidPositionDeltasShape(
                    expectedBatch: 2, actualShape: [3, 1])
            ) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }
    }

    @Test func scalarPositionDeltaIsCopiedIntoEveryRowWithoutCrossRowState() throws {
        let model = PackedVerificationModel()
        model.returnsScalarPositionDelta = true
        let cache = PackedVerificationCache(offsets: [1, 4])

        let output = try verifyMTPPackedTargets(
            model: model,
            tokens: MLXArray([
                1, 2,
                3, 4,
            ]).reshaped(2, 2),
            rowMaps: [
                .init(rowIndex: 7, queryOffset: 1, inputCount: 2, proposalCount: 1),
                .init(rowIndex: 3, queryOffset: 4, inputCount: 2, proposalCount: 1),
            ],
            cache: [cache],
            requireContinuationState: true)

        let first = try #require(output.rows[0].continuationState)
        let second = try #require(output.rows[1].continuationState)
        #expect(first.positionDeltas?.shape == [1])
        #expect(first.positionDeltas?.asArray(Int.self) == [9])
        #expect(second.positionDeltas?.shape == [1])
        #expect(second.positionDeltas?.asArray(Int.self) == [9])
        #expect(cache.finalizeCallCount == 1)
    }

    @Test func sharedKVSpanMustReachUnboundedOffsetButMayMatchBoundedCap() throws {
        let tokens = MLXArray([1, 2]).reshaped(1, 2)
        let maps = [
            MTPPackedVerificationRowMap(
                rowIndex: 7, queryOffset: 4, inputCount: 2, proposalCount: 1)
        ]

        do {
            let model = PackedVerificationModel()
            model.sharedKVWidth = 5
            let cache = PackedVerificationCache(offsets: [4])
            #expect(
                throws: MTPPackedVerificationError.insufficientSharedKVSequenceSpan(
                    layerType: "full_attention",
                    rowIndex: 7,
                    expected: 6,
                    actual: 5)
            ) {
                try verifyMTPPackedTargets(
                    model: model,
                    tokens: tokens,
                    rowMaps: maps,
                    cache: [cache],
                    requireContinuationState: true)
            }
            #expect(cache.finalizeCallCount == 1)
        }

        do {
            let model = PackedVerificationModel()
            model.sharedKVWidth = 5
            let cache = PackedVerificationCache(offsets: [4], maxSize: 5)
            let output = try verifyMTPPackedTargets(
                model: model,
                tokens: tokens,
                rowMaps: maps,
                cache: [cache],
                requireContinuationState: true)

            let state = try #require(output.rows[0].continuationState)
            let keys = try #require(state.sharedKV["full_attention"]?.0)
            #expect(keys.shape == [1, 1, 5, 1])
            #expect(state.sharedKVOffsets == ["full_attention": 6])
            #expect(state.queryOffset == 6)
            #expect(cache.finalizeCallCount == 1)
        }
    }

    @Test func continuationStateRejectsPostForwardRowIdentityMismatch() throws {
        let model = PackedVerificationModel()
        model.wrongPostForwardOffsetRow = 1
        let cache = PackedVerificationCache(offsets: [1, 4])

        #expect(
            throws: MTPPackedVerificationError.postForwardCacheOffsetMismatch(
                cacheIndex: 0, rowIndex: 3, expected: 6, actual: 7)
        ) {
            try verifyMTPPackedTargets(
                model: model,
                tokens: MLXArray([
                    1, 2,
                    3, 4,
                ]).reshaped(2, 2),
                rowMaps: [
                    .init(rowIndex: 7, queryOffset: 1, inputCount: 2, proposalCount: 1),
                    .init(rowIndex: 3, queryOffset: 4, inputCount: 2, proposalCount: 1),
                ],
                cache: [cache],
                requireContinuationState: true)
        }

        #expect(cache.finalizeCallCount == 1)
        #expect(cache.activeLengths == nil)
        #expect(cache.activeRowMaps == nil)
    }
}
