// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private let packedOutputStateKey = LMOutput.Key<Int>("tests.mtp.packed.output")

private final class PackedVerificationCache: BaseKVCache, MTPPackedVerificationCache {
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

    init(offsets: [Int]) {
        self.batchOffset = MLXArray(offsets)
        super.init()
    }

    override var ropeOffset: RoPEOffset { .batch(batchOffset) }

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

    override func prepare(lengths: [Int]?) {
        prepareCallCount += 1
        preparedLengths = lengths
        activeLengths = lengths
    }

    override func finalize() {
        finalizeCallCount += 1
        activeLengths = nil
        activeRowMaps = nil
    }

    override func copy() -> any KVCache {
        PackedVerificationCache(offsets: batchOffset.asArray(Int.self))
    }
}

private final class PackedVerificationModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    private(set) var callCount = 0
    private(set) var receivedTokens: [Int] = []
    private(set) var receivedMask: [Int] = []
    private(set) var receivedEmitFlag: Bool?
    private(set) var receivedOpaqueState: Int?
    private(set) var observedActiveLengths: [Int]?
    private(set) var observedActiveRowMaps: [MTPPackedVerificationRowMap]?
    var returnsInvalidLogitsShape = false

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
        outputState[mtpLastHiddenStatesKey] = MLXArray(hiddenValues, [batchSize, width, 2])
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
}
