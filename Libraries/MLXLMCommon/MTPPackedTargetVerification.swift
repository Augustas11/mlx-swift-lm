// Copyright © 2026 Apple Inc.

import MLX

/// Cache lifecycle required by ``verifyMTPPackedTargets(model:tokens:rowMaps:cache:)``.
///
/// The facade calls ``prepareMTPPackedVerification(rowMaps:)`` before the
/// integer-array `prepare(lengths:)` overload, then always calls
/// ``KVCache/finalize()``. An implementation must use the complete row maps to:
///
/// - exclude each row's right padding,
/// - expose an independent absolute position for every row through
///   ``BatchPositionedKVCache/batchOffset``,
/// - treat column zero as the already-committed base token,
/// - stage only columns `1 ..< inputCount` as proposal commit candidates, and
/// - leave commit or discard decisions to a caller-owned row transaction.
///
/// ``KVCache/finalize()`` must only clear transient packed-call metadata. It
/// must not commit or discard staged writes. Plain scalar caches such as
/// `KVCacheSimple` do not satisfy this contract.
public protocol MTPPackedVerificationCache: BatchPositionedKVCache {
    /// Configure one packed verification call without committing any row.
    ///
    /// Each row has the exact layout `[lastCommitted, proposal1, ..., proposalN]`.
    /// `rowIndex` identifies scheduler-owned transaction state, while
    /// `queryOffset` is the absolute position of `lastCommitted`.
    func prepareMTPPackedVerification(rowMaps: [MTPPackedVerificationRowMap]) throws

    /// Target-native recurrent checkpoint requested for this packed call.
    ///
    /// Hybrid caches return the number of leading verification columns that
    /// are unconditionally retained. Attention-only caches use `nil`.
    var mtpPackedCheckpointIndex: Int? { get }
}

extension MTPPackedVerificationCache {
    public var mtpPackedCheckpointIndex: Int? { nil }
}

/// Maps one packed target-model row back to a scheduler-owned sequence.
///
/// Valid tokens occupy the leading ``inputCount`` columns of the packed row
/// with the exact layout `[lastCommitted, proposal1, ..., proposalN]`.
/// Therefore `inputCount` must equal `proposalCount + 1`. `proposalCount` may
/// be zero when an ordinary row contributes only its last committed token.
public struct MTPPackedVerificationRowMap: Sendable, Hashable {
    /// Scheduler-owned row identifier.
    public let rowIndex: Int

    /// Absolute position of the first valid input token in this row.
    public let queryOffset: Int

    /// Number of valid leading input columns, excluding right padding.
    public let inputCount: Int

    /// Number of proposal tokens at the end of the valid input prefix.
    public let proposalCount: Int

    public init(
        rowIndex: Int,
        queryOffset: Int,
        inputCount: Int,
        proposalCount: Int
    ) {
        self.rowIndex = rowIndex
        self.queryOffset = queryOffset
        self.inputCount = inputCount
        self.proposalCount = proposalCount
    }
}

/// Target logits for one scheduler row in a packed verification call.
public struct MTPPackedVerificationRowOutput {
    /// The scheduler mapping supplied for this packed row.
    public let map: MTPPackedVerificationRowMap

    /// Logits that score only the proposal tokens, shaped
    /// `[proposalCount, vocabularySize]`.
    public let proposalLogits: MLXArray

    /// The final valid logit row, shaped `[1, vocabularySize]`.
    ///
    /// Schedulers use this row for the bonus token only after every proposal
    /// is accepted. It is intentionally separate from ``proposalLogits``.
    public let bonusLogits: MLXArray

    /// Compatibility view of target hidden states for the valid input prefix,
    /// shaped `[inputCount, hiddenSize]`, when emitted.
    public let lastHidden: MLXArray?

    /// Exact row-local target state for another drafter round.
    ///
    /// This is populated only by the strict overload that sets
    /// `requireContinuationState`.
    public let continuationState: MTPPackedVerificationRowState?
}

/// Result of one packed target-model verification call.
public struct MTPPackedVerificationOutput {
    /// Per-row logit slices in packed-input order.
    public let rows: [MTPPackedVerificationRowOutput]
}

/// Resolution failures from ``MTPPackedMambaBatchCache``.
public enum MTPPackedMambaCacheError: Error, Equatable {
    case invalidRowMaps
    case incompatibleRowState
    case missingForwardState
    case missingCheckpoint
    case invalidRetainedInputCount(retaining: Int, inputCount: Int)
    case unsupportedPartialRetention(retaining: Int, proposalCount: Int)
}

/// One row-local recurrent-state transaction produced by a packed MTP call.
///
/// The source row remains unchanged until ``commit(retaining:)``. Dropping the
/// transaction therefore rolls back without work.
public final class MTPPackedMambaRowTransaction {
    public let inputCount: Int
    public let proposalCount: Int

    private let rowCache: MambaCache
    private let baseState: [MLXArray]
    private let finalState: [MLXArray]

    package init(
        rowCache: MambaCache,
        inputCount: Int,
        proposalCount: Int,
        baseState: [MLXArray],
        finalState: [MLXArray]
    ) {
        self.rowCache = rowCache
        self.inputCount = inputCount
        self.proposalCount = proposalCount
        self.baseState = baseState
        self.finalState = finalState
    }

    /// Commit an exact leading prefix of the packed verification input.
    ///
    /// Current hybrid Qwen checkpoints support the unconditional base column
    /// or the complete one-proposal write.
    public func commit(retaining: Int) throws {
        guard retaining > 0, retaining <= inputCount else {
            throw MTPPackedMambaCacheError.invalidRetainedInputCount(
                retaining: retaining, inputCount: inputCount)
        }
        let selected: [MLXArray]
        if retaining == inputCount {
            selected = finalState
        } else if retaining == 1, proposalCount == 1 {
            selected = baseState
        } else {
            throw MTPPackedMambaCacheError.unsupportedPartialRetention(
                retaining: retaining, proposalCount: proposalCount)
        }
        rowCache.state = selected
        eval(selected)
    }
}

/// Transactional batched recurrent cache for packed hybrid-model MTP.
///
/// The target sees a concrete `MambaCache`, while row-owned caches remain
/// isolated until the scheduler resolves each returned row transaction.
public final class MTPPackedMambaBatchCache: MambaCache, MTPPackedVerificationCache {
    private let rowCaches: [MambaCache]
    private var preparedRowMaps: [MTPPackedVerificationRowMap]?
    private var pendingTransactions: [MTPPackedMambaRowTransaction]?

    public init(rowCaches: [MambaCache]) throws {
        self.rowCaches = rowCaches
        super.init()
        try packRows()
    }

    public var batchOffset: MLXArray {
        MLXArray(
            (preparedRowMaps?.map(\.queryOffset) ?? rowCaches.map(\.offset))
                .map(Int32.init))
    }

    public var mtpPackedCheckpointIndex: Int? {
        guard let preparedRowMaps,
              preparedRowMaps.contains(where: { $0.proposalCount > 0 })
        else { return nil }
        return 1
    }

    public func prepareMTPPackedVerification(
        rowMaps: [MTPPackedVerificationRowMap]
    ) throws {
        guard rowMaps.count == rowCaches.count,
              rowMaps.allSatisfy({
                  $0.queryOffset >= 0
                      && $0.inputCount == $0.proposalCount + 1
                      && (0 ... 1).contains($0.proposalCount)
              })
        else { throw MTPPackedMambaCacheError.invalidRowMaps }
        preparedRowMaps = rowMaps
        pendingTransactions = nil
    }

    /// Capture row-local commit handles after the packed target forward.
    public func rowTransactions() throws -> [MTPPackedMambaRowTransaction] {
        if let pendingTransactions { return pendingTransactions }
        guard let preparedRowMaps,
              preparedRowMaps.count == rowCaches.count,
              state.count == 2
        else { throw MTPPackedMambaCacheError.missingForwardState }

        let completeState = state
        let needsCheckpoint = preparedRowMaps.contains(where: { $0.proposalCount > 0 })
        let checkpointState: [MLXArray]
        if needsCheckpoint {
            guard restoreSpeculativeCheckpoint(), state.count == 2 else {
                throw MTPPackedMambaCacheError.missingCheckpoint
            }
            checkpointState = state
            state = completeState
        } else {
            checkpointState = completeState
        }

        let transactions = preparedRowMaps.enumerated().map { rowIndex, map in
            MTPPackedMambaRowTransaction(
                rowCache: rowCaches[rowIndex],
                inputCount: map.inputCount,
                proposalCount: map.proposalCount,
                baseState: checkpointState.map {
                    $0[rowIndex ..< rowIndex + 1, .ellipsis]
                },
                finalState: completeState.map {
                    $0[rowIndex ..< rowIndex + 1, .ellipsis]
                })
        }
        pendingTransactions = transactions
        return transactions
    }

    private func packRows() throws {
        let rowStates = rowCaches.map(\.state)
        let slotCount = rowStates.map(\.count).max() ?? 0
        guard slotCount > 0 else { return }
        for slot in 0 ..< slotCount {
            guard let first = rowStates.first(where: { $0.indices.contains(slot) })?[slot]
            else { throw MTPPackedMambaCacheError.incompatibleRowState }
            let arrays = rowStates.map { states in
                states.indices.contains(slot)
                    ? states[slot]
                    : MLXArray.zeros(
                        [1] + Array(first.shape.dropFirst()), dtype: first.dtype)
            }
            guard arrays.allSatisfy({
                $0.ndim >= 1
                    && $0.dim(0) == 1
                    && Array($0.shape.dropFirst()) == Array(first.shape.dropFirst())
                    && $0.dtype == first.dtype
            }) else { throw MTPPackedMambaCacheError.incompatibleRowState }
            self[slot] = concatenated(arrays, axis: 0)
        }
    }
}

/// Target state for one packed row's next MTP drafter call.
///
/// Every array keeps a leading batch dimension of one. Shared K/V arrays are
/// also trimmed to the row's live chronological sequence span, so they contain
/// neither another packed row nor right-padding from this one.
public struct MTPPackedVerificationRowState {
    /// Target hidden states for this row's valid verification input, shaped
    /// `[1, inputCount, hiddenSize]`.
    public let lastHidden: MLXArray

    /// Target K/V snapshots keyed by layer type. Each tuple keeps the target's
    /// original rank with batch size one and a row-local live sequence axis.
    public let sharedKV: [String: (MLXArray, MLXArray)]

    /// Cache entry that supplied each shared K/V tuple.
    public let sharedKVSourceIndices: [String: Int]

    /// Absolute post-forward cache offset for each shared K/V tuple.
    public let sharedKVOffsets: [String: Int]

    /// Absolute position for the next drafter query. This is the resolved
    /// `full_attention` shared-K/V offset for this row.
    public let queryOffset: Int

    /// Row-sliced target position deltas, when the model emitted them.
    public let positionDeltas: MLXArray?
}

/// Validation failures from ``verifyMTPPackedTargets(model:tokens:rowMaps:cache:)``.
public enum MTPPackedVerificationError: Error, Equatable {
    case tokensMustBeMatrix(actualRank: Int)
    case emptyBatch
    case rowCountMismatch(expected: Int, actual: Int)
    case invalidRowIndex(Int)
    case duplicateRowIndex(Int)
    case invalidQueryOffset(rowIndex: Int, queryOffset: Int)
    case invalidInputCount(rowIndex: Int, inputCount: Int, paddedWidth: Int)
    case invalidProposalCount(rowIndex: Int, proposalCount: Int, inputCount: Int)
    case invalidVerificationInputCount(
        rowIndex: Int, inputCount: Int, expectedForProposalCount: Int)
    case postForwardCacheOffsetOverflow(rowIndex: Int, queryOffset: Int, inputCount: Int)
    case emptyCache
    case unsupportedCache(cacheIndex: Int)
    case cacheOffsetMismatch(cacheIndex: Int, expected: [Int], actual: [Int])
    case invalidCacheOffsetShape(cacheIndex: Int, expectedCount: Int, actualShape: [Int])
    case invalidLogitsShape(expectedBatch: Int, expectedWidth: Int, actualShape: [Int])
    case invalidLastHiddenShape(expectedBatch: Int, expectedWidth: Int, actualShape: [Int])
    case missingLastHidden
    case missingSharedKV
    case emptySharedKV
    case missingSharedKVSourceIndices
    case sharedKVSourceKeyMismatch(expected: [String], actual: [String])
    case invalidSharedKVSource(layerType: String, sourceIndex: Int)
    case invalidSharedKVShape(layerType: String, keysShape: [Int], valuesShape: [Int])
    case invalidSharedKVBatch(layerType: String, expectedBatch: Int, actualBatch: Int)
    case insufficientSharedKVSequenceSpan(
        layerType: String, rowIndex: Int, expected: Int, actual: Int)
    case invalidPostForwardCacheOffsetShape(
        cacheIndex: Int, expectedCount: Int, actualShape: [Int])
    case postForwardCacheOffsetMismatch(
        cacheIndex: Int, rowIndex: Int, expected: Int, actual: Int)
    case missingFullAttentionSharedKV
    case invalidPositionDeltasShape(expectedBatch: Int, actualShape: [Int])
    case inconsistentCheckpointIndices([Int])
    case invalidCheckpointIndex(Int)
}

/// Verify ragged MTP proposal rows with exactly one target-model call.
///
/// `tokens` must be a right-padded `[batch, width]` matrix. `rowMaps` is in
/// packed row order, while each `rowIndex` belongs to the caller's scheduler.
/// Each row's `inputCount` is scoped through `KVCache.prepare(lengths:)` and
/// `KVCache.finalize()` around the target call. `queryOffset` instead describes
/// the absolute cache position and is validated independently.
///
/// At least one cache is required. Every cache must explicitly conform to
/// ``MTPPackedVerificationCache`` and prepare the supplied row transaction
/// metadata. This fails closed rather than running ragged rows with scalar
/// positions, padding-blind writes, or shared commit/discard ownership.
///
/// The compatibility overload returns logits and optional hidden states exactly
/// as before. Call the strict overload with `requireContinuationState: true`
/// when a scheduler also needs row-isolated target state for another MTP round.
/// The external scheduler still owns each row transaction and resolves its
/// commit/discard policy after verification.
public func verifyMTPPackedTargets(
    model: any LanguageModel,
    tokens: MLXArray,
    rowMaps: [MTPPackedVerificationRowMap],
    cache: [KVCache]
) throws -> MTPPackedVerificationOutput {
    try verifyMTPPackedTargets(
        model: model,
        tokens: tokens,
        rowMaps: rowMaps,
        cache: cache,
        requireContinuationState: false)
}

/// Verify packed targets and optionally require exact row-local continuation state.
///
/// Set `requireContinuationState` for multi-round MTP. In that mode every
/// target-emitted shared K/V tensor must be batch-major and its source cache
/// must expose an exact post-forward ``BatchPositionedKVCache/batchOffset``.
/// The cache state must store each row's live chronological K/V in the leading
/// sequence prefix and any unused capacity as right padding. The facade uses
/// those offsets to remove padding and refuses missing or ambiguous state.
///
/// The overload without `requireContinuationState` preserves the original
/// single-round API and leaves ``MTPPackedVerificationRowOutput/continuationState``
/// unset.
public func verifyMTPPackedTargets(
    model: any LanguageModel,
    tokens: MLXArray,
    rowMaps: [MTPPackedVerificationRowMap],
    cache: [KVCache],
    requireContinuationState: Bool
) throws -> MTPPackedVerificationOutput {
    guard tokens.ndim == 2 else {
        throw MTPPackedVerificationError.tokensMustBeMatrix(actualRank: tokens.ndim)
    }

    let batchSize = tokens.dim(0)
    let paddedWidth = tokens.dim(1)
    guard batchSize > 0 else {
        throw MTPPackedVerificationError.emptyBatch
    }
    guard rowMaps.count == batchSize else {
        throw MTPPackedVerificationError.rowCountMismatch(
            expected: batchSize, actual: rowMaps.count)
    }
    var seenRows = Set<Int>()
    for map in rowMaps {
        guard map.rowIndex >= 0 else {
            throw MTPPackedVerificationError.invalidRowIndex(map.rowIndex)
        }
        guard seenRows.insert(map.rowIndex).inserted else {
            throw MTPPackedVerificationError.duplicateRowIndex(map.rowIndex)
        }
        guard map.queryOffset >= 0 else {
            throw MTPPackedVerificationError.invalidQueryOffset(
                rowIndex: map.rowIndex, queryOffset: map.queryOffset)
        }
        guard map.inputCount > 0, map.inputCount <= paddedWidth else {
            throw MTPPackedVerificationError.invalidInputCount(
                rowIndex: map.rowIndex,
                inputCount: map.inputCount,
                paddedWidth: paddedWidth)
        }
        guard map.proposalCount >= 0, map.proposalCount < map.inputCount else {
            throw MTPPackedVerificationError.invalidProposalCount(
                rowIndex: map.rowIndex,
                proposalCount: map.proposalCount,
                inputCount: map.inputCount)
        }
        let expectedInputCount = map.proposalCount + 1
        guard map.inputCount == expectedInputCount else {
            throw MTPPackedVerificationError.invalidVerificationInputCount(
                rowIndex: map.rowIndex,
                inputCount: map.inputCount,
                expectedForProposalCount: expectedInputCount)
        }
        let (_, postForwardOffsetOverflow) = map.queryOffset.addingReportingOverflow(
            map.inputCount)
        guard !postForwardOffsetOverflow else {
            throw MTPPackedVerificationError.postForwardCacheOffsetOverflow(
                rowIndex: map.rowIndex,
                queryOffset: map.queryOffset,
                inputCount: map.inputCount)
        }
    }

    let queryOffsets = rowMaps.map(\.queryOffset)
    guard !cache.isEmpty else {
        throw MTPPackedVerificationError.emptyCache
    }
    let packedCache = try cache.enumerated().map { cacheIndex, entry in
        guard let entry = entry as? any MTPPackedVerificationCache else {
            throw MTPPackedVerificationError.unsupportedCache(cacheIndex: cacheIndex)
        }
        return entry
    }

    let maskValues = rowMaps.flatMap { map in
        (0 ..< paddedWidth).map { column in Int32(column < map.inputCount ? 1 : 0) }
    }
    let input = LMInput.Text(
        tokens: tokens,
        mask: MLXArray(maskValues, [batchSize, paddedWidth]))

    var targetState = LMOutput.State()
    targetState[mtpEmitFlagKey] = true

    defer {
        for entry in cache {
            entry.finalize()
        }
    }
    for entry in packedCache {
        try entry.prepareMTPPackedVerification(rowMaps: rowMaps)
    }
    let checkpointIndices = Set(packedCache.compactMap(\.mtpPackedCheckpointIndex))
    guard checkpointIndices.count <= 1 else {
        throw MTPPackedVerificationError.inconsistentCheckpointIndices(
            checkpointIndices.sorted())
    }
    if let checkpointIndex = checkpointIndices.first {
        guard checkpointIndex > 0,
              rowMaps.allSatisfy({ checkpointIndex <= $0.inputCount }),
              rowMaps.contains(where: { checkpointIndex < $0.inputCount })
        else {
            throw MTPPackedVerificationError.invalidCheckpointIndex(checkpointIndex)
        }
        targetState[mtpCacheCheckpointIndexKey] = checkpointIndex
    }
    for entry in cache {
        entry.prepare(lengths: rowMaps.map(\.inputCount))
    }

    try validateMTPPackedCacheOffsets(packedCache, queryOffsets: queryOffsets)

    let output = model(input, cache: cache, state: targetState)
    guard output.logits.ndim == 3,
        output.logits.dim(0) == batchSize,
        output.logits.dim(1) == paddedWidth
    else {
        throw MTPPackedVerificationError.invalidLogitsShape(
            expectedBatch: batchSize,
            expectedWidth: paddedWidth,
            actualShape: output.logits.shape)
    }

    let lastHidden = output.state?[mtpLastHiddenStatesKey]
    if let lastHidden {
        guard lastHidden.ndim == 3,
            lastHidden.dim(0) == batchSize,
            lastHidden.dim(1) == paddedWidth
        else {
            throw MTPPackedVerificationError.invalidLastHiddenShape(
                expectedBatch: batchSize,
                expectedWidth: paddedWidth,
                actualShape: lastHidden.shape)
        }
    }

    let continuationStates = try requireContinuationState
        ? extractMTPPackedContinuationStates(
            output.state,
            rowMaps: rowMaps,
            cache: packedCache,
            batchSize: batchSize,
            paddedWidth: paddedWidth)
        : nil

    let rows = rowMaps.enumerated().map { packedRow, map in
        let rowLogits = output.logits[packedRow]
        return MTPPackedVerificationRowOutput(
            map: map,
            proposalLogits: rowLogits[0 ..< map.proposalCount, 0...],
            bonusLogits: rowLogits[(map.inputCount - 1) ..< map.inputCount, 0...],
            lastHidden: lastHidden?[
                packedRow, 0 ..< map.inputCount, 0...],
            continuationState: continuationStates?[packedRow])
    }
    return MTPPackedVerificationOutput(rows: rows)
}

private func extractMTPPackedContinuationStates(
    _ state: LMOutput.State?,
    rowMaps: [MTPPackedVerificationRowMap],
    cache: [any MTPPackedVerificationCache],
    batchSize: Int,
    paddedWidth: Int
) throws -> [MTPPackedVerificationRowState] {
    guard let lastHidden = state?[mtpLastHiddenStatesKey] else {
        throw MTPPackedVerificationError.missingLastHidden
    }
    guard lastHidden.ndim == 3,
        lastHidden.dim(0) == batchSize,
        lastHidden.dim(1) == paddedWidth
    else {
        throw MTPPackedVerificationError.invalidLastHiddenShape(
            expectedBatch: batchSize,
            expectedWidth: paddedWidth,
            actualShape: lastHidden.shape)
    }
    guard let sharedKV = state?[mtpSharedKVStatesKey] else {
        throw MTPPackedVerificationError.missingSharedKV
    }
    guard !sharedKV.isEmpty else {
        throw MTPPackedVerificationError.emptySharedKV
    }
    guard let sources = state?[mtpSharedKVSourceIndicesKey] else {
        throw MTPPackedVerificationError.missingSharedKVSourceIndices
    }
    let sharedKVKeys = sharedKV.keys.sorted()
    let sourceKeys = sources.keys.sorted()
    guard sharedKVKeys == sourceKeys else {
        throw MTPPackedVerificationError.sharedKVSourceKeyMismatch(
            expected: sharedKVKeys, actual: sourceKeys)
    }

    var postForwardOffsets = [[Int]?](repeating: nil, count: cache.count)
    for sourceIndex in Set(sources.values) {
        guard cache.indices.contains(sourceIndex) else {
            let layerType = sources
                .filter { $0.value == sourceIndex }
                .map(\.key)
                .sorted()
                .first!
            throw MTPPackedVerificationError.invalidSharedKVSource(
                layerType: layerType, sourceIndex: sourceIndex)
        }
        let offsets = cache[sourceIndex].batchOffset
        guard offsets.ndim == 1, offsets.dim(0) == batchSize else {
            throw MTPPackedVerificationError.invalidPostForwardCacheOffsetShape(
                cacheIndex: sourceIndex,
                expectedCount: batchSize,
                actualShape: offsets.shape)
        }
        let values = offsets.asArray(Int.self)
        for (packedRow, map) in rowMaps.enumerated() {
            let (expected, overflow) = map.queryOffset.addingReportingOverflow(map.inputCount)
            guard !overflow else {
                throw MTPPackedVerificationError.postForwardCacheOffsetOverflow(
                    rowIndex: map.rowIndex,
                    queryOffset: map.queryOffset,
                    inputCount: map.inputCount)
            }
            guard values[packedRow] == expected else {
                throw MTPPackedVerificationError.postForwardCacheOffsetMismatch(
                    cacheIndex: sourceIndex,
                    rowIndex: map.rowIndex,
                    expected: expected,
                    actual: values[packedRow])
            }
        }
        postForwardOffsets[sourceIndex] = values
    }

    for (layerType, pair) in sharedKV {
        let sourceIndex = sources[layerType]!
        guard cache.indices.contains(sourceIndex) else {
            throw MTPPackedVerificationError.invalidSharedKVSource(
                layerType: layerType, sourceIndex: sourceIndex)
        }
        let keys = pair.0
        let values = pair.1
        guard keys.ndim >= 3, keys.shape == values.shape, keys.dim(-2) > 0 else {
            throw MTPPackedVerificationError.invalidSharedKVShape(
                layerType: layerType,
                keysShape: keys.shape,
                valuesShape: values.shape)
        }
        guard keys.dim(0) == batchSize else {
            throw MTPPackedVerificationError.invalidSharedKVBatch(
                layerType: layerType,
                expectedBatch: batchSize,
                actualBatch: keys.dim(0))
        }
    }

    guard sharedKV["full_attention"] != nil else {
        throw MTPPackedVerificationError.missingFullAttentionSharedKV
    }

    let positionDeltas = state?[mtpPositionDeltasKey]
    if let positionDeltas {
        guard positionDeltas.ndim == 0 || positionDeltas.dim(0) == batchSize else {
            throw MTPPackedVerificationError.invalidPositionDeltasShape(
                expectedBatch: batchSize,
                actualShape: positionDeltas.shape)
        }
    }

    return try rowMaps.enumerated().map { packedRow, map in
        var rowSharedKV: [String: (MLXArray, MLXArray)] = [:]
        var rowOffsets: [String: Int] = [:]
        rowSharedKV.reserveCapacity(sharedKV.count)
        rowOffsets.reserveCapacity(sharedKV.count)

        for (layerType, pair) in sharedKV {
            let sourceIndex = sources[layerType]!
            let offsets = postForwardOffsets[sourceIndex]!
            let offset = offsets[packedRow]
            let liveLength = cache[sourceIndex].maxSize.map { Swift.min(offset, $0) } ?? offset
            guard pair.0.dim(-2) >= liveLength else {
                throw MTPPackedVerificationError.insufficientSharedKVSequenceSpan(
                    layerType: layerType,
                    rowIndex: map.rowIndex,
                    expected: liveLength,
                    actual: pair.0.dim(-2))
            }
            rowSharedKV[layerType] = (
                pair.0[packedRow ..< (packedRow + 1), .ellipsis, 0 ..< liveLength, 0...],
                pair.1[packedRow ..< (packedRow + 1), .ellipsis, 0 ..< liveLength, 0...]
            )
            rowOffsets[layerType] = offset
        }

        let queryOffset = rowOffsets["full_attention"]!
        return MTPPackedVerificationRowState(
            lastHidden: lastHidden[
                packedRow ..< (packedRow + 1), 0 ..< map.inputCount, 0...],
            sharedKV: rowSharedKV,
            sharedKVSourceIndices: sources,
            sharedKVOffsets: rowOffsets,
            queryOffset: queryOffset,
            positionDeltas: positionDeltas.map { deltas in
                deltas.ndim == 0
                    ? deltas.reshaped(1)
                    : deltas[packedRow ..< (packedRow + 1)]
            })
    }
}

private func validateMTPPackedCacheOffsets(
    _ cache: [any MTPPackedVerificationCache], queryOffsets: [Int]
) throws {
    for (cacheIndex, entry) in cache.enumerated() {
        let offsets = entry.batchOffset
        guard offsets.ndim == 1, offsets.dim(0) == queryOffsets.count else {
            throw MTPPackedVerificationError.invalidCacheOffsetShape(
                cacheIndex: cacheIndex,
                expectedCount: queryOffsets.count,
                actualShape: offsets.shape)
        }
        let actual = offsets.asArray(Int.self)
        guard actual == queryOffsets else {
            throw MTPPackedVerificationError.cacheOffsetMismatch(
                cacheIndex: cacheIndex,
                expected: queryOffsets,
                actual: actual)
        }
    }
}
