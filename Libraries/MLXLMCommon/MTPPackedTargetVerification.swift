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

    /// Target hidden states for the valid input prefix, when emitted.
    ///
    /// This is the only model state this facade consumes or exposes.
    public let lastHidden: MLXArray?
}

/// Result of one packed target-model verification call.
public struct MTPPackedVerificationOutput {
    /// Per-row logit slices in packed-input order.
    public let rows: [MTPPackedVerificationRowOutput]
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
    case emptyCache
    case unsupportedCache(cacheIndex: Int)
    case cacheOffsetMismatch(cacheIndex: Int, expected: [Int], actual: [Int])
    case invalidCacheOffsetShape(cacheIndex: Int, expectedCount: Int, actualShape: [Int])
    case invalidLogitsShape(expectedBatch: Int, expectedWidth: Int, actualShape: [Int])
    case invalidLastHiddenShape(expectedBatch: Int, expectedWidth: Int, actualShape: [Int])
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
/// This v0.1 facade deliberately creates fresh target state, requests MTP last
/// hidden states, and discards every other returned state key. It does not
/// support shared-target-KV state, model-specific position deltas, or native
/// recurrent checkpoint state. The external scheduler owns independent row
/// transactions and resolves their commit/discard policy after verification.
public func verifyMTPPackedTargets(
    model: any LanguageModel,
    tokens: MLXArray,
    rowMaps: [MTPPackedVerificationRowMap],
    cache: [KVCache]
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

    let rows = rowMaps.enumerated().map { packedRow, map in
        let rowLogits = output.logits[packedRow]
        return MTPPackedVerificationRowOutput(
            map: map,
            proposalLogits: rowLogits[0 ..< map.proposalCount, 0...],
            bonusLogits: rowLogits[(map.inputCount - 1) ..< map.inputCount, 0...],
            lastHidden: lastHidden?[
                packedRow, 0 ..< map.inputCount, 0...])
    }
    return MTPPackedVerificationOutput(rows: rows)
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
