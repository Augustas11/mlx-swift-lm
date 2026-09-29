// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen35MTPPackedError: Error {
    case emptyRows
    case incompatibleState
    case invalidAcceptedCount
}

/// Ragged batch adapter for the Qwen MTP head's row-owned KVCacheSimple state.
/// Past and appended values remain row-local; only the model-body call is
/// shared. Per-row offsets drive RoPE and the explicit mask hides both past
/// padding and right-padded commit columns.
private final class Qwen35MTPPackedCache: BaseKVCache, BatchPositionedKVCache {
    private let rowCaches: [KVCache]
    private let inputCounts: [Int]
    private let initialOffsets: [Int]
    private var packedKeys: MLXArray?
    private var packedValues: MLXArray?

    init(rowCaches: [KVCache], inputCounts: [Int]) throws {
        guard !rowCaches.isEmpty, rowCaches.count == inputCounts.count else {
            throw Qwen35MTPPackedError.emptyRows
        }
        self.rowCaches = rowCaches
        self.inputCounts = inputCounts
        self.initialOffsets = rowCaches.map(\.offset)
        super.init()
    }

    var batchOffset: MLXArray {
        MLXArray(initialOffsets.map(Int32.init))
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let finalOffsets = zip(initialOffsets, inputCounts).map(+)
        let capacity = finalOffsets.max() ?? 0
        var outputKeys = MLXArray.zeros(
            [rowCaches.count, keys.dim(1), capacity, keys.dim(3)], dtype: keys.dtype)
        var outputValues = MLXArray.zeros(
            [rowCaches.count, values.dim(1), capacity, values.dim(3)], dtype: values.dtype)
        for rowIndex in rowCaches.indices {
            let prior = rowCaches[rowIndex].state
            if prior.count == 2, initialOffsets[rowIndex] > 0 {
                outputKeys[rowIndex ..< rowIndex + 1, 0..., 0 ..< initialOffsets[rowIndex], 0...] =
                    prior[0][0..., 0..., 0 ..< initialOffsets[rowIndex], 0...]
                outputValues[rowIndex ..< rowIndex + 1, 0..., 0 ..< initialOffsets[rowIndex], 0...] =
                    prior[1][0..., 0..., 0 ..< initialOffsets[rowIndex], 0...]
            }
            let count = inputCounts[rowIndex]
            outputKeys[
                rowIndex ..< rowIndex + 1, 0...,
                initialOffsets[rowIndex] ..< finalOffsets[rowIndex], 0...
            ] = keys[rowIndex ..< rowIndex + 1, 0..., 0 ..< count, 0...]
            outputValues[
                rowIndex ..< rowIndex + 1, 0...,
                initialOffsets[rowIndex] ..< finalOffsets[rowIndex], 0...
            ] = values[rowIndex ..< rowIndex + 1, 0..., 0 ..< count, 0...]
        }
        packedKeys = outputKeys
        packedValues = outputValues
        offset = capacity
        return (outputKeys, outputValues)
    }

    override func makeMask(
        n: Int, windowSize _: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let batch = rowCaches.count
        let capacity = zip(initialOffsets, inputCounts).map(+).max() ?? 0
        let queries = batchOffset.reshaped([batch, 1, 1, 1])
            + MLXArray(Int32(0) ..< Int32(n)).reshaped([1, 1, n, 1])
        let keys = MLXArray(Int32(0) ..< Int32(capacity)).reshaped([1, 1, 1, capacity])
        let live = MLXArray(zip(initialOffsets, inputCounts).map { Int32($0 + $1) })
            .reshaped([batch, 1, 1, 1])
        return .array((keys .<= queries) & (keys .< live))
    }

    func scatter() throws -> [[MLXArray]] {
        guard let packedKeys, let packedValues else {
            throw Qwen35MTPPackedError.incompatibleState
        }
        return rowCaches.indices.map { rowIndex in
            let count = initialOffsets[rowIndex] + inputCounts[rowIndex]
            return [
                packedKeys[rowIndex ..< rowIndex + 1, 0..., 0 ..< count, 0...],
                packedValues[rowIndex ..< rowIndex + 1, 0..., 0 ..< count, 0...],
            ]
        }
    }
}

private struct Qwen35MTPPreparedPackedRow {
    let rowIndex: Int
    var state: MTPDrafterState
    let tokens: MLXArray
    let hidden: MLXArray
    let inputCount: Int
}

final class Qwen35MTPPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func newCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        hiddenStates previousHidden: MLXArray,
        cache: [KVCache],
        positionOffset: Int?
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            let faMask = createAttentionMask(h: hiddenStates, cache: layerCache)
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: nil,
                cache: layerCache,
                positionOffset: positionOffset)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, MTPPackedStatefulDrafterModel {
    public let configuration: Qwen35TextConfiguration
    public let maximumBlockSize: Int? = 2
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    private let preconvertedNorms: Bool
    private let standaloneCheckpoint: Bool

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public init(
        _ configuration: Qwen35TextConfiguration,
        preconvertedNorms: Bool = false,
        standaloneCheckpoint: Bool = false
    ) {
        self.configuration = configuration
        self.preconvertedNorms = preconvertedNorms
        self.standaloneCheckpoint = standaloneCheckpoint
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false,
        standaloneCheckpoint: Bool = false
    ) {
        self.init(
            configuration.textConfig,
            preconvertedNorms: preconvertedNorms,
            standaloneCheckpoint: standaloneCheckpoint)
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let prompt = normalizedMTPTokenBatch(promptTokens)
        let bonus = normalizedMTPColumn(firstBonus)
        guard prompt.dim(-1) > 0 else { return }

        let shifted = concatenated([prompt[0..., 1...], bonus], axis: 1)
        let hidden = targetHidden[0..., ..<shifted.dim(1), 0...]
        state.nextPosition = 0
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(shifted),
            hiddenStates: hidden,
            cache: state.cache,
            positionOffset: 0)
        state.nextPosition = shifted.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: nil,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens

        if let seed = state.seedToken {
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            return seed
        }

        state.proposalAppended = blockSize - 1
        let proposed = draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: lastToken,
            lastHidden: lastHidden,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            cache: state.cache
        ) { inputsEmbeds, hiddenStates, cache, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                positionOffset: positionOffset)
        }
        state.nextPosition += state.proposalAppended
        return proposed
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
            hiddens.append(targetHidden[0..., index ..< (index + 1), 0...])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        hiddens.append(targetHidden[0..., acceptedCount ..< (acceptedCount + 1), 0...])

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let committedTokens = concatenated(tokens, axis: 1)
        let committedHidden = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(committedTokens),
            hiddenStates: committedHidden,
            cache: state.cache,
            positionOffset: state.nextPosition)
        state.nextPosition += committedTokens.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func advanceAndProposePacked(
        target: any LanguageModel,
        rows: [MTPPackedDrafterAdvanceRow],
        sampler: any LogitSampler
    ) throws -> [MTPPackedDrafterAdvanceOutput] {
        guard !rows.isEmpty else { return [] }
        var prepared: [Qwen35MTPPreparedPackedRow] = []
        prepared.reserveCapacity(rows.count)
        for row in rows {
            guard row.acceptedCount >= 0,
                  row.targetHidden.ndim == 3,
                  row.targetHidden.dim(1) > row.acceptedCount
            else { throw Qwen35MTPPackedError.invalidAcceptedCount }
            var state = row.state
            state.cache = state.cache.map { $0.copy() }
            let keepAppended = Swift.min(row.acceptedCount, state.proposalAppended)
            let trim = state.proposalAppended - keepAppended
            if trim > 0 {
                trimPromptCache(state.cache, numTokens: trim)
                state.nextPosition -= trim
            }

            var tokens: [MLXArray] = []
            var hiddens: [MLXArray] = []
            for index in keepAppended ..< row.acceptedCount {
                tokens.append(row.draftTokens[0..., index ..< index + 1])
                hiddens.append(row.targetHidden[0..., index ..< index + 1, 0...])
            }
            tokens.append(MLXArray([Int32(row.finalToken)]).reshaped([1, 1]))
            hiddens.append(
                row.targetHidden[0..., row.acceptedCount ..< row.acceptedCount + 1, 0...])
            let tokenBatch = concatenated(tokens, axis: 1)
            prepared.append(Qwen35MTPPreparedPackedRow(
                rowIndex: row.rowIndex,
                state: state,
                tokens: tokenBatch,
                hidden: concatenated(hiddens, axis: 1),
                inputCount: tokenBatch.dim(1)
            ))
        }

        let width = prepared.map(\.inputCount).max() ?? 1
        let paddedTokens = concatenated(prepared.map { row in
            row.inputCount == width
                ? row.tokens
                : concatenated([
                    row.tokens,
                    MLXArray.zeros([1, width - row.inputCount], dtype: row.tokens.dtype),
                ], axis: 1)
        }, axis: 0)
        let hiddenSize = prepared[0].hidden.dim(2)
        let paddedHidden = concatenated(prepared.map { row in
            row.inputCount == width
                ? row.hidden
                : concatenated([
                    row.hidden,
                    MLXArray.zeros(
                        [1, width - row.inputCount, hiddenSize], dtype: row.hidden.dtype),
                ], axis: 1)
        }, axis: 0)

        guard let layerCount = prepared.first?.state.cache.count,
              prepared.allSatisfy({ $0.state.cache.count == layerCount })
        else { throw Qwen35MTPPackedError.incompatibleState }
        let packedCaches = try (0 ..< layerCount).map { layerIndex in
            try Qwen35MTPPackedCache(
                rowCaches: prepared.map { $0.state.cache[layerIndex] },
                inputCounts: prepared.map(\.inputCount)
            )
        }
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let packedHidden = mtp(
            inputsEmbeds: inputEmbedding(paddedTokens),
            hiddenStates: paddedHidden,
            cache: packedCaches,
            positionOffset: nil
        )

        for layerIndex in packedCaches.indices {
            let states = try packedCaches[layerIndex].scatter()
            for rowIndex in prepared.indices {
                prepared[rowIndex].state.cache[layerIndex].state = states[rowIndex]
            }
        }
        let proposalHidden = concatenated(prepared.enumerated().map { rowIndex, row in
            packedHidden[
                rowIndex ..< rowIndex + 1,
                row.inputCount - 1 ..< row.inputCount,
                0...
            ]
        }, axis: 0)
        let proposals = sampleMTPSeed(
            hidden: proposalHidden,
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            sampler: sampler
        )
        return prepared.enumerated().map { index, row in
            var state = row.state
            state.nextPosition += row.inputCount
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            return MTPPackedDrafterAdvanceOutput(
                rowIndex: row.rowIndex,
                proposal: proposals[index ..< index + 1, 0...],
                state: state
            )
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        qwenMTPSanitizeWeights(
            weights: weights,
            mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts,
            shiftNormWeights: !preconvertedNorms,
            standaloneCheckpoint: standaloneCheckpoint
        )
    }

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        if let model = target as? Qwen35Model {
            return (model.languageModel.model.embedTokens, model.languageModel.lmHead)
        }
        if let model = target as? Qwen35TextModel {
            return (model.model.embedTokens, model.lmHead)
        }
        fatalError(
            "Qwen35MTPDraftModel requires a Qwen35 target, got \(type(of: target))")
    }
}
