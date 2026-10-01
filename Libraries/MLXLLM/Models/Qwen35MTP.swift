// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

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

    /// Advance every row by its accepted prefix and final token with one MTP
    /// forward, then propose each row's next token from the same forward.
    ///
    /// Row `r` feeds exactly the columns ``commitDrafterState`` would:
    /// accepted proposals not already in the drafter cache, then the final
    /// token, paired with the target hidden states at the same verification
    /// columns. Rows are right-padded to the widest row; padded columns are
    /// masked out of every row's attention and dropped from its cache. The
    /// next proposal is sampled from the hidden state at each row's last valid
    /// column. Input states are not mutated; nothing is evaluated.
    public func advanceAndProposePacked(
        target: any LanguageModel,
        rows: [MTPPackedDrafterAdvanceRow],
        sampler: any LogitSampler
    ) throws -> MTPPackedDrafterAdvanceResult {
        guard !rows.isEmpty else {
            return MTPPackedDrafterAdvanceResult(
                states: [], proposals: MLXArray.zeros([0, 1], dtype: .int32))
        }
        let layerCount = mtp.layers.count
        var rowStates: [[[MLXArray]]] = []
        var columnTokens: [[Int32]] = []
        var columnHidden: [MLXArray] = []
        var nextPositions: [Int] = []
        rowStates.reserveCapacity(rows.count)
        for (index, row) in rows.enumerated() {
            let hidden = row.targetHidden
            guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0 else {
                throw MTPPackedDrafterError.invalidTargetHidden(row: index, shape: hidden.shape)
            }
            let acceptedCount = row.acceptedTokens.count
            guard acceptedCount < hidden.dim(1) else {
                throw MTPPackedDrafterError.invalidAcceptedCount(
                    row: index, acceptedCount: acceptedCount, inputCount: hidden.dim(1))
            }
            let state = row.state
            guard state.cache.count == layerCount,
                  state.cache.allSatisfy({ $0 is KVCacheSimple }),
                  state.proposalAppended >= 0
            else { throw MTPPackedDrafterError.incompatibleState(row: index) }
            let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
            let trim = state.proposalAppended - keepAppended
            // Per-row commit positions RoPE by `nextPosition`; the packed
            // forward positions by each cache's offset, so they must agree.
            let cacheOffset = state.cache[0].offset
            guard state.cache.allSatisfy({ $0.offset == cacheOffset }),
                  cacheOffset == state.nextPosition,
                  trim <= cacheOffset
            else {
                throw MTPPackedDrafterError.positionMismatch(
                    row: index, nextPosition: state.nextPosition, cacheOffset: cacheOffset)
            }
            let live = cacheOffset - trim
            rowStates.append(state.cache.map { cache in
                let kv = cache.state
                guard live > 0, kv.count == 2 else { return [] }
                return kv[0].dim(2) == live
                    ? kv : kv.map { $0[.ellipsis, ..<live, 0...] }
            })
            nextPositions.append(live)
            columnTokens.append(
                row.acceptedTokens[keepAppended...].map(Int32.init) + [Int32(row.finalToken)])
            columnHidden.append(hidden[0..., keepAppended ..< (acceptedCount + 1), 0...])
        }

        let inputCounts = columnTokens.map(\.count)
        let width = inputCounts.max() ?? 1
        let hiddenSize = columnHidden[0].dim(2)
        let tokens = MLXArray(
            columnTokens.flatMap { $0 + Array(repeating: Int32(0), count: width - $0.count) },
            [rows.count, width])
        let paddedHidden = columnHidden.map { hidden -> MLXArray in
            let pad = width - hidden.dim(1)
            return pad == 0
                ? hidden
                : concatenated(
                    [hidden, MLXArray.zeros([1, pad, hiddenSize], dtype: hidden.dtype)], axis: 1)
        }
        let packedCaches = (0 ..< layerCount).map { layer in
            MTPPackedDrafterLayerCache(
                rowStates: rowStates.map { $0[layer] }, inputCounts: inputCounts)
        }

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(tokens),
            hiddenStates: rows.count == 1 ? paddedHidden[0] : concatenated(paddedHidden, axis: 0),
            cache: packedCaches,
            positionOffset: nil)

        let seedHidden = inputCounts.enumerated().map { row, count in
            mtpHidden[row ..< row + 1, (count - 1) ..< count, 0...]
        }
        let packedSeedHidden = rows.count == 1 ? seedHidden[0] : concatenated(seedHidden, axis: 0)
        let logits = lmHead.map { $0(packedSeedHidden) }
            ?? targetEmbedTokens.asLinear(packedSeedHidden)
        let proposals = sampler.sample(logits: logits[0..., -1, 0...])
            .reshaped([rows.count, 1])

        var advancedCaches = rows.map { _ in [KVCache]() }
        for packed in packedCaches {
            guard let states = packed.rowStates() else {
                throw MTPPackedDrafterError.incompatibleState(row: 0)
            }
            for row in rows.indices {
                let cache = KVCacheSimple()
                cache.state = states[row]
                advancedCaches[row].append(cache)
            }
        }
        let states = rows.indices.map { row in
            MTPDrafterState(
                cache: advancedCaches[row],
                nextPosition: nextPositions[row] + inputCounts[row],
                seedToken: proposals[row ..< row + 1, 0...],
                seedHidden: seedHidden[row],
                proposalAppended: 0)
        }
        return MTPPackedDrafterAdvanceResult(states: states, proposals: proposals)
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
