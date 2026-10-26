// DeepSeek V4 DSpark, following DeepInfer v4_modules/heads.py and mlx-vlm's
// speculative/drafters/deepseek_v4_dspark. Context KV contains committed
// target positions only; denoising-block KV never enters the cache.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class DeepseekV4DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var embedding: Embedding
    @ModuleInfo(key: "markov_w2") var projection: Linear

    init(vocabularySize: Int, rank: Int) {
        _embedding.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: rank)
        _projection.wrappedValue = Linear(rank, vocabularySize, bias: false)
        super.init()
    }
}

final class DeepseekV4DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(hiddenSize: Int, rank: Int) {
        _projection.wrappedValue = Linear(hiddenSize + rank, 1, bias: false)
        super.init()
    }

    func callAsFunction(hidden: MLXArray, markov: MLXArray) -> MLXArray {
        // DeepInfer uses a bias-free float32 predictor. Confidence never
        // changes the target's greedy acceptance decision.
        sigmoid(deepseekV4LinearF32(projection, concatenated([hidden, markov], axis: -1)))
            .squeezed(axis: -1)
    }
}

final class DeepseekV4DSparkStage: Module {
    @ModuleInfo(key: "attn") var attn: DeepseekV4Attention
    @ModuleInfo(key: "ffn") var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: GLM5NextHyperConnection
    @ModuleInfo(key: "main_proj") var mainProj: Linear?
    @ModuleInfo(key: "main_norm") var mainNorm: RMSNorm?
    @ModuleInfo(key: "markov_head") var markovHead: DeepseekV4DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") var confidenceHead: DeepseekV4DSparkConfidenceHead?
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead?
    @ModuleInfo(key: "norm") var norm: RMSNorm?

    init(_ configuration: DeepseekV4Configuration, stageID: Int, stageCount: Int) {
        var config = configuration
        config.compressRatios = [0]
        config.hiddenLayers = 1
        config.numHashLayers = 0
        _attn.wrappedValue = DeepseekV4Attention(config, layerIdx: 0)
        _ffn.wrappedValue = DeepseekV4MoE(config, layerIdx: 0)
        _attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        func hc() -> GLM5NextHyperConnection {
            GLM5NextHyperConnection(
                hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
                hcEps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
        }
        _attnHC.wrappedValue = hc()
        _ffnHC.wrappedValue = hc()
        if stageID == 0 {
            _mainProj.wrappedValue = Linear(
                config.hiddenSize * config.dsparkTargetLayerIDs.count, config.hiddenSize, bias: false)
            _mainNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if stageID == stageCount - 1 {
            _markovHead.wrappedValue = DeepseekV4DSparkMarkovHead(
                vocabularySize: config.vocabularySize, rank: config.dsparkMarkovRank)
            _confidenceHead.wrappedValue = DeepseekV4DSparkConfidenceHead(
                hiddenSize: config.hiddenSize, rank: config.dsparkMarkovRank)
            _hcHead.wrappedValue = DeepseekV4HyperHead(config)
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: RotatingKVCache) {
        guard context.dim(1) > 0 else { return }
        let kv = attn.kvNorm(attn.wkv(context)).reshaped(context.dim(0), 1, context.dim(1), attn.headDim)
        let rotated = attn.rope(kv, offset: cache.offset)
        _ = cache.update(
            keys: rotated,
            values: MLXArray.zeros([context.dim(0), 1, context.dim(1), 0], dtype: rotated.dtype))
        // Unlike target prefill, draft attention has no earlier queries that
        // need a chunk's temporary extended window. Retain only live context.
        if let view = cache.logicalView(tail: attn.window) {
            var metadata = cache.metaState
            metadata[4] = String(view.0.dim(2))
            metadata[6] = "false"
            cache.state = [view.0, view.1]
            cache.metaState = metadata
        }
    }

    private func crossAttention(_ x: MLXArray, caches: [RotatingKVCache]) -> MLXArray {
        let (batch, width) = (x.dim(0), x.dim(1))
        var q = attn.wqB(attn.qNorm(attn.wqA(x)))
            .reshaped(batch, width, attn.heads, attn.headDim)
        q = MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: attn.normEps).transposed(0, 2, 1, 3)
        let offsets = MLXArray(caches.map { Int32($0.offset) })
        q = attn.rope(q, offsets: offsets)
        var blockKV = attn.kvNorm(attn.wkv(x)).reshaped(batch, 1, width, attn.headDim)
        blockKV = attn.rope(blockKV, offsets: offsets)
        var outputs = [MLXArray]()
        for (row, cache) in caches.enumerated() {
            let block = blockKV[row ..< (row + 1)]
            let keys = cache.logicalView(tail: attn.window).map { view -> MLXArray in
                var context = view.0
                // oMLX's DSparkContextCache follows the checkpoint's physical
                // absolute-position ring order for its block attention.
                let cutoff = cache.offset % attn.window
                if context.dim(2) == attn.window && cutoff > 0 {
                    let split = attn.window - cutoff
                    context = concatenated([
                        context[.ellipsis, split..., 0...], context[.ellipsis, ..<split, 0...],
                    ], axis: 2)
                }
                return concatenated([context, block], axis: 2)
            } ?? block
            let keyList = broadcast(
                keys.squeezed(axis: 1).expandedDimensions(axis: 1),
                to: [1, width, keys.dim(2), attn.headDim])
            let valid = MLXArray.ones([1, width, keys.dim(2)], dtype: .bool)
            outputs.append(attn.referenceAttention(q[row ..< (row + 1)], keys: keyList, valid: valid))
        }
        var out = concatenated(outputs, axis: 0).asType(x.dtype)
        out = attn.rope(out, offsets: offsets, inverse: true)
        out = out.reshaped(batch, attn.groups, -1, width, attn.headDim)
            .transposed(0, 1, 3, 2, 4).flattened(start: -2)
        if let quantized = attn.woA as? QuantizedMultiLinear {
            out = quantized(out)
        } else {
            out = (attn.woA as! MultiLinear)(out)
        }
        out = out.transposed(0, 2, 1, 3).flattened(start: -2)
        return attn.woB(out)
    }

    func callAsFunction(_ h: MLXArray, cache: RotatingKVCache, inputIDs: MLXArray) -> MLXArray {
        decodeBatch(h, caches: [cache], inputIDs: inputIDs)
    }

    func decodeBatch(_ h: MLXArray, caches: [RotatingKVCache], inputIDs: MLXArray) -> MLXArray {
        let (x, post, comb) = attnHC.normalizedBranchInput(h, norm: attnNorm)
        let h = glm5NextHCExpand(crossAttention(x, caches: caches), h, post, comb)
        let (ffnInput, ffnPost, ffnComb) = ffnHC.normalizedBranchInput(h, norm: ffnNorm)
        return glm5NextHCExpand(ffn(ffnInput, inputIDs: inputIDs), h, ffnPost, ffnComb)
    }

}

/// Target-embedded three-stage DSpark. It retains no model weights or request
/// state itself. The checkpoint adds 10.88 GB; three 128x512 bf16 KV rings
/// add 393216 bytes per row, excluding allocator overhead.
public final class DeepseekV4DSparkDraftModel: Module, MTPPackedStatefulDrafterModel {
    public let configuration: DeepseekV4Configuration
    public let maximumDepth: Int
    public var maximumBlockSize: Int? { maximumDepth + 1 }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true

    public init(_ configuration: DeepseekV4Configuration, maximumDepth: Int = 5) {
        precondition((1 ... 5).contains(maximumDepth))
        self.configuration = configuration
        self.maximumDepth = min(maximumDepth, configuration.dsparkBlockSize)
        super.init()
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: (0 ..< configuration.dsparkStageCount).map { _ in
            RotatingKVCache(maxSize: configuration.slidingWindow)
        })
    }

    private func target(_ target: any LanguageModel) -> DeepseekV4Model {
        guard let target = target as? DeepseekV4Model,
            target.mtp.count == configuration.dsparkStageCount, !target.mtp.isEmpty
        else { preconditionFailure("DSpark requires the target's embedded drafter weights") }
        return target
    }

    private func appendContext(
        target: DeepseekV4Model, hidden: MLXArray, state: inout MTPDrafterState
    ) {
        guard hidden.dim(1) > 0 else { return }
        let first = target.mtp[0]
        let context = first.mainNorm!(first.mainProj!(hidden))
        precondition(state.cache.count == target.mtp.count)
        for (stage, cache) in zip(target.mtp, state.cache) {
            stage.appendContext(context, cache: cache as! RotatingKVCache)
        }
        state.nextPosition += hidden.dim(1)
        state.proposalAppended = 0
        state.seedToken = nil
        state.proposalConfidence = nil
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus: MLXArray, positionDeltas: MLXArray?, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        appendContext(target: self.target(target), hidden: targetHidden, state: &state)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        appendContext(target: self.target(target), hidden: lastHidden, state: &state)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let result = propose(target: self.target(target), anchors: lastToken.reshaped(-1),
            caches: [state.cache], depth: min(maximumDepth, blockSize - 1))
        state.proposalConfidence = result.confidence
        state.seedToken = result.tokens
        state.proposalAppended = 0
        return result.tokens
    }

    private func propose(
        target: DeepseekV4Model, anchors: MLXArray, caches: [[KVCache]], depth: Int
    ) -> (tokens: MLXArray, confidence: MLXArray) {
        precondition(depth > 0 && depth <= maximumDepth && anchors.size == caches.count)
        let inputs = concatenated([
            anchors.expandedDimensions(axis: 1),
            MLXArray.full([anchors.size, depth - 1], values: MLXArray(Int32(configuration.dsparkNoiseTokenID))),
        ], axis: 1)
        var h = repeated(target.embed(inputs).expandedDimensions(axis: 2),
            count: configuration.hcMult, axis: 2)
        for (index, stage) in target.mtp.enumerated() {
            h = stage.decodeBatch(h, caches: caches.map { $0[index] as! RotatingKVCache }, inputIDs: inputs)
        }
        let last = target.mtp.last!
        let hidden = last.norm!(last.hcHead!(h))
        let markov = last.markovHead!
        var previous = anchors
        var tokens = [MLXArray]()
        var confidence = [MLXArray]()
        for position in 0 ..< depth {
            let embedding = markov.embedding(previous)
            // Keep head launch geometry independent of depth. The reference's
            // Markov dependency is sequential even though its backbone is not.
            let hiddenColumn = hidden[0..., position, 0...]
            let stepLogits = target.logits(hiddenColumn) + deepseekV4LinearF32(markov.projection, embedding)
            confidence.append(last.confidenceHead!(hidden: hiddenColumn, markov: embedding)
                .expandedDimensions(axis: 1))
            previous = argMax(stepLogits, axis: -1)
            tokens.append(previous.expandedDimensions(axis: 1))
        }
        return (concatenated(tokens, axis: 1), concatenated(confidence, axis: 1))
    }

    public func commitDrafterState(
        target: any LanguageModel, targetHidden: MLXArray, draftTokens: MLXArray,
        acceptedCount: Int, finalToken: MLXArray, positionDeltas: MLXArray?,
        state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        precondition(acceptedCount >= 0 && acceptedCount < targetHidden.dim(1))
        appendContext(target: self.target(target),
            hidden: targetHidden[0..., ..<(acceptedCount + 1), 0...], state: &state)
    }

    public func proposePacked(
        target: any LanguageModel, states: [MTPDrafterState], anchors: [Int], depth: Int
    ) throws -> MTPPackedDrafterAdvanceResult {
        guard states.count == anchors.count, (1 ... maximumDepth).contains(depth) else {
            throw MTPPackedDrafterError.incompatibleState(row: 0)
        }
        for (index, state) in states.enumerated() {
            guard state.cache.count == configuration.dsparkStageCount,
                state.cache.allSatisfy({ $0 is RotatingKVCache && $0.offset == state.nextPosition })
            else { throw MTPPackedDrafterError.incompatibleState(row: index) }
        }
        guard !states.isEmpty else {
            return MTPPackedDrafterAdvanceResult(states: [],
                proposals: MLXArray.zeros([0, depth], dtype: .int32))
        }
        let result = propose(target: self.target(target),
            anchors: MLXArray(anchors.map(Int32.init)), caches: states.map(\.cache), depth: depth)
        var advanced = states
        for index in advanced.indices {
            advanced[index].seedToken = result.tokens[index ..< (index + 1)]
            advanced[index].proposalConfidence = result.confidence[index ..< (index + 1)]
        }
        return MTPPackedDrafterAdvanceResult(states: advanced, proposals: result.tokens)
    }

    public func advanceAndProposePacked(
        target: any LanguageModel, rows: [MTPPackedDrafterAdvanceRow], sampler: any LogitSampler
    ) throws -> MTPPackedDrafterAdvanceResult {
        // Rows keep independently committed contexts. No batch-wide minimum
        // acceptance discards a good row's accepted prefix.
        var states = [MTPDrafterState]()
        for (index, row) in rows.enumerated() {
            guard row.targetHidden.ndim == 3, row.targetHidden.dim(0) == 1 else {
                throw MTPPackedDrafterError.invalidTargetHidden(row: index, shape: row.targetHidden.shape)
            }
            guard row.acceptedTokens.count < row.targetHidden.dim(1) else {
                throw MTPPackedDrafterError.invalidAcceptedCount(row: index,
                    acceptedCount: row.acceptedTokens.count, inputCount: row.targetHidden.dim(1))
            }
            guard row.state.cache.count == configuration.dsparkStageCount,
                row.state.cache.allSatisfy({ $0 is RotatingKVCache && $0.offset == row.state.nextPosition })
            else { throw MTPPackedDrafterError.incompatibleState(row: index) }
        }
        guard !rows.isEmpty else {
            return MTPPackedDrafterAdvanceResult(states: [],
                proposals: MLXArray.zeros([0, maximumDepth], dtype: .int32))
        }
        for row in rows {
            var state = row.state
            state.cache = state.cache.map { $0.copy() }
            commitDrafterState(target: target, targetHidden: row.targetHidden,
                draftTokens: MLXArray(row.acceptedTokens.map(Int32.init)).reshaped(1, row.acceptedTokens.count),
                acceptedCount: row.acceptedTokens.count, finalToken: MLXArray(Int32(row.finalToken)),
                positionDeltas: row.positionDeltas, state: &state, sampler: sampler)
            states.append(state)
        }
        let result = propose(target: self.target(target),
            anchors: MLXArray(rows.map { Int32($0.finalToken) }), caches: states.map(\.cache), depth: maximumDepth)
        for index in states.indices {
            states[index].seedToken = result.tokens[index ..< (index + 1)]
            states[index].proposalConfidence = result.confidence[index ..< (index + 1)]
        }
        return MTPPackedDrafterAdvanceResult(states: states, proposals: result.tokens)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { [:] }
}

public enum DeepseekV4DSparkRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType("deepseek_v4", creator: { data in
            let config = try JSONDecoder.json5().decode(DeepseekV4Configuration.self, from: data)
            try config.validateModelConfiguration()
            return DeepseekV4DSparkDraftModel(config)
        })
    }
}
