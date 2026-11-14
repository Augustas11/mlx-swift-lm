// MiniMax M3 DSpark drafter (`DSparkMiniMaxDraftModel`, the `dspark/` folder
// of a MiniMax M3.1 checkpoint). Forward semantics follow the published
// serving references for this checkpoint (SGLang `dspark_minimax.py`, vLLM
// `minimax_m3/common/dspark.py`) and DeepSpec's DSpark reference:
//
// - Context: the target's residual outputs after `dspark_target_layer_ids`,
//   concatenated, through `fc` then `hidden_norm`. Every draft layer projects
//   that context through its own k/v projections, k norm and RoPE at the
//   context's absolute positions, into a private sliding-window cache. Only
//   committed target positions enter the context.
// - Block: `[anchor, mask x (block - 1)]` embedded with the target's
//   embeddings at positions `p ..< p + block`. Queries come from the block
//   only; keys are context then block; attention inside the block is
//   bidirectional. Block K/V never enters the cache.
// - Output: `final_norm`, the target's `lm_head`, plus the vanilla Markov
//   bias `markov_w2(markov_w1(prev))`, chained left to right from the anchor.
//   Every block position proposes (row j predicts position p + j + 1).
// - Confidence: `sigmoid(proj(concat(h_j, markov_w1(prev_j))))`, the
//   conditional acceptance probability of draft j. It shapes how much is
//   verified; the target's greedy token alone decides acceptance.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The DSpark fields of a MiniMax DSpark drafter `config.json`, plus the draft
/// layers' text configuration (decoded by ``MiniMaxM3Configuration``).
public struct MiniMaxM3DSparkConfiguration: Sendable {
    public static let architecture = "DSparkMiniMaxDraftModel"

    let text: MiniMaxM3Configuration
    public let blockSize: Int
    public let markovRank: Int
    public let noiseTokenID: Int
    public let targetLayerIDs: [Int]
    /// Target decoder layer count the drafter was trained against, when the
    /// checkpoint records it (`dflash_config.num_target_layers`).
    public let targetLayerCount: Int?
    public let slidingWindow: Int
    /// MXFP8 (e4m3 weights, e8m0 scales per 32 inputs) draft-layer projections.
    public let mxfp8: Bool

    public var hiddenSize: Int { text.hiddenSize }
    public var layerCount: Int { text.hiddenLayers }
    public var vocabularySize: Int { text.vocabularySize }

    public init(data: Data) throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "MiniMaxM3DSparkConfiguration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw fail("config.json is not an object")
        }
        let textConfig = root["text_config"] as? [String: Any] ?? [:]
        func value(_ key: String) -> Any? { root[key] ?? textConfig[key] }
        func int(_ key: String) -> Int? { (value(key) as? NSNumber)?.intValue }
        func bool(_ key: String) -> Bool? { (value(key) as? NSNumber)?.boolValue }

        let architectures = (root["architectures"] as? [String] ?? [])
            + (textConfig["architectures"] as? [String] ?? [])
        guard architectures.contains(Self.architecture) else {
            throw fail("not a \(Self.architecture) checkpoint")
        }
        text = try JSONDecoder.json5().decode(MiniMaxM3Configuration.self, from: data)
        let dflash = value("dflash_config") as? [String: Any] ?? [:]

        guard let blockSize = int("dspark_block_size") ?? (dflash["block_size"] as? NSNumber)?.intValue,
            let markovRank = int("dspark_markov_rank"),
            let noise = int("dspark_noise_token_id") ?? (dflash["mask_token_id"] as? NSNumber)?.intValue,
            let layerIDs = (value("dspark_target_layer_ids") as? [NSNumber])
                ?? (dflash["target_layer_ids"] as? [NSNumber])
        else { throw fail("DSpark block size, Markov rank, noise token or target layers missing") }
        guard (value("dspark_markov_head_type") as? String ?? "vanilla") == "vanilla" else {
            throw fail("only the vanilla Markov head is supported")
        }
        guard bool("dspark_use_dense_ffn") ?? true, !(bool("dspark_hybrid_context_fusion") ?? false) else {
            throw fail("only dense-FFN DSpark without hybrid context fusion is supported")
        }
        guard bool("enable_confidence_head") ?? false, bool("confidence_head_with_markov") ?? false else {
            throw fail("the Markov-conditioned confidence head is required")
        }
        let layerTypes = (textConfig["layer_types"] as? [String]) ?? (dflash["layer_types"] as? [String])
        guard (layerTypes ?? []).allSatisfy({ $0 == "sliding_attention" }) else {
            throw fail("draft layers must all be sliding_attention")
        }
        self.blockSize = blockSize
        self.markovRank = markovRank
        self.noiseTokenID = noise
        self.targetLayerIDs = layerIDs.map(\.intValue)
        self.targetLayerCount = (dflash["num_target_layers"] as? NSNumber)?.intValue
        self.slidingWindow =
            (textConfig["sliding_window"] as? NSNumber)?.intValue
            ?? (dflash["sliding_window"] as? NSNumber)?.intValue ?? 4096

        if let quantization = root["quantization_config"] as? [String: Any] {
            guard quantization["quant_method"] as? String == "mxfp8",
                (quantization["weight_block_size"] as? [NSNumber])?.map(\.intValue) == [1, 32]
            else { throw fail("only MXFP8 [1, 32] draft quantization is supported") }
            mxfp8 = true
        } else {
            mxfp8 = false
        }
        try validate()
    }

    func validate() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "MiniMaxM3DSparkConfiguration", code: 2,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard (1 ... 16).contains(blockSize), markovRank > 0, slidingWindow > 0,
            noiseTokenID >= 0, noiseTokenID < text.vocabularySize
        else { throw fail("invalid DSpark block size, Markov rank, window or noise token") }
        guard !targetLayerIDs.isEmpty, Set(targetLayerIDs).count == targetLayerIDs.count,
            targetLayerIDs.allSatisfy({ $0 >= 0 && $0 < (targetLayerCount ?? Int.max) })
        else { throw fail("invalid DSpark target layers") }
        guard text.hiddenAct == "swigluoai", text.qkNormType == "per_head", text.useQkNorm,
            !text.attentionOutputGate, !text.moeLayers.contains(true), !text.sparseLayers.contains(true),
            text.attentionHeads % text.kvHeads == 0, text.rotaryDim <= text.headDim
        else { throw fail("unsupported DSpark draft layer layout") }
    }
}

final class MiniMaxM3DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var embedding: Embedding
    @ModuleInfo(key: "markov_w2") var projection: Linear

    init(vocabularySize: Int, rank: Int) {
        _embedding.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: rank)
        _projection.wrappedValue = Linear(rank, vocabularySize, bias: false)
        super.init()
    }
}

final class MiniMaxM3DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(inputSize: Int) {
        _projection.wrappedValue = Linear(inputSize, 1, bias: true)
        super.init()
    }

    /// The references run this predictor in float32.
    func callAsFunction(hidden: MLXArray, markov: MLXArray) -> MLXArray {
        let x = concatenated([hidden, markov], axis: -1).asType(.float32)
        var y = matmul(x, projection.weight.asType(.float32).T)
        if let bias = projection.bias { y = y + bias.asType(.float32) }
        return sigmoid(y).squeezed(axis: -1)
    }
}

/// One draft layer's committed context K/V: the newest `window` positions,
/// RoPE'd at their absolute positions. `offset` counts every committed
/// position, including those that slid out of the window.
final class MiniMaxM3DSparkContextCache: BaseKVCache {
    let window: Int
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?

    init(window: Int) {
        self.window = window
        super.init()
    }

    /// Append `[1, kvHeads, T, D]` keys and values of `committed` positions
    /// (`T <= committed`: positions already past the window are not passed).
    func append(keys newKeys: MLXArray, values newValues: MLXArray, committed: Int) {
        precondition(newKeys.dim(2) <= committed && newKeys.dim(2) == newValues.dim(2))
        offset += committed
        guard newKeys.dim(2) > 0 else { return }
        var k = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        var v = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        if k.dim(2) > window {
            k = k[0..., 0..., (k.dim(2) - window)..., 0...]
            v = v[0..., 0..., (v.dim(2) - window)..., 0...]
        }
        keys = k
        values = v
    }

    var count: Int { keys?.dim(2) ?? 0 }

    override var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count == 2 else {
                keys = nil
                values = nil
                return
            }
            keys = newValue[0]
            values = newValue[1]
        }
    }

    override var metaState: [String] {
        get { [String(offset), String(window)] }
        set {}
    }

    override func copy() -> any KVCache {
        let copied = MiniMaxM3DSparkContextCache(window: window)
        copied.offset = offset
        copied.keys = keys
        copied.values = values
        return copied
    }
}

/// Separate-artifact DSpark drafter for MiniMax M3 targets. It reads the
/// target's embeddings and `lm_head` per call and stores no request state;
/// each row's context caches live in its ``MTPDrafterState``.
public final class MiniMaxM3DSparkDraftModel: Module, MTPPackedStatefulDrafterModel {
    public let configuration: MiniMaxM3DSparkConfiguration
    /// Draft tokens returned per proposal (at most the block size).
    public let maximumDepth: Int
    public var maximumBlockSize: Int? { maximumDepth + 1 }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "final_norm") var finalNorm: MiniMaxM3RMSNorm
    let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "markov_head") var markovHead: MiniMaxM3DSparkMarkovHead
    @ModuleInfo(key: "confidence_head") var confidenceHead: MiniMaxM3DSparkConfidenceHead

    public init(_ configuration: MiniMaxM3DSparkConfiguration, maximumDepth: Int? = nil) {
        let depth = maximumDepth ?? configuration.blockSize
        precondition((1 ... configuration.blockSize).contains(depth))
        self.configuration = configuration
        self.maximumDepth = depth
        let text = configuration.text
        let hidden = text.hiddenSize
        _fc.wrappedValue = Linear(hidden * configuration.targetLayerIDs.count, hidden, bias: false)
        _hiddenNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: hidden, eps: text.rmsNormEps, gemma: text.useGemmaNorm)
        _finalNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: hidden, eps: text.rmsNormEps, gemma: text.useGemmaNorm)
        layers = (0 ..< text.hiddenLayers).map { MiniMaxM3DecoderLayer(text, layer: $0) }
        _markovHead.wrappedValue = MiniMaxM3DSparkMarkovHead(
            vocabularySize: text.vocabularySize, rank: configuration.markovRank)
        _confidenceHead.wrappedValue = MiniMaxM3DSparkConfidenceHead(
            inputSize: hidden + configuration.markovRank)
        super.init()
        if configuration.mxfp8 {
            // The checkpoint stores e4m3 weights with e8m0 scales per 32
            // inputs: MLX's mxfp8 layout, used as is (no dequantization).
            quantize(
                model: self,
                filter: { path, module in
                    guard path.hasPrefix("layers."), module is Linear, !(module is QuantizedLinear)
                    else { return nil }
                    return (groupSize: 32, bits: 8, mode: .mxfp8)
                },
                apply: { module, groupSize, bits, mode in
                    guard let linear = module as? Linear else { return nil }
                    let (outputs, inputs) = linear.shape
                    return QuantizedLinear(
                        weight: MLXArray.zeros([outputs, inputs / 4], dtype: .uint32),
                        scales: MLXArray.zeros([outputs, inputs / groupSize], dtype: .uint8),
                        biases: nil, groupSize: groupSize, bits: bits, mode: mode)
                })
        }
    }

    /// Cache every Gemma norm's `1 + weight`; call once after loading weights.
    public func prepare() {
        let norms = [hiddenNorm, finalNorm] + layers.flatMap(\.norms)
        eval(norms.compactMap { $0.prepare() })
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in
            MiniMaxM3DSparkContextCache(window: configuration.slidingWindow)
        })
    }

    private func target(_ target: any LanguageModel) -> MiniMaxM3TextModel {
        if let model = target as? MiniMaxM3Model { return model.languageModel }
        if let model = target as? MiniMaxM3TextModel { return model }
        preconditionFailure("MiniMax DSpark requires a MiniMax M3 target")
    }

    private func contextCaches(_ state: MTPDrafterState) -> [MiniMaxM3DSparkContextCache]? {
        let caches = state.cache.compactMap { $0 as? MiniMaxM3DSparkContextCache }
        guard caches.count == layers.count, state.cache.count == layers.count,
            caches.allSatisfy({ $0.offset == state.nextPosition })
        else { return nil }
        return caches
    }

    /// Commit target hidden columns `[1, T, layers x hidden]` as context.
    private func appendContext(_ hidden: MLXArray, state: inout MTPDrafterState) {
        precondition(
            hidden.ndim == 3 && hidden.dim(0) == 1
                && hidden.dim(2) == configuration.hiddenSize * configuration.targetLayerIDs.count,
            "MiniMax DSpark context must be the target's captured layers")
        let committed = hidden.dim(1)
        guard committed > 0 else { return }
        guard let caches = contextCaches(state) else {
            preconditionFailure("MiniMax DSpark state does not match the drafter")
        }
        // Positions past the window never attend again; skip their projection.
        let kept = min(committed, configuration.slidingWindow)
        let start = state.nextPosition + committed - kept
        let columns = kept == committed ? hidden : hidden[0..., (committed - kept)..., 0...]
        let context = hiddenNorm(fc(columns))
        for (layer, cache) in zip(layers, caches) {
            let attention = layer.selfAttn
            var keys = attention.kProj(context).reshaped(1, kept, attention.kvHeads, attention.headDim)
            if let kNorm = attention.kNorm { keys = kNorm(keys) }
            keys = attention.rope(keys.transposed(0, 2, 1, 3), offset: start)
            let values = attention.vProj(context)
                .reshaped(1, kept, attention.kvHeads, attention.headDim).transposed(0, 2, 1, 3)
            cache.append(keys: keys, values: values, committed: committed)
        }
        state.nextPosition += committed
        state.proposalAppended = 0
        state.seedToken = nil
        state.proposalConfidence = nil
    }

    /// One block from `anchor` at the row's next position: draft tokens and
    /// their conditional acceptance probabilities, each `[1, maximumDepth]`.
    private func propose(target: MiniMaxM3TextModel, anchor: MLXArray, state: MTPDrafterState)
        -> (tokens: MLXArray, confidence: MLXArray)
    {
        guard let caches = contextCaches(state) else {
            preconditionFailure("MiniMax DSpark state does not match the drafter")
        }
        let width = configuration.blockSize
        let position = state.nextPosition
        let anchor = anchor.reshaped(1).asType(.int32)
        let inputs = concatenated(
            [
                anchor.reshaped(1, 1),
                MLXArray.full([1, width - 1], values: MLXArray(Int32(configuration.noiseTokenID))),
            ], axis: 1)
        var h = target.model.embedTokens(inputs)
        let window = configuration.slidingWindow
        for (layer, cache) in zip(layers, caches) {
            let attention = layer.selfAttn
            let x = layer.inputLayerNorm(h)
            var q = attention.qProj(x).reshaped(1, width, attention.heads, attention.headDim)
            var k = attention.kProj(x).reshaped(1, width, attention.kvHeads, attention.headDim)
            if let qNorm = attention.qNorm { q = qNorm(q) }
            if let kNorm = attention.kNorm { k = kNorm(k) }
            q = attention.rope(q.transposed(0, 2, 1, 3), offset: position)
            k = attention.rope(k.transposed(0, 2, 1, 3), offset: position)
            let v = attention.vProj(x)
                .reshaped(1, width, attention.kvHeads, attention.headDim).transposed(0, 2, 1, 3)
            let contextCount = cache.count
            let keys = cache.keys.map { concatenated([$0, k], axis: 2) } ?? k
            let values = cache.values.map { concatenated([$0, v], axis: 2) } ?? v
            // Sliding window: block query j at p + j sees context positions
            // above p + j - window; block keys are always visible.
            var mask = MLXFast.ScaledDotProductAttentionMaskMode.none
            if contextCount + width > window {
                let columns = arange(contextCount + width, dtype: .int32).reshaped(1, -1)
                let lowest = arange(width, dtype: .int32).reshaped(-1, 1)
                    + MLXArray(Int32(contextCount - window + 1))
                mask = .array(
                    (columns .>= lowest) .|| (columns .>= MLXArray(Int32(contextCount))))
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: attention.scale, mask: mask)
            h = h + attention.oProj(attended.transposed(0, 2, 1, 3).reshaped(1, width, -1))
            h = h + layer.mlp!(layer.postAttentionLayerNorm(h))
        }
        let hidden = finalNorm(h)
        let base = target.logits(hidden).asType(.float32)
        var previous = anchor
        var tokens = [MLXArray]()
        var confidence = [MLXArray]()
        for column in 0 ..< maximumDepth {
            let embedding = markovHead.embedding(previous)
            let logits = base[0..., column, 0...] + markovHead.projection(embedding).asType(.float32)
            confidence.append(confidenceHead(hidden: hidden[0..., column, 0...], markov: embedding))
            previous = argMax(logits, axis: -1).asType(.int32)
            tokens.append(previous)
        }
        return (stacked(tokens, axis: 1), stacked(confidence, axis: 1))
    }

    private func proposeInto(
        _ state: inout MTPDrafterState, target: MiniMaxM3TextModel, anchor: MLXArray
    ) -> MLXArray {
        let result = propose(target: target, anchor: anchor, state: state)
        state.seedToken = result.tokens
        state.proposalConfidence = result.confidence
        state.proposalAppended = 0
        return result.tokens
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus: MLXArray, positionDeltas: MLXArray?, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let count = promptTokens.dim(-1)
        precondition(targetHidden.dim(1) >= count)
        appendContext(
            targetHidden.dim(1) == count ? targetHidden : targetHidden[0..., ..<count, 0...],
            state: &state)
        _ = proposeInto(&state, target: self.target(target), anchor: firstBonus)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        appendContext(lastHidden, state: &state)
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
        let tokens = proposeInto(&state, target: self.target(target), anchor: lastToken)
        let depth = min(maximumDepth, max(0, blockSize - 1))
        return tokens[0..., ..<depth]
    }

    /// The committed columns are the verify input's leading
    /// `acceptedCount + 1` positions: the last committed token and the
    /// accepted drafts. The target's next token becomes the next anchor.
    public func commitDrafterState(
        target: any LanguageModel, targetHidden: MLXArray, draftTokens: MLXArray,
        acceptedCount: Int, finalToken: MLXArray, positionDeltas: MLXArray?,
        state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        precondition(acceptedCount >= 0 && acceptedCount < targetHidden.dim(1))
        appendContext(targetHidden[0..., ..<(acceptedCount + 1), 0...], state: &state)
    }

    public func advanceAndProposePacked(
        target: any LanguageModel, rows: [MTPPackedDrafterAdvanceRow], sampler: any LogitSampler
    ) throws -> MTPPackedDrafterAdvanceResult {
        for (index, row) in rows.enumerated() {
            guard row.targetHidden.ndim == 3, row.targetHidden.dim(0) == 1,
                row.targetHidden.dim(2) == configuration.hiddenSize * configuration.targetLayerIDs.count
            else {
                throw MTPPackedDrafterError.invalidTargetHidden(row: index, shape: row.targetHidden.shape)
            }
            guard row.acceptedTokens.count < row.targetHidden.dim(1) else {
                throw MTPPackedDrafterError.invalidAcceptedCount(
                    row: index, acceptedCount: row.acceptedTokens.count,
                    inputCount: row.targetHidden.dim(1))
            }
            guard contextCaches(row.state) != nil else {
                throw MTPPackedDrafterError.incompatibleState(row: index)
            }
        }
        guard !rows.isEmpty else {
            return MTPPackedDrafterAdvanceResult(
                states: [], proposals: MLXArray.zeros([0, maximumDepth], dtype: .int32))
        }
        let target = self.target(target)
        // Rows keep independent contexts and positions; each proposes alone.
        var states = [MTPDrafterState]()
        var proposals = [MLXArray]()
        for row in rows {
            var state = row.state
            state.cache = state.cache.map { $0.copy() }
            appendContext(
                row.targetHidden[0..., ..<(row.acceptedTokens.count + 1), 0...], state: &state)
            proposals.append(proposeInto(&state, target: target, anchor: MLXArray(Int32(row.finalToken))))
            states.append(state)
        }
        return MTPPackedDrafterAdvanceResult(
            states: states, proposals: concatenated(proposals, axis: 0))
    }

    /// Strips the checkpoint's `language_model.model.dspark.` prefix and the
    /// `decoder_layer` nesting; FP8 `weight_scale_inv` becomes MLX `scales`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        for (key, value) in weights {
            var name = key
            for prefix in ["language_model.model.dspark.", "model.dspark.", "dspark."]
            where name.hasPrefix(prefix) {
                name = String(name.dropFirst(prefix.count))
                break
            }
            renamed[name.replacingOccurrences(of: ".decoder_layer.", with: ".")] = value
        }
        return MiniMaxM3Weights.renameScales(renamed)
    }
}

public enum MiniMaxM3DSparkRegistration {
    /// Registers `minimax_m3_vl` drafter checkpoints whose architecture is
    /// `DSparkMiniMaxDraftModel`; any other `minimax_m3_vl` config is refused.
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "minimax_m3_vl",
            creator: { data in
                MiniMaxM3DSparkDraftModel(try MiniMaxM3DSparkConfiguration(data: data))
            })
    }
}
