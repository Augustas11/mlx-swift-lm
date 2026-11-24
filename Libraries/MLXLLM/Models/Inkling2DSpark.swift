// Inkling 2 Mini's bundled DSpark draft head (`mtp.safetensors`, `mtp_config`).
//
// Ported from oMLX `omlx/patches/inkling2/dspark.py` (jundot/omlx v0.8.3, e72868da; Apache-2.0),
// with the cycle structure of megakernel-zoo `engines/inkling-2-mini_m5max/offline-c/engine/
// mk_draft.py` (dangxingyu/megakernel-zoo 95e315c8; Apache-2.0). Both licences permit this
// derivative; see each repository's LICENSE.
//
// The draft proposes, the target decides (SPEC-048 R005): every proposal is verified by the
// target's own forward, so nothing here can change an emitted token, only how many are accepted.
//
// Shape of one draft (block size 8 for Mini):
// - Context: each committed position contributes the target taps of ``targetLayerIds`` (the state
//   entering visits 5, 23 and 41, and the final-norm output). Draft layer `l` mixes them with
//   `softmax(fuse_weight)[l]`, RMS-normalises with its `dflare_norm`, and projects the mix to
//   context keys (`k_proj_ctx`, `k_norm_ctx`, RoPE at the position) and values (`v_proj_ctx`).
//   Every layer attends to the newest `sliding_window_size - 1` context positions (oMLX).
// - Block: the anchor (last committed token, target embedding + embed norm) and
//   `block_size - 1` copies of `mask_embed`, at positions `anchor ..< anchor + block_size`. The
//   block attends to the context and to itself in both directions.
// - Proposals: mask row `i` gives base logits through the target's LM head; a low-rank RNN
//   Markov head adds a per-step vocabulary bias that conditions draft `i` on draft `i - 1`
//   (`lm_head . hidden_out` folded once at load). Greedy argmax.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private final class Inkling2DSparkAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProjection: Linear
    @ModuleInfo(key: "k_proj") var kProjection: Linear
    @ModuleInfo(key: "v_proj") var vProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "k_proj_ctx") var contextKeyProjection: Linear
    @ModuleInfo(key: "v_proj_ctx") var contextValueProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var contextKeyNorm: RMSNorm

    let rope: RoPE

    init(_ c: Inkling2DSparkConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        let h = c.hiddenSize
        _qProjection.wrappedValue = Linear(h, heads * headDim, bias: false)
        _kProjection.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _vProjection.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _outputProjection.wrappedValue = Linear(heads * headDim, h, bias: false)
        _contextKeyProjection.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _contextValueProjection.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _contextKeyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        // nn.RoPE(head_dim, traditional=False): rotate halves.
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    /// Context keys and values of `mixed` `[B, n, H]` at positions `offset ..< offset + n`:
    /// `[B, kvHeads, n, headDim]` each.
    func context(_ mixed: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let (b, n) = (mixed.dim(0), mixed.dim(1))
        var keys = contextKeyNorm(contextKeyProjection(mixed).reshaped(b, n, kvHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)
        let values = contextValueProjection(mixed).reshaped(b, n, kvHeads, headDim)
            .transposed(0, 2, 1, 3)
        return (keys, values)
    }

    /// The block `[B, N, H]` at positions `offset ..< offset + N`, attending to `context` and to
    /// itself without a mask.
    func callAsFunction(
        _ x: MLXArray, offset: Int, context: (keys: MLXArray, values: MLXArray)?
    ) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        var q = queryNorm(qProjection(x).reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3)
        var k = keyNorm(kProjection(x).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProjection(x).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        if let context {
            k = concatenated([context.keys.asType(k.dtype), k], axis: 2)
            v = concatenated([context.values.asType(v.dtype), v], axis: 2)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return outputProjection(out.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

private final class Inkling2DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    /// Normalises this layer's mix of the target taps before the context projections.
    @ModuleInfo(key: "dflare_norm") var tapNorm: RMSNorm

    init(_ c: Inkling2DSparkConfiguration) {
        _attention.wrappedValue = Inkling2DSparkAttention(c)
        _mlp.wrappedValue = Inkling2DSparkMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _tapNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, offset: Int, context: (keys: MLXArray, values: MLXArray)?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), offset: offset, context: context)
        return h + mlp(postAttentionNorm(h))
    }
}

private final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousToken: Embedding
    @ModuleInfo(key: "rnn_proj") var recurrence: Linear
    @ModuleInfo(key: "hidden_out") var hiddenOut: Linear
    /// Normalises the mix of the last committed position's taps the recurrence reads every step.
    @ModuleInfo(key: "dflare_norm") var tapNorm: RMSNorm

    init(_ c: Inkling2DSparkConfiguration, vocabularySize: Int) {
        let (r, h) = (c.markovRank, c.hiddenSize)
        _previousToken.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: r)
        _recurrence.wrappedValue = Linear(2 * r + 2 * h, 3 * r, bias: true)
        _hiddenOut.wrappedValue = Linear(r, h, bias: true)
        _tapNorm.wrappedValue = RMSNorm(dimensions: h, eps: c.rmsNormEps)
        super.init()
    }
}

/// `lm_head . hidden_out` folded at load, outside the module tree so it is neither a checkpoint
/// key nor re-quantized: a `[vocabulary, rank]` 8-bit product with the folded bias.
private final class Inkling2DSparkFold: @unchecked Sendable {
    var projection: QuantizedLinear?
    /// `softmax(fuse_weight)` rounded to bf16 (as the reference binds it), in float32: one row per
    /// draft layer and a last row for the Markov feature.
    var alpha: MLXArray?
}

public enum Inkling2DSparkError: LocalizedError, Equatable {
    case missingDraftConfig
    case missingWeightFile(String)
    case incompatibleTarget(String)

    public var errorDescription: String? {
        switch self {
        case .missingDraftConfig: "Inkling 2 DSpark: config.json has no usable mtp_config"
        case .missingWeightFile(let name): "Inkling 2 DSpark: \(name) is missing"
        case .incompatibleTarget(let why): "Inkling 2 DSpark: incompatible target (\(why))"
        }
    }
}

/// The bundled Inkling 2 DSpark drafter. Stateless between streams: each stream's context K/V
/// lives in its ``MTPDrafterState`` (one `KVCacheSimple` per draft layer, at most
/// `sliding_window_size - 1 + compactionSlack` positions), and the mixed taps of its last
/// committed position in `seedHidden`.
public final class Inkling2DSparkDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Inkling2Configuration
    public let draft: Inkling2DSparkConfiguration

    /// Mask rows proposed per block. The trained block always runs whole; shorter proposals
    /// take its first rows.
    public var maximumBlockSize: Int? { draft.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { draft.targetLayerIds }

    @ModuleInfo(key: "layers") fileprivate var layers: [Inkling2DSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: Inkling2DSparkMarkovHead
    @ParameterInfo(key: "mask_embed") var maskEmbed: MLXArray
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray

    private let fold = Inkling2DSparkFold()

    /// Context positions kept beyond the window before the arrays are cut back (one copy per
    /// `compactionSlack` committed positions instead of one per round).
    static let compactionSlack = 256

    public init(_ configuration: Inkling2Configuration) throws {
        guard let draft = configuration.dspark else { throw Inkling2DSparkError.missingDraftConfig }
        self.configuration = configuration
        self.draft = draft
        _layers.wrappedValue = (0 ..< draft.hiddenLayers).map { _ in Inkling2DSparkLayer(draft) }
        _norm.wrappedValue = RMSNorm(dimensions: draft.hiddenSize, eps: draft.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(
            draft, vocabularySize: configuration.text.vocabularySize)
        _maskEmbed.wrappedValue = MLXArray.zeros([draft.hiddenSize])
        _fuseWeight.wrappedValue = MLXArray.zeros([draft.hiddenLayers + 1, draft.targetLayerIds.count])
        super.init()
    }

    // MARK: Loading

    public static let embeddedWeightFileName = "mtp.safetensors"

    /// Loads the draft a checkpoint bundles beside its weights and binds it to `target`: the
    /// target's `config.json` `mtp_config`, the tensors of `mtp.safetensors` (exactly the
    /// draft's keys), the draft layers' projections as 8-bit affine (group 64, as oMLX: they
    /// only propose), and the folded Markov projection.
    public static func loadEmbedded(
        targetDirectory: URL, target: Inkling2Model
    ) throws -> Inkling2DSparkDraftModel {
        let model = try Inkling2DSparkDraftModel(target.configuration)
        let url = targetDirectory.appendingPathComponent(embeddedWeightFileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Inkling2DSparkError.missingWeightFile(embeddedWeightFileName)
        }
        let weights = try loadArrays(url: url)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        quantize(model: model) { path, module in
            path.hasPrefix("layers.") && module is Linear ? (64, 8, .affine) : nil
        }
        try model.bind(target)
        eval(model)
        return model
    }

    /// ``loadEmbedded(targetDirectory:target:)`` wrapped for the shared drafter surface.
    public static func loadEmbeddedContainer(
        targetDirectory: URL, target: Inkling2Model
    ) throws -> MTPDrafterContainer {
        let model = try loadEmbedded(targetDirectory: targetDirectory, target: target)
        return MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(directory: targetDirectory), model: model))
    }

    /// Folds the target LM head into the Markov latent projection: `W_head . W_out` `[V, rank]`
    /// and `W_head . b_out` `[V]`, in float32 over 16,384-row slices, rounded to bf16 (oMLX
    /// `Inkling2MarkovHead.fold`), then stored 8-bit (megakernel-zoo `MK_DRAFT8`).
    func bind(_ target: Inkling2Model) throws {
        try validateCompatibility(with: target)
        let head: MLXArray
        if let q = target.lmHead as? QuantizedLinear {
            head = dequantized(
                q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize,
                bits: q.bits, mode: q.mode)
        } else {
            head = target.lmHead.weight
        }
        let out = markovHead.hiddenOut.weight.asType(.float32)  // [H, rank]
        guard let outBias = markovHead.hiddenOut.bias else {
            throw Inkling2DSparkError.incompatibleTarget("hidden_out has no bias")
        }
        let bias = outBias.asType(.float32)
        var rows = [MLXArray]()
        var biases = [MLXArray]()
        var start = 0
        while start < head.dim(0) {
            let slice = head[start ..< min(start + 16_384, head.dim(0))].asType(.float32)
            rows.append(matmul(slice, out).asType(.bfloat16))
            biases.append(matmul(slice, bias).asType(.bfloat16))
            eval(rows.last!, biases.last!)
            start += 16_384
        }
        fold.alpha = softmax(fuseWeight.asType(.float32), axis: -1).asType(.bfloat16)
            .asType(.float32)
        fold.projection = QuantizedLinear(
            weight: concatenated(rows, axis: 0), bias: concatenated(biases, axis: 0),
            groupSize: 64, bits: 8)
        eval(fold.projection!, fold.alpha!)
    }

    // MARK: Target

    private func inkling(_ target: any LanguageModel) -> Inkling2Model {
        guard let target = target as? Inkling2Model else {
            preconditionFailure("Inkling2DSparkDraftModel requires an Inkling2Model target, got \(type(of: target))")
        }
        return target
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard let target = target as? Inkling2Model else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Inkling2DSparkDraftModel", expected: "an Inkling2Model target",
                actual: String(describing: type(of: target)))
        }
        let text = target.configuration.text
        guard draft.markovHeadType == "rnn_hidden", !draft.queryBias, !draft.outputBias,
            text.hiddenSize == draft.hiddenSize,
            text.unpaddedVocabularySize == configuration.text.unpaddedVocabularySize,
            markovHead.previousToken.weight.dim(0) == target.lmHead.weight.dim(0),
            target.schedule.count == draft.targetLayerIds.last
        else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Inkling2DSparkDraftModel",
                expected: "the Inkling 2 checkpoint this draft was bundled with",
                actual: "hidden \(text.hiddenSize), vocabulary \(text.vocabularySize), visits \(target.schedule.count)")
        }
    }

    // MARK: State

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in KVCacheSimple() })
    }

    /// `rms_norm(sum_t taps_t * alpha[row, t], weight)` of taps `[B, n, T * H]`, float32 sums,
    /// in the taps' dtype (oMLX `mix_rmsnorm`).
    private func mix(_ taps: MLXArray, row: Int, norm: RMSNorm) -> MLXArray {
        let (b, n, width) = (taps.dim(0), taps.dim(1), taps.dim(2))
        let count = draft.targetLayerIds.count
        precondition(width == count * draft.hiddenSize, "Inkling 2 DSpark tap width \(width) != \(count) x \(draft.hiddenSize)")
        guard let alpha = fold.alpha else {
            preconditionFailure("Inkling 2 DSpark used before load")
        }
        let x = taps.reshaped(b, n, count, draft.hiddenSize).asType(.float32)
        let mixed = (x * alpha[row].reshaped(1, 1, count, 1)).sum(axis: 2)
        return MLXFast.rmsNorm(mixed, weight: norm.weight.asType(.float32), eps: draft.rmsNormEps)
            .asType(taps.dtype)
    }

    /// Appends the committed positions' taps `[B, n, T * H]` to every layer's context and keeps
    /// the Markov feature of the last one.
    private func append(_ newTaps: MLXArray, state: inout MTPDrafterState) {
        precondition(state.cache.count == layers.count, "Inkling 2 DSpark cache/layer mismatch")
        var taps = newTaps
        var n = taps.dim(1)
        guard n > 0 else { return }
        let keep = draft.slidingWindow - 1
        if n > keep, keep > 0 {
            // Only the newest `keep` positions are ever read: skip projecting the rest (a long
            // prompt would otherwise mix and project every row on the TTFT path).
            state.nextPosition += n - keep
            taps = taps[0..., (n - keep)..., 0...]
            n = keep
        }
        for (index, layer) in layers.enumerated() {
            let cache = state.cache[index]
            let (keys, values) = layer.attention.context(
                mix(taps, row: index, norm: layer.tapNorm), offset: state.nextPosition)
            _ = cache.update(keys: keys, values: values)
            // Positions older than the window are never read again (oMLX: every layer sees
            // the newest `sliding_window_size - 1`); cut them back once per slack.
            if cache.offset > keep + Self.compactionSlack {
                let kv = cache.state
                let from = kv[0].dim(2) - keep
                state.cache[index].state = [
                    kv[0][.ellipsis, from..., 0...], kv[1][.ellipsis, from..., 0...],
                ]
            }
        }
        state.nextPosition += n
        let total = newTaps.dim(1)
        state.seedHidden = mix(
            newTaps[0..., (total - 1)..., 0...], row: draft.hiddenLayers, norm: markovHead.tapNorm)
    }

    /// The newest `sliding_window_size - 1` context positions of layer `index`.
    private func context(_ state: MTPDrafterState, layer index: Int) -> (keys: MLXArray, values: MLXArray)? {
        let kv = state.cache[index].state
        guard kv.count == 2, kv[0].dim(2) > 0 else { return nil }
        let keep = draft.slidingWindow - 1
        let from = max(0, kv[0].dim(2) - keep)
        return (kv[0][.ellipsis, from..., 0...], kv[1][.ellipsis, from..., 0...])
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        _ = inkling(target)
        append(targetHidden, state: &state)
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens _: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        _ = inkling(target)
        // The verify input is [anchor, drafts...]; the anchor and the accepted drafts are now
        // committed positions, and the target's token after them is the next anchor.
        let committed = acceptedCount + 1
        precondition(
            targetHidden.dim(1) >= committed,
            "Inkling 2 target did not emit every committed DSpark context row")
        append(targetHidden[0..., ..<committed, 0...], state: &state)
    }

    // MARK: Proposals

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize _: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        preconditionFailure("Inkling 2 DSpark needs its stream state; use the stateful draftBlock")
    }

    /// Proposes `blockSize - 1` tokens after the anchor `lastToken`, at the state's next
    /// position. Sets `state.proposalConfidence` to each draft's own probability
    /// `[B, blockSize - 1]` (for depth policies; acceptance is target equality alone).
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        let target = inkling(target)
        let proposals = blockSize - 1
        let rows = draft.blockSize
        precondition((1 ..< rows).contains(proposals), "Inkling 2 DSpark proposes 1...\(rows - 1) tokens")
        guard let fold = fold.projection, let feature = state.seedHidden else {
            preconditionFailure("Inkling 2 DSpark used before load or before its prompt context")
        }

        let anchor = normalizedMTPColumn(lastToken).asType(.int32)  // [B, 1]
        let b = anchor.dim(0)
        let embedded = target.model.embedNorm(target.model.embed(anchor))
        var x = concatenated(
            [embedded, broadcast(maskEmbed.asType(embedded.dtype), to: [b, rows - 1, draft.hiddenSize])],
            axis: 1)
        let offset = state.nextPosition
        for (index, layer) in layers.enumerated() {
            x = layer(x, offset: offset, context: context(state, layer: index))
        }
        let hidden = norm(x)[0..., 1 ..< (1 + proposals), 0...]  // the mask rows

        let base = target.lmHead(hidden).asType(.float32)  // [B, proposals, V]
        let r = draft.markovRank
        let cap = configuration.text.finalLogitSoftcap
        let vocabulary = configuration.text.unpaddedVocabularySize
        // The recurrence runs in float32 and the fold is 8-bit (megakernel-zoo `_markov`,
        // `MK_DRAFT8`); oMLX keeps both in bf16. Either only changes proposals.
        var recurrent = MLXArray.zeros([b, r], dtype: .float32)
        var previous = anchor[0..., 0]
        var drafts = [MLXArray]()
        var probabilities = [MLXArray]()
        for step in 0 ..< proposals {
            let input = concatenated(
                [
                    recurrent.asType(hidden.dtype),
                    markovHead.previousToken(previous).asType(hidden.dtype),
                    hidden[0..., step, 0...],
                    feature[0..., 0, 0...].asType(hidden.dtype),
                ], axis: -1)
            let proj = markovHead.recurrence(input).asType(.float32)
            let gate = sigmoid(proj[0..., ..<r])
            recurrent = gate * recurrent + (1 - gate) * tanh(proj[0..., r ..< (2 * r)])
            let latent = tanh(proj[0..., (2 * r)...])
            var logits = base[0..., step, 0...] + fold(latent.asType(.bfloat16)).asType(.float32)
            if cap != 0 { logits = tanh(logits / cap) * cap }
            logits = logits[0..., ..<vocabulary]
            previous = argMax(logits, axis: -1).asType(.int32)
            drafts.append(previous.expandedDimensions(axis: 1))
            probabilities.append(
                exp(logits.max(axis: -1) - logSumExp(logits, axis: -1)).expandedDimensions(axis: 1))
        }
        state.proposalConfidence = concatenated(probabilities, axis: 1)
        return concatenated(drafts, axis: 1)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
