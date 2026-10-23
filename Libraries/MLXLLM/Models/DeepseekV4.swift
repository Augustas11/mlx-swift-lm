//
//  DeepseekV4.swift
//  mlx-swift-lm
//
//  DeepSeek-V4 (V4-Flash) text model: mHC hyper-connections, a single
//  512-dim KV head with grouped low-rank output, sliding-window attention
//  plus compressed KV pools (ratios 4 and 128), a lightning indexer over the
//  ratio-4 pool, sqrt-softplus `noaux_tc` routing and hash routing in the
//  first layers.
//
//  Port of mlx-vlm 0.7.10 `mlx_vlm/models/deepseek_v4/language.py` (text
//  path, no vision, no MTP). Module paths follow the checkpoint
//  (`layers.N.attn.wq_a`, `layers.N.ffn.shared_experts.w1`, ...) so the
//  per-layer MXFP4/MXFP8 quantization entries of the MLX artifact apply
//  as written. This is the generic op path; fused decode kernels come later.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    public struct RopeScaling: Codable, Sendable {
        var type: String?
        var ropeType: String?
        var factor: Float?
        var originalMaxPositionEmbeddings: Int?
        var betaFast: Float?
        var betaSlow: Float?

        enum CodingKeys: String, CodingKey {
            case type
            case ropeType = "rope_type"
            case factor
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
        }
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var nSharedExperts: Int
    var nRoutedExperts: Int
    var routedScalingFactor: Float
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var numExpertsPerTok: Int
    var normTopkProb: Bool
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: RopeScaling?
    var attentionBias: Bool
    var headDim: Int
    var scoringFunc: String
    var compressRatios: [Int]
    var compressRopeTheta: Float
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float
    var numHashLayers: Int
    var swigluLimit: Float
    var slidingWindow: Int
    var oGroups: Int
    var oLoraRank: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    var indexTopk: Int
    /// HISA hierarchical indexer selection. The reference config defaults
    /// these to 64 and 16 (`deepseek_v4/config.py`), and the 0731 artifact
    /// sets neither, so the reference oracles ran with HISA on. 0 disables it.
    var indexBlock: Int
    var indexKeep: Int
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case scoringFunc = "scoring_func"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case numHashLayers = "num_hash_layers"
        case swigluLimit = "swiglu_limit"
        case slidingWindow = "sliding_window"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case indexBlock = "index_block"
        case indexKeep = "index_keep"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v4"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 129280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 43
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1024
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        compressRopeTheta = try c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160000
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        numHashLayers = try c.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 3
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        oGroups = try c.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        oLoraRank = try c.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1024
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 64
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 512
        indexBlock = try c.decodeIfPresent(Int.self, forKey: .indexBlock) ?? 64
        indexKeep = try c.decodeIfPresent(Int.self, forKey: .indexKeep) ?? 16
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // The checkpoint lists one ratio per layer plus the MTP stages; the
        // text model takes the first `num_hidden_layers` (config.py:72-80).
        if let ratios = try c.decodeIfPresent([Int].self, forKey: .compressRatios), !ratios.isEmpty {
            compressRatios = Array(ratios.prefix(hiddenLayers))
        } else {
            let n = hiddenLayers
            compressRatios =
                [0] + (0 ..< max(n - 2, 0)).map { $0 % 2 == 1 ? 4 : 128 } + (n >= 2 ? [0] : [])
        }
    }

    func ratio(layer: Int) -> Int { compressRatios[layer] }
}

extension DeepseekV4Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "DeepseekV4Configuration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "deepseek_v4: \(message)"])
        }
        guard compressRatios.count == hiddenLayers else {
            throw fail("compress_ratios has \(compressRatios.count) entries for \(hiddenLayers) layers")
        }
        let unsupported = compressRatios.filter { ![0, 4, 128].contains($0) }
        guard unsupported.isEmpty else { throw fail("unsupported compress ratios \(unsupported)") }
        guard scoringFunc == "sqrtsoftplus" || scoringFunc == "sigmoid" || scoringFunc == "softmax"
        else { throw fail("unsupported scoring_func \(scoringFunc)") }
        guard attentionHeads % oGroups == 0 else {
            throw fail("num_attention_heads \(attentionHeads) is not a multiple of o_groups \(oGroups)")
        }
        guard qkRopeHeadDim % 2 == 0, qkRopeHeadDim <= headDim, qkRopeHeadDim <= indexHeadDim else {
            throw fail("qk_rope_head_dim \(qkRopeHeadDim) exceeds head_dim or index_head_dim")
        }
        guard numExpertsPerTok > 0, numExpertsPerTok <= nRoutedExperts, hcMult > 0,
            slidingWindow > 0, indexTopk > 0, indexBlock >= 0, indexKeep >= 0
        else { throw fail("invalid expert, hc, window or indexer sizes") }
        if let scaling = ropeScaling {
            let kind = scaling.type ?? scaling.ropeType ?? "default"
            guard ["yarn", "deepseek_yarn", "default"].contains(kind) else {
                throw fail("unsupported rope_scaling type \(kind)")
            }
            if kind != "default" {
                guard scaling.factor != nil, scaling.originalMaxPositionEmbeddings != nil else {
                    throw fail("yarn rope_scaling needs factor and original_max_position_embeddings")
                }
            }
        }
    }
}

// MARK: - Helpers

/// The most negative finite value of `dtype` (`mx.finfo(dtype).min`), so a
/// masked score stays finite in that dtype.
private func deepseekV4FinfoMin(_ dtype: DType) -> Float {
    switch dtype {
    case .bfloat16: Float(bitPattern: 0xFF7F_0000)
    case .float16: -65504
    default: -Float.greatestFiniteMagnitude
    }
}

private func deepseekV4MaskScores(_ scores: MLXArray, _ mask: MLXArray?) -> MLXArray {
    guard let mask else { return scores }
    return MLX.where(mask, scores, MLXArray(deepseekV4FinfoMin(scores.dtype)).asType(scores.dtype))
}

private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let quantized as QuantizedMultiLinear: quantized(x)
    case let dense as MultiLinear: dense(x)
    default: fatalError("Expected MultiLinear or QuantizedMultiLinear, got \(type(of: module))")
    }
}

// MARK: - RoPE

/// DeepSeek-V4 RoPE (`language.py:258-339`): traditional pairs on the last
/// `dims` channels only (the leading pairs get infinite wavelength), optional
/// YaRN frequency interpolation without an mscale term, an inverse mode for
/// the output, and a `freqScale` for compressor pools (positions are divided
/// by it, frequencies multiplied).
final class DeepseekV4RoPE {
    let dims: Int
    let freqScale: Int
    private let freqs: MLXArray
    private var cache = [Int: MLXArray]()
    private let lock = NSLock()

    init(dims: Int, base: Float, scaling: DeepseekV4Configuration.RopeScaling?, freqScale: Int = 1) {
        self.dims = dims
        self.freqScale = freqScale
        var invFreq =
            1.0 / pow(MLXArray(base), MLXArray.arange(0, dims, step: 2, dtype: .float32) / Float(dims))
        let kind = scaling?.type ?? scaling?.ropeType
        if let scaling, kind == "yarn" || kind == "deepseek_yarn",
            let factor = scaling.factor, let original = scaling.originalMaxPositionEmbeddings
        {
            let betaFast = Double(scaling.betaFast ?? 32)
            let betaSlow = Double(scaling.betaSlow ?? 1)
            func correctionDim(_ rotations: Double) -> Double {
                Double(dims) * Foundation.log(Double(original) / (rotations * 2 * Double.pi))
                    / (2 * Foundation.log(Double(base)))
            }
            let low = max(Foundation.floor(correctionDim(betaFast)), 0)
            var high = min(Foundation.ceil(correctionDim(betaSlow)), Double(dims - 1))
            if low == high { high += 0.001 }
            let ramp =
                (MLXArray.arange(dims / 2, dtype: .float32) - Float(low)) / Float(high - low)
            let smooth = 1 - clip(ramp, min: 0, max: 1)
            invFreq = invFreq / factor * (1 - smooth) + invFreq * smooth
        }
        self.freqs = 1.0 / invFreq
    }

    private func frequencies(headDim: Int, inverse: Bool) -> MLXArray {
        let key = inverse ? -headDim : headDim
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache[key] { return cached }
        var f = freqs
        if freqScale != 1 { f = f / Float(freqScale) }
        if inverse { f = -f }
        let nopePairs = (headDim - dims) / 2
        if nopePairs > 0 {
            f = concatenated([MLXArray.full([nopePairs], values: MLXArray(Float.infinity)), f])
        }
        cache[key] = f
        return f
    }

    /// Rotate `[..., T, headDim]` at positions `offset ..< offset + T`.
    func callAsFunction(_ x: MLXArray, offset: Int, inverse: Bool = false) -> MLXArray {
        let headDim = x.dim(-1)
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: true, base: nil, scale: 1,
            offset: freqScale != 1 ? offset / freqScale : offset,
            freqs: frequencies(headDim: headDim, inverse: inverse))
    }
}

// MARK: - Feed-forward

final class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var gateProj: Linear
    @ModuleInfo(key: "w2") var downProj: Linear
    @ModuleInfo(key: "w3") var upProj: Linear
    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self.activation = makeLimitedSwiGLU(limit: swigluLimit > 0 ? swigluLimit : nil)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x), upProj(x)))
    }
}

/// Router (`MoEGate`, `language.py:476-552`): `sqrt(softplus(logits))` scores;
/// hash layers take the experts from `tid2eid[token]`, the others the top-k
/// of `scores + bias` (no group restriction). The weights are the scores at
/// the chosen experts, renormalized and scaled.
final class DeepseekV4MoEGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray?

    let topK: Int
    let scoringFunc: String
    let scaling: Float
    let normalize: Bool

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        let experts = config.nRoutedExperts
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize])
        if layerIdx < config.numHashLayers {
            _tid2eid.wrappedValue = MLXArray.zeros(
                [config.vocabularySize, config.numExpertsPerTok], dtype: .int32)
        } else {
            _bias.wrappedValue = MLXArray.zeros([experts], dtype: .float32)
        }
        self.topK = config.numExpertsPerTok
        self.scoringFunc = config.scoringFunc
        self.scaling = config.routedScalingFactor
        self.normalize = config.normTopkProb
        super.init()
    }

    var isHash: Bool { tid2eid != nil }

    private func score(_ logits: MLXArray) -> MLXArray {
        switch scoringFunc {
        case "softmax": softmax(logits, axis: -1, precise: true)
        case "sigmoid": sigmoid(logits)
        default: sqrt(logAddExp(logits, MLXArray(Float(0))))
        }
    }

    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x, weight.T).asType(.float32)
        let scores = score(logits)
        let indices: MLXArray
        if let tid2eid {
            indices = take(tid2eid, inputIDs, axis: 0).asType(.int32)
        } else {
            let biased = scores + bias!
            indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
                .asType(.int32)
        }
        var weights = takeAlong(scores, indices, axis: -1)
        if scoringFunc != "softmax", normalize {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * scaling)
    }
}

final class DeepseekV4MoE: Module {
    @ModuleInfo(key: "gate") var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _gate.wrappedValue = DeepseekV4MoEGate(config, layerIdx: layerIdx)
        let limit = config.swigluLimit > 0 ? config.swigluLimit : nil
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, activationProduct: makeLimitedSwiGLU(limit: limit))
        _sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
        super.init()
    }

    /// `SwitchGLU._combine`: routed outputs weighted in their dtype, summed, plus the shared expert.
    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, inputIDs: inputIDs)
        let routed = switchMLP(x, indices)
        let combined = (routed * expandedDimensions(weights.asType(routed.dtype), axis: -1))
            .sum(axis: -2)
        return combined + sharedExperts(x)
    }
}

// MARK: - Compressor and indexer

private let deepseekV4SimpleCompress: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { kv, gate, ape in
        let weights = softmax(gate.asType(.float32) + ape, axis: -2).asType(kv.dtype)
        return (kv * weights).sum(axis: -2)
    }

/// Ratio-4 overlapping windows (`_overlap_compress_kv`, `language.py:398-415`):
/// each window also pools the first half-channels of the previous window in
/// the same call. The first window of a call has none (zeros, gate `-inf`),
/// as in the reference, which does not carry the previous half across calls.
private let deepseekV4OverlapCompress: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile { kv, gate, ape in
        let (B, W, R, D) = (kv.dim(0), kv.dim(1), kv.dim(2), kv.dim(3))
        let gate = gate + ape.asType(gate.dtype)
        let kvParts = split(kv, parts: 2, axis: -1)
        let gateParts = split(gate, parts: 2, axis: -1)
        let kvA = concatenated(
            [MLXArray.zeros([B, 1, R, D / 2], dtype: kv.dtype), kvParts[0][0..., ..<(W - 1)]], axis: 1)
        let gateA = concatenated(
            [
                MLXArray.full([B, 1, R, D / 2], values: MLXArray(-Float.infinity), type: Float.self)
                    .asType(kv.dtype),
                gateParts[0][0..., ..<(W - 1)],
            ], axis: 1)
        let mixedKV = concatenated([kvA, kvParts[1]], axis: 2)
        let mixedGate = concatenated([gateA, gateParts[1]], axis: 2)
        let weights = softmax(mixedGate, axis: -2, precise: true)
        return (mixedKV * weights).sum(axis: -2)
    }

/// Pools every complete window of `ratio` tokens into one entry
/// (`Compressor`, `language.py:598-654`): `softmax(gate + ape)`-weighted sum
/// of the window's kv, RMSNorm, then RoPE at the window's start position.
final class DeepseekV4Compressor: Module {
    let ratio: Int
    let headDim: Int
    let overlap: Bool

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let rope: DeepseekV4RoPE

    init(_ config: DeepseekV4Configuration, ratio: Int, headDim: Int) {
        self.ratio = ratio
        self.headDim = headDim
        self.overlap = ratio == 4
        let outDim = headDim * (overlap ? 2 : 1)
        _wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        _wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        _ape.wrappedValue = MLXArray.zeros([ratio, outDim], dtype: .float32)
        _norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = DeepseekV4RoPE(
            dims: config.qkRopeHeadDim, base: config.compressRopeTheta,
            scaling: config.ropeScaling, freqScale: ratio)
        super.init()
    }

    /// The whole pool `[B, P, headDim]` after adding the windows that
    /// `x` (`[B, L, hidden]`, tokens `offset ..< offset + L`) completes.
    func callAsFunction(_ x: MLXArray, cache: DeepseekV4PoolCache, offset: Int) -> MLXArray {
        let B = x.dim(0)
        let ready = cache.accumulateWindows(kv: wkv(x), gate: wgate(x), offset: offset)
        guard ready.kv.dim(1) > 0 else {
            return cache.appendPooled(MLXArray.zeros([B, 0, headDim], dtype: x.dtype))
        }
        let windows = ready.kv.dim(1) / ratio
        let kv = ready.kv.reshaped(B, windows, ratio, -1)
        let gate = ready.gate.reshaped(B, windows, ratio, -1)
        var entries =
            overlap
            ? deepseekV4OverlapCompress(kv, gate, ape) : deepseekV4SimpleCompress(kv, gate, ape)
        entries = norm(entries)
        entries = rope(expandedDimensions(entries, axis: 1), offset: ready.base).squeezed(axis: 1)
        return cache.appendPooled(entries)
    }
}

/// Lightning indexer over the ratio-4 index pool (`Indexer`,
/// `language.py:657-785`): `sum_h relu(q_h . k) * scale * w_h(x) / sqrt(H)`,
/// then the top `index_topk` pooled entries per query. With `index_block`
/// and `index_keep` set and a pool of at least `block * keep` entries the
/// selection is HISA's two-stage block filter, as in the reference.
final class DeepseekV4Indexer: Module {
    let heads: Int
    let headDim: Int
    let topK: Int
    let scale: Float
    let block: Int
    let keep: Int

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(_ config: DeepseekV4Configuration, ratio: Int) {
        self.heads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.scale = pow(Float(config.indexHeadDim), -0.5)
        self.block = config.indexBlock
        self.keep = config.indexKeep
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _compressor.wrappedValue = DeepseekV4Compressor(config, ratio: ratio, headDim: headDim)
        super.init()
    }

    /// `[B, L, k]` pooled indices, or `nil` while the pool is empty.
    func callAsFunction(
        _ x: MLXArray, qResidual: MLXArray, rope: DeepseekV4RoPE, cache: DeepseekV4PoolCache,
        offset: Int
    ) -> MLXArray? {
        let (B, L) = (x.dim(0), x.dim(1))
        let pooled = compressor(x, cache: cache, offset: offset)
        let poolCount = pooled.dim(1)
        guard poolCount > 0 else { return nil }

        var q = wqB(qResidual).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        let k = min(topK, poolCount)
        let poolMask = cache.poolMask(queries: L, offset: offset)
        let weights = weightsProj(x).asType(.float32) * pow(Float(heads), -0.5)

        if block > 0, poolCount >= block * keep, keep * block >= k {
            if L == 1 {
                return hisaDecode(q: q, pooled: pooled, weights: weights, k: k)
            }
            let visible: MLXArray
            if let poolMask {
                visible = broadcast(
                    poolMask.asType(.int32).sum(axis: -1, keepDims: false)[.newAxis], to: [B, L])
            } else {
                visible = MLXArray.full([B, L], values: MLXArray(Int32(poolCount)))
            }
            return hisaPrefill(q: q, pooled: pooled, weights: weights, k: k, visible: visible)
        }

        var scores = matmul(
            q.asType(.float32), expandedDimensions(pooled, axis: 1).swappedAxes(-1, -2).asType(.float32))
        scores = maximum(scores, 0) * scale
        scores = (scores * expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)).sum(axis: 1)
        if let poolMask {
            scores = MLX.where(
                expandedDimensions(poolMask, axis: 0), scores,
                MLXArray(-Float.greatestFiniteMagnitude))
        }
        return argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    }

    /// `_hisa_select` (`language.py:672-710`): one query per row, every pooled
    /// entry visible. Only whole blocks are candidates.
    private func hisaDecode(q: MLXArray, pooled: MLXArray, weights: MLXArray, k: Int) -> MLXArray {
        let (B, poolCount, D) = (pooled.dim(0), pooled.dim(1), pooled.dim(2))
        let blocks = poolCount / block
        let usable = blocks * block
        let wq = expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)  // [B, H, 1, 1]
        let q32 = q.asType(.float32)

        let rep = pooled[0..., ..<usable].reshaped(B, blocks, block, D).mean(axis: 2)
        let coarse = maximum(
            matmul(q32, expandedDimensions(rep, axis: 1).swappedAxes(-1, -2).asType(.float32)), 0)
        let coarseScore = (coarse * scale * wq).sum(axis: 1)  // [B, 1, blocks]
        let kept = min(keep, blocks)
        let topBlocks = argPartition(-coarseScore, kth: kept - 1, axis: -1)[.ellipsis, ..<kept]
            .asType(.int32)

        let candidates = kept * block
        let positions = (expandedDimensions(topBlocks, axis: -1) * Int32(block)
            + MLXArray.arange(block, dtype: .int32)).reshaped(B, 1, candidates)
        let gatherIndex = broadcast(
            expandedDimensions(positions.reshaped(B, candidates), axis: -1), to: [B, candidates, D])
        let candidateKeys = takeAlong(pooled, gatherIndex, axis: 1)
        let fine = maximum(
            matmul(
                q32, expandedDimensions(candidateKeys, axis: 1).swappedAxes(-1, -2).asType(.float32)),
            0)
        let fineScore = (fine * scale * wq).sum(axis: 1)  // [B, 1, candidates]
        let selected = argPartition(-fineScore, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        return takeAlong(positions, selected, axis: -1)
    }

    /// `hisa_select` (`hisa_kernel.py:15-81`): `visible[b, l]` pooled entries
    /// are causally visible to each query; the fine stage runs in tiles of
    /// 512 queries to bound memory.
    private func hisaPrefill(
        q: MLXArray, pooled: MLXArray, weights: MLXArray, k: Int, visible: MLXArray
    ) -> MLXArray {
        let (B, L) = (q.dim(0), q.dim(2))
        let (poolCount, D) = (pooled.dim(1), pooled.dim(2))
        let blocks = poolCount / block
        let usable = blocks * block
        let q32 = q.asType(.float32)
        let pooled32 = pooled.asType(.float32)
        let masked = MLXArray(Float(-1e30))

        let wk = weights * scale  // [B, L, H]
        let wkHeads = expandedDimensions(wk.transposed(0, 2, 1), axis: -1)  // [B, H, L, 1]
        let rep = pooled32[0..., ..<usable].reshaped(B, blocks, block, D).mean(axis: 2)
        let coarse = maximum(matmul(q32, expandedDimensions(rep, axis: 1).swappedAxes(-1, -2)), 0)
        var coarseScore = (coarse * wkHeads).sum(axis: 1)  // [B, L, blocks]
        let blockStart = MLXArray.arange(blocks, dtype: .int32) * Int32(block)
        coarseScore = MLX.where(
            blockStart.reshaped(1, 1, blocks) .< expandedDimensions(visible, axis: -1), coarseScore,
            masked)
        let kept = min(keep, blocks)
        let topBlocks = argPartition(-coarseScore, kth: kept - 1, axis: -1)[.ellipsis, ..<kept]
            .asType(.int32)

        let candidates = kept * block
        let within = MLXArray.arange(block, dtype: .int32)
        let tile = 512
        var parts = [MLXArray]()
        for start in stride(from: 0, to: L, by: tile) {
            let end = min(start + tile, L)
            let n = end - start
            let positions = (expandedDimensions(topBlocks[0..., start ..< end], axis: -1)
                * Int32(block) + within).reshaped(B, n, candidates)
            let gatherIndex = broadcast(
                expandedDimensions(positions.reshaped(B, n * candidates), axis: -1),
                to: [B, n * candidates, D])
            let candidateKeys = takeAlong(pooled32, gatherIndex, axis: 1).reshaped(B, n, candidates, D)
            let qTile = q32[0..., 0..., start ..< end].transposed(0, 2, 1, 3)  // [B, n, H, D]
            let fine = maximum(matmul(qTile, candidateKeys.swappedAxes(-1, -2)), 0)  // [B, n, H, C]
            var score = (fine * expandedDimensions(wk[0..., start ..< end], axis: -1)).sum(axis: 2)
            score = MLX.where(
                positions .< expandedDimensions(visible[0..., start ..< end], axis: -1), score, masked)
            if tile < L { eval(score) }
            parts.append(score)
        }
        let fineScore = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        let selected = argPartition(-fineScore, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let positions = (expandedDimensions(topBlocks, axis: -1) * Int32(block) + within)
            .reshaped(B, L, candidates)
        return takeAlong(positions, selected, axis: -1)
    }
}

// MARK: - Attention

/// One V4 attention layer (`LocalAttention` / `CompressedAttention` /
/// `SparseCompressedAttention`, `language.py:788-1030`). 64 query heads of
/// 512 attend to one KV head of 512 (keys and values are the same tensor):
/// the last 128 tokens always, plus the compressed pool (ratio 128: every
/// entry; ratio 4: every entry up to `index_topk`, then the indexer's top-k).
/// Per-head sinks join the softmax. The output is inverse-rotated (V carries
/// the rotated channels) and projected through 8 low-rank groups.
final class DeepseekV4Attention: Module {
    let ratio: Int
    let heads: Int
    let headDim: Int
    let groups: Int
    let scale: Float
    let normEps: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_a") var woA: Module
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?
    let rope: DeepseekV4RoPE

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        self.ratio = config.ratio(layer: layerIdx)
        self.heads = config.attentionHeads
        self.headDim = config.headDim
        self.groups = config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.normEps = config.rmsNormEps

        _wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kvNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _woA.wrappedValue = MultiLinear(
            inputDims: heads * headDim / groups, outputDims: config.oLoraRank, numHeads: groups)
        _woB.wrappedValue = Linear(
            groups * config.oLoraRank, config.hiddenSize, bias: config.attentionBias)
        _attnSink.wrappedValue = MLXArray.zeros([heads], dtype: .float32)
        if ratio > 0 {
            _compressor.wrappedValue = DeepseekV4Compressor(config, ratio: ratio, headDim: headDim)
        }
        if ratio == 4 {
            _indexer.wrappedValue = DeepseekV4Indexer(config, ratio: ratio)
        }
        // Local layers rotate with the plain base; compressed layers (and
        // their indexer queries) with the YaRN-scaled compress base.
        self.rope =
            ratio == 0
            ? DeepseekV4RoPE(dims: config.qkRopeHeadDim, base: config.ropeTheta, scaling: nil)
            : DeepseekV4RoPE(
                dims: config.qkRopeHeadDim, base: config.compressRopeTheta,
                scaling: config.ropeScaling)
        super.init()
    }

    /// `x` is `[B, L, hidden]` (normalized); `mask` the `[L, S]` local
    /// sliding-window mask for multi-token calls (`nil` for one token).
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: DeepseekV4LayerCache) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let offset = cache.local.offset

        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual).reshaped(B, L, heads, headDim)
        q = MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: normEps)
        q = rope(q.transposed(0, 2, 1, 3), offset: offset)
        var kv = kvNorm(wkv(x)).reshaped(B, 1, L, headDim)
        kv = rope(kv, offset: offset)

        var out = attend(x, q: q, kv: kv, qResidual: qResidual, mask: mask, cache: cache, offset: offset)
        out = rope(out, offset: offset, inverse: true)

        out = out.reshaped(B, groups, heads / groups, L, headDim).transposed(0, 1, 3, 2, 4)
            .reshaped(B, groups, L, heads / groups * headDim)
        out = callMultiLinear(woA, out)  // [B, G, L, R]
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return woB(out)
    }

    private func attend(
        _ x: MLXArray, q: MLXArray, kv: MLXArray, qResidual: MLXArray, mask: MLXArray?,
        cache: DeepseekV4LayerCache, offset: Int
    ) -> MLXArray {
        let L = x.dim(1)
        let (local, _) = cache.local.update(keys: kv, values: kv)
        let localMask = alignLocalMask(mask, length: local.dim(2))
        let sinks = attnSink.asType(q.dtype)

        guard let compressor, let poolCache = cache.pool else {
            return sdpa(q, local, mask: localMask, sinks: sinks)
        }
        let pooled = compressor(x, cache: poolCache, offset: offset)
        let poolMask = poolCache.poolMask(queries: L, offset: offset)
        let topk: MLXArray?
        if let indexer, let indexCache = cache.indexPool {
            topk = indexer(x, qResidual: qResidual, rope: rope, cache: indexCache, offset: offset)
        } else {
            topk = nil
        }

        if pooled.dim(1) == 0 {
            return sdpa(q, local, mask: localMask, sinks: sinks)
        }
        if let topk, let indexer, pooled.dim(1) > indexer.topK {
            var sparseMask: MLXArray?
            if let poolMask {
                sparseMask = expandedDimensions(
                    takeAlong(expandedDimensions(poolMask, axis: 0), topk, axis: 2), axis: 1)
            }
            return sparsePooledAttention(
                q: q, local: local, pooled: pooled, topk: topk, localMask: localMask,
                pooledMask: sparseMask, sinks: sinks)
        }
        let full = concatenated([local, expandedDimensions(pooled, axis: 1)], axis: 2)
        return sdpa(q, full, mask: extendMask(localMask, poolMask, length: full.dim(2)), sinks: sinks)
    }

    private func sdpa(_ q: MLXArray, _ kv: MLXArray, mask: MLXArray?, sinks: MLXArray) -> MLXArray {
        if let mask {
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv, scale: scale, mask: .array(mask), sinks: sinks)
        }
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv, scale: scale, mask: .none, sinks: sinks)
    }

    /// `_align_local_mask`: the mask's key axis to the local window's length.
    private func alignLocalMask(_ mask: MLXArray?, length: Int) -> MLXArray? {
        guard let mask else { return nil }
        let current = mask.dim(-1)
        if current == length { return mask }
        if current > length { return mask[.ellipsis, (current - length)...] }
        var padShape = mask.shape
        padShape[padShape.count - 1] = length - current
        return concatenated([MLXArray.ones(padShape, dtype: .bool), mask], axis: -1)
    }

    /// `_extend_mask`: append the pool's `[L, P]` visibility (all visible when `nil`).
    private func extendMask(_ mask: MLXArray?, _ poolMask: MLXArray?, length: Int) -> MLXArray? {
        guard let mask else { return nil }
        let width = length - mask.dim(-1)
        let pool = poolMask ?? MLXArray.ones([mask.dim(-2), width], dtype: .bool)
        return concatenated([mask, pool], axis: -1)
    }

    /// `_sparse_pooled_attention` (`language.py:438-473`): one softmax over the
    /// local window, the gathered top-k pooled entries and the sink.
    private func sparsePooledAttention(
        q: MLXArray, local: MLXArray, pooled: MLXArray, topk: MLXArray, localMask: MLXArray?,
        pooledMask: MLXArray?, sinks: MLXArray
    ) -> MLXArray {
        let (B, poolCount, D) = (pooled.dim(0), pooled.dim(1), pooled.dim(2))
        let (L, k) = (topk.dim(1), topk.dim(2))
        let gathered: MLXArray
        if B == 1 {
            gathered = take(pooled, topk.reshaped(-1), axis: 1).reshaped(B, L, k, D)
        } else {
            var rows = MLX.where(topk .< 0, topk + Int32(poolCount), topk)
            rows = rows + (MLXArray.arange(B, dtype: topk.dtype) * Int32(poolCount)).reshaped(B, 1, 1)
            gathered = take(pooled.reshaped(B * poolCount, D), rows.reshaped(-1), axis: 0)
                .reshaped(B, L, k, D)
        }

        let qScaled = q * scale
        let localScores = deepseekV4MaskScores(matmul(qScaled, local.swappedAxes(-1, -2)), localMask)
        var normalizer = logSumExp(localScores, axis: -1, keepDims: true)
        let qByToken = qScaled.transposed(0, 2, 1, 3)
        let pooledScores = deepseekV4MaskScores(
            matmul(qByToken, gathered.swappedAxes(-1, -2)).transposed(0, 2, 1, 3), pooledMask)
        normalizer = logAddExp(normalizer, logSumExp(pooledScores, axis: -1, keepDims: true))
        normalizer = logAddExp(normalizer, sinks.reshaped(1, -1, 1, 1))
        let localWeights = exp(localScores - normalizer)
        let pooledWeights = exp(pooledScores - normalizer)

        let out = matmul(localWeights, local)
            + matmul(pooledWeights.transposed(0, 2, 1, 3), gathered).transposed(0, 2, 1, 3)
        return out.asType(q.dtype)
    }
}

// MARK: - Hyper-connection head

/// `HyperHead` (`hyper_connection.py:437-456`): a learned sigmoid-weighted
/// collapse of the `hc_mult` streams before the final norm.
final class DeepseekV4HyperHead: Module {
    let normEps: Float
    let hcEps: Float
    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: DeepseekV4Configuration) {
        self.normEps = config.rmsNormEps
        self.hcEps = config.hcEps
        _fn.wrappedValue = MLXArray.zeros(
            [config.hcMult, config.hcMult * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([config.hcMult], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([1], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = x.asType(.float32)
        let z = MLXFast.rmsNorm(y.reshaped(B, L, H * D), weight: MLXArray.mlxNone, eps: normEps)
        let pre = sigmoid(matmul(z, fn.T) * scale + base) + hcEps
        return (expandedDimensions(pre, axis: -1) * y).sum(axis: 2).asType(x.dtype)
    }
}

// MARK: - Decoder

final class DeepseekV4Block: Module {
    @ModuleInfo(key: "attn") var attn: DeepseekV4Attention
    @ModuleInfo(key: "ffn") var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: GLM5NextHyperConnection

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _attn.wrappedValue = DeepseekV4Attention(config, layerIdx: layerIdx)
        _ffn.wrappedValue = DeepseekV4MoE(config, layerIdx: layerIdx)
        _attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        func hc() -> GLM5NextHyperConnection {
            GLM5NextHyperConnection(
                hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
                hcEps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
        }
        _attnHC.wrappedValue = hc()
        _ffnHC.wrappedValue = hc()
        super.init()
    }

    /// `h` is `[B, L, hc, hidden]`.
    func callAsFunction(
        _ h: MLXArray, mask: MLXArray?, cache: DeepseekV4LayerCache, inputIDs: MLXArray
    ) -> MLXArray {
        let (attnInput, attnPost, attnComb) = attnHC.normalizedBranchInput(h, norm: attnNorm)
        let h = glm5NextHCExpand(attn(attnInput, mask: mask, cache: cache), h, attnPost, attnComb)
        let (ffnInput, ffnPost, ffnComb) = ffnHC.normalizedBranchInput(h, norm: ffnNorm)
        return glm5NextHCExpand(ffn(ffnInput, inputIDs: inputIDs), h, ffnPost, ffnComb)
    }
}

/// A layer's caches: the 128-token window, plus the attention pool on
/// compressed layers and the indexer pool on ratio-4 layers
/// (`make_cache`, `language.py:1274-1295`).
struct DeepseekV4LayerCache {
    let local: RotatingKVCache
    let pool: DeepseekV4PoolCache?
    let indexPool: DeepseekV4PoolCache?

    init(_ cache: KVCache, ratio: Int) {
        if ratio > 0, let list = cache as? CacheList {
            local = list[0] as! RotatingKVCache
            pool = (list[1] as! DeepseekV4PoolCache)
            indexPool = ratio == 4 ? (list[2] as! DeepseekV4PoolCache) : nil
        } else {
            local = cache as! RotatingKVCache
            pool = nil
            indexPool = nil
        }
    }
}

public final class DeepseekV4Model: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: DeepseekV4Configuration

    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "head") var head: Linear

    public init(_ configuration: DeepseekV4Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _embed.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            DeepseekV4Block(configuration, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _hcHead.wrappedValue = DeepseekV4HyperHead(configuration)
        _head.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    /// Logits `[B, L, vocab]` for token ids `[B, L]`. Without a cache the
    /// call runs on fresh caches (the reference's no-cache pool path skips
    /// the pool's causal mask, so it is not reproduced).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let caches = try! cache ?? newCache(parameters: nil)
        let layerCaches = zip(caches, layers).map {
            DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
        }
        let L = inputs.dim(1)

        var h = embed(inputs)
        h = repeated(expandedDimensions(h, axis: 2), count: configuration.hcMult, axis: 2)

        // One local mask for every layer, from layer 0's window before any update.
        var mask: MLXArray?
        if L > 1,
            case .array(let array) = layerCaches[0].local.makeMask(
                n: L, windowSize: configuration.slidingWindow, returnArray: true)
        {
            mask = array
        }

        for (layer, layerCache) in zip(layers, layerCaches) {
            h = layer(h, mask: mask, cache: layerCache, inputIDs: inputs)
        }
        return head(norm(hcHead(h)))
    }

    /// The reference prefill step (mlx-vlm `generate_step` default).
    public static let prefillStepSize = 2048

    /// Prefill as mlx-vlm `generate_step` does: chunks of
    /// `min(prefillStepSize, remaining - 1)` while more than one prompt token
    /// is left, each followed by a cache eval, then the last token alone in
    /// the iterator's first forward. The ratio-4 overlap pools borrow the
    /// previous window only within one call, so the pools depend on the chunk
    /// plan; the model fixes the plan the parity oracles were made with and
    /// ignores the caller's step size and chunking.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let y = input.text
        let total = y.tokens.size
        var processed = 0
        while total - processed > 1 {
            try Task.checkCancellation()
            let n = min(Self.prefillStepSize, total - processed - 1)
            autoreleasepool {
                _ = self(y[.newAxis, processed ..< processed + n].tokens, cache: cache)
                eval(cache)
            }
            processed += n
            prefill.progress?(processed, total)
        }
        return .tokens(y[processed...])
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        layers.map { layer -> KVCache in
            let window = RotatingKVCache(maxSize: configuration.slidingWindow)
            switch layer.attn.ratio {
            case 0: return window
            case 4:
                return CacheList(
                    window, DeepseekV4PoolCache(ratio: 4), DeepseekV4PoolCache(ratio: 4))
            default:
                return CacheList(window, DeepseekV4PoolCache(ratio: layer.attn.ratio))
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        DeepseekV4Weights.sanitize(weights, configuration: configuration)
    }

    public var loraLayers: [Module] { layers }
}

// MARK: - Weight sanitization

enum DeepseekV4Weights {
    /// Map the MLX artifact (`TensorFold/DeepSeek-V4-Flash-0731-MXFP4-MLX`
    /// layout, `language.py:1297-1409`) to module paths: drop `mtp.*` and
    /// layers past `num_hidden_layers`, stack per-expert weights into
    /// `switch_mlp`, group `wo_a`, move hyper-connection tensors under
    /// `attn_hc`/`ffn_hc`/`hc_head`, and make `tid2eid` int32.
    static func sanitize(
        _ weights: [String: MLXArray], configuration: DeepseekV4Configuration
    ) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        var experts = [String: [Int: MLXArray]]()
        let projections = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

        for (key, value) in weights {
            if key.hasPrefix("mtp.") { continue }
            let parts = key.split(separator: ".").map(String.init)
            if parts.count > 1, parts[0] == "layers", let layer = Int(parts[1]),
                layer >= configuration.hiddenLayers
            {
                continue
            }

            // layers.N.ffn.experts.E.wK.suffix
            if parts.count == 7, parts[0] == "layers", parts[2] == "ffn", parts[3] == "experts",
                let expert = Int(parts[4]), let projection = projections[parts[5]]
            {
                let stacked = "layers.\(parts[1]).ffn.switch_mlp.\(projection).\(parts[6])"
                experts[stacked, default: [:]][expert] = value
                continue
            }

            var mapped = key
            switch key {
            case "hc_head_fn": mapped = "hc_head.fn"
            case "hc_head_base": mapped = "hc_head.base"
            case "hc_head_scale": mapped = "hc_head.scale"
            default:
                for branch in ["attn", "ffn"] {
                    for parameter in ["fn", "base", "scale"] {
                        mapped = mapped.replacingOccurrences(
                            of: ".hc_\(branch)_\(parameter)", with: ".\(branch)_hc.\(parameter)")
                    }
                }
            }
            var tensor = value
            if mapped.hasSuffix(".tid2eid") {
                tensor = tensor.asType(.int32)
            }
            if mapped.contains(".attn.wo_a."), tensor.ndim == 2 {
                tensor = tensor.reshaped(configuration.oGroups, configuration.oLoraRank, -1)
            }
            result[mapped] = tensor
        }

        for (key, byExpert) in experts {
            guard byExpert.count == configuration.nRoutedExperts else {
                // Leave the gap for update(verify:) to report.
                for (expert, value) in byExpert { result["\(key).\(expert)"] = value }
                continue
            }
            result[key] = stacked((0 ..< configuration.nRoutedExperts).map { byExpert[$0]! })
        }
        return result
    }
}
