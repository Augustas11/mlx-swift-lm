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
    /// HISA hierarchical indexer selection (block size and blocks kept), an
    /// mlx-vlm approximation of the indexer. DeepSeek's reference selects the
    /// exact top `index_topk` of every pooled entry (`inference/model.py:
    /// 426-433`), so both default to 0 (exact flat top-k) and HISA runs only
    /// when a config sets both explicitly.
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
        indexBlock = try c.decodeIfPresent(Int.self, forKey: .indexBlock) ?? 0
        indexKeep = try c.decodeIfPresent(Int.self, forKey: .indexKeep) ?? 0
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

/// The kernel's unnormalized attention weights as the value product reads
/// them: `exp(s - max)` rounded to bf16 (`acc_s_cast`, `inference/kernel.py:
/// 305, 340-343`), the normalizer and accumulation staying float32. The
/// kernel takes its running max over 64-key blocks in its key order; this
/// takes the row max over all keys (the kernel's final max).
private func deepseekV4KernelWeights(_ weights: MLXArray) -> MLXArray {
    weights.asType(.bfloat16).asType(.float32)
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

// MARK: - Float32 linear

/// `x @ w.T` as float32 for `M` rows of float32 `x` and a bfloat16 `w`
/// `[N, K]`: each simdgroup reads one row of `w` once and accumulates every
/// row of `x` in float32 (bfloat16 widens to float32 exactly). A row's sum
/// does not depend on `M`, so batched decode rows match one-row calls.
private let deepseekV4LinearF32Kernel = MLXFast.metalKernel(
    name: "deepseek_v4_linear_f32",
    inputNames: ["x", "w"],
    outputNames: ["out"],
    source: """
        const uint n = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        if (n >= uint(N)) { return; }
        const device uint4* row = reinterpret_cast<const device uint4*>(w) + n * uint(K / 8);
        float acc[M];
        for (int m = 0; m < M; ++m) { acc[m] = 0.0f; }
        for (uint i = lane; i < uint(K / 8); i += 32) {
            const uint4 p = row[i];
            const float4 lo = float4(
                as_type<float>(p.x << 16), as_type<float>(p.x & 0xFFFF0000u),
                as_type<float>(p.y << 16), as_type<float>(p.y & 0xFFFF0000u));
            const float4 hi = float4(
                as_type<float>(p.z << 16), as_type<float>(p.z & 0xFFFF0000u),
                as_type<float>(p.w << 16), as_type<float>(p.w & 0xFFFF0000u));
            for (int m = 0; m < M; ++m) {
                const device float4* xr = reinterpret_cast<const device float4*>(x + m * K) + 2 * i;
                acc[m] += dot(xr[0], lo) + dot(xr[1], hi);
            }
        }
        for (int m = 0; m < M; ++m) {
            const float total = simd_sum(acc[m]);
            if (lane == 0) { out[m * N + n] = total; }
        }
        """)

/// `layer(x)` with float32 output, as the reference's float32 linears
/// (`F.linear(x.float(), weight.float())`) compute it, without a float32
/// copy of a bfloat16 weight: up to 8 rows on the GPU use the kernel above;
/// longer inputs (prompt chunks) and the CPU widen the weight per call. Like
/// the kernel, a decode row's result does not depend on the other rows.
func deepseekV4LinearF32(_ layer: Linear, _ x: MLXArray) -> MLXArray {
    let x32 = x.asType(.float32)
    if let quantized = layer as? QuantizedLinear { return quantized(x32) }
    let weight = layer.weight
    let (K, N) = (x.dim(-1), weight.dim(0))
    let rows = x.size / max(K, 1)
    var out: MLXArray
    if weight.dtype == .bfloat16, (1 ... 8).contains(rows), K % 8 == 0,
        (N + 7) / 8 * 256 <= Int(Int32.max), Device.defaultDevice().deviceType == .gpu
    {
        out = deepseekV4LinearF32Kernel(
            [x32.reshaped(rows, K), weight],
            template: [("M", rows), ("K", K), ("N", N)],
            grid: ((N + 7) / 8 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, N]],
            outputDTypes: [.float32]
        )[0].reshaped(Array(x.shape.dropLast()) + [N])
    } else if (2 ... 8).contains(rows), Device.defaultDevice().deviceType == .cpu {
        // CPU GEMM rounds a row differently from the one-row GEMV; decode rows
        // go one at a time so a batched row equals its serial self.
        let widened = weight.asType(.float32).T
        let flat = x32.reshaped(rows, K)
        out = concatenated((0 ..< rows).map { matmul(flat[$0 ..< ($0 + 1)], widened) }, axis: 0)
            .reshaped(Array(x.shape.dropLast()) + [N])
    } else {
        out = matmul(x32, weight.asType(.float32).T)
    }
    if let bias = layer.bias { out = out + bias.asType(.float32) }
    return out
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

    /// Each batch row starts at its own absolute token position.
    func callAsFunction(_ x: MLXArray, offsets: MLXArray, inverse: Bool = false) -> MLXArray {
        let headDim = x.dim(-1)
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: true, base: nil, scale: 1,
            offset: freqScale != 1 ? offsets.floorDivide(Int32(freqScale)) : offsets,
            freqs: frequencies(headDim: headDim, inverse: inverse))
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

    /// The SwiGLU in float32, back to the activation dtype for `w2`
    /// (`Expert.forward`, `inference/model.py:601-611`).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(
            activation(gateProj(x).asType(.float32), upProj(x).asType(.float32)).asType(x.dtype))
    }
}

/// Router (`Gate`, `inference/model.py:555-585`): `sqrt(softplus(logits))`
/// scores; hash layers take the experts from `tid2eid[token]`, the others
/// the top-k of `scores + bias` (no group restriction). The weights are the
/// scores at the chosen experts, renormalized and scaled. The logits are
/// float32 from a float32 weight (`linear(x.float(), weight.float())`;
/// `sanitize` converts the checkpoint's bf16 gate once at load).
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
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize], dtype: .float32)
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
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
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
    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _gate.wrappedValue = DeepseekV4MoEGate(config, layerIdx: layerIdx)
        let limit = config.swigluLimit > 0 ? config.swigluLimit : nil
        self.activation = makeLimitedSwiGLU(limit: limit)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, activationProduct: activation)
        _sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
        super.init()
    }

    /// `MoE.forward` (`inference/model.py:638-649`): each routed expert's
    /// SwiGLU in float32, scaled by its routing weight before `w2`
    /// (`Expert.forward`, model.py:601-611), the expert outputs summed in
    /// float32 and the shared expert added in float32. The projections go
    /// through `switch_mlp`'s gathered matmuls, sorted by expert for long
    /// inputs as `SwitchGLU` does.
    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, inputIDs: inputIDs)
        let k = indices.dim(-1)
        let sorted = indices.size >= 64
        var inputs = expandedDimensions(x, axes: [-2, -3])
        var experts = indices
        var routeWeights = expandedDimensions(weights, axes: [-1, -2])
        var inverseOrder: MLXArray?
        if sorted {
            let flat = indices.flattened()
            let order = argSort(flat)
            inverseOrder = argSort(order)
            inputs = inputs.flattened(start: 0, end: -3)[order.floorDivide(k)]
            experts = flat[order]
            routeWeights = weights.flattened()[order].reshaped(-1, 1, 1)
        }
        let gateOut = switchMLP.gateProjection(inputs, experts, sortedIndices: sorted)
        let up = switchMLP.upProjection(inputs, experts, sortedIndices: sorted)
        let hidden = (activation(gateOut.asType(.float32), up.asType(.float32)) * routeWeights)
            .asType(x.dtype)
        var routed = switchMLP.downProjection(hidden, experts, sortedIndices: sorted)
        if let inverseOrder {
            routed = scatterUnsort(x: routed, invOrder: inverseOrder, shape: indices.shape)
        }
        let combined = routed.squeezed(axis: -2).asType(.float32).sum(axis: -2)
        return (combined + sharedExperts(x).asType(.float32)).asType(x.dtype)
    }
}

// MARK: - Compressor and indexer

/// Non-overlapping windows: float32 `softmax(gate + ape)`-weighted sum.
private let deepseekV4SimpleCompress: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { kv, gate, ape in
        let weights = softmax(gate.asType(.float32) + ape.asType(.float32), axis: -2)
        return (kv.asType(.float32) * weights).sum(axis: -2)
    }

/// Ratio-4 overlapping windows (`Compressor.overlap_transform` and the decode
/// state, `inference/model.py:316-323, 353-360`): window `w` pools its own
/// second half-channels and the first half-channels of window `w - 1`. The
/// first window of a call takes its predecessor from `previous`, the state
/// carried across calls (zeros with gate `-inf` before any window, as at the
/// start of a prompt), so the pooled entries do not depend on how the tokens
/// were split into calls. `kv`/`gate` are `[B, W, ratio, D]`, `previous`
/// `[B, ratio, D / 2]` without `ape`. Returns the entries and the call's last
/// window as the next state.
private func deepseekV4OverlapCompress(
    kv: MLXArray, gate: MLXArray, ape: MLXArray, previous: (kv: MLXArray, gate: MLXArray)
) -> (entries: MLXArray, last: (kv: MLXArray, gate: MLXArray)) {
    let (B, W, R, D) = (kv.dim(0), kv.dim(1), kv.dim(2), kv.dim(3))
    let kvParts = split(kv, parts: 2, axis: -1)
    let gateParts = split(gate, parts: 2, axis: -1)
    let apeParts = split(ape, parts: 2, axis: -1)
    let kvA = concatenated(
        [previous.kv.reshaped(B, 1, R, D / 2).asType(kv.dtype), kvParts[0][0..., ..<(W - 1)]],
        axis: 1)
    let gateA = concatenated(
        [previous.gate.reshaped(B, 1, R, D / 2).asType(gate.dtype), gateParts[0][0..., ..<(W - 1)]],
        axis: 1)
    let mixedKV = concatenated([kvA, kvParts[1]], axis: 2).asType(.float32)
    let mixedGate = concatenated(
        [gateA + apeParts[0].asType(gate.dtype), gateParts[1] + apeParts[1].asType(gate.dtype)],
        axis: 2
    ).asType(.float32)
    let weights = softmax(mixedGate, axis: -2, precise: true)
    let last = (
        kv: kvParts[0][0..., W - 1].asType(.float32), gate: gateParts[0][0..., W - 1].asType(.float32)
    )
    return ((mixedKV * weights).sum(axis: -2), last)
}

/// The overlap state before a row's first window: zeros, gate `-inf`.
private func deepseekV4OverlapStart(rows: Int, ratio: Int, width: Int) -> (kv: MLXArray, gate: MLXArray) {
    (
        MLXArray.zeros([rows, ratio, width], dtype: .float32),
        MLXArray.full([rows, ratio, width], values: MLXArray(-Float.infinity))
    )
}

/// Pools every complete window of `ratio` tokens into one entry
/// (`Compressor`, `inference/model.py:286-375`): `softmax(gate + ape)`-weighted
/// sum of the window's kv, RMSNorm, then RoPE at the window's start position.
/// The projections, buffered rows, `ape`, softmax and sum are float32; the
/// entry returns to the activation dtype before the norm (`kv.to(dtype)`).
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

    /// The carried overlap state of `cache`, `[1 or B, ratio, headDim]`.
    private func overlapState(_ cache: DeepseekV4PoolCache) -> (kv: MLXArray, gate: MLXArray) {
        if let kv = cache.overlapKV, let gate = cache.overlapGate { return (kv, gate) }
        let rows = cache.bufferRows ?? 1
        return deepseekV4OverlapStart(rows: rows, ratio: ratio, width: headDim)
    }

    /// Projections and completed-window compression are shared across decode rows.
    func decodeBatch(_ x: MLXArray, caches: [DeepseekV4PoolCache], offsets: [Int]) -> [MLXArray] {
        let kv = deepseekV4LinearF32(wkv, x)
        let gate = deepseekV4LinearF32(wgate, x)
        var readyKV = [MLXArray]()
        var readyGate = [MLXArray]()
        var completed = [Int]()
        var bases = [Int32]()
        for row in caches.indices {
            let ready = caches[row].accumulateWindows(
                kv: kv[row ..< (row + 1)], gate: gate[row ..< (row + 1)], offset: offsets[row])
            if ready.kv.dim(1) > 0 {
                completed.append(row)
                bases.append(Int32(ready.base))
                readyKV.append(ready.kv)
                readyGate.append(ready.gate)
            }
        }
        if !completed.isEmpty {
            let kv = concatenated(readyKV, axis: 0).reshaped(completed.count, 1, ratio, -1)
            let gate = concatenated(readyGate, axis: 0).reshaped(completed.count, 1, ratio, -1)
            var entries: MLXArray
            if overlap {
                let previous = completed.map { overlapState(caches[$0]) }
                let compressed = deepseekV4OverlapCompress(
                    kv: kv, gate: gate, ape: ape,
                    previous: (
                        concatenated(previous.map { $0.kv }, axis: 0),
                        concatenated(previous.map { $0.gate }, axis: 0)
                    ))
                entries = compressed.entries
                for (slot, row) in completed.enumerated() {
                    caches[row].overlapKV = compressed.last.kv[slot ..< (slot + 1)]
                    caches[row].overlapGate = compressed.last.gate[slot ..< (slot + 1)]
                }
            } else {
                entries = deepseekV4SimpleCompress(kv, gate, ape)
            }
            entries = rope(
                expandedDimensions(norm(entries.asType(x.dtype)), axis: 1), offsets: MLXArray(bases))
                .squeezed(axis: 1)
            for (slot, row) in completed.enumerated() {
                _ = caches[row].appendPooled(entries[slot ..< (slot + 1)])
            }
        }
        let empty = MLXArray.zeros([1, 0, headDim], dtype: x.dtype)
        return caches.map { $0.pooled ?? empty }
    }

    /// The whole pool `[B, P, headDim]` after adding the windows that
    /// `x` (`[B, L, hidden]`, tokens `offset ..< offset + L`) completes.
    func callAsFunction(_ x: MLXArray, cache: DeepseekV4PoolCache, offset: Int) -> MLXArray {
        let B = x.dim(0)
        let ready = cache.accumulateWindows(
            kv: deepseekV4LinearF32(wkv, x), gate: deepseekV4LinearF32(wgate, x), offset: offset)
        guard ready.kv.dim(1) > 0 else {
            return cache.appendPooled(MLXArray.zeros([B, 0, headDim], dtype: x.dtype))
        }
        let windows = ready.kv.dim(1) / ratio
        let kv = ready.kv.reshaped(B, windows, ratio, -1)
        let gate = ready.gate.reshaped(B, windows, ratio, -1)
        var entries: MLXArray
        if overlap {
            let compressed = deepseekV4OverlapCompress(
                kv: kv, gate: gate, ape: ape, previous: overlapState(cache))
            entries = compressed.entries
            cache.overlapKV = compressed.last.kv
            cache.overlapGate = compressed.last.gate
        } else {
            entries = deepseekV4SimpleCompress(kv, gate, ape)
        }
        entries = norm(entries.asType(x.dtype))
        entries = rope(expandedDimensions(entries, axis: 1), offset: ready.base).squeezed(axis: 1)
        return cache.appendPooled(entries)
    }
}

/// Lightning indexer over the ratio-4 index pool (`Indexer`,
/// `language.py:657-785`): `sum_h relu(q_h . k) * scale * w_h(x) / sqrt(H)`,
/// then the exact top `min(index_topk, P)` pooled entries per query
/// (`inference/model.py:426-433`). Only a config that sets `index_block` and
/// `index_keep` swaps in HISA's two-stage block filter for pools of at least
/// `block * keep` entries.
final class DeepseekV4Indexer: Module {
    let heads: Int
    let headDim: Int
    let topK: Int
    let scale: Float
    let block: Int
    let keep: Int

    /// Bytes of float32 indexer scores one prefill tile may hold.
    var scoreWorkspaceBytes = 512 << 20

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

    /// Ragged pools share one query/weight projection and one scoring graph.
    /// Each row gets the top `min(index_topk, P)` of its own pool in
    /// descending score order (the order of the reference's `topk`); entries
    /// at ranks past a row's pool size are padding. The exact selection only:
    /// a HISA configuration does not batch (`DeepseekV4Model.batchesExactly`).
    func decodeBatch(
        _ x: MLXArray, qResidual: MLXArray, rope: DeepseekV4RoPE, pools: [MLXArray],
        offsets: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let counts = pools.map { $0.dim(1) }
        let poolCount = counts.max()!
        let k = min(topK, poolCount)
        let pooled = concatenated(pools.map { pool in
            let padding = poolCount - pool.dim(1)
            return padding == 0 ? pool : concatenated([
                pool, MLXArray.zeros([1, padding, headDim], dtype: pool.dtype),
            ], axis: 1)
        }, axis: 0)
        let q = rope(
            wqB(qResidual).reshaped(B, 1, heads, headDim).transposed(0, 2, 1, 3), offsets: offsets)
            .asType(.float32)
        let weights = weightsProj(x).asType(.float32) * pow(Float(heads), -0.5)
        let wq = expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)
        let lengths = MLXArray(counts.map(Int32.init)).reshaped(B, 1, 1)
        var scores = maximum(
            matmul(q, expandedDimensions(pooled, axis: 1).swappedAxes(-1, -2).asType(.float32)), 0)
        scores = (scores * scale * wq).sum(axis: 1)
        scores = MLX.where(
            MLXArray.arange(poolCount, dtype: .int32).reshaped(1, 1, poolCount) .< lengths,
            scores, MLXArray(-Float.infinity))
        return argSort(-scores, axis: -1)[.ellipsis, ..<k].asType(.int32)
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

        let keys = expandedDimensions(pooled, axis: 1).swappedAxes(-1, -2).asType(.float32)
        func exactTopK(_ q: MLXArray, _ weights: MLXArray, _ poolMask: MLXArray?) -> MLXArray {
            var scores = maximum(matmul(q.asType(.float32), keys), 0) * scale
            scores = (scores * expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)).sum(axis: 1)
            if let poolMask {
                scores = MLX.where(
                    expandedDimensions(poolMask, axis: 0), scores,
                    MLXArray(-Float.greatestFiniteMagnitude))
            }
            return argSort(-scores, axis: -1)[.ellipsis, ..<k].asType(.int32)
        }
        // The `[B, H, tile, P]` float32 scores stay within the workspace
        // budget however long the context grows; each tile keeps only its
        // selection.
        let tile = max(1, scoreWorkspaceBytes / max(1, B * heads * poolCount * 4))
        guard L > tile else { return exactTopK(q, weights, poolMask) }
        var parts = [MLXArray]()
        for start in stride(from: 0, to: L, by: tile) {
            let end = min(start + tile, L)
            let part = exactTopK(
                q[0..., 0..., start ..< end], weights[0..., start ..< end],
                poolMask.map { $0[start ..< end] })
            eval(part)
            parts.append(part)
        }
        return concatenated(parts, axis: 1)
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
    let window: Int

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
        self.window = config.slidingWindow

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

    /// One token per row. Projections and the attention arithmetic are
    /// batched; each row keeps its own window, pools and selection, and its
    /// key list is the one `attend` builds for it alone, so a batched row
    /// computes exactly what it computes serially.
    func decodeBatch(_ x: MLXArray, caches: [DeepseekV4LayerCache]) -> MLXArray {
        let B = x.dim(0)
        let offsets = caches.map { $0.local.offset }
        let offsetArray = MLXArray(offsets.map(Int32.init))
        let qResidual = qNorm(wqA(x))
        let q = rope(
            MLXFast.rmsNorm(
                wqB(qResidual).reshaped(B, 1, heads, headDim),
                weight: MLXArray.mlxNone, eps: normEps).transposed(0, 2, 1, 3),
            offsets: offsetArray)
        let kv = rope(kvNorm(wkv(x)).reshaped(B, 1, 1, headDim), offsets: offsetArray)
        let emptyValues = MLXArray.zeros([B, 1, 1, 0], dtype: kv.dtype)
        let windows = caches.enumerated().map { row, cache in
            Self.temporal(
                cache.local.update(
                    keys: kv[row ..< (row + 1)], values: emptyValues[row ..< (row + 1)]).0,
                cache: cache.local)
        }

        // Each row's compressed list: [C, D] keys and the count of valid entries.
        var compressed = Array(repeating: (keys: MLXArray?.none, valid: 0), count: B)
        if let compressor {
            let pools = compressor.decodeBatch(x, caches: caches.map { $0.pool! }, offsets: offsets)
            // The index pool must advance even before selection becomes necessary.
            let indexPools = indexer?.compressor.decodeBatch(
                x, caches: caches.map { $0.indexPool! }, offsets: offsets)
            if let indexer, let indexPools {
                if indexPools.contains(where: { $0.dim(1) > 0 }) {
                    let selected = indexer.decodeBatch(
                        x, qResidual: qResidual, rope: rope, pools: indexPools, offsets: offsetArray)
                    for row in 0 ..< B where pools[row].dim(1) > 0 {
                        let count = min(indexer.topK, pools[row].dim(1))
                        compressed[row] = (
                            take(pools[row][0], selected[row, 0, ..<count], axis: 0), count
                        )
                    }
                }
            } else {
                for row in 0 ..< B where pools[row].dim(1) > 0 {
                    compressed[row] = (pools[row][0], pools[row].dim(1))
                }
            }
        }
        let width = compressed.map { $0.keys?.dim(0) ?? 0 }.max()!
        var keys = [MLXArray]()
        var valid = [MLXArray]()
        for row in 0 ..< B {
            let (windowKeys, windowValid) = windowList(
                windows[row][0, 0], first: offsets[row] + 1 - windows[row].dim(2),
                positions: [offsets[row]])
            var rowKeys = [windowKeys[0]]
            var rowValid = [windowValid[0]]
            if width > 0 {
                let entries = compressed[row].keys ?? MLXArray.zeros([0, headDim], dtype: kv.dtype)
                let padding = width - entries.dim(0)
                rowKeys.append(
                    padding == 0
                        ? entries
                        : concatenated([
                            entries, MLXArray.zeros([padding, headDim], dtype: entries.dtype),
                        ]))
                rowValid.append(MLXArray.arange(width, dtype: .int32) .< Int32(compressed[row].valid))
            }
            keys.append(concatenated(rowKeys, axis: 0))
            valid.append(concatenated(rowValid, axis: 0))
        }
        let attended = referenceAttention(
            q, keys: expandedDimensions(stacked(keys, axis: 0), axis: 1),
            valid: expandedDimensions(stacked(valid, axis: 0), axis: 1)
        ).asType(q.dtype)
        var out = rope(attended, offsets: offsetArray, inverse: true)
        out = out.reshaped(B, groups, heads / groups, 1, headDim).transposed(0, 1, 3, 2, 4)
            .reshaped(B, groups, 1, heads / groups * headDim)
        out = callMultiLinear(woA, out)
        return woB(out.transposed(0, 2, 1, 3).reshaped(B, 1, -1))
    }

    /// `RotatingKVCache` returns a one-token update in storage (ring) order;
    /// the window list needs temporal order. Multi-token updates already are.
    static func temporal(_ window: MLXArray, cache: RotatingKVCache) -> MLXArray {
        let meta = cache.metaState
        guard meta.count >= 7, let idx = Int(meta[4]), Int(meta[0]) == 0 else { return window }
        let length = window.dim(2)
        if idx >= length { return window }
        if meta[6] == "true" {
            return concatenated([window[.ellipsis, idx..., 0...], window[.ellipsis, ..<idx, 0...]], axis: 2)
        }
        return window[.ellipsis, ..<idx, 0...]
    }

    /// The reference's window list (`get_window_topk_idxs`, `inference/
    /// model.py:258-270`) for queries at `positions`: `slidingWindow` slots in
    /// temporal order, the last `window` positions up to the query, or
    /// positions `0...p` followed by invalid slots early on. `local` is
    /// `[S, D]` in temporal order, `local[0]` at position `first`. Returns
    /// `[L, W, D]` keys and `[L, W]` validity.
    private func windowList(_ local: MLXArray, first: Int, positions: [Int]) -> (MLXArray, MLXArray) {
        let S = local.dim(0)
        var index = [Int32]()
        var valid = [Bool]()
        for p in positions {
            for j in 0 ..< window {
                let position = p >= window - 1 ? p - window + 1 + j : j
                let ok = position <= p && position - first >= 0 && position - first < S
                index.append(Int32(ok ? position - first : 0))
                valid.append(ok)
            }
        }
        let L = positions.count
        return (
            take(local, MLXArray(index), axis: 0).reshaped(L, window, -1),
            MLXArray(valid).reshaped(L, window)
        )
    }

    /// Bytes of gathered float32 keys one prompt tile may hold.
    var keyWorkspaceBytes = 512 << 20

    /// The reference attention for the queries at `offset ..< offset + L`:
    /// each query's key list is its window list followed by its compressed
    /// list (every visible entry of a ratio-128 pool in order; the indexer's
    /// top-k in descending score order on ratio-4 layers), as `sparse_attn`
    /// receives them.
    private func attend(
        _ x: MLXArray, q: MLXArray, kv: MLXArray, qResidual: MLXArray, mask: MLXArray?,
        cache: DeepseekV4LayerCache, offset: Int
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let (updated, _) = cache.local.update(
            keys: kv, values: MLXArray.zeros([kv.dim(0), 1, L, 0], dtype: kv.dtype))
        let local = L == 1 ? Self.temporal(updated, cache: cache.local) : updated
        let first = offset + L - local.dim(2)

        // [B, L, C] compressed indices and validity, or nil without a pool.
        var list: (indices: MLXArray, valid: MLXArray, pooled: MLXArray)?
        if let compressor, let poolCache = cache.pool {
            let pooled = compressor(x, cache: poolCache, offset: offset)
            let visible = MLXArray((offset ..< (offset + L)).map { Int32(($0 + 1) / ratio) })
                .reshaped(1, L, 1)
            if let indexer, let indexCache = cache.indexPool {
                if let topk = indexer(x, qResidual: qResidual, rope: rope, cache: indexCache, offset: offset) {
                    let topk = topk.asType(.int32)
                    list = (topk, topk .< visible, pooled)
                }
            } else if pooled.dim(1) > 0 {
                let indices = broadcast(
                    MLXArray.arange(pooled.dim(1), dtype: .int32).reshaped(1, 1, -1),
                    to: [B, L, pooled.dim(1)])
                list = (indices, indices .< visible, pooled)
            }
        }

        let K = window + (list?.indices.dim(2) ?? 0)
        let tile = max(1, min(L, keyWorkspaceBytes / max(1, B * K * headDim * 4)))
        var parts = [MLXArray]()
        for start in stride(from: 0, to: L, by: tile) {
            let end = min(start + tile, L)
            let n = end - start
            var keys = [MLXArray]()
            var valid = [MLXArray]()
            for row in 0 ..< B {
                let (windowKeys, windowValid) = windowList(
                    local[row, 0], first: first, positions: Array((offset + start) ..< (offset + end)))
                var rowKeys = [windowKeys]
                var rowValid = [windowValid]
                if let list {
                    let indices = list.indices[row, start ..< end]
                    let safe = maximum(indices, Int32(0))
                    rowKeys.append(
                        take(list.pooled[row], safe.flattened(), axis: 0)
                            .reshaped(n, indices.dim(-1), headDim))
                    rowValid.append(list.valid[row, start ..< end])
                }
                keys.append(concatenated(rowKeys, axis: 1))
                valid.append(concatenated(rowValid, axis: 1))
            }
            let part = referenceAttention(
                q[0..., 0..., start ..< end], keys: stacked(keys, axis: 0),
                valid: stacked(valid, axis: 0)
            ).asType(q.dtype)
            if tile < L { eval(part) }
            parts.append(part)
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    /// `sparse_attn` (`inference/kernel.py:294-350`) for queries `[B, H, L, D]`
    /// over per-query keys `[B, L, K, D]` (values are the keys) with validity
    /// `[B, L, K]`: float32 scores; over 64-key blocks in list order, a
    /// running max, `exp(s - running max)` rounded to bf16 for the value
    /// product, and the float32 accumulator and normalizer rescaled when the
    /// max rises; the float32 sink joins the normalizer at the end.
    func referenceAttention(_ q: MLXArray, keys: MLXArray, valid: MLXArray) -> MLXArray {
        let (B, H, L) = (q.dim(0), q.dim(1), q.dim(2))
        let K = keys.dim(2)
        let block = 64
        let blocks = (K + block - 1) / block
        let keys32 = keys.asType(.float32)
        var scores = matmul(
            q.transposed(0, 2, 1, 3).asType(.float32) * scale, keys32.swappedAxes(-1, -2))  // [B, L, H, K]
        scores = MLX.where(expandedDimensions(valid, axis: 2), scores, MLXArray(-Float.infinity))
        if blocks * block > K {
            scores = concatenated(
                [scores, MLXArray.full([B, L, H, blocks * block - K], values: MLXArray(-Float.infinity))],
                axis: -1)
        }
        let blocked = scores.reshaped(B, L, H, blocks, block)
        let running = cummax(blocked.max(axis: -1), axis: -1)  // [B, L, H, blocks]
        let peak = running[.ellipsis, (blocks - 1)...]  // [B, L, H, 1]
        let weights = exp(blocked - expandedDimensions(running, axis: -1))
        let rescale = exp(running - peak)
        let normalizer = (weights.sum(axis: -1) * rescale).sum(axis: -1, keepDims: true)
            + exp(attnSink.asType(.float32).reshaped(1, 1, H, 1) - peak)
        let valueWeights = (deepseekV4KernelWeights(weights) * expandedDimensions(rescale, axis: -1))
            .reshaped(B, L, H, blocks * block)[.ellipsis, ..<K]
        return (matmul(valueWeights, keys32) / normalizer).transposed(0, 2, 1, 3)
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

    func decodeBatch(_ x: MLXArray) -> MLXArray {
        guard Device.defaultDevice().deviceType == .cpu, x.dim(0) > 1 else { return self(x) }
        return concatenated((0 ..< x.dim(0)).map { self(x[$0 ..< ($0 + 1)]) }, axis: 0)
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

    func decodeBatch(
        _ h: MLXArray, caches: [DeepseekV4LayerCache], inputIDs: MLXArray
    ) -> MLXArray {
        let (attnInput, attnPost, attnComb) = decodeBranchInput(attnHC, h, norm: attnNorm)
        let h = glm5NextHCExpand(
            attn.decodeBatch(attnInput, caches: caches), h, attnPost, attnComb)
        let (ffnInput, ffnPost, ffnComb) = decodeBranchInput(ffnHC, h, norm: ffnNorm)
        return glm5NextHCExpand(ffn(ffnInput, inputIDs: inputIDs), h, ffnPost, ffnComb)
    }

    private func decodeBranchInput(
        _ hc: GLM5NextHyperConnection, _ h: MLXArray, norm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray) {
        // CPU GEMM changes the float32 mix reduction at eight rows; a tiny
        // post/comb delta can change a later bf16 rounding and greedy token.
        // GPU fused HC already evaluates each row with serial arithmetic.
        guard Device.defaultDevice().deviceType == .cpu, h.dim(0) > 1 else {
            return hc.normalizedBranchInput(h, norm: norm)
        }
        let rows = (0 ..< h.dim(0)).map {
            hc.normalizedBranchInput(h[$0 ..< ($0 + 1)], norm: norm)
        }
        return (
            concatenated(rows.map { $0.0 }, axis: 0),
            concatenated(rows.map { $0.1 }, axis: 0),
            concatenated(rows.map { $0.2 }, axis: 0)
        )
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
        return logits(norm(hcHead(h)))
    }

    /// One token per independently prefetched row. Each row owns ordinary
    /// `newCache` caches and may join, leave or change batch position between
    /// calls. Prefill each row with `prepare` before admitting its last token.
    /// Only projections and MoE are batched; ragged attention never pads or
    /// changes the reference pooling boundaries.
    /// Whether `decodeBatch` rows equal their serial runs exactly. An
    /// explicitly configured HISA indexer (approximate, never the default)
    /// does not batch; serve it serially.
    public var batchesExactly: Bool {
        !(configuration.indexBlock > 0 && configuration.indexKeep > 0)
    }

    public func decodeBatch(_ inputs: MLXArray, rowCaches: [[KVCache]]) -> MLXArray {
        precondition(batchesExactly, "DeepSeek V4 with a HISA indexer does not batch")
        precondition(inputs.ndim == 2 && inputs.dim(1) == 1)
        precondition((1 ... 8).contains(inputs.dim(0)), "DeepSeek V4 decode supports 1...8 rows")
        precondition(rowCaches.count == inputs.dim(0))
        precondition(rowCaches.allSatisfy { $0.count == layers.count })
        var h = repeated(
            expandedDimensions(embed(inputs), axis: 2), count: configuration.hcMult, axis: 2)
        for (i, layer) in layers.enumerated() {
            let caches = rowCaches.map { DeepseekV4LayerCache($0[i], ratio: layer.attn.ratio) }
            h = layer.decodeBatch(h, caches: caches, inputIDs: inputs)
        }
        return logits(norm(hcHead.decodeBatch(h)))
    }

    /// Float32 logits (`ParallelHead`, `inference/model.py:728-735`:
    /// `F.linear(x.float(), weight)` with a float32 head) from the bf16 head
    /// weight, without a resident float32 copy.
    private func logits(_ x: MLXArray) -> MLXArray {
        deepseekV4LinearF32(head, x)
    }

    /// The reference prefill step (mlx-vlm `generate_step` default).
    public static let prefillStepSize = 2048

    /// Prefill as mlx-vlm `generate_step` does: chunks of
    /// `min(prefillStepSize, remaining - 1)` while more than one prompt token
    /// is left, each followed by a cache eval, then the last token alone in
    /// the iterator's first forward. The pooled entries no longer depend on
    /// the chunk plan (the overlap window is carried across calls), but the
    /// plan stays fixed, whatever step size and chunking the caller asks
    /// for, so prompt snapshots taken at its boundaries resume bit for bit.
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
    /// `attn_hc`/`ffn_hc`/`hc_head`, make `tid2eid` int32 and the router
    /// gate float32.
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
            // The router computes float32 logits; convert its weight once here.
            if mapped.hasSuffix(".ffn.gate.weight") {
                tensor = tensor.asType(.float32)
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
