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
    var dsparkBlockSize: Int
    var dsparkNoiseTokenID: Int
    var dsparkTargetLayerIDs: [Int]
    var dsparkMarkovRank: Int
    var dsparkStageCount: Int = 0

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
        case dsparkBlockSize = "dspark_block_size"
        case dsparkNoiseTokenID = "dspark_noise_token_id"
        case dsparkTargetLayerIDs = "dspark_target_layer_ids"
        case dsparkMarkovRank = "dspark_markov_rank"
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
        dsparkBlockSize = try c.decodeIfPresent(Int.self, forKey: .dsparkBlockSize) ?? 5
        dsparkNoiseTokenID = try c.decodeIfPresent(Int.self, forKey: .dsparkNoiseTokenID) ?? 128799
        dsparkTargetLayerIDs = try c.decodeIfPresent([Int].self, forKey: .dsparkTargetLayerIDs) ?? []
        dsparkMarkovRank = try c.decodeIfPresent(Int.self, forKey: .dsparkMarkovRank) ?? 256

        // The checkpoint lists one ratio per layer plus the MTP stages; the
        // text model takes the first `num_hidden_layers` (config.py:72-80).
        if let ratios = try c.decodeIfPresent([Int].self, forKey: .compressRatios), !ratios.isEmpty {
            compressRatios = Array(ratios.prefix(hiddenLayers))
            if !dsparkTargetLayerIDs.isEmpty {
                let stages = ratios.dropFirst(hiddenLayers)
                guard !stages.isEmpty, stages.allSatisfy({ $0 == 0 }) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .compressRatios, in: c,
                        debugDescription: "DSpark needs uncompressed stage ratios after the target layers")
                }
                dsparkStageCount = stages.count
            }
        } else {
            let n = hiddenLayers
            compressRatios =
                [0] + (0 ..< max(n - 2, 0)).map { $0 % 2 == 1 ? 4 : 128 } + (n >= 2 ? [0] : [])
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(qLoraRank, forKey: .qLoraRank)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(scoringFunc, forKey: .scoringFunc)
        try c.encode(compressRatios + Array(repeating: 0, count: dsparkStageCount), forKey: .compressRatios)
        try c.encode(compressRopeTheta, forKey: .compressRopeTheta)
        try c.encode(hcMult, forKey: .hcMult)
        try c.encode(hcSinkhornIters, forKey: .hcSinkhornIters)
        try c.encode(hcEps, forKey: .hcEps)
        try c.encode(numHashLayers, forKey: .numHashLayers)
        try c.encode(swigluLimit, forKey: .swigluLimit)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(oGroups, forKey: .oGroups)
        try c.encode(oLoraRank, forKey: .oLoraRank)
        try c.encode(indexNHeads, forKey: .indexNHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexTopk, forKey: .indexTopk)
        try c.encode(indexBlock, forKey: .indexBlock)
        try c.encode(indexKeep, forKey: .indexKeep)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(dsparkBlockSize, forKey: .dsparkBlockSize)
        try c.encode(dsparkNoiseTokenID, forKey: .dsparkNoiseTokenID)
        try c.encode(dsparkTargetLayerIDs, forKey: .dsparkTargetLayerIDs)
        try c.encode(dsparkMarkovRank, forKey: .dsparkMarkovRank)
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
        if !dsparkTargetLayerIDs.isEmpty {
            guard dsparkStageCount == 3, (1 ... 7).contains(dsparkBlockSize),
                dsparkNoiseTokenID >= 0, dsparkNoiseTokenID < vocabularySize,
                dsparkMarkovRank > 0, Set(dsparkTargetLayerIDs).count == dsparkTargetLayerIDs.count,
                dsparkTargetLayerIDs.allSatisfy({ (0 ..< hiddenLayers).contains($0) })
            else { throw fail("invalid DSpark stages, target layers, block size or Markov head") }
        }
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

/// DeepSeek multiplies the learned weight before rounding back to the activation dtype.
final class DeepseekV4RMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        let inverse = rsqrt(square(value).mean(axis: -1, keepDims: true) + eps)
        return (value * inverse * weight.asType(.float32)).asType(x.dtype)
    }
}

/// The reference rounds the inverse RMS to the query dtype before multiplying.
func deepseekV4QueryHeadNorm(_ q: MLXArray, eps: Float) -> MLXArray {
    let inverse = rsqrt(square(q.asType(.float32)).mean(axis: -1, keepDims: true) + eps)
    return q * inverse.asType(q.dtype)
}

/// Round each float32 stream product before reducing, as in DeepSeek's HC post.
func deepseekV4HCExpand(
    _ x: MLXArray, _ residual: MLXArray, _ post: MLXArray, _ comb: MLXArray
) -> MLXArray {
    let expanded = expandedDimensions(post, axis: -1)
        * expandedDimensions(x.asType(.float32), axis: 2)
    let products = expandedDimensions(comb, axis: -1)
        * expandedDimensions(residual.asType(.float32), axis: -2)
    return (expanded + products.sum(axis: 2)).asType(x.dtype)
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
        const device uint4* row = reinterpret_cast<const device uint4*>(w) + size_t(n) * size_t(K / 8);
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
                const device float4* xr = reinterpret_cast<const device float4*>(x + size_t(m) * size_t(K)) + 2 * i;
                acc[m] += dot(xr[0], lo) + dot(xr[1], hi);
            }
        }
        for (int m = 0; m < M; ++m) {
            const float total = simd_sum(acc[m]);
            if (lane == 0) { out[size_t(m) * size_t(N) + n] = total; }
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

/// Small packed verification uses the decode kernel geometry. Unsupported
/// dense float32 products retain one-row GEMV order on both CPU and GPU.
private func deepseekV4PackedLinear(_ layer: Linear, _ x: MLXArray) -> MLXArray {
    let denseFloat32 = x.dtype == .float32 && !(layer is QuantizedLinear)
    if x.dim(0) > 1, Device.defaultDevice().deviceType == .cpu || denseFloat32 {
        return concatenated((0 ..< x.dim(0)).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0)
    }
    return deepseekV4Project(layer, x)
}

/// The production bf16-weight float32 kernel is already row-independent.
/// Only dense float32 weights need the serial primitive in packed verification.
private func deepseekV4PackedLinearF32(_ layer: Linear, _ x: MLXArray) -> MLXArray {
    if x.dim(0) > 1, x.dtype == .float32, layer.weight.dtype == .float32,
        !(layer is QuantizedLinear)
    {
        return concatenated((0 ..< x.dim(0)).map {
            deepseekV4LinearF32(layer, x[$0 ..< ($0 + 1)])
        }, axis: 0)
    }
    return deepseekV4LinearF32(layer, x)
}

// MARK: - RoPE

/// DeepSeek-V4 RoPE (`language.py:258-339`): traditional pairs on the last
/// `dims` channels only, optional YaRN frequency interpolation without an
/// mscale term, an inverse mode for
/// the output, and a `freqScale` for compressor pools (positions are divided
/// by it, frequencies multiplied).
final class DeepseekV4RoPE {
    let dims: Int
    let freqScale: Int
    private let inverseFrequencies: MLXArray
    private let legacyFloat32Wavelengths: MLXArray

    init(dims: Int, base: Float, scaling: DeepseekV4Configuration.RopeScaling?, freqScale: Int = 1) {
        self.dims = dims
        self.freqScale = freqScale
        // Match the reference's float32 frequency initialization, without a
        // GPU pow approximation or a reciprocal-to-wavelength round trip.
        var invFreq = (0 ..< (dims / 2)).map {
            Float(1) / Foundation.pow(base, Float(2 * $0) / Float(dims))
        }
        // CPU RoPE already uses precise trig. Retain its established float32
        // frequency graph and call geometry for the CPU parity oracle.
        var legacyInvFreq =
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
            let legacyLow = max(Foundation.floor(correctionDim(betaFast)), 0)
            var legacyHigh = min(Foundation.ceil(correctionDim(betaSlow)), Double(dims - 1))
            let low = Float(legacyLow)
            var high = Float(legacyHigh)
            if low == high { high += 0.001 }
            if legacyLow == legacyHigh { legacyHigh += 0.001 }
            let legacyRamp = (MLXArray.arange(dims / 2, dtype: .float32) - Float(legacyLow))
                / Float(legacyHigh - legacyLow)
            let legacySmooth = 1 - clip(legacyRamp, min: 0, max: 1)
            legacyInvFreq = legacyInvFreq / factor * (1 - legacySmooth) + legacyInvFreq * legacySmooth
            for index in invFreq.indices {
                let ramp = min(max((Float(index) - low) / (high - low), 0), 1)
                let smooth = 1 - ramp
                invFreq[index] = invFreq[index] / factor * (1 - smooth) + invFreq[index] * smooth
            }
        }
        self.inverseFrequencies = MLXArray(invFreq)
        self.legacyFloat32Wavelengths = 1.0 / legacyInvFreq
    }

    private func legacyFloat32Frequencies(headDim: Int, inverse: Bool) -> MLXArray {
        var frequencies = legacyFloat32Wavelengths
        if freqScale != 1 { frequencies = frequencies / Float(freqScale) }
        if inverse { frequencies = -frequencies }
        let prefixPairs = (headDim - dims) / 2
        if prefixPairs > 0 {
            frequencies = concatenated([
                MLXArray.full([prefixPairs], values: MLXArray(Float.infinity)), frequencies,
            ])
        }
        return frequencies
    }

    private func rotate(_ x: MLXArray, positions: MLXArray, inverse: Bool) -> MLXArray {
        let length = x.dim(-2)
        let prefix = x.dim(-1) - dims
        let positionShape =
            positions.ndim == 1
            ? Array(repeating: 1, count: x.ndim - 2) + [length, dims / 2]
            : [positions.dim(0)] + Array(repeating: 1, count: x.ndim - 3) + [length, dims / 2]
        let angles = (expandedDimensions(positions.asType(.float32), axis: -1) * inverseFrequencies)
            .reshaped(positionShape)
        let (cosine, forwardSine) = deepseekV4RoPEPhases(angles)
        let sine = inverse ? -forwardSine : forwardSine
        let pairs = x[.ellipsis, prefix...].asType(.float32)
            .reshaped(Array(x.shape.dropLast()) + [dims / 2, 2])
        let first = pairs[.ellipsis, 0]
        let second = pairs[.ellipsis, 1]
        let rotated = stacked(
            [first * cosine - second * sine, first * sine + second * cosine], axis: -1
        ).reshaped(Array(x.shape.dropLast()) + [dims]).asType(x.dtype)
        return prefix == 0 ? rotated : concatenated([x[.ellipsis, ..<prefix], rotated], axis: -1)
    }

    /// Each batch row starts at its own absolute token position.
    func callAsFunction(_ x: MLXArray, offsets: MLXArray, inverse: Bool = false) -> MLXArray {
        if Device.defaultDevice().deviceType == .cpu, x.dtype == .float32 {
            return MLXFast.RoPE(
                x, dimensions: x.dim(-1), traditional: true, base: nil, scale: 1,
                offset: freqScale != 1 ? offsets.floorDivide(Int32(freqScale)) : offsets,
                freqs: legacyFloat32Frequencies(headDim: x.dim(-1), inverse: inverse))
        }
        let starts = freqScale != 1 ? offsets.floorDivide(Int32(freqScale)) : offsets
        let positions = expandedDimensions(starts, axis: -1)
            + MLXArray.arange(x.dim(-2), dtype: .int32)
        return rotate(x, positions: positions * Int32(freqScale), inverse: inverse)
    }

    /// Rotate `[..., T, headDim]` at positions `offset ..< offset + T`.
    func callAsFunction(_ x: MLXArray, offset: Int, inverse: Bool = false) -> MLXArray {
        if Device.defaultDevice().deviceType == .cpu, x.dtype == .float32 {
            return MLXFast.RoPE(
                x, dimensions: x.dim(-1), traditional: true, base: nil, scale: 1,
                offset: freqScale != 1 ? offset / freqScale : offset,
                freqs: legacyFloat32Frequencies(headDim: x.dim(-1), inverse: inverse))
        }
        let positions = MLXArray.arange(x.dim(-2), dtype: .int32) + Int32(offset / freqScale)
        return rotate(x, positions: positions * Int32(freqScale), inverse: inverse)
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
// Adapted from MLX 0.32.4 kernels/utils.h log1p; uses precise::log to
// match the frozen oracle library. Copyright © 2023 Apple Inc.; see NOTICE.
private final class DeepseekV4RouterLog1pKernel: Sendable {
    static let shared = DeepseekV4RouterLog1pKernel()
    let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_router_reference_log1p", inputNames: ["x"], outputNames: ["y"],
        source: """
            uint i = thread_position_in_grid.x;
            float v = x[i];
            float vp1 = 1.0f + v;
            y[i] = vp1 == 1.0f ? v : v * (metal::precise::log(vp1) / (vp1 - 1.0f));
            """, ensureRowContiguous: true)
}

/// Preserve the frozen reference's float32 score graph. The bundled native
/// log used by log1p differs by ulps from its precise-log oracle, changing
/// routing weights and weighted bf16 expert rounding even with exact logits.
func deepseekV4RouterSoftplusScores(_ logits: MLXArray) -> MLXArray {
    let e = exp(-abs(logits))
    let logarithm: MLXArray
    if Device.defaultDevice().deviceType == .gpu, e.dtype == .float32, e.size > 0 {
        logarithm = DeepseekV4RouterLog1pKernel.shared.kernel(
            [e], grid: (e.size, 1, 1), threadGroup: (min(e.size, 256), 1, 1),
            outputShapes: [e.shape], outputDTypes: [.float32])[0]
    } else {
        logarithm = log1p(e)
    }
    return sqrt(logarithm + maximum(logits, 0))
}

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
        default: deepseekV4RouterSoftplusScores(logits)
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
    private let swigluLimit: Float

    init(_ config: DeepseekV4Configuration, layerIdx: Int) {
        _gate.wrappedValue = DeepseekV4MoEGate(config, layerIdx: layerIdx)
        let limit = config.swigluLimit > 0 ? config.swigluLimit : nil
        self.activation = makeLimitedSwiGLU(limit: limit)
        self.swigluLimit = config.swigluLimit
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
    /// float32 and the shared expert added in float32. Prefill preserves the
    /// reference's per-expert quantized GEMMs; decode retains gathered matmuls.
    func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, inputIDs: inputIDs)
        if let output = quantizedPrefill(x, indices: indices, weights: weights) {
            return output
        }
        return gatheredForward(x, indices: indices, weights: weights)
    }

    private func gatheredForward(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
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

    /// The float32 router's multi-vector GEMV rounds differently from the
    /// ordinary one-vector kernel. Keep this small projection positionwise;
    /// packed attention and expert projections still share their weight reads.
    func packedGate(_ x: MLXArray, inputIDs: MLXArray) -> (MLXArray, MLXArray) {
        guard (2 ... 8).contains(x.dim(0)) else { return gate(x, inputIDs: inputIDs) }
        let rows = (0 ..< x.dim(0)).map { row in
            gate(x[row ..< (row + 1)], inputIDs: inputIDs[row ..< (row + 1)])
        }
        return (concatenated(rows.map { $0.0 }, axis: 0),
                concatenated(rows.map { $0.1 }, axis: 0))
    }

    func packed(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
        let tokens = x.size / x.dim(-1)
        guard (1 ... 8).contains(tokens), x.dtype == .bfloat16,
            Device.defaultDevice().deviceType == .gpu,
            GPU.deviceInfo().architecture == "applegpu_g15d",
            let gateProjection = switchMLP.gateProjection as? QuantizedSwitchLinear,
            let upProjection = switchMLP.upProjection as? QuantizedSwitchLinear,
            let downProjection = switchMLP.downProjection as? QuantizedSwitchLinear
        else { return self(x, inputIDs: inputIDs) }
        func supported(_ projection: QuantizedSwitchLinear) -> Bool {
            guard projection.mode == .mxfp4, projection.bits == 4,
                projection.groupSize == 32 else { return false }
            let parts = projection.quantizedParts
            return parts.bias == nil && parts.biases == nil
        }
        guard supported(gateProjection), supported(upProjection), supported(downProjection)
        else { return self(x, inputIDs: inputIDs) }

        let (indices, weights) = packedGate(x, inputIDs: inputIDs)
        // The aligned MXFP4 gathered vector kernel uses the same reduction
        // as per-expert qmv while preserving the original route-slot order.
        return gatheredForward(x, indices: indices, weights: weights)
    }

    private func quantizedPrefill(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray? {
        let tokens = x.size / x.dim(-1)
        guard tokens > 8, x.dtype == .bfloat16,
            Device.defaultDevice().deviceType == .gpu,
            let gateProjection = switchMLP.gateProjection as? QuantizedSwitchLinear,
            let upProjection = switchMLP.upProjection as? QuantizedSwitchLinear,
            let downProjection = switchMLP.downProjection as? QuantizedSwitchLinear,
            gateProjection.mode == .mxfp4, upProjection.mode == .mxfp4,
            downProjection.mode == .mxfp4,
            gateProjection.quantizedParts.bias == nil,
            upProjection.quantizedParts.bias == nil,
            downProjection.quantizedParts.bias == nil
        else { return nil }

        let k = indices.dim(-1)
        var assignments = [[Int32]](repeating: [], count: gateProjection.quantizedParts.weight.dim(0))
        for (slot, expert) in indices.asType(.int32).asArray(Int32.self).enumerated() {
            assignments[Int(expert)].append(Int32(slot))
        }
        let flat = x.reshaped(tokens, x.dim(-1))
        let routeWeights = weights.flattened()
        var combined = MLXArray.zeros(flat.shape, dtype: .float32)
        func project(_ projection: QuantizedSwitchLinear, _ input: MLXArray, expert: Int) -> MLXArray {
            let parts = projection.quantizedParts
            if projection.groupSize == 32, projection.bits == 4, parts.biases == nil,
                let reference = deepseekV4ReferencePrefillQMM(
                    input, weight: parts.weight[expert], scales: parts.scales[expert])
            {
                return reference
            }
            return quantizedMM(
                input, parts.weight[expert], scales: parts.scales[expert],
                biases: parts.biases.map { $0[expert] }, transpose: true,
                groupSize: projection.groupSize, bits: projection.bits, mode: projection.mode)
        }
        for (expert, slots) in assignments.enumerated() where !slots.isEmpty {
            let routes = MLXArray(slots)
            let rows = routes.floorDivide(Int32(k))
            let input = flat[rows]
            var gate = project(gateProjection, input, expert: expert).asType(.float32)
            var up = project(upProjection, input, expert: expert).asType(.float32)
            if swigluLimit > 0 {
                gate = minimum(gate, swigluLimit)
                up = clip(up, min: -swigluLimit, max: swigluLimit)
            }
            let hidden = (expandedDimensions(routeWeights[routes], axis: -1)
                * ((gate * sigmoid(gate)) * up)).asType(x.dtype)
            let output = project(downProjection, hidden, expert: expert).asType(.float32)
            combined = combined.at[rows].add(output)
        }
        return (combined.reshaped(x.shape) + sharedExperts(x).asType(.float32)).asType(x.dtype)
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
        _norm.wrappedValue = DeepseekV4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
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
        appendProjected(
            kv: deepseekV4LinearF32(wkv, x), gate: deepseekV4LinearF32(wgate, x),
            dtype: x.dtype, cache: cache, offset: offset)
    }

    /// Consume one position's already-batched projections completely before
    /// advancing another position of the same cache (including overlap carry).
    func appendProjected(
        kv projectedKV: MLXArray, gate projectedGate: MLXArray, dtype: DType,
        cache: DeepseekV4PoolCache, offset: Int
    ) -> MLXArray {
        let B = projectedKV.dim(0)
        let ready = cache.accumulateWindows(kv: projectedKV, gate: projectedGate, offset: offset)
        guard ready.kv.dim(1) > 0 else {
            return cache.appendPooled(MLXArray.zeros([B, 0, headDim], dtype: dtype))
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
        entries = norm(entries.asType(dtype))
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
            deepseekV4Project(wqB, qResidual).reshaped(B, 1, heads, headDim).transposed(0, 2, 1, 3),
            offsets: offsets)
            .asType(.float32)
        let weights = deepseekV4LinearF32(weightsProj, x) * (scale * pow(Float(heads), -0.5))
        let wq = expandedDimensions(weights.swappedAxes(-1, -2), axis: -1)
        let lengths = MLXArray(counts.map(Int32.init)).reshaped(B, 1, 1)
        var scores = maximum(
            matmul(q, expandedDimensions(pooled, axis: 1).swappedAxes(-1, -2).asType(.float32)), 0)
        scores = (scores * wq).sum(axis: 1)
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

        var q = deepseekV4Project(wqB, qResidual).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        let weights = deepseekV4LinearF32(weightsProj, x) * (scale * pow(Float(heads), -0.5))
        return selectProjected(q, weights: weights, pooled: pooled, cache: cache, offset: offset)
    }

    /// Selection sees only this position's committed pool prefix. Projections
    /// may be shared with later verification positions, visibility may not.
    /// Weights are float32 and already include scale * heads**-0.5, as in
    /// the reference's LinearF32 weights_proj; never round them to bf16.
    func selectProjected(
        _ q: MLXArray, weights: MLXArray, pooled: MLXArray,
        cache: DeepseekV4PoolCache, offset: Int
    ) -> MLXArray? {
        let (B, L, poolCount) = (q.dim(0), q.dim(2), pooled.dim(1))
        guard poolCount > 0 else { return nil }
        let k = min(topK, poolCount)
        let poolMask = cache.poolMask(queries: L, offset: offset)

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
            var scores = maximum(matmul(q.asType(.float32), keys), 0)
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
        let coarseScore = (coarse * wq).sum(axis: 1)  // [B, 1, blocks]
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
        let fineScore = (fine * wq).sum(axis: 1)  // [B, 1, candidates]
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

        let wk = weights  // [B, L, H], already includes both reference scales
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
        _qNorm.wrappedValue = DeepseekV4RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kvNorm.wrappedValue = DeepseekV4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
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

        let qResidual = qNorm(deepseekV4Project(wqA, x))
        var q = deepseekV4Project(wqB, qResidual).reshaped(B, L, heads, headDim)
        q = deepseekV4QueryHeadNorm(q, eps: normEps)
        q = rope(q.transposed(0, 2, 1, 3), offset: offset)
        var kv = kvNorm(deepseekV4Project(wkv, x)).reshaped(B, 1, L, headDim)
        kv = rope(kv, offset: offset)

        var out = attend(x, q: q, kv: kv, qResidual: qResidual, mask: mask, cache: cache, offset: offset)
        out = rope(out, offset: offset, inverse: true)

        out = out.reshaped(B, groups, heads / groups, L, headDim).transposed(0, 1, 3, 2, 4)
            .reshaped(B, groups, L, heads / groups * headDim)
        out = callMultiLinear(woA, out)  // [B, G, L, R]
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return deepseekV4Project(woB, out)
    }

    /// One token per row. Projections and the attention arithmetic are
    /// batched; each row keeps its own window, pools and selection, and its
    /// key list is the one `attend` builds for it alone, so a batched row
    /// computes exactly what it computes serially.
    func decodeBatch(_ x: MLXArray, caches: [DeepseekV4LayerCache]) -> MLXArray {
        let B = x.dim(0)
        let offsets = caches.map { $0.local.offset }
        let offsetArray = MLXArray(offsets.map(Int32.init))
        let qResidual = qNorm(deepseekV4Project(wqA, x))
        let q = rope(
            deepseekV4QueryHeadNorm(
                deepseekV4Project(wqB, qResidual).reshaped(B, 1, heads, headDim),
                eps: normEps).transposed(0, 2, 1, 3),
            offsets: offsetArray)
        let kv = rope(kvNorm(deepseekV4Project(wkv, x)).reshaped(B, 1, 1, headDim), offsets: offsetArray)
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
        return deepseekV4Project(woB, out.transposed(0, 2, 1, 3).reshaped(B, 1, -1))
    }

    /// Flattened row-major verification positions. All weight-bound work is
    /// projected once; only window/pool/indexer state advances positionwise,
    /// as in mlx-vlm's `DECODE_BLOCK_SIZE` verification path.
    func verifyPacked(
        _ x: MLXArray, caches: [KVCache], rowLengths: [Int]
    ) -> (output: MLXArray, prefixes: [[KVCache]]) {
        let M = x.dim(0)
        let layerCaches = caches.map { DeepseekV4LayerCache($0, ratio: ratio) }
        let offsets = zip(layerCaches, rowLengths).flatMap { cache, count in
            (0 ..< count).map { cache.local.offset + $0 }
        }
        let offsetArray = MLXArray(offsets.map(Int32.init))
        let qResidual = qNorm(deepseekV4PackedLinear(wqA, x))
        let q = rope(
            deepseekV4QueryHeadNorm(
                deepseekV4PackedLinear(wqB, qResidual).reshaped(M, 1, heads, headDim),
                eps: normEps).transposed(0, 2, 1, 3),
            offsets: offsetArray)
        let kv = rope(
            kvNorm(deepseekV4PackedLinear(wkv, x)).reshaped(M, 1, 1, headDim),
            offsets: offsetArray)
        let projectedPool = compressor.map {
            (kv: deepseekV4PackedLinearF32($0.wkv, x), gate: deepseekV4PackedLinearF32($0.wgate, x))
        }
        let projectedIndex = indexer.map { indexer in
            (
                kv: deepseekV4PackedLinearF32(indexer.compressor.wkv, x),
                gate: deepseekV4PackedLinearF32(indexer.compressor.wgate, x),
                q: rope(
                    deepseekV4PackedLinear(indexer.wqB, qResidual)
                        .reshaped(M, 1, indexer.heads, indexer.headDim).transposed(0, 2, 1, 3),
                    offsets: offsetArray),
                weights: deepseekV4PackedLinearF32(indexer.weightsProj, x)
                    * (indexer.scale * pow(Float(indexer.heads), -0.5))
            )
        }
        var attended = [MLXArray]()
        var prefixes = rowLengths.map { _ in [KVCache]() }
        var packed = 0
        for row in rowLengths.indices {
            for _ in 0 ..< rowLengths[row] {
                let range = packed ..< (packed + 1)
                let part = attend(
                    x[range], q: q[range], kv: kv[range], qResidual: qResidual[range], mask: nil,
                    cache: layerCaches[row], offset: offsets[packed],
                    poolProjection: projectedPool.map { (kv: $0.kv[range], gate: $0.gate[range]) },
                    indexProjection: projectedIndex.map {
                        (kv: $0.kv[range], gate: $0.gate[range], q: $0.q[range], weights: $0.weights[range])
                    })
                attended.append(part)
                prefixes[row].append(caches[row].copy())
                packed += 1
            }
        }
        var out = rope(concatenated(attended, axis: 0), offsets: offsetArray, inverse: true)
        out = out.reshaped(M, groups, heads / groups, 1, headDim).transposed(0, 1, 3, 2, 4)
            .reshaped(M, groups, 1, heads / groups * headDim)
        let denseFloat32Output = out.dtype == .float32 && woA is MultiLinear
        if M > 1, Device.defaultDevice().deviceType == .cpu || denseFloat32Output {
            out = concatenated((0 ..< M).map { callMultiLinear(woA, out[$0 ..< ($0 + 1)]) }, axis: 0)
        } else {
            out = callMultiLinear(woA, out)
        }
        return (
            deepseekV4PackedLinear(woB, out.transposed(0, 2, 1, 3).reshaped(M, 1, -1)),
            prefixes
        )
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
    func windowList(_ local: MLXArray, first: Int, positions: [Int]) -> (MLXArray, MLXArray) {
        let S = local.dim(0)
        // A wrapped one-token window already has the reference temporal order.
        // Reusing it avoids a gather and host index construction per layer/row.
        if positions.count == 1, S == window, first == positions[0] - window + 1 {
            return (
                expandedDimensions(local, axis: 0),
                MLXArray.ones([1, window], dtype: .bool)
            )
        }
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
        cache: DeepseekV4LayerCache, offset: Int,
        poolProjection: (kv: MLXArray, gate: MLXArray)? = nil,
        indexProjection: (kv: MLXArray, gate: MLXArray, q: MLXArray, weights: MLXArray)? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let (updated, _) = cache.local.update(
            keys: kv, values: MLXArray.zeros([kv.dim(0), 1, L, 0], dtype: kv.dtype))
        let local = L == 1 ? Self.temporal(updated, cache: cache.local) : updated
        let first = offset + L - local.dim(2)

        // [B, L, C] compressed indices and validity, or nil without a pool.
        var list: (indices: MLXArray, valid: MLXArray, pooled: MLXArray)?
        if let compressor, let poolCache = cache.pool {
            let pooled: MLXArray
            if let projected = poolProjection {
                pooled = compressor.appendProjected(
                    kv: projected.kv, gate: projected.gate, dtype: x.dtype,
                    cache: poolCache, offset: offset)
            } else {
                pooled = compressor(x, cache: poolCache, offset: offset)
            }
            let visible = MLXArray((offset ..< (offset + L)).map { Int32(($0 + 1) / ratio) })
                .reshaped(1, L, 1)
            if let indexer, let indexCache = cache.indexPool {
                let topk: MLXArray?
                if let projected = indexProjection {
                    let indexPool = indexer.compressor.appendProjected(
                        kv: projected.kv, gate: projected.gate, dtype: x.dtype,
                        cache: indexCache, offset: offset)
                    topk = indexer.selectProjected(
                        projected.q, weights: projected.weights, pooled: indexPool,
                        cache: indexCache, offset: offset)
                } else {
                    topk = indexer(
                        x, qResidual: qResidual, rope: rope, cache: indexCache, offset: offset)
                }
                if let topk {
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
        if let fused = deepseekV4SparseAttention(
            q, keys: keys, valid: valid, sink: attnSink, scale: scale)
        {
            return fused
        }
        return graphAttention(q, keys: keys, valid: valid)
    }

    /// Float32 graph oracle and fallback for prefill, CPU and unsupported shapes.
    func graphAttention(_ q: MLXArray, keys: MLXArray, valid: MLXArray) -> MLXArray {
        let (B, H, L) = (q.dim(0), q.dim(1), q.dim(2))
        let K = keys.dim(2)
        let block = 64
        let blocks = (K + block - 1) / block
        let keys32 = keys.asType(.float32)
        var scores = matmul(
            q.transposed(0, 2, 1, 3).asType(.float32), keys32.swappedAxes(-1, -2)) * scale  // [B, L, H, K]
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
        _attnNorm.wrappedValue = DeepseekV4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ffnNorm.wrappedValue = DeepseekV4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        func hc() -> GLM5NextHyperConnection {
            GLM5NextHyperConnection(
                hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
                hcEps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize,
                normWeightInFloat32: true, normalizeMixAfterProjection: true)
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
        let h = deepseekV4HCExpand(attn(attnInput, mask: mask, cache: cache), h, attnPost, attnComb)
        let (ffnInput, ffnPost, ffnComb) = ffnHC.normalizedBranchInput(h, norm: ffnNorm)
        return deepseekV4HCExpand(ffn(ffnInput, inputIDs: inputIDs), h, ffnPost, ffnComb)
    }

    func decodeBatch(
        _ h: MLXArray, caches: [DeepseekV4LayerCache], inputIDs: MLXArray
    ) -> MLXArray {
        let (attnInput, attnPost, attnComb) = decodeBranchInput(attnHC, h, norm: attnNorm)
        let h = deepseekV4HCExpand(
            attn.decodeBatch(attnInput, caches: caches), h, attnPost, attnComb)
        let (ffnInput, ffnPost, ffnComb) = decodeBranchInput(ffnHC, h, norm: ffnNorm)
        return deepseekV4HCExpand(ffn(ffnInput, inputIDs: inputIDs), h, ffnPost, ffnComb)
    }

    func verifyPacked(
        _ h: MLXArray, caches: [KVCache], rowLengths: [Int], inputIDs: MLXArray
    ) -> (output: MLXArray, prefixes: [[KVCache]]) {
        let (attnInput, attnPost, attnComb) = packedBranchInput(attnHC, h, norm: attnNorm)
        let attention = attn.verifyPacked(attnInput, caches: caches, rowLengths: rowLengths)
        let h = deepseekV4HCExpand(attention.output, h, attnPost, attnComb)
        let (ffnInput, ffnPost, ffnComb) = packedBranchInput(ffnHC, h, norm: ffnNorm)
        let projected: MLXArray
        if h.dim(0) > 1, Device.defaultDevice().deviceType == .cpu || h.dtype == .float32 {
            projected = concatenated((0 ..< h.dim(0)).map {
                ffn(ffnInput[$0 ..< ($0 + 1)], inputIDs: inputIDs[$0 ..< ($0 + 1)])
            }, axis: 0)
        } else {
            projected = ffn.packed(ffnInput, inputIDs: inputIDs)
        }
        return (deepseekV4HCExpand(projected, h, ffnPost, ffnComb), attention.prefixes)
    }

    /// Dense float32 HC falls back to a graph whose GEMM depends on row
    /// count. Keep its serial primitive without changing ordinary CB arithmetic.
    private func packedBranchInput(
        _ hc: GLM5NextHyperConnection, _ h: MLXArray, norm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard h.dtype == .float32, h.dim(0) > 1 else {
            return decodeBranchInput(hc, h, norm: norm)
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
    public var hasMTPWeights: Bool { !mtp.isEmpty }
    let configuration: DeepseekV4Configuration

    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "head") var head: Linear
    @ModuleInfo(key: "mtp") var mtp: [DeepseekV4DSparkStage]

    public init(_ configuration: DeepseekV4Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _embed.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            DeepseekV4Block(configuration, layerIdx: $0)
        }
        _norm.wrappedValue = DeepseekV4RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _hcHead.wrappedValue = DeepseekV4HyperHead(configuration)
        _head.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        _mtp.wrappedValue = (0 ..< configuration.dsparkStageCount).map {
            DeepseekV4DSparkStage(
                configuration, stageID: $0, stageCount: configuration.dsparkStageCount)
        }
        super.init()
    }

    /// Logits `[B, L, vocab]` for token ids `[B, L]`. Without a cache the
    /// call runs on fresh caches (the reference's no-cache pool path skips
    /// the pool's causal mask, so it is not reproduced).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, emitMTPState: false).logits
    }

    private func forward(
        _ inputs: MLXArray, cache: [KVCache]?, emitMTPState: Bool
    ) -> LMOutput {
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

        var captured = [Int: MLXArray]()
        for (index, item) in zip(layers, layerCaches).enumerated() {
            h = item.0(h, mask: mask, cache: item.1, inputIDs: inputs)
            if emitMTPState, configuration.dsparkTargetLayerIDs.contains(index) {
                captured[index] = h.mean(axis: 2)
            }
        }
        return output(logits: logits(norm(hcHead(h))), captured: captured, emitMTPState: emitMTPState)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(input.tokens, cache: cache, emitMTPState: state?[mtpEmitFlagKey] ?? false)
    }

    private func output(
        logits: MLXArray, captured: [Int: MLXArray], emitMTPState: Bool
    ) -> LMOutput {
        guard emitMTPState, !configuration.dsparkTargetLayerIDs.isEmpty else {
            return LMOutput(logits: logits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = concatenated(
            configuration.dsparkTargetLayerIDs.map { captured[$0]! }, axis: -1)
        return LMOutput(logits: logits, state: state)
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
        decodeBatchOutput(inputs, rowCaches: rowCaches, emitMTPState: false).logits
    }

    public func decodeBatchOutput(
        _ inputs: MLXArray, rowCaches: [[KVCache]], emitMTPState: Bool
    ) -> LMOutput {
        precondition(batchesExactly, "DeepSeek V4 with a HISA indexer does not batch")
        precondition(inputs.ndim == 2 && inputs.dim(1) == 1)
        precondition((1 ... 8).contains(inputs.dim(0)), "DeepSeek V4 decode supports 1...8 rows")
        precondition(rowCaches.count == inputs.dim(0))
        precondition(rowCaches.allSatisfy { $0.count == layers.count })
        // The serial forward has the same cache geometry without row slicing,
        // stacking or padding when the scheduler currently owns one row.
        if rowCaches.count == 1 {
            return forward(inputs, cache: rowCaches[0], emitMTPState: emitMTPState)
        }
        var captured = [Int: MLXArray]()
        var h = repeated(
            expandedDimensions(embed(inputs), axis: 2), count: configuration.hcMult, axis: 2)
        for (i, layer) in layers.enumerated() {
            let caches = rowCaches.map { DeepseekV4LayerCache($0[i], ratio: layer.attn.ratio) }
            h = layer.decodeBatch(h, caches: caches, inputIDs: inputs)
            if emitMTPState, configuration.dsparkTargetLayerIDs.contains(i) {
                captured[i] = h.mean(axis: 2)
            }
        }
        return output(
            logits: logits(norm(hcHead.decodeBatch(h))), captured: captured,
            emitMTPState: emitMTPState)
    }

    /// Exact packed ragged verification of at most eight total positions,
    /// with independent cache snapshots for every possible acceptance prefix.
    public func verifyBatchOutput(
        _ inputs: [[Int]], rowCaches: [[KVCache]], emitMTPState: Bool = true
    ) -> (output: LMOutput, cachePrefixes: [[[KVCache]]], rowOffsets: [Int]) {
        verifyPackedBatchOutput(inputs, rowCaches: rowCaches, emitMTPState: emitMTPState)
    }

    /// Shared packed implementation for the public serving verifier.
    func verifyPackedBatchOutput(
        _ inputs: [[Int]], rowCaches: [[KVCache]], emitMTPState: Bool = true
    ) -> (output: LMOutput, cachePrefixes: [[[KVCache]]], rowOffsets: [Int]) {
        let lengths = inputs.map { $0.count }
        let M = lengths.reduce(0, +)
        precondition(batchesExactly, "DeepSeek V4 with a HISA indexer does not verify in a batch")
        precondition(!inputs.isEmpty && lengths.allSatisfy { $0 > 0 })
        precondition((1 ... 8).contains(M), "DeepSeek V4 packed verify supports 1...8 positions")
        precondition(rowCaches.count == inputs.count)
        precondition(rowCaches.allSatisfy { $0.count == layers.count })
        // CPU graph reductions can round differently with packed leading
        // dimensions. Preserve the ordinary one-position graph there; the
        // qualified GPU verifier below remains packed.
        if Device.defaultDevice().deviceType == .cpu {
            return verifyCPUBatchOutput(inputs, rowCaches: rowCaches, emitMTPState: emitMTPState)
        }
        let working = rowCaches.map { $0.map { $0.copy() } }
        let tokens = MLXArray(inputs.flatMap { $0 }.map(Int32.init)).reshaped(M, 1)
        var h = repeated(
            expandedDimensions(embed(tokens), axis: 2), count: configuration.hcMult, axis: 2)
        var captured = [Int: MLXArray]()
        var prefixes = lengths.map { length in (0 ..< length).map { _ in [KVCache]() } }
        for (index, layer) in layers.enumerated() {
            let result = layer.verifyPacked(
                h, caches: working.map { $0[index] }, rowLengths: lengths, inputIDs: tokens)
            h = result.output
            for row in lengths.indices {
                for column in 0 ..< lengths[row] {
                    prefixes[row][column].append(result.prefixes[row][column])
                }
            }
            if emitMTPState, configuration.dsparkTargetLayerIDs.contains(index) {
                captured[index] = h.mean(axis: 2).reshaped(1, M, configuration.hiddenSize)
            }
        }
        var offsets = [Int]()
        var offset = 0
        for length in lengths {
            offsets.append(offset)
            offset += length
        }
        let collapsed: MLXArray
        // The small float32 HC projection must retain the ordinary GEMV order.
        if M > 1 {
            collapsed = concatenated((0 ..< M).map { hcHead(h[$0 ..< ($0 + 1)]) }, axis: 0)
        } else {
            collapsed = hcHead.decodeBatch(h)
        }
        let result = output(
            logits: deepseekV4PackedLinearF32(head, norm(collapsed)).reshaped(1, M, vocabularySize),
            captured: captured, emitMTPState: emitMTPState)
        return (result, prefixes, offsets)
    }

    private func verifyCPUBatchOutput(
        _ inputs: [[Int]], rowCaches: [[KVCache]], emitMTPState: Bool
    ) -> (output: LMOutput, cachePrefixes: [[[KVCache]]], rowOffsets: [Int]) {
        var logits = [MLXArray]()
        var hidden = [MLXArray]()
        var prefixes = [[[KVCache]]]()
        var offsets = [Int]()
        for row in inputs.indices {
            offsets.append(logits.count)
            let working = rowCaches[row].map { $0.copy() }
            var rowPrefixes = [[KVCache]]()
            for token in inputs[row] {
                let result = forward(
                    MLXArray([Int32(token)]).reshaped(1, 1), cache: working,
                    emitMTPState: emitMTPState)
                logits.append(result.logits)
                if let captured = result.state?[mtpLastHiddenStatesKey] {
                    hidden.append(captured)
                }
                rowPrefixes.append(working.map { $0.copy() })
            }
            prefixes.append(rowPrefixes)
        }
        var state: LMOutput.State?
        if !hidden.isEmpty {
            state = LMOutput.State()
            state?[mtpLastHiddenStatesKey] = concatenated(hidden, axis: 1)
        }
        return (LMOutput(logits: concatenated(logits, axis: 1), state: state), prefixes, offsets)
    }

    /// Float32 logits (`ParallelHead`, `inference/model.py:728-735`:
    /// `F.linear(x.float(), weight)` with a float32 head) from the bf16 head
    /// weight, without a resident float32 copy.
    func logits(_ x: MLXArray) -> MLXArray {
        deepseekV4LinearF32(head, x)
    }

    /// Bound reference-shaped prompt forwards to the qualified 32K context.
    /// Smaller chunks change quantized expert and float32 projection reductions.
    public static let prefillStepSize = 32768

    /// Keep each complete reference chunk intact, including the final chunk.
    /// Return its logits directly so the iterator does not split off one token.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let y = input.text
        let total = y.tokens.size
        guard total > 0 else { return .tokens(y) }
        var processed = 0
        while total - processed > Self.prefillStepSize {
            try Task.checkCancellation()
            let end = processed + Self.prefillStepSize
            autoreleasepool {
                _ = self(y[.newAxis, processed ..< end], cache: cache, state: state)
                eval(cache)
            }
            processed = end
            prefill.progress?(processed, total)
        }
        try Task.checkCancellation()
        let result = self(y[.newAxis, processed...], cache: cache, state: state)
        eval(result.logits)
        eval(cache)
        prefill.progress?(total, total)
        return .logits(result)
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
    /// layout, `language.py:1297-1409`) to module paths: retain configured
    /// DSpark stages, drop layers past `num_hidden_layers`, stack experts into
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
            let parts = key.split(separator: ".").map(String.init)
            if parts.first == "mtp" {
                guard parts.count > 1, let stage = Int(parts[1]),
                    (0 ..< configuration.dsparkStageCount).contains(stage)
                else { continue }
            }
            if parts.count > 1, parts[0] == "layers", let layer = Int(parts[1]),
                layer >= configuration.hiddenLayers
            {
                continue
            }

            // layers.N.ffn.experts.E.wK.suffix
            if parts.count == 7, ["layers", "mtp"].contains(parts[0]),
                parts[2] == "ffn", parts[3] == "experts",
                let expert = Int(parts[4]), let projection = projections[parts[5]]
            {
                let stacked = "\(parts[0]).\(parts[1]).ffn.switch_mlp.\(projection).\(parts[6])"
                experts[stacked, default: [:]][expert] = value
                continue
            }

            var mapped = key
            switch key {
            case "hc_head_fn": mapped = "hc_head.fn"
            case "hc_head_base": mapped = "hc_head.base"
            case "hc_head_scale": mapped = "hc_head.scale"
            default:
                for parameter in ["fn", "base", "scale"] {
                    mapped = mapped.replacingOccurrences(
                        of: ".hc_head_\(parameter)", with: ".hc_head.\(parameter)")
                }
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
