//
//  GLM5Next.swift
//  mlx-swift-lm
//
//  GLM-5-Next (GLM-5.3-Flash) text model: mHC hyper-connections, Kimi Delta
//  Attention (KDA) linear layers, and DeepSeek sparse attention (MLA latent KV
//  plus a pooled-key lightning indexer).
//
//  Port of mlx-vlm v0.7.8 `mlx_vlm/models/glm5_next/language.py`. Module
//  names follow the MLX checkpoints in circulation (separate q/k/v and
//  `forget_gate.*` projections, as in mlx-vlm#2030).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Shared helpers

private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let quantized as QuantizedMultiLinear: quantized(x)
    case let dense as MultiLinear: dense(x)
    default: fatalError("Expected MultiLinear or QuantizedMultiLinear, got \(type(of: module))")
    }
}

/// `post * x + comb^T @ residual`, the residual expansion of one mHC branch.
let glm5NextHCExpand: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { x, residual, post, comb in
        let expanded =
            expandedDimensions(post, axis: -1) * expandedDimensions(x, axis: 2).asType(.float32)
        let mixed = matmul(comb.swappedAxes(-1, -2), residual.asType(.float32))
        return (expanded + mixed).asType(x.dtype)
    }

func makeLimitedSwiGLU(limit: Float?) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    guard let limit else { return compiledSiluProduct }
    return compile(shapeless: true) { gate, up in
        silu(minimum(gate, limit)) * clip(up, min: -limit, max: limit)
    }
}

// MARK: - Hyper-connections (mHC)

/// A decoder input between half-layers during one-token decode: either h
/// itself, or a branch output whose expand `h = expand(y, residual, post,
/// comb)` is still pending (`mm = comb^T @ float(residual)`).
enum GLM5NextHCState {
    case materialized(MLXArray)
    case deferred(y: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray, mm: MLXArray)

    func materialize() -> MLXArray {
        switch self {
        case .materialized(let h): h
        case .deferred(let y, let residual, let post, let comb, _):
            glm5NextHCExpand(y, residual, post, comb)
        }
    }
}

final class GLM5NextHyperConnection: Module {
    let hcMult: Int
    let normEps: Float
    let hcEps: Float
    let sinkhornIterations: Int

    /// Decode (up to 8 tokens per row, any number of rows) runs the mix GEMV
    /// and the Sinkhorn + collapse + branch RMSNorm as two kernels, as the
    /// reference; every token is computed on its own.
    var fusedDecode = true

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    private let sinkhorn: @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)

    convenience init(_ config: GLM5NextConfiguration) {
        self.init(
            hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
            hcEps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
    }

    /// Also DeepSeek-V4's `attn_hc`/`ffn_hc`, which share the math.
    init(hcMult hc: Int, sinkhornIterations iterations: Int, hcEps eps: Float, normEps: Float, hiddenSize: Int) {
        self.hcMult = hc
        self.normEps = normEps
        self.hcEps = eps
        self.sinkhornIterations = iterations

        let mix = (2 + hc) * hc
        _fn.wrappedValue = MLXArray.zeros([mix, hc * hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([3], dtype: .float32)

        self.sinkhorn = compile { mixes, scale, base in
            let mixes = mixes.asType(.float32)
            let scale = scale.asType(.float32)
            let base = base.asType(.float32)
            let pre =
                sigmoid(mixes[.ellipsis, ..<hc] * scale[0] + base[..<hc]) + eps
            let post =
                2 * sigmoid(mixes[.ellipsis, hc ..< (2 * hc)] * scale[1] + base[hc ..< (2 * hc)])
            let leading = Array(mixes.shape.dropLast())
            var comb =
                mixes[.ellipsis, (2 * hc)...].reshaped(leading + [hc, hc]) * scale[2]
                + base[(2 * hc)...].reshaped(hc, hc)
            comb = softmax(comb, axis: -1, precise: true) + eps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            for _ in 0 ..< max(iterations - 1, 0) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            }
            return (pre, post, comb)
        }
        super.init()
    }

    /// Collapse `[B, L, hc, D]` to `[B, L, D]`; returns `(collapsed, post, comb)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = x.asType(.float32)
        let z = MLXFast.rmsNorm(y.reshaped(B, L, H * D), weight: MLXArray.mlxNone, eps: normEps)
        let mixes = matmul(z, fn.T)
        let (pre, post, comb) = sinkhorn(mixes, scale, base)
        let collapsed = (expandedDimensions(pre, axis: -1) * y).sum(axis: 2).asType(x.dtype)
        return (collapsed, post, comb)
    }

    /// `norm(collapsed)`, `post` and `comb` for a branch: the reference
    /// model's decode kernels for one sequence, the op path otherwise.
    func normalizedBranchInput(_ x: MLXArray, norm: RMSNorm) -> (MLXArray, MLXArray, MLXArray) {
        if fusedDecode, !training, x.dim(1) <= 8, hcMult == 4,
            let mixes = glm5NextHCMix(x, fn: fn, eps: normEps),
            let fused = glm5NextHCNorm(
                x, mixes: mixes, scale: scale, base: base, normWeight: norm.weight,
                sinkhornIterations: sinkhornIterations, hcEps: hcEps, normEps: norm.eps)
        {
            return fused
        }
        let (collapsed, post, comb) = self(x)
        return (norm(collapsed), post, comb)
    }

    func deferredDecodeSupported(dtype: DType, norm: RMSNorm) -> Bool {
        fusedDecode && !training && hcMult == 4
            && glm5NextHCDeferredSupported(
                dtype: dtype, width: fn.dim(1) / hcMult, fn: fn, scale: scale, base: base,
                normWeight: norm.weight)
    }

    /// One-token-per-row branch input as `glm5NextHCPre` + `glm5NextHCPost`:
    /// `(normalized, h, post, comb, mm)`, the values of the expand, mix and
    /// Sinkhorn + collapse + RMSNorm chain.
    func deferredBranchInput(_ state: GLM5NextHCState, norm: RMSNorm) -> (
        MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
    ) {
        let (normalized, mixes, h): (MLXArray, MLXArray, MLXArray)
        switch state {
        case .materialized(let x):
            (normalized, mixes, h) = glm5NextHCPre(
                x: x, deferred: nil, fn: fn, scale: scale, base: base, normWeight: norm.weight,
                normEps: normEps, hcEps: hcEps, branchNormEps: norm.eps)
        case .deferred(let y, _, let post, _, let mm):
            (normalized, mixes, h) = glm5NextHCPre(
                x: nil, deferred: (y, mm, post), fn: fn, scale: scale, base: base,
                normWeight: norm.weight, normEps: normEps, hcEps: hcEps,
                branchNormEps: norm.eps)
        }
        let (post, comb, mm) = glm5NextHCPost(
            mixes: mixes, scale: scale, base: base, residual: h,
            sinkhornIterations: sinkhornIterations, hcEps: hcEps)
        // Encode the post kernel before the branch so it runs beside it.
        return (depends(input: normalized, dependencies: [post]), h, post, comb, mm)
    }
}

/// An affine quantized projection's parts for the fused decode kernels.
func glm5NextAffine(_ module: Module) -> GLM5NextAffine? {
    switch module {
    case let switchLinear as QuantizedSwitchLinear:
        let parts = switchLinear.quantizedParts
        guard switchLinear.mode == .affine, parts.bias == nil, let biases = parts.biases
        else { return nil }
        return GLM5NextAffine(
            weight: parts.weight, scales: parts.scales, biases: biases,
            bits: switchLinear.bits, groupSize: switchLinear.groupSize)
    case let linear as QuantizedLinear:
        guard linear.mode == .affine, linear.bias == nil, let biases = linear.biases
        else { return nil }
        return GLM5NextAffine(
            weight: linear.weight, scales: linear.scales, biases: biases,
            bits: linear.bits, groupSize: linear.groupSize)
    default:
        return nil
    }
}

/// Tokens per row of a `[B, L, D]` activation (the rows of a `[T, D]` one
/// count as one-token rows): the fused decode kernels serve rows of up to 8.
func glm5NextDecodeRowLength(_ x: MLXArray) -> Int {
    x.ndim >= 3 ? x.dim(-2) : 1
}

/// `linear(x)` with `glm5NextQMVBatch` when `x` carries 2...8 tokens and the
/// projection is a covered affine one (each weight tile read once for every
/// token; every row bitwise equal to MLX's qmv of that token), else `linear(x)`.
func glm5NextProject(_ linear: Linear, _ x: MLXArray) -> MLXArray {
    let tokens = x.size / x.dim(-1)
    guard (2 ... 8).contains(tokens), let parts = glm5NextAffine(linear),
        let y = glm5NextQMVBatch(x.reshaped(tokens, x.dim(-1)), parts)
    else { return linear(x) }
    return y.reshaped(Array(x.shape.dropLast()) + [y.dim(-1)])
}

// MARK: - Feed-forward

final class GLM5NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(_ config: GLM5NextConfiguration, intermediateSize: Int? = nil) {
        let hidden = intermediateSize ?? config.intermediateSize
        _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
        self.activation = makeLimitedSwiGLU(limit: config.swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        glm5NextProject(
            downProj, activation(glm5NextProject(gateProj, x), glm5NextProject(upProj, x)))
    }
}

final class GLM5NextMoEGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    private let select: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray)
    let topK: Int
    let nGroup: Int
    let scaling: Float
    let normalize: Bool

    /// Rows of up to 8 tokens route with `glm5NextMoERouter` (two kernels),
    /// each token as a one-token GEMV.
    var fusedDecode = true

    init(_ config: GLM5NextConfiguration) {
        let experts = config.nRoutedExperts
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize], dtype: .float32)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([experts], dtype: .float32)

        let topK = config.numExpertsPerTok
        let nGroup = config.nGroup
        let topkGroup = config.topkGroup
        let scaling = config.routedScalingFactor
        let normalize = config.normTopkProb
        self.topK = topK
        self.nGroup = nGroup
        self.scaling = scaling
        self.normalize = normalize
        self.select = compile { logits, bias in
            let scores = sigmoid(logits.asType(.float32))
            var choice = scores + bias
            if nGroup > 1 {
                let shape = choice.shape
                let grouped = choice.reshaped(Array(shape.dropLast()) + [nGroup, -1])
                let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
                let dropCount = nGroup - topkGroup
                if dropCount > 0 {
                    let drop = argPartition(groupScores, kth: dropCount - 1, axis: -1)[
                        .ellipsis, ..<dropCount]
                    var keep = MLXArray.ones(groupScores.shape, dtype: .bool)
                    keep = putAlong(keep, drop, values: MLXArray(false), axis: -1)
                    let expanded = repeated(
                        expandedDimensions(keep, axis: -1), count: grouped.dim(-1), axis: -1
                    ).reshaped(shape)
                    choice = MLX.where(expanded, choice, MLXArray(-Float.infinity))
                }
            }
            let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, indices, axis: -1)
            if topK > 1, normalize {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            return (indices, weights * scaling)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let rows = x.shape.dropLast().reduce(1, *)
        if fusedDecode, !training, nGroup <= 1, glm5NextDecodeRowLength(x) <= 8,
            let (indices, scores) = glm5NextMoERouter(
                x.reshaped(rows, x.dim(-1)), weight: weight, bias: eScoreCorrectionBias,
                topK: topK, scaling: scaling, normalize: normalize)
        {
            let shape = Array(x.shape.dropLast()) + [topK]
            return (indices.reshaped(shape), scores.reshaped(shape))
        }
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        return select(logits, eScoreCorrectionBias)
    }
}

final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLM5NextMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP?
    let limit: Float?

    /// Lab knob (off): several-token decode reads each routed expert's
    /// weights once for every token routed to it (bitwise equal to the
    /// per-token kernels). Slower than per-token on the M5 4-layer cut.
    var shareExpertLoads = false

    init(_ config: GLM5NextConfiguration) {
        _gate.wrappedValue = GLM5NextMoEGate(config)
        self.limit = config.swigluLimit
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activationProduct: makeLimitedSwiGLU(limit: config.swigluLimit))
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = GLM5NextMLP(
                config, intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }

    /// The expert half of decode (rows of up to 8 tokens) as
    /// `glm5NextMoEExperts` (two kernels after the router logits), each token
    /// as a one-token decode; the gate's `fusedDecode` controls it.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fused = fusedDecodeStep(x) {
            return fused
        }
        let (indices, scores) = gate(x)
        let routed = switchMLP(x, indices)
        var y = weightedExpertSum(routed, scores).asType(x.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }

    private func fusedDecodeStep(_ x: MLXArray) -> MLXArray? {
        let rows = x.shape.dropLast().reduce(1, *)
        guard gate.fusedDecode, !gate.training, glm5NextDecodeRowLength(x) <= 8,
            gate.nGroup <= 1, let limit, let sharedExperts
        else { return nil }
        let x2 = x.reshaped(rows, x.dim(-1))
        guard glm5NextMoERouterCovered(x2, weight: gate.weight, bias: gate.eScoreCorrectionBias),
            let gateParts = glm5NextAffine(switchMLP.gateProjection),
            let upParts = glm5NextAffine(switchMLP.upProjection),
            let downParts = glm5NextAffine(switchMLP.downProjection),
            let sharedGate = glm5NextAffine(sharedExperts.gateProj),
            let sharedUp = glm5NextAffine(sharedExperts.upProj),
            let sharedDown = glm5NextAffine(sharedExperts.downProj)
        else { return nil }
        let (sig, biased) = glm5NextRouterLogits(
            x2, weight: gate.weight, bias: gate.eScoreCorrectionBias)
        return glm5NextMoEExperts(
            x2, sig: sig, biased: biased, topK: gate.topK, scaling: gate.scaling,
            normalize: gate.normalize, limit: limit,
            gate: gateParts, up: upParts, down: downParts,
            sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
            shareExpertLoads: shareExpertLoads
        )?.reshaped(x.shape)
    }

}

// MARK: - Kimi Delta Attention (linear layers)

final class GLM5NextGatedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    /// mlx-vlm's `Glm5NextRMSNormGated` op graph, which the fused decode kernel replays.
    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = (xf * xf).mean(axis: -1, keepDims: true)
        let normed = weight.asType(.float32) * (xf * rsqrt(variance + eps))
        return (normed * sigmoid(gate.asType(.float32))).asType(x.dtype)
    }
}

/// `exp(lower * sigmoid(exp(A_log) * (a + dt_bias)))` (safe gate) or
/// `exp(-exp(A_log) * softplus(a + dt_bias))`, per key channel.
private let glm5NextSafeDecay: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { aLog, a, dtBias, lowerBound in
        exp(lowerBound * sigmoid(exp(aLog.asType(.float32)) * (a + dtBias)))
    }

private let glm5NextSoftplusDecay: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { aLog, a, dtBias in
        exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    }

final class GLM5NextForgetGate: Module {
    let numHeads: Int
    let headDim: Int
    let lowerBound: Float?

    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.linearNumHeads
        self.headDim = config.linearHeadDim
        self.lowerBound = config.linearLowerBound
        let projection = numHeads * headDim
        _fAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projection, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([numHeads], dtype: .float32)
        _dtBias.wrappedValue = MLXArray.zeros([projection], dtype: .float32)
        super.init()
    }

    /// The low-rank gate projection `[B, L, H * Dk]`, before the decay.
    func project(_ x: MLXArray) -> MLXArray {
        fBProj(fAProj(x))
    }

    /// Multiplicative per-channel decay `[B, L, H, Dk]` in float32.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let a = project(x).reshaped(B, L, numHeads, headDim)
        let aLog = aLog.reshaped(numHeads, 1)
        let dtBias = dtBias.reshaped(numHeads, headDim)
        if let lowerBound {
            return glm5NextSafeDecay(aLog, a, dtBias, MLXArray(lowerBound))
        }
        return glm5NextSoftplusDecay(aLog, a, dtBias)
    }
}

final class GLM5NextLinearAttention: Module {
    let numHeads: Int
    let headDim: Int
    let projectionDim: Int
    let convKernelSize: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "forget_gate") var forgetGate: GLM5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: GLM5NextGatedRMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.linearNumHeads
        self.headDim = config.linearHeadDim
        self.projectionDim = numHeads * headDim
        self.convKernelSize = config.linearConvKernelDim
        let hidden = config.hiddenSize
        let convDim = 3 * projectionDim

        _qProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _forgetGate.wrappedValue = GLM5NextForgetGate(config)
        _bProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _oNorm.wrappedValue = GLM5NextGatedRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, hidden, bias: false)
        super.init()
    }

    /// Decode (up to 8 tokens per row, any number of rows) runs
    /// `glm5NextKDADecodeStep` (one dispatch for the conv, norms, gates,
    /// delta rule and gated norm of every row).
    var fusedDecode = true

    func callAsFunction(_ x: MLXArray, cache: MambaCache?, checkpointAfter: Int? = nil) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        if let split = checkpointAfter, let cache, L > 1 {
            return verifyColumns(x, cache: cache, checkpointAfter: split)
        }
        if fusedDecode, !training, let cache, L <= 8, let y = fusedDecodeStep(x, cache: cache) {
            return y
        }
        let mixed = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)

        let convState =
            cache?[0]
            ?? MLXArray.zeros([B, convKernelSize - 1, 3 * projectionDim], dtype: x.dtype)
        let convInput = concatenated([convState, mixed], axis: 1)
        let convOut = silu(conv1d(convInput))

        let parts = split(convOut, indices: [projectionDim, 2 * projectionDim], axis: -1)
        let shape = [B, L, numHeads, headDim]
        // mlx-vlm `_l2norm`: x * rsqrt(sum(x^2) + 1e-6) in float32; q also scaled by Dk^-1/2.
        func l2norm(_ x: MLXArray) -> MLXArray {
            x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
        }
        let q =
            (l2norm(parts[0].reshaped(shape).asType(.float32)) * Float(pow(Double(headDim), -0.5)))
            .asType(x.dtype)
        let k = l2norm(parts[1].reshaped(shape).asType(.float32)).asType(x.dtype)
        let v = parts[2].reshaped(shape)

        let decay = forgetGate(x)
        let beta = sigmoid(bProj(x))
        let (out, state) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: decay, beta: beta, state: cache?[1], useKernel: !training)

        if let cache {
            cache[0] = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])
            cache[1] = state
            cache.advance(L)
        }

        let gate = gBProj(gAProj(x)).reshaped(shape)
        return glm5NextProject(oProj, oNorm(out, gate: gate).reshaped(B, L, -1))
    }

    /// MTP verification: every column computed as its one-token decode step
    /// (so each verified column equals the serial decode of that token), and
    /// the state after every kept prefix `k < L` saved for an in-place rewind
    /// (`MambaCache.saveSpeculativePrefixes`). The fused path projects all
    /// columns in one multi-qmv and runs every column in one KDA dispatch;
    /// each prefix state is a lazy replay of that kernel over its first `k`
    /// columns, evaluated only if a rewind restores it. A right-padded column
    /// (prepared lengths) never advances a row's state.
    private func verifyColumns(_ x: MLXArray, cache: MambaCache, checkpointAfter _: Int)
        -> MLXArray
    {
        let (B, L) = (x.dim(0), x.dim(1))
        // Valid columns per row of a right-padded packed verification (`nil`
        // when every column is valid).
        let valid = cache.makeMask(N: L)
        let validCounts = valid.map { $0.asType(.int32).sum(axis: 1) }
        if fusedDecode, !training, let lowerBound = forgetGate.lowerBound, convKernelSize == 4,
            let fb = glm5NextAffine(forgetGate.fBProj), let gb = glm5NextAffine(gBProj),
            let projected = inputProjections(x.reshaped(B * L, 1, x.dim(-1)))
        {
            let parts = projected.map { $0.reshaped(B, L, -1) }
            let convBase =
                cache[0]
                ?? MLXArray.zeros([B, convKernelSize - 1, 3 * projectionDim], dtype: x.dtype)
            let stateBase =
                cache[1] ?? MLXArray.zeros([B, numHeads, headDim, headDim], dtype: .float32)
            func run(_ k: Int) -> (y: MLXArray, convState: MLXArray, state: MLXArray)? {
                func cols(_ a: MLXArray) -> MLXArray { k == L ? a : a[0..., ..<k] }
                return glm5NextKDAVerifyStep(
                    q: cols(parts[0]), k: cols(parts[1]), v: cols(parts[2]), fa: cols(parts[3]),
                    ga: cols(parts[4]), fb: fb, gb: gb, bRaw: cols(parts[5]),
                    convState: convBase, convWeight: conv1d.weight, aLog: forgetGate.aLog,
                    dtBias: forgetGate.dtBias, normWeight: oNorm.weight, state: stateBase,
                    heads: numHeads, headDim: headDim, lowerBound: lowerBound,
                    normEps: oNorm.eps,
                    valid: validCounts.map { k == L ? $0 : minimum($0, Int32(k)) })
            }
            if let step = run(L) {
                cache[0] = step.convState
                cache[1] = step.state
                cache.advance(L)
                var prefixes = [Int: [MLXArray]]()
                for k in 1 ..< L {
                    // Same shapes and widths as the full step, so covered too.
                    let replay = run(k)!
                    prefixes[k] = [replay.convState, replay.state]
                }
                cache.saveSpeculativePrefixes(prefixes, width: L)
                return glm5NextProject(oProj, step.y)
            }
        }
        func keep(_ c: Int, _ new: MLXArray, _ old: MLXArray?) -> MLXArray {
            guard let valid, let old else { return new }
            let shape = [B] + Array(repeating: 1, count: new.ndim - 1)
            return MLX.where(valid[0..., c ..< (c + 1)].reshaped(shape), new, old)
        }
        var outputs = [MLXArray]()
        var prefixes = [Int: [MLXArray]]()
        for c in 0 ..< L {
            let (convBefore, stateBefore) = (cache[0], cache[1])
            outputs.append(self(x[0..., c ..< (c + 1)], cache: cache))
            cache[0] = keep(c, cache[0]!, convBefore)
            cache[1] = keep(c, cache[1]!, stateBefore)
            if c + 1 < L { prefixes[c + 1] = [cache[0]!, cache[1]!] }
        }
        cache.saveSpeculativePrefixes(prefixes, width: L)
        return concatenated(outputs, axis: 1)
    }

    /// One token per row: the q, k, v, f_a, g_a and b projections `[B, 1, N]`,
    /// one `glm5NextMultiQMV` dispatch per (bits, group size); a group the
    /// kernel does not cover runs its modules. `nil` unless every module is
    /// affine.
    private func inputProjections(_ x: MLXArray) -> [MLXArray]? {
        let modules: [Module] = [qProj, kProj, vProj, forgetGate.fAProj, gAProj, bProj]
        let parts = modules.compactMap(glm5NextAffine)
        guard parts.count == modules.count else { return nil }
        let B = x.dim(0)
        let x2 = x.reshaped(B, x.dim(-1))
        var outputs = [MLXArray?](repeating: nil, count: modules.count)
        var groups: [[Int]] = []
        for i in parts.indices {
            if let g = groups.firstIndex(where: {
                (parts[$0[0]].bits, parts[$0[0]].groupSize) == (parts[i].bits, parts[i].groupSize)
            }) {
                groups[g].append(i)
            } else {
                groups.append([i])
            }
        }
        for group in groups {
            if let ys = glm5NextMultiQMV(x2, group.map { parts[$0] }) {
                for (i, y) in zip(group, ys) { outputs[i] = y.reshaped(B, 1, -1) }
            } else {
                for i in group { outputs[i] = (modules[i] as! UnaryLayer)(x) }
            }
        }
        return outputs.map { $0! }
    }

    private func fusedDecodeStep(_ x: MLXArray, cache: MambaCache) -> MLXArray? {
        guard let lowerBound = forgetGate.lowerBound else { return nil }
        let (B, L) = (x.dim(0), x.dim(1))
        let convState =
            cache[0] ?? MLXArray.zeros([B, convKernelSize - 1, 3 * projectionDim], dtype: x.dtype)
        let state =
            cache[1] ?? MLXArray.zeros([B, numHeads, headDim, headDim], dtype: .float32)
        guard convKernelSize == 4 else { return nil }
        let fused: (y: MLXArray, convState: MLXArray, state: MLXArray)?
        if L == 1, let projected = inputProjections(x),
            let fb = glm5NextAffine(forgetGate.fBProj), let gb = glm5NextAffine(gBProj),
            let step = glm5NextKDADecodeStep(
                q: projected[0], k: projected[1], v: projected[2], fa: projected[3],
                ga: projected[4], fb: fb, gb: gb, bRaw: projected[5],
                convState: convState, convWeight: self.conv1d.weight,
                aLog: forgetGate.aLog, dtBias: forgetGate.dtBias, normWeight: oNorm.weight,
                state: state, heads: numHeads, headDim: headDim,
                lowerBound: lowerBound, normEps: oNorm.eps)
        {
            fused = step
        } else {
            fused = glm5NextKDADecodeStep(
                q: qProj(x), k: kProj(x), v: vProj(x),
                aPre: forgetGate.project(x), gatePre: gBProj(gAProj(x)), bRaw: bProj(x),
                convState: convState, convWeight: self.conv1d.weight,
                aLog: forgetGate.aLog, dtBias: forgetGate.dtBias, normWeight: oNorm.weight,
                state: state, heads: numHeads, headDim: headDim,
                lowerBound: lowerBound, normEps: oNorm.eps)
        }
        guard let fused else { return nil }
        cache[0] = fused.convState
        cache[1] = fused.state
        cache.advance(L)
        return glm5NextProject(oProj, fused.y)
    }
}

// MARK: - DeepSeek sparse attention (MLA + lightning indexer)

final class GLM5NextIndexer: Module {
    let numHeads: Int
    let headDim: Int
    let indexTopk: Int
    let kpool: Int
    let alwaysSelectTail: Bool
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var compressApe: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    /// Queries scored per chunk: at most 512, and at most ~2^26 score elements
    /// (`[B, rows, heads, pools]` in float32) so long contexts stay bounded.
    func scoreChunk(poolCount: Int) -> Int {
        max(1, min(512, (1 << 26) / max(numHeads * poolCount, 1)))
    }

    init(_ config: GLM5NextConfiguration) {
        self.numHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.indexTopk = config.indexTopk
        self.kpool = config.indexKpool
        self.alwaysSelectTail = config.indexKpoolAlwaysSelectTail
        self.softmaxScale = pow(Float(headDim), -0.5)
        _wqB.wrappedValue = Linear(config.qLoraRank, numHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, numHeads, bias: false)
        _compressApe.wrappedValue = MLXArray.zeros([kpool, headDim], dtype: .float32)
        _compressGate.wrappedValue = MLXArray.zeros(
            [headDim, config.hiddenSize], dtype: .float32)
        super.init()
    }

    /// Fold this chunk's keys into the pooled-key cache; returns all pooled keys.
    /// `keys`: `kNorm(wk(x))` when the caller computed it for more columns.
    func updatePools(_ x: MLXArray, cache: any GLM5NextLatentCache, keys: MLXArray? = nil)
        -> MLXArray?
    {
        let keys = keys ?? kNorm(wk(x))
        let gates = matmul(x.asType(.float32), compressGate.asType(.float32).T).asType(keys.dtype)
        if let (readyKeys, readyGates) = cache.takeCompleteWindows(keys: keys, gates: gates) {
            cache.appendPooled(compress(keys: readyKeys, gates: readyGates))
        }
        return cache.pooledKeys
    }

    private func compress(keys: MLXArray, gates: MLXArray) -> MLXArray {
        let (B, S, D) = (keys.dim(0), keys.dim(1), keys.dim(2))
        let groupedKeys = keys.reshaped(B, S / kpool, kpool, D)
        let logits = gates.reshaped(B, S / kpool, kpool, D).asType(.float32) + compressApe
        let probs = softmax(logits, axis: 2, precise: true).asType(keys.dtype)
        return (probs * groupedKeys).sum(axis: 2)
    }

    /// Token indices `[B, L, index_topk + kpool - 1]` (`-1` = unused) attended by
    /// the last `L` queries of a `kvLength`-token sequence.
    /// One decode token selects with `glm5NextDSASelect` (two kernels).
    var fusedDecode = true

    /// `q`, `weights`: `wq_b(qResid)` and `weights_proj(x)` when the caller
    /// computed them for more columns.
    func select(
        _ x: MLXArray, qResid: MLXArray, pooledKeys: MLXArray?, kvLength: Int,
        q precomputedQ: MLXArray? = nil, weights precomputedWeights: MLXArray? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        if fusedDecode, !training, B == 1, L == 1, let pooledKeys,
            let fused = glm5NextDSASelect(
                q: precomputedQ ?? wqB(qResid), weights: precomputedWeights ?? weightsProj(x),
                weightScale: pow(Float(numHeads), -0.5) * softmaxScale, pooledKeys: pooledKeys,
                kvLength: kvLength, heads: numHeads, dim: headDim, indexTopk: indexTopk,
                kpool: kpool, tail: alwaysSelectTail)
        {
            return fused
        }
        let start = kvLength - L
        let queryPositions = MLXArray(Int32(start) ..< Int32(kvLength))
        let poolCount = pooledKeys?.dim(1) ?? 0
        let selectK = min(indexTopk / kpool, poolCount)

        var topk: MLXArray
        if let pooledKeys, poolCount > 0 {
            let poolEnd =
                MLXArray(Int32(0) ..< Int32(poolCount)) * Int32(kpool)
                + Int32(kpool - 1)
            let candidates = broadcast(
                expandedDimensions(poolEnd, axes: [0, 1])
                    .<= expandedDimensions(queryPositions, axes: [0, 2]),
                to: [B, L, poolCount])

            var selected: MLXArray
            if selectK == poolCount {
                selected = broadcast(
                    MLXArray(Int32(0) ..< Int32(poolCount)).reshaped(1, 1, poolCount),
                    to: [B, L, poolCount])
            } else {
                let q = (precomputedQ ?? wqB(qResid)).reshaped(B, L, numHeads, headDim)
                    .asType(.float32)
                let weights =
                    (precomputedWeights ?? weightsProj(x)).asType(.float32)
                    * (pow(Float(numHeads), -0.5) * softmaxScale)
                let poolT = expandedDimensions(pooledKeys.asType(.float32), axis: 1)
                    .swappedAxes(-1, -2)
                var chunks = [MLXArray]()
                let chunk = scoreChunk(poolCount: poolCount)
                for c0 in stride(from: 0, to: L, by: chunk) {
                    let c1 = min(c0 + chunk, L)
                    var scores = maximum(matmul(q[0..., c0 ..< c1], poolT), 0)
                    scores = (scores * expandedDimensions(weights[0..., c0 ..< c1], axis: -1))
                        .sum(axis: 2)
                    scores = MLX.where(
                        candidates[0..., c0 ..< c1], scores,
                        MLXArray(-Float.greatestFiniteMagnitude))
                    chunks.append(
                        argPartition(-scores, kth: selectK - 1, axis: -1)[.ellipsis, ..<selectK]
                            .asType(.int32))
                }
                selected = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 1)
            }
            var valid = takeAlong(candidates, selected, axis: -1)

            // argPartition fixes the set, not its order: keep pools chronological
            // with invalid slots last.
            let order = argSort(MLX.where(valid, selected, Int32(poolCount)), axis: -1)
            selected = takeAlong(selected, order, axis: -1)
            valid = takeAlong(valid, order, axis: -1)

            let offsets = MLXArray(Int32(0) ..< Int32(kpool))
            topk = (expandedDimensions(selected, axis: -1) * Int32(kpool) + offsets)
                .reshaped(B, L, selectK * kpool)
            let tokenValid = repeated(expandedDimensions(valid, axis: -1), count: kpool, axis: -1)
                .reshaped(B, L, selectK * kpool)
            topk = MLX.where(tokenValid, topk, Int32(-1))
        } else {
            topk = MLXArray.zeros([B, L, 0], dtype: .int32)
        }

        if topk.dim(-1) < indexTopk {
            let padding = MLXArray.zeros([B, L, indexTopk - topk.dim(-1)], dtype: .int32) - 1
            topk = concatenated([topk, padding], axis: -1)
        }
        topk = topk[.ellipsis, ..<indexTopk]

        if alwaysSelectTail && kpool > 1 {
            // Tokens after the last complete pool each query can see.
            let tailWidth = kpool - 1
            var tail = [Int32]()
            tail.reserveCapacity(L * tailWidth)
            for position in start ..< kvLength {
                let visible = position + 1
                let count = visible % kpool
                for j in 0 ..< tailWidth {
                    let index = visible - count + j
                    tail.append(j < count && index < kvLength ? Int32(index) : -1)
                }
            }
            let tailArray = broadcast(
                MLXArray(tail, [1, L, tailWidth]), to: [B, L, tailWidth])
            topk = concatenated([topk, tailArray], axis: -1)
        }
        return topk.asType(.int32)
    }
}

final class GLM5NextAttention: Module {
    let numHeads: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let indexTopk: Int
    let scale: Float

    /// Skip the indexer's selection while the whole context fits in `index_topk`:
    /// it would select every causal token, so DSA is dense MLA there.
    var denseBypass = true

    /// Queries per gathered sparse-attention block.
    static let sparseChunk = 64

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: GLM5NextIndexer?

    init(_ config: GLM5NextConfiguration, sharedIndexer: Bool) {
        self.numHeads = config.attentionHeads
        self.qHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.indexTopk = config.indexTopk
        self.scale = pow(Float(qHeadDim), -0.5)
        let hidden = config.hiddenSize
        let kvLoraRank = config.kvLoraRank

        _qAProj.wrappedValue = Linear(hidden, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProj.wrappedValue = Linear(hidden, kvLoraRank, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, hidden, bias: config.attentionBias)
        if !sharedIndexer {
            _indexer.wrappedValue = GLM5NextIndexer(config)
        }
        super.init()
    }

    /// Returns the projected output and each row's token selection (`nil` when
    /// dense) so a following `shared` indexer layer can reuse it.
    ///
    /// A `GLM5NextBatchedLatentCache` holds one cache per row of `x`: the
    /// projections run on all rows at once, then each row attends against its
    /// own cache with its own context length (dense/sparse choice, fused
    /// selection, SDPA), exactly as a serial decode of that row. Any other
    /// cache is one `GLM5NextLatentCache` for all of `x`.
    ///
    /// Batched rows match their serial runs bit for bit while a forward
    /// carries at most 8 flattened tokens (rows x tokens per row): past the
    /// MLX core's qmv batch limit (12 on M3 Ultra, 13 on M5 for these
    /// projection sizes) quantized matmuls switch to qmm.
    ///
    /// `perColumn` (MTP verification) attends each row's columns one at a
    /// time, each as a one-token decode step (fused selection, unmasked
    /// SDPA), so a verified column equals the serial decode of that token;
    /// selections are then carried per (row, column). `rowLengths` limits
    /// each row to its leading valid columns (right-padded packed rows); the
    /// padded columns attend nothing and write no cache.
    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, previousTopk: [MLXArray?], perColumn: Bool = false,
        rowLengths: [Int]? = nil
    ) -> (MLXArray, [MLXArray?]) {
        let (B, L) = (x.dim(0), x.dim(1))
        let rows: [any GLM5NextLatentCache]
        var rowLengths = rowLengths
        switch cache {
        case let batch as any GLM5NextBatchedLatentCache:
            precondition(
                batch.rowCaches.count == B,
                "GLM5Next batched cache has \(batch.rowCaches.count) rows for a batch of \(B)")
            rows = batch.rowCaches
            if rowLengths == nil, let packed = batch as? GLM5NextLatentCacheBatch {
                rowLengths = packed.rowLengths
            }
        case let serial as any GLM5NextLatentCache:
            rows = [serial]
        case nil:
            rows = [GLM5NextAttentionCache(kpool: indexer?.kpool ?? 1)]
        default:
            fatalError(
                "GLM5Next sparse layer needs a GLM5NextLatentCache, got \(type(of: cache!))")
        }

        let qResid = qALayerNorm(glm5NextProject(qAProj, x))
        let q = glm5NextProject(qBProj, qResid).reshaped(B, L, numHeads, qHeadDim)
            .transposed(0, 2, 1, 3)
        let latentIn = expandedDimensions(kvALayerNorm(glm5NextProject(kvAProj, x)), axis: 1)
        let qLatent = callMultiLinear(embedQ, q)

        precondition(rowLengths == nil || rowLengths!.count == rows.count)
        if perColumn && rows.count > 1 && B * L <= 8 {
            let (all, selections) = attendPackedColumns(
                x: x, qResid: qResid, qLatent: qLatent, latentIn: latentIn,
                caches: rows, previousTopk: previousTopk, rowLengths: rowLengths)
            let output = callMultiLinear(unembedOut, all)
                .transposed(0, 2, 1, 3)
                .reshaped(B, L, numHeads * vHeadDim)
            return (glm5NextProject(oProj, output), selections)
        }
        var attended = [MLXArray]()
        var selections = [MLXArray?]()
        for (index, rowCache) in rows.enumerated() {
            func row(_ a: MLXArray) -> MLXArray { rows.count == 1 ? a : a[index ..< (index + 1)] }
            let (rx, rq, rl) = (row(x), row(qResid), row(latentIn))
            let rqLatent = row(qLatent)
            if perColumn {
                let valid = rowLengths?[index] ?? L
                // The indexer's projections for every column at once (row-
                // independent); the gates, selection and attention per column.
                let indexerKeys = indexer.map { $0.kNorm(glm5NextProject($0.wk, rx)) }
                let indexerQ = indexer.map { glm5NextProject($0.wqB, rq) }
                let indexerWeights = indexer.map { glm5NextProject($0.weightsProj, rx) }
                var columns = [MLXArray]()
                for c in 0 ..< L {
                    let slot = index * L + c
                    guard c < valid else {
                        // A padded column: attends nothing, writes no cache.
                        columns.append(MLXArray.zeros(like: columns[0]))
                        selections.append(nil)
                        continue
                    }
                    func col(_ a: MLXArray?) -> MLXArray? { a?[0..., c ..< (c + 1)] }
                    let (out, topk) = attendRow(
                        x: rx[0..., c ..< (c + 1)], qResid: rq[0..., c ..< (c + 1)],
                        qLatent: rqLatent[0..., 0..., c ..< (c + 1)],
                        latentIn: rl[0..., 0..., c ..< (c + 1)], cache: rowCache,
                        previousTopk: slot < previousTopk.count ? previousTopk[slot] : nil,
                        indexerInputs: (col(indexerKeys), col(indexerQ), col(indexerWeights)))
                    columns.append(out)
                    selections.append(topk)
                }
                attended.append(columns.count == 1 ? columns[0] : concatenated(columns, axis: 2))
                continue
            }
            let valid = rowLengths?[index] ?? L
            precondition(valid >= 1 && valid <= L, "GLM5Next row length \(valid) outside 1...\(L)")
            let (out, topk) = attendRow(
                x: valid == L ? rx : rx[0..., ..<valid],
                qResid: valid == L ? rq : rq[0..., ..<valid],
                qLatent: valid == L ? rqLatent : rqLatent[0..., 0..., ..<valid],
                latentIn: valid == L ? rl : rl[0..., 0..., ..<valid],
                cache: rowCache, previousTopk: index < previousTopk.count ? previousTopk[index] : nil)
            if valid < L {
                let padding = MLXArray.zeros(
                    [out.dim(0), out.dim(1), L - valid, out.dim(3)], dtype: out.dtype)
                attended.append(concatenated([out, padding], axis: 2))
            } else {
                attended.append(out)
            }
            selections.append(topk)
        }
        let all = attended.count == 1 ? attended[0] : concatenated(attended, axis: 0)
        let output = callMultiLinear(unembedOut, all)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * vHeadDim)
        return (glm5NextProject(oProj, output), selections)
    }

    /// Project the cohort together, then batch compatible one-token attention
    /// calls. Cache updates and selections remain row-local and column-ordered.
    private func attendPackedColumns(
        x: MLXArray, qResid: MLXArray, qLatent: MLXArray, latentIn: MLXArray,
        caches: [any GLM5NextLatentCache], previousTopk: [MLXArray?], rowLengths: [Int]?
    ) -> (MLXArray, [MLXArray?]) {
        let (B, L) = (x.dim(0), x.dim(1))
        let keys = indexer.map { $0.kNorm(glm5NextProject($0.wk, x)) }
        let queries = indexer.map { glm5NextProject($0.wqB, qResid) }
        let weights = indexer.map { glm5NextProject($0.weightsProj, x) }
        var outputs = Array(repeating: [MLXArray](), count: B)
        var selections = [MLXArray?](repeating: nil, count: B * L)
        for c in 0 ..< L {
            var active = [Int]()
            var latents = [MLXArray]()
            var topks = [MLXArray?]()
            var qs = [MLXArray]()
            for r in 0 ..< B {
                let valid = rowLengths?[r] ?? L
                precondition(valid >= 1 && valid <= L)
                guard c < valid else {
                    outputs[r].append(MLXArray.zeros(like: outputs[r][0]))
                    continue
                }
                func col(_ a: MLXArray) -> MLXArray { a[r ..< (r + 1), c ..< (c + 1)] }
                let slot = r * L + c
                let prepared = prepareAttention(
                    x: col(x), qResid: col(qResid),
                    latentIn: latentIn[r ..< (r + 1), 0..., c ..< (c + 1)],
                    cache: caches[r], previousTopk: slot < previousTopk.count ? previousTopk[slot] : nil,
                    indexerInputs: (keys.map(col), queries.map(col), weights.map(col)))
                active.append(r)
                latents.append(prepared.latent)
                topks.append(prepared.topk)
                qs.append(qLatent[r ..< (r + 1), 0..., c ..< (c + 1)])
                selections[slot] = prepared.topk
            }
            // Equal shapes retain the same unpadded attention and reduction
            // route as a serial token. Ragged contexts form separate groups.
            var groups = [[Int]]()
            for i in active.indices {
                if let g = groups.firstIndex(where: {
                    latents[$0[0]].shape == latents[i].shape
                        && topks[$0[0]]?.shape == topks[i]?.shape
                }) {
                    groups[g].append(i)
                } else {
                    groups.append([i])
                }
            }
            for group in groups {
                let q = group.count == 1 ? qs[group[0]] : concatenated(group.map { qs[$0] }, axis: 0)
                let latent = group.count == 1
                    ? latents[group[0]] : concatenated(group.map { latents[$0] }, axis: 0)
                let out: MLXArray
                if let first = topks[group[0]] {
                    let topk = group.count == 1
                        ? first : concatenated(group.map { topks[$0]! }, axis: 0)
                    out = sparseAttention(q, latent: latent, topk: topk)
                } else {
                    out = MLXFast.scaledDotProductAttention(
                        queries: q, keys: latent, values: latent, scale: scale, mask: .none)
                }
                for (index, i) in group.enumerated() {
                    outputs[active[i]].append(
                        group.count == 1 ? out : out[index ..< (index + 1)])
                }
            }
        }
        let attended = outputs.map { $0.count == 1 ? $0[0] : concatenated($0, axis: 2) }
        return (concatenated(attended, axis: 0), selections)
    }

    /// One cache's attention: append the latent, update the indexer's pools,
    /// select (or attend densely while the context fits in `index_topk`).
    private func attendRow(
        x: MLXArray, qResid: MLXArray, qLatent: MLXArray, latentIn: MLXArray,
        cache: any GLM5NextLatentCache, previousTopk: MLXArray?,
        indexerInputs: (keys: MLXArray?, q: MLXArray?, weights: MLXArray?) = (nil, nil, nil)
    ) -> (MLXArray, MLXArray?) {
        let L = x.dim(1)
        let (latent, topk) = prepareAttention(
            x: x, qResid: qResid, latentIn: latentIn, cache: cache,
            previousTopk: previousTopk, indexerInputs: indexerInputs)
        if let topk {
            return (sparseAttention(qLatent, latent: latent, topk: topk), topk)
        }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
            L == 1 ? .none : .array(createCausalMask(n: L, offset: cache.offset - L))
        let out = MLXFast.scaledDotProductAttention(
            queries: qLatent, keys: latent, values: latent, scale: scale, mask: mask)
        return (out, nil)
    }

    private func prepareAttention(
        x: MLXArray, qResid: MLXArray, latentIn: MLXArray,
        cache: any GLM5NextLatentCache, previousTopk: MLXArray?,
        indexerInputs: (keys: MLXArray?, q: MLXArray?, weights: MLXArray?)
    ) -> (latent: MLXArray, topk: MLXArray?) {
        let latent = cache.appendLatent(latentIn)
        let kvLength = cache.offset
        let dense = denseBypass && kvLength <= indexTopk

        var topk: MLXArray?
        if let indexer {
            let pooled = indexer.updatePools(x, cache: cache, keys: indexerInputs.keys)
            if !dense {
                topk = indexer.select(
                    x, qResid: qResid, pooledKeys: pooled, kvLength: kvLength,
                    q: indexerInputs.q, weights: indexerInputs.weights)
            }
        } else if !dense {
            guard let previousTopk else {
                fatalError("GLM5Next shared-indexer layer has no previous top-k selection")
            }
            topk = previousTopk
        }

        return (latent, topk)
    }

    /// Absorbed-MLA attention over each query's selected latent rows.
    /// `qLatent: [B, H, L, R]`, `latent: [B, 1, T, R]`, `topk: [B, L, K]`.
    private func sparseAttention(_ qLatent: MLXArray, latent: MLXArray, topk: MLXArray)
        -> MLXArray
    {
        let (B, H, L, R) = (qLatent.dim(0), qLatent.dim(1), qLatent.dim(2), qLatent.dim(3))
        let T = latent.dim(2)
        let K = topk.dim(-1)
        let flatLatent = latent.reshaped(B * T, R)
        let batchOffsets = (MLXArray(Int32(0) ..< Int32(B)) * Int32(T)).reshaped(B, 1, 1)

        var outputs = [MLXArray]()
        for c0 in stride(from: 0, to: L, by: Self.sparseChunk) {
            let c1 = min(c0 + Self.sparseChunk, L)
            let rows = c1 - c0
            let indices = topk[0..., c0 ..< c1]
            let valid = logicalAnd(indices .>= 0, indices .< Int32(T))
            let safe = clip(indices, min: 0, max: max(T - 1, 0)) + batchOffsets
            let selected = take(flatLatent, safe.flattened(), axis: 0)
                .reshaped(B * rows, 1, K, R)
            let queries = qLatent[0..., 0..., c0 ..< c1]
                .transposed(0, 2, 1, 3)
                .reshaped(B * rows, H, 1, R)
            let out = MLXFast.scaledDotProductAttention(
                queries: queries, keys: selected, values: selected, scale: scale,
                mask: .array(valid.reshaped(B * rows, 1, 1, K)))
            outputs.append(out.reshaped(B, rows, H, R).transposed(0, 2, 1, 3))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }
}

// MARK: - Decoder

final class GLM5NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: GLM5NextHyperConnection

    init(_ config: GLM5NextConfiguration, layerIdx: Int) {
        self.isLinear = config.layerTypes[layerIdx] == "linear_attention"
        if isLinear {
            _selfAttn.wrappedValue = GLM5NextLinearAttention(config)
        } else {
            _selfAttn.wrappedValue = GLM5NextAttention(
                config, sharedIndexer: config.indexerTypes[layerIdx] == "shared")
        }
        if config.mlpLayerTypes[layerIdx] == "sparse" {
            _mlp.wrappedValue = GLM5NextMoE(config)
        } else {
            _mlp.wrappedValue = GLM5NextMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnHC.wrappedValue = GLM5NextHyperConnection(config)
        _ffnHC.wrappedValue = GLM5NextHyperConnection(config)
        super.init()
    }

    /// `x: [B, L, hc_mult, D]`.
    /// `checkpointAfter` marks MTP verification: attention runs column by
    /// column (see `GLM5NextAttention`, `GLM5NextLinearAttention`).
    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, previousTopk: [MLXArray?], checkpointAfter: Int? = nil
    ) -> (MLXArray, [MLXArray?]) {
        let (normed, post, comb) = attnHC.normalizedBranchInput(x, norm: inputLayerNorm)
        let attention: MLXArray
        var topk = previousTopk
        if isLinear {
            attention = (selfAttn as! GLM5NextLinearAttention)(
                normed, cache: linearCache(cache), checkpointAfter: checkpointAfter)
        } else {
            (attention, topk) = (selfAttn as! GLM5NextAttention)(
                normed, cache: cache, previousTopk: previousTopk,
                perColumn: checkpointAfter != nil && x.dim(1) > 1)
        }
        let h = glm5NextHCExpand(attention, x, post, comb)

        let (ffnNormed, ffnPost, ffnComb) = ffnHC.normalizedBranchInput(
            h, norm: postAttentionLayerNorm)
        let ffnOut = (mlp as! UnaryLayer)(ffnNormed)
        return (glm5NextHCExpand(ffnOut, h, ffnPost, ffnComb), topk)
    }

    func deferredDecodeSupported(dtype: DType) -> Bool {
        attnHC.deferredDecodeSupported(dtype: dtype, norm: inputLayerNorm)
            && ffnHC.deferredDecodeSupported(dtype: dtype, norm: postAttentionLayerNorm)
    }

    /// One-token decode with both hyper-connection expands deferred into the
    /// next half-layer's pre kernel.
    func decodeDeferred(
        _ state: GLM5NextHCState, cache: KVCache?, previousTopk: [MLXArray?],
        checkpointAfter: Int? = nil
    ) -> (GLM5NextHCState, [MLXArray?]) {
        let (normed, h, post, comb, mm) = attnHC.deferredBranchInput(state, norm: inputLayerNorm)
        let attention: MLXArray
        var topk = previousTopk
        if isLinear {
            attention = (selfAttn as! GLM5NextLinearAttention)(
                normed, cache: linearCache(cache), checkpointAfter: checkpointAfter)
        } else {
            (attention, topk) = (selfAttn as! GLM5NextAttention)(
                normed, cache: cache, previousTopk: previousTopk,
                perColumn: checkpointAfter != nil && normed.dim(1) > 1)
        }
        let (ffnNormed, ffnH, ffnPost, ffnComb, ffnMM) = ffnHC.deferredBranchInput(
            .deferred(y: attention, residual: h, post: post, comb: comb, mm: mm),
            norm: postAttentionLayerNorm)
        let ffnOut = (mlp as! UnaryLayer)(ffnNormed)
        return (
            .deferred(y: ffnOut, residual: ffnH, post: ffnPost, comb: ffnComb, mm: ffnMM), topk
        )
    }

    private func linearCache(_ cache: KVCache?) -> MambaCache? {
        guard let cache else { return nil }
        guard let mamba = cache as? MambaCache else {
            fatalError("GLM5Next linear layer needs MambaCache, got \(type(of: cache))")
        }
        return mamba
    }
}

// MARK: - Multi-token prediction head (loaded, not run)

final class GLM5NextMTPBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GLM5NextAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM5NextConfiguration) {
        _selfAttn.wrappedValue = GLM5NextAttention(config, sharedIndexer: false)
        if config.nRoutedExperts > 0 {
            _mlp.wrappedValue = GLM5NextMoE(config)
        } else {
            _mlp.wrappedValue = GLM5NextMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// The checkpoint's nextn layer: `enorm`/`hnorm` fusion, a plain (no mHC)
/// decoder block, and the shared-head `norm`. Kept for a later MTP drafter.
public final class GLM5NextMTPLayer: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "block") var block: GLM5NextMTPBlock

    init(_ config: GLM5NextConfiguration) {
        let dim = config.hiddenSize
        _enorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _ehProj.wrappedValue = Linear(2 * dim, dim, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _block.wrappedValue = GLM5NextMTPBlock(config)
        super.init()
    }
}

// MARK: - Model

public final class GLM5NextModelInner: Module {
    let hcMult: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [GLM5NextDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM5NextConfiguration) {
        self.hcMult = config.hcMult
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            GLM5NextDecoderLayer(config, layerIdx: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Decode of up to 8 tokens per row (any number of rows; each token on
    /// its own) passes deferred hyper-connection state between half-layers
    /// (`GLM5NextDecoderLayer.decodeDeferred`) where MLX computes the expand's
    /// comb product without NAX (`glm5NextHCDeferredExact`).
    var deferredDecode = true

    /// During decode and verify forwards (up to 8 tokens per row), start
    /// evaluating the hidden state every this many layers while the rest of
    /// the graph is built (oMLX's staged eval; 0 = off). Scheduling only;
    /// values are unchanged. On the M3 Ultra it cut a plain step from 28.6 to
    /// 25.3 ms.
    var stagedEvalLayers = 8

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        norm(hidden(inputs, cache: cache))
    }

    /// The hidden state after the last layer, streams averaged, before the
    /// final norm `[B, L, D]`: what the MTP head consumes. `checkpointAfter`
    /// runs an MTP verification (every column as a one-token decode step,
    /// recurrent checkpoint after that many columns).
    func hidden(_ inputs: MLXArray, cache: [KVCache]?, checkpointAfter: Int? = nil) -> MLXArray {
        var h = repeated(expandedDimensions(embedTokens(inputs), axis: 2), count: hcMult, axis: 2)

        if deferredDecode, glm5NextHCDeferredExact, inputs.dim(1) <= 8,
            layers.allSatisfy({ $0.deferredDecodeSupported(dtype: h.dtype) })
        {
            var state = GLM5NextHCState.materialized(h)
            var topk = [MLXArray?]()
            for (index, layer) in layers.enumerated() {
                (state, topk) = layer.decodeDeferred(
                    state, cache: cache?[index], previousTopk: topk,
                    checkpointAfter: checkpointAfter)
                if stagedEvalLayers > 0, (index + 1) % stagedEvalLayers == 0,
                    index + 1 < layers.count, case .deferred(let y, _, _, _, let mm) = state
                {
                    asyncEval(y, mm)
                }
            }
            return state.materialize().mean(axis: 2)
        }

        var topk = [MLXArray?]()
        for (index, layer) in layers.enumerated() {
            (h, topk) = layer(
                h, cache: cache?[index], previousTopk: topk, checkpointAfter: checkpointAfter)
            if stagedEvalLayers > 0, inputs.dim(1) <= 8, (index + 1) % stagedEvalLayers == 0,
                index + 1 < layers.count
            {
                asyncEval(h)
            }
        }
        return h.mean(axis: 2)
    }
}

/// Text model for `glm5_next_text` checkpoints (`model.*`, `lm_head.*`, `mtp.*`).
public final class GLM5NextTextModel: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: GLM5NextConfiguration

    @ModuleInfo(key: "model") public var model: GLM5NextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    /// Multi-token-prediction layers, or a parameterless placeholder per layer
    /// when the checkpoint ships none.
    @ModuleInfo(key: "mtp") public var mtp: [Module]

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _model.wrappedValue = GLM5NextModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        _mtp.wrappedValue = (0 ..< configuration.numNextnPredictLayers).map { _ in
            GLM5NextMTPLayer(configuration)
        }
        super.init()
    }

    /// Whether real MTP weights were loaded.
    public var hasMTPWeights: Bool {
        !mtp.isEmpty && mtp.allSatisfy { $0 is GLM5NextMTPLayer }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model(inputs, cache: cache))
    }

    func logits(_ normalized: MLXArray) -> MLXArray {
        if let lmHead {
            return glm5NextProject(lmHead, normalized)
        }
        return model.embedTokens.asLinear(normalized)
    }

    /// With `mtpEmitFlagKey` set: the pre-final-norm hidden state the MTP head
    /// consumes (`mtpLastHiddenStatesKey`, `[B, L, D]`) and the last sparse
    /// layer's latent as `sharedKV["full_attention"]` (keys and values both
    /// `[B, 1, T, kv_lora_rank]`, rows right-padded to the longest). A
    /// `mtpCacheCheckpointIndexKey` makes the forward an MTP verification:
    /// each column runs as a one-token decode step and every KDA cache keeps
    /// its state after that many columns as the speculative checkpoint.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: self(input.tokens, cache: cache))
        }
        let hidden = model.hidden(
            input.tokens, cache: cache, checkpointAfter: state?[mtpCacheCheckpointIndexKey])
        var outState = state ?? LMOutput.State()
        outState[mtpLastHiddenStatesKey] = hidden
        if let cache, let source = model.layers.lastIndex(where: { !$0.isLinear }),
            let (latent, offset) = glm5NextSharedLatent(cache[source])
        {
            outState[mtpSharedKVStatesKey] = ["full_attention": (latent, latent)]
            outState[mtpSharedKVOffsetsKey] = ["full_attention": offset]
            outState[mtpSharedKVSourceIndicesKey] = ["full_attention": source]
        }
        return LMOutput(logits: logits(model.norm(hidden)), state: outState)
    }

    /// One cache per decoder layer. Sparse-attention layers keep the whole
    /// context, so `maxKVSize` and KV quantization do not apply to them.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { layer -> KVCache in
            if layer.isLinear {
                return MambaCache()
            }
            let attention = layer.selfAttn as! GLM5NextAttention
            return GLM5NextAttentionCache(
                kpool: attention.indexer?.kpool ?? configuration.indexKpool)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = GLM5NextWeights.sanitize(weights, configuration: configuration)
        if configuration.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("lm_head.") }
        }
        let hasMTP = weights.keys.contains { $0.hasPrefix("mtp.") }
        if !hasMTP && !mtp.isEmpty {
            _ = update(
                modules: ModuleChildren.unflattened(mtp.indices.map { ("mtp.\($0)", Module()) }))
        }
        return weights
    }

    public var loraLayers: [Module] { model.layers }
}

/// A sparse layer cache's latent `[B, 1, T, R]` (batched rows right-padded to
/// the longest) and its offset (the longest row's), or `nil` while empty.
func glm5NextSharedLatent(_ cache: KVCache) -> (MLXArray, Int)? {
    let rows: [any GLM5NextLatentCache]
    switch cache {
    case let batch as any GLM5NextBatchedLatentCache: rows = batch.rowCaches
    case let serial as any GLM5NextLatentCache: rows = [serial]
    default: return nil
    }
    let latents = rows.map(\.validLatent)
    guard let longest = rows.map(\.offset).max(), longest > 0,
        let template = latents.compactMap({ $0 }).first
    else { return nil }
    if rows.count == 1, let latent = latents[0] { return (latent, longest) }
    let padded = latents.map { latent -> MLXArray in
        let length = latent?.dim(2) ?? 0
        let pad = MLXArray.zeros(
            [1, template.dim(1), longest - length, template.dim(3)], dtype: template.dtype)
        guard let latent else { return pad }
        return length == longest ? latent : concatenated([latent, pad], axis: 2)
    }
    return (concatenated(padded, axis: 0), longest)
}

extension GLM5NextTextModel: SpeculativeCacheRewindModel {
    /// Up to 7 verified tokens are rewound in place (a verify carries at most
    /// 8 columns): sparse layers trim exactly, KDA layers restore the state
    /// of the kept prefix the verify saved.
    public var maximumNativeTargetCacheRewind: Int { 7 }
}

/// `glm5_next` checkpoints: the text model nested under `language_model`, with
/// vision tensors alongside (dropped).
public final class GLM5NextModel: Module, LLMModel {
    public let vocabularySize: Int

    @ModuleInfo(key: "language_model") public var languageModel: GLM5NextTextModel

    public init(_ configuration: GLM5NextConfiguration) {
        self.vocabularySize = configuration.vocabularySize
        _languageModel.wrappedValue = GLM5NextTextModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        languageModel(input, cache: cache, state: state)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let text = languageModel.sanitize(weights: GLM5NextWeights.textKeys(weights))
        return Dictionary(uniqueKeysWithValues: text.map { ("language_model." + $0.key, $0.value) })
    }

    public var loraLayers: [Module] { languageModel.loraLayers }
}

// MARK: - Weight sanitization

enum GLM5NextWeights {
    /// Map any supported checkpoint layout to text-model keys (`model.*`,
    /// `lm_head.*`, `mtp.*`) and drop vision tensors.
    static func textKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if let mapped = textKey(key) {
                result[mapped] = value
            }
        }
        return result
    }

    private static func textKey(_ key: String) -> String? {
        let visionPrefixes = ["vision_model.", "vision_tower.", "visual.", "model.visual."]
        if visionPrefixes.contains(where: { key.hasPrefix($0) }) {
            return nil
        }
        if key.hasPrefix("model.language_model.") {
            return "model." + key.dropFirst("model.language_model.".count)
        }
        if key.hasPrefix("language_model.") {
            return String(key.dropFirst("language_model.".count))
        }
        return key
    }

    static func sanitize(
        _ weights: [String: MLXArray], configuration config: GLM5NextConfiguration
    ) -> [String: MLXArray] {
        var weights = textKeys(weights)
        weights = remapNextnLayers(weights, config: config)
        weights = renameParts(weights)
        stackExperts(&weights)
        splitKVB(&weights, config: config)
        dequantizeRouterGates(&weights, hiddenSize: config.hiddenSize)

        if config.numNextnPredictLayers == 0 {
            weights = weights.filter { !$0.key.hasPrefix("mtp.") }
        }
        for (key, value) in weights where keepsFloat32(key) && value.dtype != .float32 {
            weights[key] = value.asType(.float32)
        }
        return weights
    }

    /// DeepSeek-style nextn layers stored as `model.layers.<L + i>.*` become `mtp.<i>.*`.
    private static func remapNextnLayers(
        _ weights: [String: MLXArray], config: GLM5NextConfiguration
    ) -> [String: MLXArray] {
        let fusion = [
            "eh_proj.": "eh_proj.", "enorm.": "enorm.", "hnorm.": "hnorm.",
            "shared_head.norm.": "norm.",
        ]
        var result = [String: MLXArray]()
        for (key, value) in weights {
            guard key.hasPrefix("model.layers.") else {
                result[key] = value
                continue
            }
            let rest = key.dropFirst("model.layers.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else {
                result[key] = value
                continue
            }
            guard index >= config.hiddenLayers else {
                result[key] = value
                continue
            }
            let nextn = index - config.hiddenLayers
            guard nextn < config.numNextnPredictLayers else { continue }
            let suffix = String(rest[rest.index(after: dot)...])
            if suffix.hasPrefix("shared_head.head.") || suffix.hasPrefix("embed_tokens.") {
                continue
            }
            if let (source, target) = fusion.first(where: { suffix.hasPrefix($0.key) }) {
                result["mtp.\(nextn).\(target)\(suffix.dropFirst(source.count))"] = value
            } else {
                result["mtp.\(nextn).block.\(suffix)"] = value
            }
        }
        return result
    }

    /// Hugging Face names to module names: `hc_attn_*`, top-level forget-gate
    /// parameters, and per-projection short convolutions.
    private static func renameParts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let forgetGateParts = [
            "A_log", "dt_bias",
            "f_a_proj.weight", "f_a_proj.scales", "f_a_proj.biases",
            "f_b_proj.weight", "f_b_proj.scales", "f_b_proj.biases",
        ]
        var result = [String: MLXArray]()
        var convParts = [String: [Character: MLXArray]]()
        for (key, value) in weights {
            var key =
                key
                .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
                .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")
            if let part = ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"]
                .first(where: { key.hasSuffix(".self_attn." + $0) })
            {
                let prefix = String(key.dropLast(part.count))
                convParts[prefix, default: [:]][part.first!] = value
                continue
            }
            if let part = forgetGateParts.first(where: { key.hasSuffix(".self_attn." + $0) }) {
                key = String(key.dropLast(part.count)) + "forget_gate." + part
            }
            result[key] = value
        }
        for (prefix, parts) in convParts {
            if let q = parts["q"], let k = parts["k"], let v = parts["v"] {
                result[prefix + "conv1d.weight"] = concatenated([q, k, v], axis: 0)
            } else {
                for (name, value) in parts {
                    result[prefix + "\(name)_conv1d.weight"] = value
                }
            }
        }
        for (key, value) in result
        where key.hasSuffix("conv1d.weight") && value.ndim == 3 && value.dim(-1) != 1 {
            result[key] = value.movedAxis(source: 2, destination: 1)
        }
        return result
    }

    /// `<prefix>.mlp.experts.<e>.<proj>.<part>` to `<prefix>.mlp.switch_mlp.<proj>.<part>`.
    private static func stackExperts(_ weights: inout [String: MLXArray]) {
        var groups = [String: [Int: String]]()
        for key in weights.keys {
            guard let range = key.range(of: ".mlp.experts.") else { continue }
            let rest = key[range.upperBound...]
            guard let dot = rest.firstIndex(of: "."), let expert = Int(rest[..<dot]) else {
                continue
            }
            let target =
                key[..<range.lowerBound] + ".mlp.switch_mlp." + rest[rest.index(after: dot)...]
            groups[String(target), default: [:]][expert] = key
        }
        for (target, members) in groups {
            let ordered = members.keys.sorted()
            guard ordered == Array(0 ..< ordered.count) else { continue }
            weights[target] = stacked(ordered.map { weights.removeValue(forKey: members[$0]!)! })
        }
    }

    /// `kv_b_proj` (`[H * (nope + v), kv_lora_rank]`) to `embed_q` / `unembed_out`.
    private static func splitKVB(_ weights: inout [String: MLXArray], config: GLM5NextConfiguration)
    {
        let heads = config.attentionHeads
        let nope = config.qkNopeHeadDim
        let rank = config.kvLoraRank
        for key in Array(weights.keys) where key.hasSuffix(".self_attn.kv_b_proj.weight") {
            let prefix = String(key.dropLast("kv_b_proj.weight".count))
            var value = weights.removeValue(forKey: key)!
            var quantization: (bits: Int, groupSize: Int)?
            if let scales = weights.removeValue(forKey: prefix + "kv_b_proj.scales") {
                let biases = weights.removeValue(forKey: prefix + "kv_b_proj.biases")
                let bits = value.dim(-1) * 32 / rank
                let groupSize = rank / scales.dim(-1)
                value = dequantized(
                    value, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
                quantization = (bits, groupSize)
            }
            value = value.reshaped(heads, nope + config.vHeadDim, rank)
            let wk = contiguous(value[0..., ..<nope, 0...].swappedAxes(-1, -2))
            let wv = contiguous(value[0..., nope..., 0...])
            for (name, matrix) in [("embed_q", wk), ("unembed_out", wv)] {
                if let quantization {
                    let (q, s, b) = MLX.quantized(
                        matrix, groupSize: quantization.groupSize, bits: quantization.bits)
                    weights[prefix + name + ".weight"] = q
                    weights[prefix + name + ".scales"] = s
                    weights[prefix + name + ".biases"] = b
                } else {
                    weights[prefix + name + ".weight"] = matrix
                }
            }
        }
    }

    /// The router has no quantized module; restore an affine-quantized gate to float32.
    private static func dequantizeRouterGates(_ weights: inout [String: MLXArray], hiddenSize: Int)
    {
        for key in Array(weights.keys) where key.hasSuffix(".mlp.gate.scales") {
            let prefix = String(key.dropLast("scales".count))
            guard let packed = weights[prefix + "weight"] else { continue }
            let scales = weights.removeValue(forKey: key)!
            let biases = weights.removeValue(forKey: prefix + "biases")
            weights[prefix + "weight"] = dequantized(
                packed, scales: scales, biases: biases,
                groupSize: hiddenSize / scales.dim(-1),
                bits: packed.dim(-1) * 32 / hiddenSize
            ).asType(.float32)
        }
    }

    private static func keepsFloat32(_ key: String) -> Bool {
        key.contains(".attn_hc.") || key.contains(".ffn_hc.")
            || key.hasSuffix("A_log") || key.hasSuffix("dt_bias")
            || key.hasSuffix("mlp.gate.weight") || key.hasSuffix("e_score_correction_bias")
    }
}

// MARK: - Configuration

public struct GLM5NextConfiguration: Codable, Sendable {
    struct LinearAttentionConfiguration: Codable, Sendable {
        var numHeads: Int?
        var headDim: Int?
        var shortConvKernelSize: Int?
        var gateLowerBound: Float?
        var safeGate: Bool?
        var kdaLayers: [Int]?
        var fullAttnLayers: [Int]?

        enum CodingKeys: String, CodingKey {
            case numHeads = "num_heads"
            case headDim = "head_dim"
            case shortConvKernelSize = "short_conv_kernel_size"
            case gateLowerBound = "gate_lower_bound"
            case safeGate = "safe_gate"
            case kdaLayers = "kda_layers"
            case fullAttnLayers = "full_attn_layers"
        }
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var numNextnPredictLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int
    var nRoutedExperts: Int
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var normTopkProb: Bool
    var rmsNormEps: Float
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var indexTopk: Int
    var indexHeadDim: Int
    var indexNHeads: Int
    var indexKpool: Int
    var indexKpoolAlwaysSelectTail: Bool
    var layerTypes: [String]
    var mlpLayerTypes: [String]
    var indexerTypes: [String]
    var swigluLimit: Float?
    var linearHeadDim: Int
    var linearNumHeads: Int
    var linearConvKernelDim: Int
    var linearLowerBound: Float?
    var hcMult: Int
    var hcEps: Float
    var hcSinkhornIters: Int
    var firstKDenseReplace: Int
    var scoringFunc: String
    var topkMethod: String
    var mlaUseNope: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case indexTopk = "index_topk"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexKpool = "index_kpool"
        case indexKpoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case indexerTypes = "indexer_types"
        case swigluLimit = "swiglu_limit"
        case linearHeadDim = "linear_head_dim"
        case linearNumHeads = "linear_num_heads"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearLowerBound = "linear_lower_bound"
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case firstKDenseReplace = "first_k_dense_replace"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case mlaUseNope = "mla_use_nope"
    }

    private enum ExtraKeys: String, CodingKey {
        case textConfig = "text_config"
        case linearAttnConfig = "linear_attn_config"
        case indexTopkPattern = "index_topk_pattern"
        case indexTopkFreq = "index_topk_freq"
        case indexSkipTopkOffset = "index_skip_topk_offset"
    }

    public init(from decoder: Decoder) throws {
        // glm5_next nests the text model under text_config; glm5_next_text is flat.
        let outer = try decoder.container(keyedBy: ExtraKeys.self)
        let nested = outer.contains(.textConfig)
        let c =
            nested
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)
        let extra =
            nested ? try outer.nestedContainer(keyedBy: ExtraKeys.self, forKey: .textConfig) : outer

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm5_next_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 154880
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 45
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 288
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1536
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 0
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 256
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 256
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexKpool = try c.decodeIfPresent(Int.self, forKey: .indexKpool) ?? 16
        indexKpoolAlwaysSelectTail =
            try c.decodeIfPresent(Bool.self, forKey: .indexKpoolAlwaysSelectTail) ?? true
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        linearHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearHeadDim) ?? 128
        linearNumHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumHeads) ?? 64
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        linearLowerBound =
            c.contains(.linearLowerBound)
            ? try c.decodeIfPresent(Float.self, forKey: .linearLowerBound) : -5.0
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 3
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? true

        let linear = try extra.decodeIfPresent(
            LinearAttentionConfiguration.self, forKey: .linearAttnConfig)
        if let linear {
            linearHeadDim = linear.headDim ?? linearHeadDim
            linearNumHeads = linear.numHeads ?? linearNumHeads
            linearConvKernelDim = linear.shortConvKernelSize ?? linearConvKernelDim
            linearLowerBound = linear.gateLowerBound ?? linearLowerBound
            if linear.safeGate ?? true, linearLowerBound == nil {
                linearLowerBound = -5.0
            }
        }

        let layers = hiddenLayers
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else if let kda = linear?.kdaLayers {
            let kdaSet = Set(kda)
            layerTypes = (0 ..< layers).map {
                kdaSet.contains($0) ? "linear_attention" : "deepseek_sparse_attention"
            }
        } else {
            layerTypes = (0 ..< layers).map {
                $0 % 4 != 3 ? "linear_attention" : "deepseek_sparse_attention"
            }
        }
        layerTypes = layerTypes.map { $0 == "full_attention" ? "deepseek_sparse_attention" : $0 }

        if let types = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = types
        } else {
            let dense = min(firstKDenseReplace, layers)
            mlpLayerTypes =
                Array(repeating: "dense", count: dense)
                + Array(repeating: "sparse", count: layers - dense)
        }

        if let types = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = types
        } else if let pattern = try? extra.decodeIfPresent(String.self, forKey: .indexTopkPattern) {
            indexerTypes = pattern.map { $0 == "S" ? "shared" : "full" }
        } else if let pattern = try? extra.decodeIfPresent(
            [String].self, forKey: .indexTopkPattern)
        {
            indexerTypes = pattern
        } else {
            let frequency = max(try extra.decodeIfPresent(Int.self, forKey: .indexTopkFreq) ?? 1, 1)
            let offset = try extra.decodeIfPresent(Int.self, forKey: .indexSkipTopkOffset) ?? 2
            indexerTypes = (0 ..< layers).map {
                max($0 - offset + 1, 0) % frequency == 0 ? "full" : "shared"
            }
        }
    }
}

extension GLM5NextConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "GLM5NextConfiguration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard attentionHeads == kvHeads else {
            throw fail("num_attention_heads must equal num_key_value_heads for GLM-5-Next")
        }
        guard layerTypes.count == hiddenLayers, mlpLayerTypes.count == hiddenLayers,
            indexerTypes.count == hiddenLayers
        else {
            throw fail("layer_types, mlp_layer_types and indexer_types need one entry per layer")
        }
        guard qkRopeHeadDim == 0, mlaUseNope else {
            throw fail("GLM-5-Next requires NoPE sparse attention (qk_rope_head_dim == 0)")
        }
        // The indexer keeps `index_topk / index_kpool` whole pools
        // (`selectK - 1` is an argPartition kth), pads to `index_topk`, and
        // the DSA select kernel takes both as template sizes and writes the
        // `index_kpool - 1` tail from one 1024-thread threadgroup.
        guard indexKpool >= 1, indexKpool <= Self.maximumIndexKpool else {
            throw fail("index_kpool must be in 1...\(Self.maximumIndexKpool)")
        }
        guard indexTopk >= indexKpool, indexTopk <= Self.maximumIndexTopk else {
            throw fail("index_topk must be in index_kpool...\(Self.maximumIndexTopk)")
        }
        guard indexTopk % indexKpool == 0 else {
            throw fail("index_topk must be divisible by index_kpool")
        }
        guard scoringFunc == "sigmoid", topkMethod == "noaux_tc" else {
            throw fail("GLM-5-Next routing supports sigmoid noaux_tc only")
        }
        let supported: Set = ["linear_attention", "deepseek_sparse_attention"]
        guard layerTypes.allSatisfy(supported.contains) else {
            throw fail("unsupported layer type in \(layerTypes)")
        }
        // Any other spelling would silently pick the dense MLP or a full
        // indexer.
        guard mlpLayerTypes.allSatisfy(["dense", "sparse"].contains) else {
            throw fail("unsupported mlp layer type in \(mlpLayerTypes)")
        }
        guard indexerTypes.allSatisfy(["full", "shared"].contains) else {
            throw fail("unsupported indexer type in \(indexerTypes)")
        }
        // A shared indexer reuses the selection of the nearest earlier
        // sparse-attention layer, so the first one must index for itself.
        if let first = layerTypes.indices.first(where: { layerTypes[$0] != "linear_attention" }),
            indexerTypes[first] == "shared"
        {
            throw fail("the first sparse-attention layer cannot use a shared indexer")
        }
        if mlpLayerTypes.contains("sparse") {
            guard nRoutedExperts >= 1, numExpertsPerTok >= 1, numExpertsPerTok <= nRoutedExperts
            else {
                throw fail("sparse MLP layers need 1 <= num_experts_per_tok <= n_routed_experts")
            }
        }
    }

    /// Upper bounds for the sparse indexer. GLM-5.3-Flash uses 2048 / 4.
    static let maximumIndexTopk = 65_536
    static let maximumIndexKpool = 1_024
}

// MARK: - Chat conventions

extension GLM5NextTextModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}

extension GLM5NextModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}

extension GLM5NextModel: SpeculativeCacheRewindModel {
    public var maximumNativeTargetCacheRewind: Int { 7 }
}
