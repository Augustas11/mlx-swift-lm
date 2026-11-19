//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  Qwen3.8-Flash-Next (`qwen4_exp`): qwen3_next-style gated delta / gated
//  attention layers with QSA sparse attention, hyper-connections, and an
//  n-gram embedding (PLE). Text only.
//
//  Port of mlx-lm `mlx_lm/models/qwen4_exp.py` (ml-explore/mlx-lm c4c9c762).
//  The n-gram table is kept in the checkpoint's row shards and gathered per
//  shard, as mlx-vlm's `ShardedEmbedding` does, instead of the reference's
//  per-head tables: the looked-up rows are the same, and loading copies
//  nothing.
//
//  Not in this port: the MTP head (its tensors are dropped on load), the
//  vision tower (dropped), FP8 checkpoints. Batched decode (continuous
//  batching) runs each row as its serial forward would: see
//  `Qwen4ExpBatchedSparseCache` and `qwen4ExpRowwise`. Anything else the checkpoint holds that the port does not
//  understand fails the load.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Norms

/// RMSNorm, optionally per `groupSize` slice; `weight` holds the folded `1 + w`.
final class Qwen4ExpRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.eps = eps
        self.groupSize = groupSize
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let groupSize else {
            return MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }
        let shape = x.shape
        let grouped = x.reshaped(Array(shape.dropLast()) + [-1, groupSize])
        let normed = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
        return normed.reshaped(shape) * weight
    }
}

/// Gated RMSNorm; the checkpoint stores `weight` as is.
final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let sigmoidGate: Bool

    init(dimensions: Int, activation: String, eps: Float) {
        self.eps = eps
        self.sigmoidGate = activation == "sigmoid"
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let out = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let g = gate.asType(.float32)
        return (out * (sigmoidGate ? sigmoid(g) : silu(g))).asType(x.dtype)
    }
}

// MARK: - Helpers

/// The last `n` steps of `x` along axis 1; a right padded row ends at its length.
func qwen4ExpKeepLast(_ x: MLXArray, n: Int, lengths: MLXArray?) -> MLXArray {
    let total = x.dim(1)
    guard let lengths else {
        // A copy: a slice would keep all of `x` alive in the cache.
        return contiguous(x[0..., (total - n)...])
    }
    var positions =
        clip(lengths, min: 0, max: total - n)[0..., .newAxis].asType(.int32)
        + MLXArray(Int32(0) ..< Int32(n))
    if x.ndim == 3 {
        positions = positions[.ellipsis, .newAxis]
    }
    return takeAlong(x, positions, axis: 1)
}

/// `layer(x)` with each row of a multi-row batch computed alone when the
/// layer is a plain (unquantized) `Linear`. Metal runs a one-row float matmul
/// as GEMV and a many-row one as GEMM, and the two round differently (about
/// 1e-3 relative in float32), so batching rows would change each row's
/// result. A quantized layer runs every row at once: its vector kernel gives
/// each row the same result at any row count up to the device's vector limit.
func qwen4ExpRowwise(_ layer: Linear, _ x: MLXArray) -> MLXArray {
    let rows = x.dim(0)
    if layer is QuantizedLinear { return layer(x) }
    let call = {
        rows > 1 ? concatenated((0 ..< rows).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0) : layer(x)
    }
    // Lab timing: every unquantized linear is its own section, at any row count.
    guard LabSectionTimer.enabled else { return call() }
    return LabSectionTimer.run("fp_linear.\(layer.weight.dim(1))x\(layer.weight.dim(0))", [], call)
}

/// Add `x` to every hyper-connection stream, scaled per stream.
func qwen4ExpInject(_ hyper: MLXArray, _ x: MLXArray, _ inject: MLXArray) -> MLXArray {
    let scaled = expandedDimensions(x, axis: -2) * expandedDimensions(inject, axis: -1)
    return hyper + scaled.reshaped(Array(x.shape.dropLast()) + [-1])
}

// MARK: - QSA sparse attention

final class Qwen4ExpQSAIndexer: Module {
    let nHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopk: Int

    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayernorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: Qwen4ExpRMSNorm

    let rope: RoPE
    /// Block `n` starts at position `n * compressRatio`.
    let blockRope: RoPE

    init(_ args: Qwen4ExpConfiguration) {
        nHeads = args.indexerNHeads
        headDim = args.indexerHeadDim
        tokenBudget = args.indexerBudget
        compressRatio = args.indexerCompressRatio
        blockTopk = tokenBudget / compressRatio
        _indexQKProj.wrappedValue = Linear(
            args.hiddenSize, (args.indexerNHeads + args.indexerKVHeads) * args.indexerHeadDim,
            bias: false)
        _qLayernorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kLayernorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        let rotary = args.rotaryDimensions
        rope = RoPE(dimensions: rotary, traditional: false, base: args.ropeTheta)
        blockRope = RoPE(
            dimensions: rotary, traditional: false, base: args.ropeTheta,
            scale: Float(args.indexerCompressRatio))
        super.init()
    }

    /// The sparse attention mask `[B, 1, L, T]`, or `nil` when every visible
    /// token fits in the budget (the causal mask is enough).
    func callAsFunction(_ x: MLXArray, cache: (any Qwen4ExpSparseCache)?) -> MLXArray? {
        select(qk: qwen4ExpRowwise(indexQKProj, x), cache: cache)
    }

    /// The sparse mask from the projected `index_qk_proj` output of the new
    /// tokens; with a cache, of one sequence.
    func select(qk: MLXArray, cache: (any Qwen4ExpSparseCache)?) -> MLXArray? {
        let B = qk.dim(0)
        let S = qk.dim(1)
        let r = compressRatio
        let split = nHeads * headDim
        let q = qk[.ellipsis, ..<split].reshaped(B, S, nHeads, headDim)
        // One key head (`indexer_kv_heads` is 1).
        var rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)

        let offset = cache?.indexerOffset ?? 0
        if let cache {
            rawK = cache.updateIndexer(keys: rawK)
        }
        let kvLen = rawK.dim(1)

        if kvLen <= tokenBudget {
            return nil
        }

        let nBlocks = kvLen / r
        let blocks = rawK[0..., ..<(nBlocks * r), 0...]
        var pooled = blocks.reshaped(B, nBlocks, r, headDim)
        pooled = pooled.asType(.float32).mean(axis: 2).asType(rawK.dtype)
        pooled = blockRope(kLayernorm(pooled), offset: 0)

        var qh = qLayernorm(q).transposed(0, 2, 1, 3)
        qh = rope(qh, offset: offset)

        // Per block: the sum over heads of relu(q.k).
        var scores = einsum("bhsd,bnd->bsnh", qh.asType(.float32), pooled.asType(.float32))
        scores = maximum(scores, 0).sum(axis: -1) / Float(headDim).squareRoot()

        // Only blocks fully in the query's past.
        let qPos = (MLXArray(Int32(offset)) + MLXArray(Int32(0) ..< Int32(S))).reshaped(1, S)
        let nComplete = floorDivide(maximum(qPos + 1, 0), r)
        var visible =
            MLXArray(Int32(0) ..< Int32(nBlocks)) .< expandedDimensions(nComplete, axis: -1)
        visible = broadcast(visible, to: [B, S, nBlocks])
        scores = which(visible, scores, MLXArray(-Float.infinity))

        let k = min(blockTopk, nBlocks)
        let top = stopGradient(argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        let picked = takeAlong(visible, top, axis: -1)

        // The tokens of the picked blocks; the others go to a dropped column.
        var tok =
            expandedDimensions(top.asType(.int32), axis: -1) * r
            + MLXArray(Int32(0) ..< Int32(r))
        tok = which(expandedDimensions(picked, axis: -1), tok, MLXArray(Int32(kvLen)))
        tok = tok.reshaped(B, S, -1)
        var keep = MLXArray.zeros([B, S, kvLen + 1], dtype: .bool)
        keep = putAlong(keep, tok, values: MLXArray(true), axis: -1)[.ellipsis, ..<kvLen]

        // The reference also keeps each query's own partial block.
        let ownStart = nComplete * r
        let cols = MLXArray(Int32(0) ..< Int32(kvLen))
        let qCols = MLXArray(Int32(kvLen - S) ..< Int32(kvLen))[0..., .newAxis]
        let own = logicalAnd(cols .>= expandedDimensions(ownStart, axis: -1), cols .<= qCols)
        return expandedDimensions(logicalOr(keep, own), axis: 1)
    }
}

final class Qwen4ExpAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer

    let rope: RoPE

    init(_ args: Qwen4ExpConfiguration) {
        nHeads = args.attentionHeads
        nKVHeads = args.kvHeads
        headDim = args.headDim
        scale = pow(Float(headDim), -0.5)
        let d = args.hiddenSize
        // q_proj also carries the output gate.
        _qProj.wrappedValue = Linear(d, nHeads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(d, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, d, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        rope = RoPE(dimensions: args.rotaryDimensions, traditional: false, base: args.ropeTheta)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)
        super.init()
    }

    /// A `Qwen4ExpBatchedSparseCache` holds one cache per row of `x`
    /// (continuous batching): the projections, norms, RoPE (one offset per
    /// row) and output gate run on all rows at once, then each row's cache
    /// update, indexer block selection and attention run against its own
    /// cache, bit for bit as a serial forward of that row. Any other cache
    /// serves all of `x`.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        if let batch = cache as? any Qwen4ExpBatchedSparseCache {
            return batchedAttention(x, batch: batch)
        }
        let (q, k, v, indexQK) = LabSectionTimer.run(
            "attn.proj", outputs: { [$0.0, $0.1, $0.2, $0.3] }
        ) {
            (
                qwen4ExpRowwise(qProj, x), qwen4ExpRowwise(kProj, x), qwen4ExpRowwise(vProj, x),
                qwen4ExpRowwise(indexer.indexQKProj, x)
            )
        }
        let merged = LabSectionTimer.run("attn.rows") {
            attend(
                q: q, k: k, v: v, indexQK: indexQK, mask: mask,
                cache: cache as? any Qwen4ExpSparseCache)
        }
        return LabSectionTimer.run("attn.o_proj") { qwen4ExpRowwise(oProj, merged) }
    }

    /// Tests set this to run the batched cache's oracle: each row's norms,
    /// RoPE and output gate inside its own `attend`, as a serial forward does.
    nonisolated(unsafe) static var perRowOracle = false

    private func batchedAttention(_ x: MLXArray, batch: any Qwen4ExpBatchedSparseCache)
        -> MLXArray
    {
        let rows = batch.rowCaches
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(
            rows.count == B, "Qwen4Exp batched cache has \(rows.count) rows for a batch of \(B)")
        let lengths = batch.rowLengths
        precondition(lengths == nil || lengths!.count == B)

        let (q, k, v, indexQK) = LabSectionTimer.run(
            "attn.proj", outputs: { [$0.0, $0.1, $0.2, $0.3] }
        ) {
            (
                qwen4ExpRowwise(qProj, x), qwen4ExpRowwise(kProj, x), qwen4ExpRowwise(vProj, x),
                qwen4ExpRowwise(indexer.indexQKProj, x)
            )
        }

        // Every row's position as one RoPE offset array; a row cache with
        // per-sequence offsets of its own takes the per-row path.
        let offsets: [Int32]? =
            Self.perRowOracle
            ? nil
            : rows.reduce(into: [Int32]()) { all, row in
                if case .scalar(let offset) = row.ropeOffset { all.append(Int32(offset)) }
            }
        guard let offsets, offsets.count == B else {
            let attended = perRowAttention(q: q, k: k, v: v, indexQK: indexQK, batch: batch)
            return LabSectionTimer.run("attn.o_proj") { qwen4ExpRowwise(oProj, attended) }
        }

        // Norms, RoPE and the output gate are per token and per head: one op
        // over all rows computes each element as the row's own forward does.
        let (qRoped, kRoped, vHeads, gate) = LabSectionTimer.run(
            "attn.norm_rope", outputs: { [$0.0, $0.1, $0.2, $0.3] }
        ) {
            let (q0, k0, v0, gate) = heads(q: q, k: k, v: v)
            let offset = RoPEOffset.batch(MLXArray(offsets))
            return (
                applyRotaryPosition(rope, to: q0, offset: offset),
                applyRotaryPosition(rope, to: k0, offset: offset), v0, sigmoid(gate)
            )
        }

        let attended = LabSectionTimer.run("attn.rows") { () -> MLXArray in
            var outputs = [MLXArray]()
            for (index, rowCache) in rows.enumerated() {
                let n = lengths?[index] ?? L
                precondition(n >= 1 && n <= L, "Qwen4Exp row length \(n) outside 1...\(L)")
                // `[B, H, L, D]` heads and `[B, L, X]` indexer input.
                func rowHeads(_ a: MLXArray) -> MLXArray {
                    if B == 1 && n == L { return a }
                    return n == L
                        ? a[index ..< (index + 1)] : a[index ..< (index + 1), 0..., ..<n]
                }
                func row(_ a: MLXArray) -> MLXArray {
                    if B == 1 && n == L { return a }
                    return n == L ? a[index ..< (index + 1)] : a[index ..< (index + 1), ..<n]
                }
                // A lone forward of the row: no mask for one token, causal past it.
                let out = attendRoped(
                    q: rowHeads(qRoped), k: rowHeads(kRoped), v: rowHeads(vHeads),
                    indexQK: row(indexQK), mask: n == 1 ? .none : .causal, cache: rowCache)
                outputs.append(padded(out, to: L))
            }
            // Padded columns are zero, and zero times the gate stays zero.
            return (B == 1 ? outputs[0] : concatenated(outputs, axis: 0)) * gate
        }
        return LabSectionTimer.run("attn.o_proj") { qwen4ExpRowwise(oProj, attended) }
    }

    /// Each row's whole attention against its own cache, one row at a time:
    /// the oracle of the batched norms, RoPE and gate.
    private func perRowAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, indexQK: MLXArray,
        batch: any Qwen4ExpBatchedSparseCache
    ) -> MLXArray {
        let rows = batch.rowCaches
        let lengths = batch.rowLengths
        let (B, L) = (q.dim(0), q.dim(1))
        return LabSectionTimer.run("attn.rows") { () -> MLXArray in
            var outputs = [MLXArray]()
            for (index, rowCache) in rows.enumerated() {
                let n = lengths?[index] ?? L
                precondition(n >= 1 && n <= L, "Qwen4Exp row length \(n) outside 1...\(L)")
                func row(_ a: MLXArray) -> MLXArray {
                    if B == 1 && n == L { return a }
                    return n == L ? a[index ..< (index + 1)] : a[index ..< (index + 1), ..<n]
                }
                let out = attend(
                    q: row(q), k: row(k), v: row(v), indexQK: row(indexQK),
                    mask: n == 1 ? .none : .causal, cache: rowCache)
                outputs.append(padded(out, to: L))
            }
            return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
        }
    }

    /// A row's `[1, n, X]` output with zero columns up to `L`: padded columns
    /// attend nothing and wrote no cache.
    private func padded(_ out: MLXArray, to L: Int) -> MLXArray {
        let n = out.dim(1)
        guard n < L else { return out }
        let padding = MLXArray.zeros([1, L - n, out.dim(2)], dtype: out.dtype)
        return concatenated([out, padding], axis: 1)
    }

    /// The normed `[B, H, S, D]` queries and keys (before RoPE), the values,
    /// and the output gate `[B, S, H * D]`, from the projections.
    private func heads(q qGated: MLXArray, k kProjected: MLXArray, v vProjected: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let B = qGated.dim(0)
        let S = qGated.dim(1)
        let qSplit = qGated.reshaped(B, S, nHeads, -1).split(parts: 2, axis: -1)
        let gate = qSplit[1].reshaped(B, S, -1)
        let q = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let k = kNorm(kProjected.reshaped(B, S, nKVHeads, -1)).transposed(0, 2, 1, 3)
        let v = vProjected.reshaped(B, S, nKVHeads, -1).transposed(0, 2, 1, 3)
        return (q, k, v, gate)
    }

    /// The gated attention output (before `o_proj`) from the projected
    /// queries (with the output gate), keys, values and indexer `qk`.
    private func attend(
        q qGated: MLXArray, k kProjected: MLXArray, v vProjected: MLXArray, indexQK: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: (any Qwen4ExpSparseCache)?
    ) -> MLXArray {
        let offset = cache?.ropeOffset
        var (q, k, v, gate) = heads(q: qGated, k: kProjected, v: vProjected)

        (q, k) = LabSectionTimer.run("attn.row.norm_rope", outputs: { [$0.0, $0.1] }) {
            (
                applyRotaryPosition(rope, to: q, offset: offset),
                applyRotaryPosition(rope, to: k, offset: offset)
            )
        }
        gate = sigmoid(gate)
        return attendRoped(q: q, k: k, v: v, indexQK: indexQK, mask: mask, cache: cache) * gate
    }

    /// The attention output `[B, S, H * D]` (before the gate) from roped
    /// `[B, H, S, D]` queries and keys, the values and the indexer `qk`.
    private func attendRoped(
        q: MLXArray, k kNew: MLXArray, v vNew: MLXArray, indexQK: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: (any Qwen4ExpSparseCache)?
    ) -> MLXArray {
        let (B, S) = (q.dim(0), q.dim(2))
        var (k, v) = (kNew, vNew)
        if let cache {
            (k, v) = LabSectionTimer.run("attn.row.cache_update", outputs: { [$0.0, $0.1] }) {
                cache.update(keys: k, values: v)
            }
        }

        // The sparse mask is causal and has no padding: it replaces `mask`.
        var maskMode = mask
        let sparse = LabSectionTimer.run(
            "attn.row.indexer", outputs: { $0.map { [$0] } ?? [] }
        ) { indexer.select(qk: indexQK, cache: cache) }
        if let sparse {
            maskMode = .array(sparse)
        }

        return LabSectionTimer.run("attn.row.sdpa") {
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: maskMode)
            return out.transposed(0, 2, 1, 3).reshaped(B, S, -1)
        }
    }
}

// MARK: - Gated delta net

final class Qwen4ExpGatedDeltaNet: Module {
    let nV: Int
    let nK: Int
    let dk: Int
    let dv: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen4ExpConfiguration) {
        nV = args.linearNumValueHeads
        nK = args.linearNumKeyHeads
        dk = args.linearKeyHeadDim
        dv = args.linearValueHeadDim
        keyDim = dk * nK
        valueDim = dv * nV
        convKernelSize = args.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        let d = args.hiddenSize

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(d, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(d, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(d, nV, bias: false)
        _inProjA.wrappedValue = Linear(d, nV, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([nV])
        _aLog.wrappedValue = MLXArray.zeros([nV])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: dv, activation: args.outputGateType, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, d, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let (projectedQKV, z, b, a) = LabSectionTimer.run(
            "gdn.in_proj", outputs: { [$0.0, $0.1, $0.2, $0.3] }
        ) {
            (
                qwen4ExpRowwise(inProjQKV, x), qwen4ExpRowwise(inProjZ, x).reshaped(B, S, nV, dv),
                qwen4ExpRowwise(inProjB, x), qwen4ExpRowwise(inProjA, x)
            )
        }

        let deltaOut = LabSectionTimer.run(
            "gdn.conv_delta", outputs: { [$0] + (cache?.state ?? []) }
        ) { () -> MLXArray in
            var mixedQKV = projectedQKV
            let convState =
                cache?[0]
                ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype)
            if let mask {
                mixedQKV = which(expandedDimensions(mask, axis: -1), mixedQKV, 0)
            }
            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache {
                cache[0] = qwen4ExpKeepLast(
                    convInput, n: convKernelSize - 1, lengths: cache.currentLengths)
            }
            let convOut = silu(conv1d(convInput))

            let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            var q = parts[0].reshaped(B, S, nK, dk)
            var k = parts[1].reshaped(B, S, nK, dk)
            let v = parts[2].reshaped(B, S, nV, dv)

            // mlx-lm `normalize_qk(q, k, inv_scale=dk**-0.5, eps=1e-6)`.
            let invScale = pow(Float(dk), -0.5)
            let rmsEps = 1e-6 * invScale * invScale
            q = (invScale * invScale) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: rmsEps)
            k = invScale * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: rmsEps)

            let (out, state) = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                state: cache?[1], mask: mask, useKernel: !training)
            if let cache {
                cache[1] = state
                cache.advance(S)
            }
            return out
        }
        return LabSectionTimer.run("gdn.out") {
            qwen4ExpRowwise(outProj, norm(deltaOut, gate: z).reshaped(B, S, -1))
        }
    }
}

// MARK: - MoE

final class Qwen4ExpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hidden: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        qwen4ExpRowwise(
            downProj, silu(qwen4ExpRowwise(gateProj, x)) * qwen4ExpRowwise(upProj, x))
    }
}

final class Qwen4ExpSparseMoeBlock: Module, UnaryLayer {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen4ExpConfiguration) {
        topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExpert.wrappedValue = Qwen4ExpMLP(
            dimensions: args.hiddenSize, hidden: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (idx, w) = LabSectionTimer.run("moe.router", [x], outputs: { [$0.0, $0.1] }) {
            () -> (MLXArray, MLXArray) in
            let logits = qwen4ExpRowwise(gate, x.asType(.float32))
            let idx = stopGradient(
                argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
            // The softmax of the top-k logits: the reference's renormalized top-k.
            return (idx, softmax(takeAlong(logits, idx, axis: -1), axis: -1))
        }
        let routed = LabSectionTimer.run("moe.experts") {
            (switchMLP(x, idx) * expandedDimensions(w, axis: -1)).sum(axis: -2).asType(x.dtype)
        }
        let shared = LabSectionTimer.run("moe.shared") {
            sigmoid(qwen4ExpRowwise(sharedExpertGate, x)) * sharedExpert(x)
        }
        return routed + shared
    }
}

// MARK: - Hyper-connections

final class Qwen4ExpGatedResidual: Module {
    let hc: Int
    let d: Int
    let lowrank: Int

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixWeightUp: Linear
    /// The checkpoint's block inject. The reference merges it into the mix
    /// down; kept apart here, every output row is the same dot product.
    @ModuleInfo(key: "block_inject_weight") var blockInjectWeight: Linear?

    init(_ args: Qwen4ExpConfiguration, useCombine: Bool) {
        hc = args.hcCount
        d = args.hiddenSize
        lowrank = args.hcLowrank
        let hcDim = hc * d
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _inputMixWeightDown.wrappedValue = Linear(hcDim, lowrank, bias: false)
        _inputMixWeightUp.wrappedValue = Linear(lowrank, hcDim, bias: false)
        if useCombine {
            _blockInjectWeight.wrappedValue = Linear(hcDim, hc, bias: false)
        }
        super.init()
    }

    private func mixed(_ normed: MLXArray) -> MLXArray {
        var w = silu(qwen4ExpRowwise(inputMixWeightDown, normed) / Float(hc))
        w = sigmoid(qwen4ExpRowwise(inputMixWeightUp, w))
        let lead = Array(w.shape.dropLast())
        w = w.reshaped(lead + [hc, d])
        return (w * normed.reshaped(lead + [hc, d])).mean(axis: -2)
    }

    /// The branch input and the per-stream inject scale.
    func combine(_ hyper: MLXArray) -> (MLXArray, MLXArray) {
        let normed = hcNorm(hyper)
        guard let blockInjectWeight else {
            fatalError("Qwen4ExpGatedResidual.combine needs block_inject_weight")
        }
        let inject = 2 * sigmoid(qwen4ExpRowwise(blockInjectWeight, normed) / Float(hc))
        return (mixed(normed), inject)
    }

    /// The streams merged into one hidden state (the final mixer).
    func merge(_ hyper: MLXArray) -> MLXArray {
        mixed(hcNorm(hyper))
    }
}

// MARK: - N-gram embedding (PLE)

/// The n-gram table in equal row shards (`shards.{i}`), gathered per shard.
final class Qwen4ExpShardedEmbedding: Module {
    let totalRows: Int
    let dimensions: Int
    let baseRows: Int
    let remainder: Int

    @ModuleInfo(key: "shards") var shards: [Embedding]

    init(totalRows: Int, dimensions: Int, shardCount: Int) {
        precondition(shardCount >= 1 && shardCount <= totalRows)
        self.totalRows = totalRows
        self.dimensions = dimensions
        let base = totalRows / shardCount
        let extra = totalRows % shardCount
        self.baseRows = base
        self.remainder = extra
        _shards.wrappedValue = (0 ..< shardCount).map { i in
            Embedding(embeddingCount: base + (i < extra ? 1 : 0), dimensions: dimensions)
        }
        super.init()
    }

    func shardRows(_ index: Int) -> Int { baseRows + (index < remainder ? 1 : 0) }

    /// The shard of global row `row` and the row within it.
    func locate(_ row: Int) -> (shard: Int, local: Int) {
        let wide = remainder * (baseRows + 1)
        if row < wide {
            return (row / (baseRows + 1), row % (baseRows + 1))
        }
        let rest = row - wide
        return (remainder + rest / baseRows, rest % baseRows)
    }

    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let flat = indices.reshaped(-1).asType(.int64)
        // One small host read keeps each step from gathering against every shard.
        let host = flat.asArray(Int64.self)
        var positions = [Int: [Int]]()
        var locals = [Int: [Int32]]()
        for (position, value) in host.enumerated() {
            let row = Int(value)
            guard row >= 0, row < totalRows else {
                fatalError("Qwen4Exp n-gram row \(row) is outside the table of \(totalRows) rows")
            }
            let (shard, local) = locate(row)
            positions[shard, default: []].append(position)
            locals[shard, default: []].append(Int32(local))
        }
        var order = [Int]()
        var pieces = [MLXArray]()
        for shard in positions.keys.sorted() {
            order.append(contentsOf: positions[shard]!)
            pieces.append(shards[shard](MLXArray(locals[shard]!)))
        }
        guard !pieces.isEmpty else {
            return shards[0](flat.asType(.int32)).reshaped(indices.shape + [dimensions])
        }
        var inverse = [Int32](repeating: 0, count: order.count)
        for (i, position) in order.enumerated() {
            inverse[position] = Int32(i)
        }
        let gathered = concatenated(pieces, axis: 0)
        return gathered.take(MLXArray(inverse), axis: 0).reshaped(indices.shape + [dimensions])
    }
}

final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNgram: Int
    let ngramHeads: Int
    let eosTokenId: Int
    let headVocabSizes: [Int]
    let headOffsets: [Int]
    let multipliers: [Int]

    @ModuleInfo(key: "ngram_embedding") var table: Qwen4ExpShardedEmbedding

    init(_ args: Qwen4ExpConfiguration, pleLayerIndex: Int) {
        ngramSize = args.ngramSize
        headsPerNgram = args.headsPerNgram
        ngramHeads = (args.ngramSize - 1) * args.headsPerNgram
        eosTokenId = args.eosTokenId
        let constants = Qwen4ExpNGramHash.constants(args, pleLayerIndex: pleLayerIndex)
        headVocabSizes = constants.sizes
        headOffsets = constants.offsets
        multipliers = constants.multipliers
        _table.wrappedValue = Qwen4ExpShardedEmbedding(
            totalRows: constants.paddedRows, dimensions: args.pleEmbedDim / ngramHeads,
            shardCount: args.splitNgramParts)
        super.init()
    }

    /// x[t], x[t-1], ... for the new tokens; `history` starts with the n-1
    /// tokens before them. Older tokens past an EOS read as EOS.
    private func ngramTokens(_ history: MLXArray) -> [MLXArray] {
        let n = ngramSize
        let newCount = history.dim(1) - (n - 1)
        let eos = MLXArray(Int64(eosTokenId))
        var tokens = [
            history[0..., (n - 1)...],
            history[0..., (n - 2) ..< (n - 2 + newCount)],
        ]
        if n > 2 {
            for s in 2 ..< n {
                let older = history[0..., (n - 1 - s) ..< (n - 1 - s + newCount)]
                tokens.append(which(tokens[tokens.count - 1] .== eos, eos, older))
            }
        }
        return tokens
    }

    func callAsFunction(_ ids: MLXArray, cache: ArraysCache?) -> MLXArray {
        let embedded = table(rows(ids, cache: cache))
        return embedded.reshaped(Array(embedded.shape.dropLast(2)) + [-1])
    }

    /// The table rows `[B, L, heads]` the new tokens read, one per head
    /// (each head's hash plus its offset); updates the n-gram context in
    /// cache slot 3.
    func rows(_ ids: MLXArray, cache: ArraysCache?) -> MLXArray {
        let n = ngramSize
        let eos = MLXArray(Int32(eosTokenId)).asType(ids.dtype)
        // Slot 3 holds `token ^ eos`: XOR with eos again gives the token, and
        // the zeros that batching puts in a new row read as EOS.
        let previous: MLXArray
        if let cache, let stored = cache[3] {
            previous = bitwiseXOr(stored, eos)
        } else {
            previous = MLXArray.full([ids.dim(0), n - 1], values: eos)
        }
        let history = concatenated([previous, ids], axis: 1)
        if let cache {
            cache[3] = bitwiseXOr(
                qwen4ExpKeepLast(history, n: n - 1, lengths: cache.currentLengths), eos)
        }

        let tokens = ngramTokens(history.asType(.int64))
        let sizes = MLXArray(headVocabSizes.map { Int64($0) })
        let offsets = MLXArray(headOffsets.map { Int64($0) })

        // Step s gives the hash of the (s+1)-gram.
        var mixed = tokens[0] * MLXArray(Int64(multipliers[0]))
        var rows = [MLXArray]()
        for s in 1 ..< n {
            mixed = bitwiseXOr(mixed, tokens[s] * MLXArray(Int64(multipliers[s])))
            let lo = (s - 1) * headsPerNgram
            rows.append(
                MLX.remainder(
                    expandedDimensions(mixed, axis: -1), sizes[lo ..< (lo + headsPerNgram)]))
        }
        return concatenated(rows, axis: -1) + offsets
    }
}

/// The n-gram hash constants, rebuilt from the config as the reference does.
enum Qwen4ExpNGramHash {
    static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let m1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let m2: UInt64 = 0x94D0_49BB_1331_11EB
    static let prime1 = 10007

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ gamma
        v = (v ^ (v >> 30)) &* m1
        v = (v ^ (v >> 27)) &* m2
        return v ^ (v >> 31)
    }

    /// Odd hash multipliers, small enough that `token * multiplier` fits in int64.
    static func multipliers(vocabSize: Int, ngramSize: Int, seed: Int) -> [Int] {
        let half = max(1, UInt64(Int64.max) / UInt64(max(vocabSize, 1)) / 2)
        return (0 ..< ngramSize).map { i in
            let mixedSeed = UInt64(bitPattern: Int64(seed)) &+ gamma &* UInt64(i + 1)
            return Int(2 * (splitmix64(mixedSeed) % half) + 1)
        }
    }

    static func isPrime(_ v: Int) -> Bool {
        if v < 2 { return false }
        if v % 2 == 0 { return v == 2 }
        var d = 3
        while d * d <= v {
            if v % d == 0 { return false }
            d += 2
        }
        return true
    }

    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var p = start
        for _ in 0 ..< count {
            p += 1
            while !isPrime(p) { p += 1 }
        }
        return p
    }

    static func constants(_ args: Qwen4ExpConfiguration, pleLayerIndex: Int) -> (
        sizes: [Int], offsets: [Int], multipliers: [Int], paddedRows: Int
    ) {
        let heads = (args.ngramSize - 1) * args.headsPerNgram
        var sizes = [Int]()
        var offsets = [Int]()
        var total = 0
        // Each head's size is the next prime after the previous head's.
        var previous = args.ngramVocabSizeBase - 1
        var skip = pleLayerIndex * heads
        while skip > 0 {
            previous = nthPrimeAfter(previous, count: 1)
            skip -= 1
        }
        for _ in 0 ..< heads {
            previous = nthPrimeAfter(previous, count: 1)
            sizes.append(previous)
            offsets.append(total)
            total += previous
        }
        let divisor = max(args.ngramVocabSizeDivisor, 1)
        let padded = (total + divisor - 1) / divisor * divisor
        let mults = multipliers(
            vocabSize: args.vocabularySize, ngramSize: args.ngramSize,
            seed: args.seed + prime1 * pleLayerIndex)
        return (sizes, offsets, mults, padded)
    }
}

final class Qwen4ExpPLELayer: Module {
    let d: Int
    let hc: Int
    let shortConvStateLength: Int

    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ args: Qwen4ExpConfiguration, pleLayerIndex: Int) {
        d = args.hiddenSize
        hc = args.hcCount
        let hcDim = d * hc
        let k = args.pleConvKernelSize
        let dilation = args.ngramSize
        shortConvStateLength = (k - 1) * dilation
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(args, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, hcDim, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, d, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: hcDim, outputChannels: hcDim, kernelSize: k, stride: 1, padding: 0,
            dilation: dilation, groups: hcDim, bias: false)
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let n = shortConvStateLength
        let state =
            cache?[2] ?? MLXArray.zeros([x.dim(0), n, x.dim(-1)], dtype: x.dtype)
        let full = concatenated([state, x], axis: 1)
        if let cache {
            cache[2] = qwen4ExpKeepLast(full, n: n, lengths: cache.currentLengths)
        }
        return silu(conv1d(full))
    }

    func callAsFunction(
        _ hidden: MLXArray, ids: MLXArray, cache: ArraysCache?, mask: MLXArray?
    ) -> MLXArray {
        var ids = ids
        if let mask {
            // Padding reads as EOS, like the start of a prompt.
            ids = which(mask, ids, MLXArray(Int32(pleEmbedding.eosTokenId)).asType(ids.dtype))
        }
        let emb = pleEmbedding(ids, cache: cache)

        let lead = Array(hidden.shape.dropLast())
        let key = normKey(qwen4ExpRowwise(keyProj, emb)).reshaped(lead + [hc, d])
        let value = qwen4ExpRowwise(valueProj, emb)
        let query = normQuery(hidden).reshaped(lead + [hc, d])

        var gate = (key * query).sum(axis: -1, keepDims: true) / Float(d).squareRoot()
        gate = sqrt(maximum(abs(gate), Float(1e-6))) * sign(gate)
        var gated = sigmoid(gate) * expandedDimensions(value, axis: -2)
        gated = gated.reshaped(lead + [-1])
        if let mask {
            gated = which(expandedDimensions(mask, axis: -1), gated, 0)
        }
        return gated + shortConv(normConv(gated), cache: cache)
    }
}

// MARK: - Decoder

final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpConfiguration, layerIdx: Int) {
        isLinear = args.layerTypes[layerIdx] == "linear_attention"
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(args)
        // `ple_layer_ids` counts layers from 1.
        if let pleIndex = args.pleLayerIds.firstIndex(of: layerIdx + 1) {
            _ple.wrappedValue = Qwen4ExpPLELayer(args, pleLayerIndex: pleIndex)
        }
        _attnHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: true)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: true)
        super.init()
    }

    func callAsFunction(
        _ h: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, convMask: MLXArray?,
        cache: KVCache?, ids: MLXArray
    ) -> MLXArray {
        var h = h
        let linearCache: ArraysCache?
        let attentionCache: KVCache?
        if let cache {
            if isLinear {
                guard let c = cache as? ArraysCache else {
                    fatalError("Qwen4Exp linear-attention layer needs an ArraysCache, got \(type(of: cache))")
                }
                linearCache = c
                attentionCache = nil
            } else {
                guard cache is any Qwen4ExpSparseCache || cache is any Qwen4ExpBatchedSparseCache
                else {
                    fatalError("Qwen4Exp attention layer needs a Qwen4ExpSparseCache, got \(type(of: cache))")
                }
                linearCache = nil
                attentionCache = cache
            }
        } else {
            linearCache = nil
            attentionCache = nil
        }

        if let ple {
            let hIn = h
            h = LabSectionTimer.run("ple", [hIn]) {
                hIn + ple(hIn, ids: ids, cache: linearCache, mask: convMask)
            }
        }

        // `LabSectionTimer` sections are pass-throughs unless lab timing is on.
        let hAttn = h
        var (x, inject) = LabSectionTimer.run("hc.combine", [hAttn], outputs: { [$0.0, $0.1] }) {
            attnHyperConnection.combine(hAttn)
        }
        if let linearAttn {
            x = linearAttn(x, mask: convMask, cache: linearCache)
        } else {
            x = selfAttn!(x, mask: mask, cache: attentionCache)
        }
        let (xAttn, injectAttn) = (x, inject)
        h = LabSectionTimer.run("hc.inject") { qwen4ExpInject(hAttn, xAttn, injectAttn) }

        let hMLP = h
        (x, inject) = LabSectionTimer.run("hc.combine", outputs: { [$0.0, $0.1] }) {
            mlpHyperConnection.combine(hMLP)
        }
        let moe = mlp(x)
        let injectMLP = inject
        return LabSectionTimer.run("hc.inject") { qwen4ExpInject(hMLP, moe, injectMLP) }
    }
}

public final class Qwen4ExpModelInner: Module {
    let hc: Int
    let firstAttention: Int
    let firstLinear: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    /// No final norm: the mixer normalizes the streams before it merges them.
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpConfiguration) {
        hc = args.hcCount
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
            Qwen4ExpDecoderLayer(args, layerIdx: $0)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: false)
        firstAttention = args.layerTypes.firstIndex(of: "full_attention") ?? 0
        firstLinear = args.layerTypes.firstIndex(of: "linear_attention") ?? 0
        super.init()
    }

    func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = LabSectionTimer.run("embed") { embedTokens(ids) }
        let n = h.dim(1)
        // Each kind of layer shares the mask of its first layer.
        let mask = makeAttentionMask(n: n, cache: cache?[firstAttention])
        // The deltanet and PLE must skip the padding.
        let convMask = (cache?[firstLinear] as? ArraysCache)?.makeMask(N: n)

        h = tiled(h, repetitions: [1, 1, hc])
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, convMask: convMask, cache: cache?[i], ids: ids)
        }
        let hOut = h
        return LabSectionTimer.run("hc.mixer") { hyperConnectionMixer.merge(hOut) }
    }
}

// MARK: - Models

/// `qwen4_exp_text` checkpoints: `model.*` and `lm_head.*`.
public final class Qwen4ExpTextModel: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: Qwen4ExpConfiguration

    @ModuleInfo(key: "model") public var model: Qwen4ExpModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: Qwen4ExpConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _model.wrappedValue = Qwen4ExpModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Lab timing: a decode forward (one token per row) evaluates its
        // logits inside the bracket; a prefill forward leaves them lazy.
        let decode = inputs.dim(1) == 1
        return LabSectionTimer.forward(
            tag: "rows=\(inputs.dim(0)),L=\(inputs.dim(1))", outputs: { decode ? [$0] : [] }
        ) { () -> MLXArray in
            let out = model(inputs, cache: cache)
            return LabSectionTimer.run("lm_head", outputs: { decode ? [$0] : [] }) {
                () -> MLXArray in
                if let lmHead {
                    return qwen4ExpRowwise(lmHead, out)
                }
                return model.embedTokens.asLinear(out)
            }
        }
    }

    /// One cache per decoder layer: attention keys and values plus indexer
    /// keys for full attention; conv and recurrent state for the deltanet,
    /// plus the PLE conv state and n-gram context on a PLE layer.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { layer -> KVCache in
            if !layer.isLinear {
                return Qwen4ExpAttentionCache()
            }
            return layer.ple == nil ? MambaCache() : ArraysCache(size: 4)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Qwen4ExpWeights.sanitize(weights, model: self)
    }

    public var loraLayers: [Module] { model.layers }
}

extension Qwen4ExpTextModel: PerLayerQuantizationSanitizing {
    public func sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization)
        -> BaseConfiguration.PerLayerQuantization
    {
        Qwen4ExpWeights.sanitize(perLayerQuantization: perLayerQuantization, modulePrefix: "")
    }
}

/// `qwen4_exp` checkpoints: the text model under `language_model`, with the
/// vision tower (dropped) and the MTP head (dropped) alongside.
public final class Qwen4ExpModel: Module, LLMModel {
    public let vocabularySize: Int

    @ModuleInfo(key: "language_model") public var languageModel: Qwen4ExpTextModel

    public init(_ configuration: Qwen4ExpConfiguration) {
        self.vocabularySize = configuration.vocabularySize
        _languageModel.wrappedValue = Qwen4ExpTextModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let text = languageModel.sanitize(weights: weights)
        return Dictionary(uniqueKeysWithValues: text.map { ("language_model." + $0.key, $0.value) })
    }

    public var loraLayers: [Module] { languageModel.loraLayers }
}

extension Qwen4ExpModel: PerLayerQuantizationSanitizing {
    public func sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization)
        -> BaseConfiguration.PerLayerQuantization
    {
        Qwen4ExpWeights.sanitize(
            perLayerQuantization: perLayerQuantization, modulePrefix: "language_model.")
    }
}

// MARK: - Weight sanitization

enum Qwen4ExpWeights {
    /// Norms stored as `w` for a `1 + w` scale; `linear_attn.norm` is not one.
    static let foldOneSuffixes = [
        "q_layernorm.weight", "k_layernorm.weight", "q_norm.weight", "k_norm.weight",
        "hc_norm.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight",
    ]
    static let ngramConstantSuffixes = [
        "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_vocab_sizes",
        "ple_embedding.ngram_heads_offsets",
    ]
    /// A load this port must refuse names its reason under this prefix: the
    /// model has no such parameter, so the weight update fails with it.
    static let refusalPrefix = "qwen4_exp_refused."

    /// A checkpoint key as a text-model key (`model.*`, `lm_head.*`), or
    /// `nil` for a tensor this text-only port drops (vision tower, MTP head).
    static func textKey(_ key: String) -> String? {
        if key.hasPrefix("mtp.") || key.hasPrefix("model.mtp.")
            || key.hasPrefix("language_model.mtp.")
        {
            return nil
        }
        let vision = ["model.visual.", "visual.", "vision_tower.", "vision_model."]
        if vision.contains(where: { key.hasPrefix($0) }) {
            return nil
        }
        var key = key
        if key.hasPrefix("model.language_model.") {
            key = "model." + key.dropFirst("model.language_model.".count)
        } else if key.hasPrefix("language_model.") {
            key = String(key.dropFirst("language_model.".count))
        }
        return renameShards(key)
    }

    /// `.ngram_embedding.shard_{i}.` (HF) to `.ngram_embedding.shards.{i}.`.
    static func renameShards(_ key: String) -> String {
        guard let range = key.range(of: ".ngram_embedding.shard_") else { return key }
        let rest = key[range.upperBound...]
        guard let dot = rest.firstIndex(of: "."), Int(rest[..<dot]) != nil else { return key }
        return String(key[..<range.lowerBound]) + ".ngram_embedding.shards."
            + String(rest[..<dot]) + String(rest[dot...])
    }

    /// The layer index of `model.layers.{i}.*`.
    static func layerIndex(_ key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let rest = key[range.upperBound...]
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[..<dot])
    }

    static func sanitize(_ weights: [String: MLXArray], model: Qwen4ExpTextModel)
        -> [String: MLXArray]
    {
        let config = model.configuration
        // HF and mlx-vlm (flat layout, `shards.{i}` n-gram names) store these
        // norms as `w` for a `1 + w` scale; other checkpoints hold `1 + w`.
        let fold = weights.keys.contains {
            $0.hasPrefix("model.language_model.")
                || ($0.hasPrefix("language_model.") && $0.contains(".ngram_embedding.shards."))
        }

        var out = [String: MLXArray]()
        var refusals = [String]()
        for (original, value) in weights {
            guard let key = textKey(original) else { continue }

            if let suffix = ngramConstantSuffixes.first(where: { key.hasSuffix($0) }) {
                if !ngramConstantMatches(key: key, suffix: suffix, value: value, model: model) {
                    refusals.append("ngram_hash_constants_do_not_match_config.\(key)")
                }
                continue
            }
            if key.hasSuffix(".ngram_embedding.weight_scale") {
                // mlx-vlm drops it once the table is in (dequantized) shards;
                // without shards it scales an FP8 table, which is not supported.
                let prefix = String(key.dropLast("weight_scale".count))
                let hasShards = weights.keys.contains {
                    textKey($0)?.hasPrefix(prefix + "shards.") ?? false
                }
                if !hasShards {
                    refusals.append("fp8_ngram_table_not_supported.\(key)")
                }
                continue
            }
            if key.contains(".mlp.experts.") {
                if key.hasSuffix("mlp.experts.gate_up_proj"), value.ndim == 3 {
                    let base = String(key.dropLast("experts.gate_up_proj".count))
                    let mid = value.dim(-2) / 2
                    out[base + "switch_mlp.gate_proj.weight"] = value[.ellipsis, ..<mid, 0...]
                    out[base + "switch_mlp.up_proj.weight"] = value[.ellipsis, mid..., 0...]
                    continue
                }
                if key.hasSuffix("mlp.experts.down_proj"), value.ndim == 3 {
                    let base = String(key.dropLast("experts.down_proj".count))
                    out[base + "switch_mlp.down_proj.weight"] = value
                    continue
                }
                refusals.append("unsupported_expert_layout.\(key)")
                continue
            }

            var v = value
            // torch (C, 1, K) to mlx (C, K, 1); a converted weight has shape[1] > 1.
            if key.hasSuffix("conv1d.weight"), v.ndim == 3, v.dim(1) == 1 {
                v = v.transposed(0, 2, 1)
            }
            if fold, foldOneSuffixes.contains(where: { key.hasSuffix($0) }) {
                v = 1 + v
            }
            out[key] = v
        }

        if config.tieWordEmbeddings {
            out = out.filter { !$0.key.hasPrefix("lm_head.") }
        }
        for reason in refusals {
            out[refusalPrefix + reason] = MLXArray(0)
        }
        return out
    }

    /// Whether a checkpoint's n-gram hash constant equals the one rebuilt from
    /// the config for that layer.
    static func ngramConstantMatches(
        key: String, suffix: String, value: MLXArray, model: Qwen4ExpTextModel
    ) -> Bool {
        guard let layer = layerIndex(key), layer < model.model.layers.count,
            let embedding = model.model.layers[layer].ple?.pleEmbedding
        else { return false }
        let rebuilt: [Int]
        switch suffix {
        case "ple_embedding.layer_multipliers": rebuilt = embedding.multipliers
        case "ple_embedding.ngram_heads_vocab_sizes": rebuilt = embedding.headVocabSizes
        default: rebuilt = embedding.headOffsets
        }
        let stored = value.asType(.int64).asArray(Int64.self).map { Int($0) }
        return stored == rebuilt
    }

    /// `quantization` with every checkpoint key also entered under the module
    /// path it names. An entry already keyed by a module path wins.
    static func sanitize(
        perLayerQuantization: BaseConfiguration.PerLayerQuantization, modulePrefix: String
    ) -> BaseConfiguration.PerLayerQuantization {
        var result = perLayerQuantization
        for (key, option) in perLayerQuantization.perLayerQuantization {
            // A layer key has no tensor suffix; map it as a weight name.
            guard let mapped = textKey(key + ".weight") else { continue }
            let path = modulePrefix + String(mapped.dropLast(".weight".count))
            if result.perLayerQuantization[path] == nil {
                result.perLayerQuantization[path] = option
            }
        }
        return result
    }
}

// MARK: - Configuration

public struct Qwen4ExpConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var ropeType: String
    var eosTokenId: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var normTopkProb: Bool?
    var linearNumKeyHeads: Int
    var linearNumValueHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var hcCount: Int
    var hcLowrank: Int
    var indexerNHeads: Int
    var indexerKVHeads: Int
    var indexerHeadDim: Int
    var indexerBudget: Int
    var indexerCompressRatio: Int
    var ngramSize: Int
    var headsPerNgram: Int
    var ngramVocabSizeBase: Int
    var ngramVocabSizeDivisor: Int
    var splitNgramParts: Int
    var pleEmbedDim: Int
    var pleLayerIds: [Int]
    var pleConvKernelSize: Int
    var seed: Int
    var layerTypes: [String]
    var outputGateType: String
    var hiddenAct: String
    var tieWordEmbeddings: Bool
    var attentionBias: Bool

    var rotaryDimensions: Int { Int(Float(headDim) * partialRotaryFactor) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case eosTokenId = "eos_token_id"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerNHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case ngramVocabSizeDivisor = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case pleEmbedDim = "ple_embed_dim"
        case pleLayerIds = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case seed
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case outputGateType = "output_gate_type"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
    }

    enum OuterKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
    }

    /// `eos_token_id` as an int or a list (the first entry counts).
    private struct EOS: Decodable {
        let value: Int
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let v = try? c.decode(Int.self) {
                value = v
            } else {
                let list = try c.decode([Int].self)
                guard let first = list.first else {
                    throw DecodingError.dataCorruptedError(
                        in: c, debugDescription: "eos_token_id is an empty list")
                }
                value = first
            }
        }
    }

    public init(from decoder: Decoder) throws {
        // qwen4_exp nests the text model under text_config; qwen4_exp_text is flat.
        let outer = try decoder.container(keyedBy: OuterKeys.self)
        let c =
            outer.contains(.textConfig)
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)

        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        func number(_ value: StringOrNumber?) -> Float? {
            switch value {
            case .int(let v): Float(v)
            case .float(let v): v
            default: nil
            }
        }
        let flatTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        guard let theta = number(rope?["rope_theta"]) ?? flatTheta else {
            throw DecodingError.keyNotFound(
                CodingKeys.ropeParameters,
                .init(
                    codingPath: c.codingPath,
                    debugDescription: "qwen4_exp needs rope_parameters.rope_theta"))
        }
        ropeTheta = theta
        let flatPartial = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        partialRotaryFactor = number(rope?["partial_rotary_factor"]) ?? flatPartial ?? 1.0
        if case .string(let type) = rope?["type"] ?? rope?["rope_type"] {
            ropeType = type
        } else {
            ropeType = "default"
        }

        eosTokenId = try c.decode(EOS.self, forKey: .eosTokenId).value
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb)
        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        hcCount = try c.decode(Int.self, forKey: .hcCount)
        hcLowrank = try c.decode(Int.self, forKey: .hcLowrank)
        indexerNHeads = try c.decode(Int.self, forKey: .indexerNHeads)
        indexerKVHeads = try c.decode(Int.self, forKey: .indexerKVHeads)
        indexerHeadDim = try c.decode(Int.self, forKey: .indexerHeadDim)
        indexerBudget = try c.decode(Int.self, forKey: .indexerBudget)
        indexerCompressRatio = try c.decode(Int.self, forKey: .indexerCompressRatio)
        ngramSize = try c.decode(Int.self, forKey: .ngramSize)
        headsPerNgram = try c.decode(Int.self, forKey: .headsPerNgram)
        ngramVocabSizeBase = try c.decode(Int.self, forKey: .ngramVocabSizeBase)
        // mlx-vlm pads the table to this multiple and splits it in
        // `split_ngram_parts` row shards; the load checks both against the
        // checkpoint's shard shapes.
        ngramVocabSizeDivisor =
            try c.decodeIfPresent(Int.self, forKey: .ngramVocabSizeDivisor) ?? 1
        splitNgramParts = try c.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 1
        pleEmbedDim = try c.decode(Int.self, forKey: .pleEmbedDim)
        pleLayerIds = try c.decode([Int].self, forKey: .pleLayerIds)
        pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        // config.json has no `seed` in every build; the reference defaults it.
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? hiddenAct
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        let decodedTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        if decodedTypes.isEmpty {
            let k = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
            layerTypes = (0 ..< hiddenLayers).map {
                ($0 + 1) % k == 0 ? "full_attention" : "linear_attention"
            }
        } else {
            layerTypes = decodedTypes
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
        try c.encode(eosTokenId, forKey: .eosTokenId)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try c.encodeIfPresent(normTopkProb, forKey: .normTopkProb)
        try c.encode(linearNumKeyHeads, forKey: .linearNumKeyHeads)
        try c.encode(linearNumValueHeads, forKey: .linearNumValueHeads)
        try c.encode(linearKeyHeadDim, forKey: .linearKeyHeadDim)
        try c.encode(linearValueHeadDim, forKey: .linearValueHeadDim)
        try c.encode(linearConvKernelDim, forKey: .linearConvKernelDim)
        try c.encode(hcCount, forKey: .hcCount)
        try c.encode(hcLowrank, forKey: .hcLowrank)
        try c.encode(indexerNHeads, forKey: .indexerNHeads)
        try c.encode(indexerKVHeads, forKey: .indexerKVHeads)
        try c.encode(indexerHeadDim, forKey: .indexerHeadDim)
        try c.encode(indexerBudget, forKey: .indexerBudget)
        try c.encode(indexerCompressRatio, forKey: .indexerCompressRatio)
        try c.encode(ngramSize, forKey: .ngramSize)
        try c.encode(headsPerNgram, forKey: .headsPerNgram)
        try c.encode(ngramVocabSizeBase, forKey: .ngramVocabSizeBase)
        try c.encode(ngramVocabSizeDivisor, forKey: .ngramVocabSizeDivisor)
        try c.encode(splitNgramParts, forKey: .splitNgramParts)
        try c.encode(pleEmbedDim, forKey: .pleEmbedDim)
        try c.encode(pleLayerIds, forKey: .pleLayerIds)
        try c.encode(pleConvKernelSize, forKey: .pleConvKernelSize)
        try c.encode(seed, forKey: .seed)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(outputGateType, forKey: .outputGateType)
        try c.encode(hiddenAct, forKey: .hiddenAct)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(attentionBias, forKey: .attentionBias)
    }
}

extension Qwen4ExpConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "Qwen4ExpConfiguration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "qwen4_exp: " + message])
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "linear_attention" })
        else {
            throw fail("layer_types needs one full_attention or linear_attention entry per layer")
        }
        guard layerTypes.contains("full_attention"), layerTypes.contains("linear_attention") else {
            throw fail("the port expects both full and linear attention layers")
        }
        guard ropeType == "default" else {
            throw fail("rope type \(ropeType) is not supported (the reference uses plain RoPE)")
        }
        guard !attentionBias else { throw fail("attention_bias is not supported") }
        guard hiddenAct == "silu" else { throw fail("hidden_act \(hiddenAct) is not supported") }
        guard outputGateType == "sigmoid" || outputGateType == "silu" else {
            throw fail("output_gate_type \(outputGateType) is not supported")
        }
        // The reference always renormalizes the top-k (a softmax over them).
        if normTopkProb == false {
            throw fail("norm_topk_prob false is not supported")
        }
        guard indexerKVHeads == 1 else { throw fail("indexer_kv_heads must be 1") }
        guard indexerCompressRatio >= 1, indexerBudget >= indexerCompressRatio else {
            throw fail("indexer_budget must be at least indexer_compress_ratio")
        }
        guard rotaryDimensions >= 2, rotaryDimensions <= indexerHeadDim,
            rotaryDimensions <= headDim
        else {
            throw fail("the rotary dimensions must fit the attention and indexer heads")
        }
        guard numExpertsPerTok >= 1, numExpertsPerTok <= numExperts else {
            throw fail("num_experts_per_tok must be in 1...num_experts")
        }
        guard linearNumValueHeads % linearNumKeyHeads == 0 else {
            throw fail("linear_num_value_heads must be a multiple of linear_num_key_heads")
        }
        guard hcCount >= 1, hcLowrank >= 1 else { throw fail("hc_count and hc_lowrank must be >= 1") }
        guard ngramSize >= 2, headsPerNgram >= 1 else {
            throw fail("ngram_size must be >= 2 and heads_per_ngram >= 1")
        }
        let ngramHeads = (ngramSize - 1) * headsPerNgram
        guard pleEmbedDim % ngramHeads == 0 else {
            throw fail("ple_embed_dim must be a multiple of the n-gram head count")
        }
        guard splitNgramParts >= 1 else { throw fail("split_ngram_parts must be >= 1") }
        guard Set(pleLayerIds).count == pleLayerIds.count else {
            throw fail("ple_layer_ids has duplicates")
        }
        for id in pleLayerIds {
            // A PLE layer keeps its state in the linear-attention cache.
            guard id >= 1, id <= hiddenLayers, layerTypes[id - 1] == "linear_attention" else {
                throw fail("ple layer \(id) must be a linear-attention layer")
            }
        }
        // `token * multiplier` must fit in int64.
        guard vocabularySize >= 1, vocabularySize < Int(Int32.max) else {
            throw fail("vocab_size out of range")
        }
    }
}

// MARK: - Chat conventions

extension Qwen4ExpTextModel {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}

extension Qwen4ExpModel {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
