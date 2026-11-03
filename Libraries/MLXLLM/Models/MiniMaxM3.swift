//
//  MiniMaxM3.swift
//  mlx-swift-lm
//
//  MiniMax M3 (M3.1-Flash) text model: GQA attention with per-head Gemma
//  QK-norm and partial RoPE, a lightning indexer that picks the top key blocks
//  each query attends (block-sparse attention), and sigmoid-routed MoE with a
//  shared expert and a clamped SwiGLU ("swigluoai").
//
//  Port of mlx-vlm 0.7.11 `mlx_vlm/models/minimax_m3_vl/language.py` (MIT),
//  text path only, cross-checked against the transformers
//  `modeling_minimax_m3_vl.py`. The BF16 rounding order of the norm, the
//  activation and the expert combine follows mlx-vlm so greedy decoding
//  matches it.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Norm and activation

/// An array derived from parameters at `prepare()`. A reference type, so
/// module reflection never sees it as a parameter.
final class MiniMaxM3DerivedArray: @unchecked Sendable {
    var value: MLXArray?
}

/// RMSNorm; the Gemma form scales by `1 + weight`, added in float32, and
/// rounds once to the activation dtype.
final class MiniMaxM3RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    let gemma: Bool
    let scale = MiniMaxM3DerivedArray()

    init(dimensions: Int, eps: Float, gemma: Bool) {
        self.weight = gemma ? MLXArray.zeros([dimensions]) : MLXArray.ones([dimensions])
        self.eps = eps
        self.gemma = gemma
        super.init()
    }

    /// Cache `1 + weight` in float32; call again after any weight update.
    func prepare() -> MLXArray? {
        scale.value = gemma ? weight.asType(.float32) + 1 : nil
        return scale.value
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard gemma else {
            return MLXFast.rmsNorm(x, weight: weight, eps: eps).asType(x.dtype)
        }
        if let w = scale.value, let out = miniMaxM3RMSNorm(x, scale: w, eps: eps) {
            return out
        }
        let w = scale.value ?? (weight.asType(.float32) + 1)
        return MLXFast.rmsNorm(x, weight: w, eps: eps).asType(x.dtype)
    }
}

/// `MLXFast.rmsNorm` of a half-precision `x` against a float32 scale, cast
/// back to `x`'s dtype, in one launch instead of three (cast to float32,
/// float32 norm, cast back). Each row replays MLX's `rms_single_row` (rows
/// up to 4096) or `rms_looped` arithmetic with the same threadgroup, so the
/// result is bitwise the three-op chain.
private let miniMaxM3RMSNormSource = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint axis_size = AXIS;
        uint gid = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint lsize = threads_per_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        const device T* xr = x + gid * size_t(axis_size) + lid * N_READS;
        const device float* wr = w + lid * N_READS;
        device T* outr = out + gid * size_t(axis_size) + lid * N_READS;

        float acc = 0;
        float thread_x[N_READS];
        if (!LOOPED) {
          if (lid * N_READS + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              thread_x[i] = xr[i];
              acc += thread_x[i] * thread_x[i];
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              thread_x[i] = (lid * N_READS + i < axis_size) ? (float)xr[i] : 0;
              acc += thread_x[i] * thread_x[i];
            }
          }
        } else {
          for (uint r = 0; r < axis_size; r += lsize * N_READS) {
            if (r + lid * N_READS + N_READS <= axis_size) {
              for (int i = 0; i < N_READS; i++) {
                float xi = xr[i + r];
                acc += xi * xi;
              }
            } else {
              for (int i = 0; i < N_READS; i++) {
                if ((r + lid * N_READS + i) < axis_size) {
                  float xi = xr[i + r];
                  acc += xi * xi;
                }
              }
            }
          }
        }
        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!LOOPED) {
          for (int i = 0; i < N_READS; i++) {
            if (lid * N_READS + i < axis_size) {
              float normed = thread_x[i] * local_inv_mean[0];
              outr[i] = static_cast<T>(wr[i] * normed);
            }
          }
        } else {
          for (uint r = 0; r < axis_size; r += lsize * N_READS) {
            for (int i = 0; i < N_READS; i++) {
              if ((r + lid * N_READS + i) < axis_size) {
                float normed = float(xr[r + i]) * local_inv_mean[0];
                outr[r + i] = static_cast<T>(wr[r + i] * normed);
              }
            }
          }
        }
    """

private final class MiniMaxM3RMSNormKernel: @unchecked Sendable {
    static let shared = MiniMaxM3RMSNormKernel()

    /// MLX's `RMS_LOOPED_LIMIT` and its looped threadgroup.
    static let loopedLimit = 4096
    static let loopedThreads = 1024

    let kernel = MLXFast.metalKernel(
        name: "minimax_m3_rms_norm",
        inputNames: ["x", "w", "eps"],
        outputNames: ["out"],
        source: miniMaxM3RMSNormSource,
        header: "#include <metal_simdgroup>\nusing namespace metal;\n")
}

func miniMaxM3RMSNorm(_ x: MLXArray, scale: MLXArray, eps: Float) -> MLXArray? {
    let axis = x.dim(-1)
    guard x.ndim >= 1, axis > 0, x.dtype == .bfloat16 || x.dtype == .float16,
        scale.dtype == .float32, scale.shape == [axis]
    else { return nil }
    let rows = x.size / axis
    guard rows > 0 else { return nil }
    let looped = axis > MiniMaxM3RMSNormKernel.loopedLimit
    let threads =
        looped ? MiniMaxM3RMSNormKernel.loopedThreads : ((axis + 3) / 4 + 31) / 32 * 32
    return MiniMaxM3RMSNormKernel.shared.kernel(
        [x, scale, MLXArray([eps])],
        template: [("T", x.dtype), ("AXIS", axis), ("LOOPED", looped)],
        grid: (threads, rows, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [x.shape],
        outputDTypes: [x.dtype])[0]
}

/// The per-head norms of up to four `[B, L, H_t, D]` tensors (`x0` ...
/// `x3`, `H_t` heads each, scale `w_t`), written as one `[B, sum H_t, L, D]`
/// tensor: the concatenated, head-major input the attention ropes in one
/// call. Each row is `miniMaxM3RMSNormSource`'s single-row arithmetic.
private let miniMaxM3HeadNormsSource = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int HEADS = H0 + H1 + H2 + H3;
        const uint axis_size = AXIS;
        uint row = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        // Output rows are (b, head, l); find the source tensor and its row.
        uint l = row % SEQ;
        uint head = (row / SEQ) % HEADS;
        uint b = row / (SEQ * HEADS);
        const device T* xr;
        const device float* wr;
        if (head < H0) {
          xr = x0 + ((b * SEQ + l) * H0 + head) * size_t(axis_size);
          wr = w0;
        } else if (head < H0 + H1) {
          xr = x1 + ((b * SEQ + l) * H1 + (head - H0)) * size_t(axis_size);
          wr = w1;
        } else if (head < H0 + H1 + H2) {
          xr = x2 + ((b * SEQ + l) * H2 + (head - H0 - H1)) * size_t(axis_size);
          wr = w2;
        } else {
          xr = x3 + ((b * SEQ + l) * H3 + (head - H0 - H1 - H2)) * size_t(axis_size);
          wr = w3;
        }
        xr += lid * N_READS;
        wr += lid * N_READS;
        device T* outr = out + row * size_t(axis_size) + lid * N_READS;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        float acc = 0;
        float thread_x[N_READS];
        if (lid * N_READS + N_READS <= axis_size) {
          for (int i = 0; i < N_READS; i++) {
            thread_x[i] = xr[i];
            acc += thread_x[i] * thread_x[i];
          }
        } else {
          for (int i = 0; i < N_READS; i++) {
            thread_x[i] = (lid * N_READS + i < axis_size) ? (float)xr[i] : 0;
            acc += thread_x[i] * thread_x[i];
          }
        }
        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < N_READS; i++) {
          if (lid * N_READS + i < axis_size) {
            float normed = thread_x[i] * local_inv_mean[0];
            outr[i] = static_cast<T>(wr[i] * normed);
          }
        }
    """

private final class MiniMaxM3HeadNormsKernel: @unchecked Sendable {
    static let shared = MiniMaxM3HeadNormsKernel()

    let kernel = MLXFast.metalKernel(
        name: "minimax_m3_head_norms",
        inputNames: ["x0", "x1", "x2", "x3", "w0", "w1", "w2", "w3", "eps"],
        outputNames: ["out"],
        source: miniMaxM3HeadNormsSource,
        header: "#include <metal_simdgroup>\nusing namespace metal;\n")
}

/// Normalize each `[B, L, H_t, D]` tensor in `parts` with its float32
/// `scales[t]` and return them as one `[B, sum H_t, L, D]` tensor (heads in
/// `parts` order), bitwise each norm cast back to the input dtype and
/// concatenated and transposed. `nil` when the shapes are not covered.
func miniMaxM3HeadNorms(_ parts: [MLXArray], scales: [MLXArray], eps: Float) -> MLXArray? {
    guard (1 ... 4).contains(parts.count), scales.count == parts.count,
        let first = parts.first, first.ndim == 4,
        first.dtype == .bfloat16 || first.dtype == .float16
    else { return nil }
    let (B, L, D) = (first.dim(0), first.dim(1), first.dim(3))
    guard D <= MiniMaxM3RMSNormKernel.loopedLimit,
        parts.allSatisfy({
            $0.ndim == 4 && $0.dim(0) == B && $0.dim(1) == L && $0.dim(3) == D
                && $0.dtype == first.dtype
        }),
        scales.allSatisfy({ $0.dtype == .float32 && $0.shape == [D] })
    else { return nil }
    var heads = parts.map { $0.dim(2) }
    var inputs = parts
    var weights = scales
    while inputs.count < 4 {
        heads.append(0)
        inputs.append(first)
        weights.append(scales[0])
    }
    let total = heads.reduce(0, +)
    let threads = ((D + 3) / 4 + 31) / 32 * 32
    return MiniMaxM3HeadNormsKernel.shared.kernel(
        inputs + weights + [MLXArray([eps])],
        template: [
            ("T", first.dtype), ("AXIS", D), ("SEQ", L),
            ("H0", heads[0]), ("H1", heads[1]), ("H2", heads[2]), ("H3", heads[3]),
        ],
        grid: (threads, B * total * L, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[B, total, L, D]],
        outputDTypes: [first.dtype])[0]
}

/// `(clip(up, ±limit) + beta) * g * sigmoid(alpha * g)` with `g = min(gate,
/// limit)`, each step rounded to the activation dtype as eager mlx-vlm does.
func makeMiniMaxM3SwiGLU(alpha: Float, limit: Float, beta: Float)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: true) { gate, up in
        let dtype = gate.dtype
        let glu = clip(gate, max: limit)
        let linear = clip(up, min: -limit, max: limit)
        let scaled = (glu.asType(.float32) * alpha).asType(dtype)
        let sig = sigmoid(scaled.asType(.float32)).asType(dtype)
        let g = (glu.asType(.float32) * sig.asType(.float32)).asType(dtype)
        let l = (linear.asType(.float32) + beta).asType(dtype)
        return (l.asType(.float32) * g.asType(.float32)).asType(dtype)
    }
}

final class MiniMaxM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(hiddenSize: Int, intermediateSize: Int, activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x), upProj(x)))
    }
}

// MARK: - MoE

final class MiniMaxM3SparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let routedScalingFactor: Float
    let sigmoidScoring: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiniMaxM3MLP?
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray?

    /// `[gates, bias] -> [indices, weights]`: top-k on the biased scores,
    /// weights from the unbiased ones, normalized.
    private let route: @Sendable ([MLXArray]) -> [MLXArray]
    /// `(outputs [.., k, D], weights [.., k]) -> [.., D]`.
    private let combine: @Sendable (MLXArray, MLXArray) -> MLXArray

    init(_ config: MiniMaxM3Configuration, activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray) {
        self.topK = config.numExpertsPerTok
        self.routedScalingFactor = config.routedScalingFactor
        self.sigmoidScoring = config.scoringFunc == "sigmoid"
        _gate.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.intermediateSize,
            numExperts: config.numLocalExperts, activationProduct: activation)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = MiniMaxM3MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.sharedIntermediateSize,
                activation: activation)
        }
        if config.useRoutingBias {
            _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.numLocalExperts])
        }

        let k = topK
        let sigmoidScoring = sigmoidScoring
        route = compile { arrays in
            let gates = arrays[0]
            let scores = sigmoidScoring ? sigmoid(gates) : softmax(gates, axis: -1, precise: true)
            let biased = arrays.count > 1 ? scores + arrays[1] : scores
            let indices = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var weights = takeAlong(scores, indices, axis: -1)
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            // Ascending expert order: the reference accumulates experts by ID.
            let order = argSort(indices, axis: -1)
            return [takeAlong(indices, order, axis: -1), takeAlong(weights, order, axis: -1)]
        }
        let factor = routedScalingFactor
        combine = compile { outputs, weights in
            // Each weighted expert output and each running sum round to the
            // activation dtype; the routed scale applies last.
            let dtype = outputs.dtype
            let weighted = (outputs.asType(.float32) * expandedDimensions(weights, axis: -1))
                .asType(dtype)
            var result = weighted[.ellipsis, 0, 0...]
            for i in 1 ..< weighted.dim(-2) {
                result = (result.asType(.float32) + weighted[.ellipsis, i, 0...].asType(.float32))
                    .asType(dtype)
            }
            return (result.asType(.float32) * factor).asType(dtype)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x).asType(.float32)
        let routed = route([gates] + (eScoreCorrectionBias.map { [$0] } ?? []))
        var y = combine(switchMLP(x, routed[0]), routed[1])
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Block-sparse attention

/// Block-sparse attention over the selected key blocks of each query, one
/// threadgroup per (row, head, query) with an online softmax in float32.
/// Port of mlx-vlm's `_minimax_m3_sparse_prefill_attention` kernel.
private let miniMaxM3SparseAttentionSource = """
        uint row_idx = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int qk_per_thread = D_SIZE / BD;
        constexpr int v_per_thread = D_SIZE / BD;

        typedef float U;
        thread U q[qk_per_thread];
        thread U o[v_per_thread];
        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];

        int key_length = int(k_size[0]);
        int query_idx = int(row_idx % Q_LEN);
        int batch_head_idx = int(row_idx / Q_LEN);
        int batch_idx = batch_head_idx / NUM_Q_HEADS;
        int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
        int kv_head_idx = q_head_idx / GQA_FACTOR;

        const device T* qptr =
            queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
            int(simd_lid) * qk_per_thread;
        device T* optr =
            out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
            int(simd_gid) * v_per_thread;

        U s = U(scale[0]);
        for (int i = 0; i < qk_per_thread; i++) {
            q[i] = s * static_cast<U>(qptr[i]);
        }
        for (int i = 0; i < v_per_thread; i++) {
            o[i] = 0;
        }

        int qpos = int(q_positions[batch_idx * Q_LEN + query_idx]);
        int index_head_idx = q_head_idx / INDEX_GROUP_SIZE;
        int blocks_offset =
            ((batch_idx * NUM_INDEX_HEADS + index_head_idx) * Q_LEN + query_idx) *
            TOPK_BLOCKS;

        U max_score = -3.4028234663852886e38f;
        U sum_exp_score = 0;

        for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
             selected_idx += BN) {
            int block_slot = selected_idx / BLOCK_SIZE;
            int block_offset = selected_idx - block_slot * BLOCK_SIZE;
            int block_idx = int(block_indices[blocks_offset + block_slot]);
            int key_pos = block_idx * BLOCK_SIZE + block_offset;
            bool valid = block_idx >= 0 && key_pos < key_length && key_pos <= qpos;

            U score = -3.4028234663852886e38f;
            if (valid) {
                const device T* kptr =
                    keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                             key_pos) *
                            D_SIZE) +
                    int(simd_lid) * qk_per_thread;
                score = 0;
                for (int j = 0; j < qk_per_thread; j++) {
                    score += q[j] * static_cast<U>(kptr[j]);
                }
                score = simd_sum(score);
            }

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = valid ? fast::exp(score - new_max) : U(0);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            if (valid) {
                const device T* vptr =
                    values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                               key_pos) *
                              D_SIZE) +
                    int(simd_lid) * v_per_thread;
                for (int j = 0; j < v_per_thread; j++) {
                    o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
                }
            } else {
                for (int j = 0; j < v_per_thread; j++) {
                    o[j] = o[j] * factor;
                }
            }
        }

        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_score = max_scores[simd_lid];
        U new_max = simd_max(max_score);
        U factor = fast::exp(max_score - new_max);
        U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

        for (int i = 0; i < v_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < v_per_thread; i++) {
                optr[i] = static_cast<T>(o[i]);
            }
        }
    """

private final class MiniMaxM3SparseAttentionKernel: @unchecked Sendable {
    static let shared = MiniMaxM3SparseAttentionKernel()

    let kernel = MLXFast.metalKernel(
        name: "minimax_m3_sparse_attention",
        inputNames: [
            "queries", "keys", "values", "block_indices", "q_positions", "scale", "k_size",
        ],
        outputNames: ["out"],
        source: miniMaxM3SparseAttentionSource,
        header: "#include <metal_simdgroup>\nusing namespace metal;\n")
}

/// `[B, Hq, L, D]` attention of `queries` over the keys of the selected
/// blocks (`blockIndices`: `[B, Hi, L, K]`, `-1` unused) at or before each
/// query position; `nil` when the shapes are not covered.
func miniMaxM3SparseAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, blockIndices: MLXArray,
    queryPositions: MLXArray, scale: Float, blockSize: Int
) -> MLXArray? {
    guard queries.ndim == 4, keys.ndim == 4, values.shape == keys.shape,
        blockIndices.ndim == 4, queryPositions.ndim == 2,
        queries.dtype == .bfloat16 || queries.dtype == .float16,
        keys.dtype == queries.dtype, values.dtype == queries.dtype
    else { return nil }
    let (B, qHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let kvHeads = keys.dim(1)
    let indexHeads = blockIndices.dim(1)
    let topk = blockIndices.dim(3)
    guard blockIndices.dim(0) == B, blockIndices.dim(2) == L, queryPositions.shape == [B, L],
        keys.dim(3) == D, D % 32 == 0, qHeads % kvHeads == 0, qHeads % indexHeads == 0
    else { return nil }
    let keyLength = keys.dim(2)
    return MiniMaxM3SparseAttentionKernel.shared.kernel(
        [
            contiguous(queries), contiguous(keys), contiguous(values),
            contiguous(blockIndices.asType(.int32)), contiguous(queryPositions.asType(.int32)),
            MLXArray([scale]), MLXArray([Int32(keyLength)]),
        ],
        template: [
            ("T", queries.dtype), ("D_SIZE", D), ("Q_LEN", L), ("NUM_Q_HEADS", qHeads),
            ("NUM_KV_HEADS", kvHeads), ("GQA_FACTOR", qHeads / kvHeads),
            ("NUM_INDEX_HEADS", indexHeads), ("INDEX_GROUP_SIZE", qHeads / indexHeads),
            ("BLOCK_SIZE", blockSize), ("TOPK_BLOCKS", topk),
            ("SELECTED_LENGTH", topk * blockSize),
        ],
        grid: (1024, B * qHeads * L, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [queries.shape],
        outputDTypes: [queries.dtype])[0]
}

private let miniMaxM3MaskIndexScores: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { scores, keyPositions, queryPositions in
        MLX.where(keyPositions .<= queryPositions, scores, MLXArray(-Float.infinity))
    }

final class MiniMaxM3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    let sparse: Bool
    let indexHeads: Int
    let indexDim: Int
    let blockSize: Int
    let topkBlocks: Int
    let initBlocks: Int
    let localBlocks: Int

    /// Query/key length ratios at which the reference switches from a
    /// block mask over dense SDPA to the sparse kernel (prefill, decode).
    static let prefillKernelRatio = 7
    static let decodeKernelRatio = 64

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MiniMaxM3RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: MiniMaxM3RMSNorm?

    @ModuleInfo(key: "index_q_proj") var indexQProj: Linear?
    @ModuleInfo(key: "index_k_proj") var indexKProj: Linear?
    @ModuleInfo(key: "index_q_norm") var indexQNorm: MiniMaxM3RMSNorm?
    @ModuleInfo(key: "index_k_norm") var indexKNorm: MiniMaxM3RMSNorm?

    let rope: RoPE

    init(_ config: MiniMaxM3Configuration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        sparse = config.sparseLayers[layer]
        indexHeads = config.indexHeads
        indexDim = config.indexHeadDim
        blockSize = config.blockSize
        topkBlocks = config.topkBlocks
        initBlocks = config.initBlocks
        localBlocks = config.localBlocks

        let h = config.hiddenSize
        _qProj.wrappedValue = Linear(h, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(h, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, h, bias: false)
        if config.useQkNorm {
            _qNorm.wrappedValue = MiniMaxM3RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
            _kNorm.wrappedValue = MiniMaxM3RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
        }
        if sparse {
            _indexQProj.wrappedValue = Linear(h, indexHeads * indexDim, bias: false)
            _indexKProj.wrappedValue = Linear(h, indexDim, bias: false)
            _indexQNorm.wrappedValue = MiniMaxM3RMSNorm(
                dimensions: indexDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
            _indexKNorm.wrappedValue = MiniMaxM3RMSNorm(
                dimensions: indexDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
        }
        rope = RoPE(dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    var norms: [MiniMaxM3RMSNorm] {
        [qNorm, kNorm, indexQNorm, indexKNorm].compactMap { $0 }
    }

    /// Top `topkBlocks` key blocks (`[B, Hi, L, K]`, int32, `-1` unused) for
    /// the queries at `qStart ..< qStart + L`: the block score is the max
    /// causal index score in the block; the `localBlocks` blocks ending at the
    /// query's own block are always kept.
    func selectBlocks(queries: MLXArray, keys: MLXArray, qStart: Int, sorted: Bool) -> MLXArray {
        let (B, H, L) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let total = keys.dim(2)
        var scores = matmul(queries.asType(.float32), keys.asType(.float32).swappedAxes(-1, -2))
        let queryPositions = arange(qStart, qStart + L, dtype: .int32).reshaped(1, 1, L, 1)
        scores = miniMaxM3MaskIndexScores(
            scores * scale, arange(total, dtype: .int32), queryPositions)

        let fullBlocks = total / blockSize
        let tail = total % blockSize
        let numBlocks = fullBlocks + (tail > 0 ? 1 : 0)
        var blockScores = scores[.ellipsis, ..<(fullBlocks * blockSize)]
            .reshaped(B, H, L, fullBlocks, blockSize).max(axis: -1)
        if tail > 0 {
            blockScores = concatenated(
                [blockScores, scores[.ellipsis, (fullBlocks * blockSize)...].max(axis: -1, keepDims: true)],
                axis: -1)
        }

        let current = floorDivide(queryPositions, MLXArray(Int32(blockSize)))
        let blocks = arange(numBlocks, dtype: .int32)
        let causal = blocks .<= current
        let negInf = MLXArray(-Float.infinity)
        var selected = MLX.where(causal .&& (blockScores .== blockScores), blockScores, negInf)
        if initBlocks > 0 {
            selected = MLX.where(
                (blocks .< MLXArray(Int32(initBlocks))) .&& causal, MLXArray(Float(1e30)),
                selected)
        }
        if localBlocks > 0 {
            let first = maximum(current - MLXArray(Int32(localBlocks - 1)), MLXArray(Int32(0)))
            selected = MLX.where((blocks .>= first) .&& causal, MLXArray(Float(1e29)), selected)
        }
        let unused = MLXArray(Int32(numBlocks))
        var indices = argPartition(-selected, kth: topkBlocks - 1, axis: -1)[
            .ellipsis, ..<topkBlocks
        ].asType(.int32)
        indices = MLX.where(indices .<= current, indices, unused)
        if sorted {
            indices = MLX.sorted(indices, axis: -1)
        }
        return MLX.where(indices .== unused, MLXArray(Int32(-1)), indices)
    }

    /// `[B, heads, L, total]` keep mask of the selected blocks, causal.
    func blockMask(_ blockIndices: MLXArray, total: Int, qStart: Int) -> MLXArray {
        let (B, H, L) = (blockIndices.dim(0), blockIndices.dim(1), blockIndices.dim(2))
        let numBlocks = (total + blockSize - 1) / blockSize
        // Unused slots land in a discarded sentinel column.
        let sentinel = MLXArray(Int32(numBlocks))
        let safe = MLX.where(
            (blockIndices .>= MLXArray(Int32(0))) .&& (blockIndices .< sentinel), blockIndices,
            sentinel)
        let blockKeep = putAlong(
            MLXArray.zeros([B, H, L, numBlocks + 1], dtype: .bool), safe,
            values: MLXArray(true), axis: -1)
        let keyPositions = arange(total, dtype: .int32)
        var keep = blockKeep.take(floorDivide(keyPositions, MLXArray(Int32(blockSize))), axis: -1)
        let queryPositions = arange(qStart, qStart + L, dtype: .int32).reshaped(L, 1)
        keep = keep .&& (keyPositions.reshaped(1, total) .<= queryPositions)
        return repeated(keep, count: heads / H, axis: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, heads, headDim)
        var k = kProj(x).reshaped(B, L, kvHeads, headDim)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache?.ropeOffset
        let qStart = cache?.offset ?? 0
        // Up to topkBlocks blocks every query attends its whole prefix, so
        // the index queries are only needed past that length.
        let selecting = qStart + L > blockSize * topkBlocks

        // Every roped tensor with the norm that precedes its RoPE.
        var raw: [(MLXArray, MiniMaxM3RMSNorm?)] = [(q, qNorm), (k, kNorm)]
        if sparse, let indexKProj, let indexKNorm {
            raw.append((indexKProj(x).reshaped(B, L, 1, indexDim), indexKNorm))
            if selecting, let indexQProj, let indexQNorm {
                raw.append((indexQProj(x).reshaped(B, L, indexHeads, indexDim), indexQNorm))
            }
        }
        let counts = raw.map { $0.0.dim(2) }

        // RoPE is per head and position, so the heads of every roped tensor
        // share one call (a partial-dims RoPE is a copy plus a rotation). The
        // head norms write that concatenated, head-major input in one launch.
        var headMajor: MLXArray?
        let norms = raw.compactMap { $0.1 }
        let scales = norms.compactMap { $0.scale.value }
        if indexDim == headDim, scales.count == raw.count,
            norms.allSatisfy({ $0.eps == norms[0].eps })
        {
            headMajor = miniMaxM3HeadNorms(raw.map { $0.0 }, scales: scales, eps: norms[0].eps)
        }
        func normed() -> [MLXArray] { raw.map { x, norm in norm.map { $0(x) } ?? x } }
        let roped: [MLXArray]
        if indexDim == headDim {
            let all = applyRotaryPosition(
                rope,
                to: headMajor ?? concatenated(normed(), axis: 2).transposed(0, 2, 1, 3),
                offset: offset)
            var start = 0
            roped = counts.map { count in
                defer { start += count }
                return all[0..., start ..< start + count]
            }
        } else {
            roped = normed().map {
                applyRotaryPosition(rope, to: $0.transposed(0, 2, 1, 3), offset: offset)
            }
        }
        q = roped[0]
        k = roped[1]

        var keys = k
        var values = v
        if let cache {
            (keys, values) = cache.update(keys: k, values: v)
        }

        var attentionMask = mask
        if roped.count > 2 {
            var ik = roped[2]
            if let cache {
                guard let sparseCache = cache as? MiniMaxM3Cache else {
                    fatalError("MiniMax M3 sparse layers need a MiniMaxM3Cache, got \(type(of: cache))")
                }
                ik = sparseCache.updateIndex(ik)
            }
            let total = ik.dim(2)
            if roped.count > 3 {
                let iq = roped[3]
                let selectedLength = blockSize * topkBlocks
                let useKernel =
                    L > 1
                    ? B == 1 && total >= selectedLength * Self.prefillKernelRatio
                    : total >= selectedLength * Self.decodeKernelRatio
                let blocks = selectBlocks(queries: iq, keys: ik, qStart: qStart, sorted: useKernel)
                if useKernel {
                    let positions = broadcast(
                        arange(qStart, qStart + L, dtype: .int32).reshaped(1, L), to: [B, L])
                    if let output = miniMaxM3SparseAttention(
                        queries: q, keys: keys, values: values, blockIndices: blocks,
                        queryPositions: positions, scale: scale, blockSize: blockSize)
                    {
                        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
                    }
                }
                attentionMask = .array(blockMask(blocks, total: total, qStart: qStart))
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: attentionMask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Decoder

final class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM3SparseMoeBlock?
    @ModuleInfo(key: "mlp") var mlp: MiniMaxM3MLP?

    init(_ config: MiniMaxM3Configuration, layer: Int) {
        _selfAttn.wrappedValue = MiniMaxM3Attention(config, layer: layer)
        _inputLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
        _postAttentionLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
        let activation = makeMiniMaxM3SwiGLU(
            alpha: config.swigluAlpha, limit: config.swigluLimit, beta: config.swigluBeta)
        if config.moeLayers[layer] {
            _blockSparseMoe.wrappedValue = MiniMaxM3SparseMoeBlock(config, activation: activation)
        } else {
            _mlp.wrappedValue = MiniMaxM3MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.denseIntermediateSize,
                activation: activation)
        }
        super.init()
    }

    var norms: [MiniMaxM3RMSNorm] {
        [inputLayerNorm, postAttentionLayerNorm] + selfAttn.norms
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        if let blockSparseMoe {
            return h + blockSparseMoe(normed)
        }
        return h + mlp!(normed)
    }
}

public final class MiniMaxM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MiniMaxM3RMSNorm

    init(_ config: MiniMaxM3Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { MiniMaxM3DecoderLayer(config, layer: $0) }
        _norm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
        super.init()
    }

    var norms: [MiniMaxM3RMSNorm] { layers.flatMap(\.norms) + [norm] }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// `minimax_m3` checkpoints: `model.*` and `lm_head.*`.
public final class MiniMaxM3TextModel: Module, LLMModel {
    public let vocabularySize: Int
    let configuration: MiniMaxM3Configuration

    @ModuleInfo(key: "model") public var model: MiniMaxM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: MiniMaxM3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _model.wrappedValue = MiniMaxM3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Cache every Gemma norm's `1 + weight`.
    public func prepare() throws {
        eval(model.norms.compactMap { $0.prepare() })
    }

    /// Sparse layers keep the whole context for the indexer, so `maxKVSize`
    /// and KV quantization do not apply to them.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { $0.selfAttn.sparse ? MiniMaxM3Cache() : KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = MiniMaxM3Weights.stackExperts(
            MiniMaxM3Weights.renameScales(weights), configuration: configuration, prefix: "model")
        if configuration.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("lm_head.") }
        }
        return weights
    }

    public var loraLayers: [Module] { model.layers }
}

/// `minimax_m3_vl` checkpoints: the text model under `language_model`, with
/// the vision tower and projectors alongside (dropped).
public final class MiniMaxM3Model: Module, LLMModel {
    public let vocabularySize: Int

    @ModuleInfo(key: "language_model") public var languageModel: MiniMaxM3TextModel

    public init(_ configuration: MiniMaxM3Configuration) {
        self.vocabularySize = configuration.vocabularySize
        _languageModel.wrappedValue = MiniMaxM3TextModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare() throws {
        try languageModel.prepare()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var text = [String: MLXArray]()
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("model.language_model.") {
                key = "language_model." + key.dropFirst("model.language_model.".count)
            }
            guard key.hasPrefix("language_model.") else { continue }
            text[String(key.dropFirst("language_model.".count))] = value
        }
        return Dictionary(
            uniqueKeysWithValues: languageModel.sanitize(weights: text).map {
                ("language_model." + $0.key, $0.value)
            })
    }

    public var loraLayers: [Module] { languageModel.loraLayers }
}

// MARK: - Weight sanitization

enum MiniMaxM3Weights {
    /// FP8 checkpoints: `weight_scale_inv` becomes `scales` and its uint8
    /// weight is viewed as uint32 words, as MLX's quantized layers expect.
    static func renameScales(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for key in weights.keys where key.hasSuffix(".weight_scale_inv") {
            let base = String(key.dropLast(".weight_scale_inv".count))
            if let weight = weights[base + ".weight"], weight.dtype == .uint8,
                weight.dim(-1) % 4 == 0
            {
                weights[base + ".weight"] = weight.view(dtype: .uint32)
            }
            weights[base + ".scales"] = weights.removeValue(forKey: key)
        }
        return weights
    }

    /// Stack per-expert `w1`/`w2`/`w3` tensors into `switch_mlp`.
    static func stackExperts(
        _ weights: [String: MLXArray], configuration: MiniMaxM3Configuration, prefix: String
    ) -> [String: MLXArray] {
        var weights = weights
        for layer in 0 ..< configuration.hiddenLayers {
            let moe = "\(prefix).layers.\(layer).block_sparse_moe"
            for (source, target) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for suffix in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.numLocalExperts).map {
                        "\(moe).experts.\($0).\(source).\(suffix)"
                    }
                    guard keys.allSatisfy({ weights[$0] != nil }) else { continue }
                    weights["\(moe).switch_mlp.\(target).\(suffix)"] = stacked(
                        keys.map { weights.removeValue(forKey: $0)! })
                }
            }
        }
        return weights
    }
}

// MARK: - Configuration

public struct MiniMaxM3Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var denseIntermediateSize: Int
    var sharedIntermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var rotaryDim: Int
    var ropeScaling: [String: StringOrNumber]?
    var vocabularySize: Int
    var tieWordEmbeddings: Bool
    var hiddenAct: String
    var swigluAlpha: Float
    var swigluBeta: Float
    var swigluLimit: Float
    var useQkNorm: Bool
    var qkNormType: String
    var useGemmaNorm: Bool
    var attentionOutputGate: Bool
    var numLocalExperts: Int
    var numExpertsPerTok: Int
    var nSharedExperts: Int
    var scoringFunc: String
    var useRoutingBias: Bool
    var routedScalingFactor: Float
    var packSharedExpert: Bool
    var nvfp4Experts: Bool
    /// Per layer: MoE (else dense MLP).
    var moeLayers: [Bool]
    /// Per layer: block-sparse attention with its own indexer (else dense).
    var sparseLayers: [Bool]
    var indexHeads: Int
    var indexHeadDim: Int
    var blockSize: Int
    var topkBlocks: Int
    var initBlocks: Int
    var localBlocks: Int
    var sparseScoreType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case ropeScaling = "rope_scaling"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
        case swigluAlpha = "swiglu_alpha"
        case swigluBeta = "swiglu_beta"
        case swigluLimit = "swiglu_limit"
        case useQkNorm = "use_qk_norm"
        case qkNormType = "qk_norm_type"
        case useGemmaNorm = "use_gemma_norm"
        case attentionOutputGate = "attention_output_gate"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nSharedExperts = "n_shared_experts"
        case scoringFunc = "scoring_func"
        case useRoutingBias = "use_routing_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case packSharedExpert = "pack_shared_expert"
        case nvfp4Experts = "nvfp4_experts"
        case moeLayers = "moe_layers"
        case sparseLayers = "sparse_layers"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case blockSize = "index_block_size"
        case topkBlocks = "index_topk_blocks"
        case initBlocks = "index_init_blocks"
        case localBlocks = "index_local_blocks"
        case sparseScoreType = "sparse_score_type"
    }

    private enum ExtraKeys: String, CodingKey {
        case textConfig = "text_config"
        case partialRotaryFactor = "partial_rotary_factor"
        case moeLayerFreq = "moe_layer_freq"
        case mlpLayerTypes = "mlp_layer_types"
        case layerTypes = "layer_types"
        case sparseAttentionConfig = "sparse_attention_config"
    }

    private struct SparseAttentionConfig: Codable {
        var useSparseAttention: Bool?
        var sparseIndexDim: Int?
        var sparseNumIndexHeads: Int?
        var sparseTopkBlocks: Int?
        var sparseBlockSize: Int?
        var sparseScoreType: String?
        var sparseInitBlock: Int?
        var sparseLocalBlock: Int?
        var sparseAttentionFreq: [Int]?
        var sparseDisableIndexValue: [Int]?

        enum CodingKeys: String, CodingKey {
            case useSparseAttention = "use_sparse_attention"
            case sparseIndexDim = "sparse_index_dim"
            case sparseNumIndexHeads = "sparse_num_index_heads"
            case sparseTopkBlocks = "sparse_topk_blocks"
            case sparseBlockSize = "sparse_block_size"
            case sparseScoreType = "sparse_score_type"
            case sparseInitBlock = "sparse_init_block"
            case sparseLocalBlock = "sparse_local_block"
            case sparseAttentionFreq = "sparse_attention_freq"
            case sparseDisableIndexValue = "sparse_disable_index_value"
        }
    }

    /// Defaults and derived fields follow mlx-vlm's `TextConfig.__post_init__`.
    public init(from decoder: Decoder) throws {
        // minimax_m3_vl nests the text model under text_config; minimax_m3 is flat.
        let outer = try decoder.container(keyedBy: ExtraKeys.self)
        let nested = outer.contains(.textConfig)
        let c =
            nested
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)
        let extra =
            nested ? try outer.nestedContainer(keyedBy: ExtraKeys.self, forKey: .textConfig) : outer

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        denseIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) ?? 12288
        sharedIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 60
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 5_000_000
        let partialRotaryFactor =
            try extra.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        rotaryDim =
            try c.decodeIfPresent(Int.self, forKey: .rotaryDim)
            ?? Int(Float(headDim) * partialRotaryFactor)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200_064
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "swigluoai"
        swigluAlpha = try c.decodeIfPresent(Float.self, forKey: .swigluAlpha) ?? 1.702
        swigluBeta = try c.decodeIfPresent(Float.self, forKey: .swigluBeta) ?? 1.0
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 7.0
        useQkNorm = try c.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        qkNormType = try c.decodeIfPresent(String.self, forKey: .qkNormType) ?? "per_head"
        useGemmaNorm = try c.decodeIfPresent(Bool.self, forKey: .useGemmaNorm) ?? true
        attentionOutputGate =
            try c.decodeIfPresent(Bool.self, forKey: .attentionOutputGate) ?? false
        numLocalExperts = try c.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 128
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        useRoutingBias = try c.decodeIfPresent(Bool.self, forKey: .useRoutingBias) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.0
        // mlx-vlm packs the shared expert into switch_mlp when unset and the
        // sizes allow it.
        packSharedExpert =
            try c.decodeIfPresent(Bool.self, forKey: .packSharedExpert)
            ?? (nSharedExperts == 1 && sharedIntermediateSize == intermediateSize)
        nvfp4Experts = try c.decodeIfPresent(Bool.self, forKey: .nvfp4Experts) ?? false

        let layers = hiddenLayers
        let defaultFrequency = (0 ..< layers).map { $0 < min(3, layers) ? 0 : 1 }
        let layerTypes = try extra.decodeIfPresent([String].self, forKey: .layerTypes)
        var moeFrequency = try extra.decodeIfPresent([Int].self, forKey: .moeLayerFreq) ?? []
        if moeFrequency.isEmpty {
            if let mlpTypes = try extra.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
                moeFrequency = mlpTypes.map { $0 == "sparse" ? 1 : 0 }
            } else {
                moeFrequency = defaultFrequency
            }
        }
        // Layers past the list are MoE, as in the reference.
        moeLayers = (0 ..< layers).map { $0 >= moeFrequency.count || moeFrequency[$0] != 0 }

        let typeFrequency = layerTypes?.map { $0 == "minimax_m3_sparse" ? 1 : 0 }
        let sparseConfig = try extra.decodeIfPresent(
            SparseAttentionConfig.self, forKey: .sparseAttentionConfig)
        let useSparse: Bool
        let sparseFrequency: [Int]
        if let sparseConfig {
            let frequency =
                sparseConfig.sparseAttentionFreq ?? typeFrequency
                ?? sparseConfig.sparseDisableIndexValue
            useSparse = sparseConfig.useSparseAttention ?? (frequency != nil)
            sparseFrequency = frequency ?? []
        } else {
            useSparse = true
            sparseFrequency = typeFrequency ?? defaultFrequency
        }
        sparseLayers = (0 ..< layers).map {
            useSparse && $0 < sparseFrequency.count && sparseFrequency[$0] != 0
        }
        let aliasHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads)
        let aliasDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim)
        let aliasBlock = try c.decodeIfPresent(Int.self, forKey: .blockSize)
        let aliasTopk = try c.decodeIfPresent(Int.self, forKey: .topkBlocks)
        let aliasLocal = try c.decodeIfPresent(Int.self, forKey: .localBlocks)
        indexHeads = sparseConfig?.sparseNumIndexHeads ?? aliasHeads ?? 4
        indexHeadDim = sparseConfig?.sparseIndexDim ?? aliasDim ?? 128
        blockSize = sparseConfig?.sparseBlockSize ?? aliasBlock ?? 128
        topkBlocks = sparseConfig?.sparseTopkBlocks ?? aliasTopk ?? 16
        initBlocks = sparseConfig?.sparseInitBlock ?? 0
        localBlocks = sparseConfig?.sparseLocalBlock ?? aliasLocal ?? 1
        sparseScoreType = sparseConfig?.sparseScoreType ?? "max"
    }
}

extension MiniMaxM3Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "MiniMaxM3Configuration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard hiddenAct == "swigluoai" else {
            throw fail("MiniMax M3 supports hidden_act swigluoai only, got \(hiddenAct)")
        }
        guard !attentionOutputGate else {
            throw fail("MiniMax M3 attention_output_gate is not supported")
        }
        guard qkNormType == "per_head" else {
            throw fail("MiniMax M3 supports per_head QK norm only, got \(qkNormType)")
        }
        if let type = ropeScaling?["type"] ?? ropeScaling?["rope_type"],
            type != .string("default")
        {
            throw fail("MiniMax M3 rope_scaling \(type) is not supported")
        }
        guard !packSharedExpert else {
            throw fail("MiniMax M3 checkpoints with the shared expert packed into switch_mlp are not supported")
        }
        guard !nvfp4Experts else {
            throw fail("MiniMax M3 NVFP4 routed experts are not supported")
        }
        guard scoringFunc == "sigmoid" || scoringFunc == "softmax" else {
            throw fail("MiniMax M3 scoring_func must be sigmoid or softmax")
        }
        guard numExpertsPerTok >= 1, numExpertsPerTok <= numLocalExperts else {
            throw fail("MiniMax M3 needs 1 <= num_experts_per_tok <= num_local_experts")
        }
        guard attentionHeads % kvHeads == 0, rotaryDim <= headDim, rotaryDim % 2 == 0 else {
            throw fail("MiniMax M3 head layout is inconsistent")
        }
        if sparseLayers.contains(true) {
            guard sparseScoreType == "max" else {
                throw fail("MiniMax M3 supports sparse_score_type max only")
            }
            guard indexHeads >= 1, attentionHeads % indexHeads == 0 else {
                throw fail("MiniMax M3 index heads must divide the attention heads")
            }
            guard blockSize >= 1, topkBlocks >= 1, initBlocks >= 0, localBlocks >= 0,
                rotaryDim <= indexHeadDim
            else {
                throw fail("MiniMax M3 sparse attention sizes are invalid")
            }
        }
    }
}
