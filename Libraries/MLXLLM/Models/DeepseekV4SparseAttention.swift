// Copyright © 2026 Apple Inc.

import MLX

/// Canonical float32 attention graph, with bf16-rounded block weights.
func deepseekV4GraphAttention(
    _ q: MLXArray, keys: MLXArray, valid: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray {
    let (B, H, L) = (q.dim(0), q.dim(1), q.dim(2))
    let K = keys.dim(2)
    let block = 64
    let blocks = (K + block - 1) / block
    let keys32 = keys.asType(.float32)
    var scores = matmul(
        q.transposed(0, 2, 1, 3).asType(.float32), keys32.swappedAxes(-1, -2)) * scale
    scores = MLX.where(expandedDimensions(valid, axis: 2), scores, MLXArray(-Float.infinity))
    if blocks * block > K {
        scores = concatenated(
            [scores, MLXArray.full([B, L, H, blocks * block - K], values: MLXArray(-Float.infinity))],
            axis: -1)
    }
    let blocked = scores.reshaped(B, L, H, blocks, block)
    let running = cummax(blocked.max(axis: -1), axis: -1)
    let peak = running[.ellipsis, (blocks - 1)...]
    // Empty leading blocks contribute zero, including an entirely empty list.
    // Finite score maxima retain the reference graph's exact arithmetic.
    let safeRunning = MLX.where(running .== MLXArray(-Float.infinity), MLXArray(Float(0)), running)
    let safePeak = MLX.where(peak .== MLXArray(-Float.infinity), MLXArray(Float(0)), peak)
    let weights = exp(blocked - expandedDimensions(safeRunning, axis: -1))
    let rescale = exp(running - safePeak)
    let normalizer = (weights.sum(axis: -1) * rescale).sum(axis: -1, keepDims: true)
        + exp(sink.asType(.float32).reshaped(1, 1, H, 1) - safePeak)
    let valueWeights = (weights.asType(.bfloat16).asType(.float32)
        * expandedDimensions(rescale, axis: -1))
        .reshaped(B, L, H, blocks * block)[.ellipsis, ..<K]
    return (matmul(valueWeights, keys32) / normalizer).transposed(0, 2, 1, 3)
}

/// Keep each decode query's GEMM geometry identical to ordinary single-row decode.
/// Large prefill and unsupported inputs retain their existing fallback.
func deepseekV4SparseAttention(
    _ q: MLXArray, keys: MLXArray, valid: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu,
        q.ndim == 4, keys.ndim == 4, valid.ndim == 3,
        [DType.float32, .float16, .bfloat16].contains(q.dtype),
        [DType.float32, .float16, .bfloat16].contains(keys.dtype), valid.dtype == .bool
    else { return nil }
    let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let K = keys.dim(2)
    guard (1 ... 8).contains(B), (1 ... 128).contains(H), (1 ... 8).contains(L),
        (32 ... 1024).contains(D), D % 32 == 0, (1 ... 65536).contains(K),
        keys.shape == [B, L, K, D], valid.shape == [B, L, K], sink.size == H
    else { return nil }

    var rows = [MLXArray]()
    for row in 0 ..< B {
        var queries = [MLXArray]()
        for position in 0 ..< L {
            let rowQ = contiguous(q[row ..< (row + 1), 0..., position ..< (position + 1), 0...])
            let rowKeys = contiguous(keys[row ..< (row + 1), position ..< (position + 1), 0..., 0...])
            let rowValid = contiguous(valid[row ..< (row + 1), position ..< (position + 1), 0...])
            queries.append(
                deepseekV4QueryAttention(
                    rowQ, keys: rowKeys, valid: rowValid, sink: sink, scale: scale)
                    ?? deepseekV4GraphAttention(
                        rowQ, keys: rowKeys, valid: rowValid, sink: sink, scale: scale))
        }
        rows.append(queries.count == 1 ? queries[0] : concatenated(queries, axis: 2))
    }
    return rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
}

/// Everything in `deepseekV4GraphAttention` between its two matmuls for one
/// query, in one kernel: scaled and masked scores, 64-key block maxima, the
/// running maximum, block weights and their sequential sums, the rescale,
/// the sequential normalizer with the sink, and the bf16-rounded rescaled
/// value weights. Each threadgroup owns one head. Products are rounded
/// before any add, as in the separate graph ops.
private let deepseekV4AttentionWeightsSource = """
    const uint h = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const int K = sizes[0];
    const int nb = (K + 63) / 64;
    threadgroup float run[64];
    threadgroup float rescale[64];
    threadgroup float product[64];
    threadgroup float safe_peak;
    auto s = raw + size_t(h) * K;
    const float sc = as_type<float>(sizes[1]);
    const float ninf = -INFINITY;
    if (t < uint(nb)) {
      float m = ninf;
      for (int j = int(t) * 64; j < min(K, int(t) * 64 + 64); ++j) {
        float v = valid[j] ? metal::fma(s[j], sc, 0.0f) : ninf;
        m = metal::max(m, v);
      }
      run[t] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      for (int b = 1; b < nb; ++b) { run[b] = metal::max(run[b - 1], run[b]); }
      float p = run[nb - 1];
      safe_peak = p == ninf ? 0.0f : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < uint(nb)) {
      const float r = run[t];
      const float safe = r == ninf ? 0.0f : r;
      const float scale_b = metal::precise::exp(r - safe_peak);
      float total = 0.0f;
      for (int j = int(t) * 64; j < int(t) * 64 + 64; ++j) {
        float w = 0.0f;
        if (j < K) {
          float v = valid[j] ? metal::fma(s[j], sc, 0.0f) : ninf;
          w = metal::precise::exp(v - safe);
          weights[size_t(h) * K + j] =
              metal::fma(static_cast<float>(static_cast<bfloat16_t>(w)), scale_b, 0.0f);
        }
        total = w + total;
      }
      rescale[t] = scale_b;
      product[t] = metal::fma(total, scale_b, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float total = 0.0f;
      for (int b = 0; b < nb; ++b) { total = product[b] + total; }
      normalizer[h] = total + metal::precise::exp(static_cast<float>(sink[h]) - safe_peak);
    }
    """

private final class DeepseekV4AttentionKernels: Sendable {
    static let shared = DeepseekV4AttentionKernels()
    let weights = MLXFast.metalKernel(
        name: "deepseek_v4_attention_weights",
        inputNames: ["raw", "valid", "sink", "sizes"],
        outputNames: ["weights", "normalizer"], source: deepseekV4AttentionWeightsSource)
}

/// `deepseekV4GraphAttention` for one query (`q: [1, H, 1, D]`, keys
/// `[1, 1, K, D]`, validity `[1, 1, K]`) with the same two matmuls and the
/// softmax steps between them fused, or `nil` outside the replayed shapes.
/// At most 64 blocks keep both block-axis sums on MLX's sequential
/// small-row reduction.
func deepseekV4QueryAttention(
    _ q: MLXArray, keys: MLXArray, valid: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray? {
    let (H, D, K) = (q.dim(1), q.dim(3), keys.dim(2))
    guard Device.defaultDevice().deviceType == .gpu, q.ndim == 4, q.dim(0) == 1, q.dim(2) == 1,
        keys.shape == [1, 1, K, D], valid.shape == [1, 1, K], valid.dtype == .bool,
        (1 ... 4096).contains(K), H > 1, sink.size == H,
        [DType.float32, .float16, .bfloat16].contains(sink.dtype)
    else { return nil }
    let keys32 = keys.asType(.float32)
    let raw = matmul(q.transposed(0, 2, 1, 3).asType(.float32), keys32.swappedAxes(-1, -2))
    let fused = DeepseekV4AttentionKernels.shared.weights(
        // `sizes` is [K, scale bits]: a one-element input would bind as a
        // constant scalar rather than a buffer.
        [raw, valid, sink, MLXArray([Int32(K), Int32(bitPattern: scale.bitPattern)])],
        grid: (64, H, 1), threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, H, K], [1, 1, H, 1]], outputDTypes: [.float32, .float32])
    return (matmul(fused[0], keys32) / fused[1]).transposed(0, 2, 1, 3)
}
