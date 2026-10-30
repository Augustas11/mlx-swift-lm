// Copyright © 2026 Apple Inc.

import MLX

/// One independent threadgroup per query/head. In particular, changing the
/// batch size never changes a dot product's reduction or its key order.
/// MLXFast's default ensureRowContiguous packs views before this flat indexing;
/// it preserves each input's dtype (in particular, bf16 keys stay bf16).
private let deepseekV4SparseAttentionKernel = MLXFast.metalKernel(
    name: "deepseek_v4_sparse_attention",
    inputNames: ["q", "keys", "valid", "sink", "scale"], outputNames: ["out"],
    source: """
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint h = threadgroup_position_in_grid.x % H;
        const uint l = threadgroup_position_in_grid.x / H;
        const uint b = threadgroup_position_in_grid.y;
        const size_t qi = ((size_t(b) * H + h) * L + l) * D;
        const size_t ki = (size_t(b) * L + l) * K;
        threadgroup float scores[64];
        threadgroup float weights[64];
        threadgroup float state[3]; // running max, accumulator rescale, denominator
        float acc[(D + 255) / 256];
        for (uint j = 0; j < (D + 255) / 256; ++j) { acc[j] = 0.0f; }
        if (tid == 0) { state[0] = -INFINITY; state[2] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint start = 0; start < K; start += 64) {
            for (uint j = sg; j < 64; j += 8) {
                const uint k = start + j;
                float dot = 0.0f;
                const bool present = k < K && valid[ki + k];
                if (present) {
                    for (uint d = lane; d < D; d += 32) {
                        const float query = float(q[qi + d]);
                        dot += query * float(keys[(ki + k) * D + d]);
                    }
                }
                dot = simd_sum(dot);
                // The reference scales the completed float32 score reduction.
                // MLX lowers the rank-zero scale input to a scalar value.
                dot *= scale;
                if (lane == 0) { scores[j] = present ? dot : -INFINITY; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float peak = state[0];
                for (uint j = 0; j < 64; ++j) { peak = max(peak, scores[j]); }
                // An empty first block must not evaluate exp(-inf - -inf).
                const float rescale = state[0] == -INFINITY ? 0.0f : exp(state[0] - peak);
                float norm = state[2] * rescale;
                for (uint j = 0; j < 64; ++j) {
                    const float w = scores[j] == -INFINITY ? 0.0f : exp(scores[j] - peak);
                    norm += w;
                    // Round-to-nearest-even bf16, without materializing widened keys.
                    const uint bits = as_type<uint>(w);
                    weights[j] = as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
                }
                state[0] = peak;
                state[1] = rescale;
                state[2] = norm;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < (D + 255) / 256; ++j) {
                const uint d = tid + j * 256;
                if (d < D) {
                    float value = acc[j] * state[1];
                    for (uint k = 0; k < 64 && start + k < K; ++k) {
                        value += weights[k] * float(keys[(ki + start + k) * D + d]);
                    }
                    acc[j] = value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Sink participates only in the denominator, never in the bf16 rounding.
        const float peak = max(state[0], float(sink[h]));
        const float rescale = state[0] == -INFINITY ? 0.0f : exp(state[0] - peak);
        const float norm = state[2] * rescale + exp(float(sink[h]) - peak);
        for (uint j = 0; j < (D + 255) / 256; ++j) {
            const uint d = tid + j * 256;
            if (d < D) { out[qi + d] = (acc[j] * rescale) / norm; }
        }
        """
)

/// Decode and bounded packed verification use the same arithmetic for every
/// row. Large prefill and unsupported inputs retain the graph implementation.
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
    return deepseekV4SparseAttentionKernel(
        [q, keys, valid, sink.asType(.float32), MLXArray(scale)],
        template: [("H", H), ("L", L), ("D", D), ("K", K)],
        grid: (H * L * 256, B, 1), threadGroup: (256, 1, 1),
        outputShapes: [q.shape], outputDTypes: [.float32])[0]
}
