// Fused attention inputs for L2-clipped attention: one Metal dispatch that takes a layer's query,
// relative-query, key and value projections to the arrays attention reads:
//
//     q = round(l2_clip(q * q_scale, q_clip)), then round(q * tau)   -> [B, heads, L, D]
//     r = round(l2_clip(r * r_scale, r_clip)), then round(r * tau)   -> [B, L, heads, d_rel]
//     k = round(l2_clip(k, q_clip))                                  -> [B, kv_heads, L, D]
//     v = round(rms_norm(v, None, eps))  (or v unchanged)            -> [B, kv_heads, L, D]
//
// with `l2_clip(v, t) = v * min(t / (|v| + 1e-6), 1)` in float32, `tau` an optional per-position
// temperature and `round` the cast to the projections' dtype.
//
// Ported from Ollama's Inkling 2 relative-attention prologue (Ollama pull request 19843,
// `mlx/relative_attention_inputs.go`; MIT licence, Copyright (c) Ollama), which replaces about a
// dozen clip, norm, scaling and layout dispatches per layer visit with one. This port keeps
// mlx-vlm's roundings (each clip result is rounded before the temperature multiplies it, as
// mlx-vlm's separate `l2_clip` and log-scaling steps do) and leaves the relative bias to the
// caller's band-mask kernel. Its norms reduce in their own order, so it is not bit-identical to
// the op-by-op graph; callers gate it on greedy token parity.

import Foundation
import MLX

private let clippedAttentionInputHeader = """
    // The sum of x over a threadgroup of Groups SIMD groups; every thread gets the total.
    template <uint Groups>
    inline float clipped_inputs_total(float x, threadgroup float* partial, uint lane, uint sg) {
        float s = simd_sum(x);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint g = 0; g < Groups; ++g) total += partial[g];
        return total;
    }
    """

/// One threadgroup of `HeadDim` threads per (head, row); heads below `KVHeads` also produce
/// their key and value rows.
private let clippedAttentionInputSource = """
        constexpr uint Groups = HeadDim / 32;
        const uint i = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint h = threadgroup_position_in_grid.y;
        const uint rt = threadgroup_position_in_grid.z;
        const uint L = q_shape[1];
        const uint b = rt / L, l = rt % L;
        threadgroup float partial[Groups];
        const float qclip = as_type<float>(QClipBits);
        const float rclip = as_type<float>(RClipBits);
        const float tau = HasTau ? float(temperature[l]) : 1.0f;

        // Queries.
        const float qv = float(q[size_t(rt) * (Heads * HeadDim) + h * HeadDim + i])
            * float(q_scale[h * HeadDim + i]);
        const float qn = metal::precise::sqrt(clipped_inputs_total<Groups>(qv * qv, partial, lane, sg));
        float qc = float(T(qv * min(qclip / (qn + 1e-6f), 1.0f)));
        if (HasTau) qc = float(T(qc * tau));
        q_out[((size_t(b) * Heads + h) * L + l) * HeadDim + i] = T(qc);

        // Relative queries.
        const float rv = i < RelDim
            ? float(r[size_t(rt) * (Heads * RelDim) + h * RelDim + i]) * float(r_scale[h * RelDim + i])
            : 0.0f;
        const float rn = metal::precise::sqrt(clipped_inputs_total<Groups>(rv * rv, partial, lane, sg));
        if (i < RelDim) {
            float rc = float(T(rv * min(rclip / (rn + 1e-6f), 1.0f)));
            if (HasTau) rc = float(T(rc * tau));
            r_out[(size_t(rt) * Heads + h) * RelDim + i] = T(rc);
        }

        // Keys and values (uniform per threadgroup, so every thread takes the same branch).
        if (h < KVHeads) {
            const size_t at = size_t(rt) * (KVHeads * HeadDim) + h * HeadDim + i;
            const size_t to = ((size_t(b) * KVHeads + h) * L + l) * HeadDim + i;
            const float kv = float(k[at]);
            const float kn = metal::precise::sqrt(clipped_inputs_total<Groups>(kv * kv, partial, lane, sg));
            k_out[to] = T(kv * min(qclip / (kn + 1e-6f), 1.0f));
            const float vv = float(v[at]);
            if (ValueNorm) {
                const float ms = clipped_inputs_total<Groups>(vv * vv, partial, lane, sg);
                v_out[to] = T(vv * metal::precise::rsqrt(ms / float(HeadDim) + as_type<float>(EpsBits)));
            } else {
                v_out[to] = T(vv);
            }
        }
    """

/// Fused L2-clipped attention inputs (see the file header).
package enum ClippedAttentionInputKernels {
    private static let kernel = MLXFast.metalKernel(
        name: "clipped_attention_inputs",
        inputNames: ["q", "r", "k", "v", "q_scale", "r_scale", "temperature"],
        outputNames: ["q_out", "k_out", "v_out", "r_out"],
        source: clippedAttentionInputSource, header: clippedAttentionInputHeader)

    private static func bits(_ value: Float) -> Int { Int(Int32(bitPattern: value.bitPattern)) }

    /// The prepared inputs, or nil when the kernel does not apply: projections `[B, L, heads * D]`
    /// (`q`), `[B, L, heads * dRel]` (`r`) and `[B, L, kvHeads * D]` (`k`, `v`), all of one
    /// floating dtype; scales `[heads, D]` and `[heads, dRel]`; `temperature` float32 `[L]`;
    /// `D` a multiple of 32 up to 1024, `dRel <= D`, `kvHeads <= heads`, on the GPU.
    ///
    /// - Returns: `q [B, heads, L, D]`, `k` and `v [B, kvHeads, L, D]`, `r [B, L, heads, dRel]`.
    package static func prepare(
        q: MLXArray, r: MLXArray, k: MLXArray, v: MLXArray, queryScale: MLXArray,
        relativeScale: MLXArray, temperature: MLXArray?, heads: Int, kvHeads: Int, headDim: Int,
        relativeDim: Int, queryClip: Float, relativeClip: Float, valueNormEps: Float?
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray)? {
        let dtype = q.dtype
        guard Device.defaultDevice().deviceType == .gpu, dtype == .bfloat16 || dtype == .float16,
            [r, k, v].allSatisfy({ $0.dtype == dtype }), q.ndim == 3, r.ndim == 3, k.ndim == 3,
            v.ndim == 3, headDim % 32 == 0, headDim <= 1024, relativeDim >= 1,
            relativeDim <= headDim, kvHeads >= 1, kvHeads <= heads
        else { return nil }
        let (batch, length) = (q.dim(0), q.dim(1))
        guard q.dim(2) == heads * headDim, r.shape == [batch, length, heads * relativeDim],
            k.shape == [batch, length, kvHeads * headDim], v.shape == k.shape,
            queryScale.size == heads * headDim, relativeScale.size == heads * relativeDim
        else { return nil }
        if let temperature {
            guard temperature.dtype == .float32, temperature.size == length else { return nil }
        }
        let out = kernel(
            [q, r, k, v, queryScale, relativeScale, temperature ?? queryScale],
            template: [
                ("T", dtype), ("Heads", heads), ("KVHeads", kvHeads), ("HeadDim", headDim),
                ("RelDim", relativeDim), ("HasTau", temperature != nil),
                ("ValueNorm", valueNormEps != nil), ("QClipBits", bits(queryClip)),
                ("RClipBits", bits(relativeClip)), ("EpsBits", bits(valueNormEps ?? 0)),
            ],
            grid: (headDim, heads, batch * length), threadGroup: (headDim, 1, 1),
            outputShapes: [
                [batch, heads, length, headDim], [batch, kvHeads, length, headDim],
                [batch, kvHeads, length, headDim], [batch, length, heads, relativeDim],
            ],
            outputDTypes: [dtype, dtype, dtype, dtype])
        return (out[0], out[1], out[2], out[3])
    }
}
