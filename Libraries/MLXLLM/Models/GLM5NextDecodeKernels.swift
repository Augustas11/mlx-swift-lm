//
//  GLM5NextDecodeKernels.swift
//  mlx-swift-lm
//
//  Fused one-sequence decode kernels for GLM-5-Next (GLM-5.3-Flash).
//
//  The Metal sources are adapted from oMLX v0.7.1rc1
//  (`omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py`,
//  https://github.com/jundot/omlx), Copyright the omlx contributors, licensed
//  under the Apache License, Version 2.0
//  (http://www.apache.org/licenses/LICENSE-2.0). Modifications: inputs are
//  the separate q/k/v projections and precomputed gate projections instead of
//  one fused projection, and every state input is always present.
//
//  Each kernel replays the arithmetic of the op graph it replaces (rounding
//  points, reduction order, MLX's Sigmoid functor), so the fused decode path
//  and `GLM5NextLinearAttention`'s op path agree; `GLM5NextTests` checks it.
//

import Foundation
import MLX

// MARK: - KDA decode step

/// MLX's Sigmoid functor.
private let glm5NextKernelHeader = """
    template <typename T>
    inline T glm_sigmoid(T x) {
      auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
      return (x < 0) ? y : 1 - y;
    }
    """

/// One threadgroup (1024 threads) per head. Steps: short conv + SiLU, q/k
/// l2norm, forget-gate decay and beta, the per-channel gated delta rule (as
/// `gated_delta_step_vec`), and the gated RMSNorm.
private let glm5NextKDASource = """
      constexpr int CK = 4;
      constexpr int NROW = CK - 1 + TOK;
      constexpr int NP = 3 * QKV;
      constexpr int NPL = DK / 32;
      const uint tid = thread_position_in_threadgroup.x;
      const uint lane = thread_index_in_simdgroup;
      const uint sg = simdgroup_index_in_threadgroup;
      const int h = int(threadgroup_position_in_grid.x);
      const float q_scale = consts[0];
      const float l2_eps = consts[1];
      const float norm_eps = consts[2];
      const float lower = consts[3];
      const float inv_n = consts[4];

      threadgroup T qs[TOK][DK];
      threadgroup T ks[TOK][DK];
      threadgroup T vs[TOK][DK];
      threadgroup T ys[TOK][DK];
      threadgroup float gs[TOK][DK];
      threadgroup T betas[TOK];

      // 1. Short conv over [conv_state; q|k|v] and SiLU.
      if (tid < uint(3 * DK)) {
        const int part = int(tid) / DK;
        const int i = int(tid) % DK;
        const int gc = part * QKV + h * DK + i;
        const device T* src = part == 0 ? q_in : (part == 1 ? k_in : v_in);
        T win[NROW];
        for (int r = 0; r < CK - 1; r++) {
          win[r] = conv_state[r * NP + gc];
        }
        for (int t = 0; t < TOK; t++) {
          win[CK - 1 + t] = src[t * QKV + h * DK + i];
        }
        const device T* w = conv_w + gc * CK;
        for (int t = 0; t < TOK; t++) {
          float acc = 0.0;
          for (int j = 0; j < CK; ++j) {
            acc += static_cast<float>(win[t + j]) * w[j];
          }
          T co = static_cast<T>(acc);
          T sgm = glm_sigmoid<T>(co);
          T sv = co * sgm;
          if (part == 0) {
            qs[t][i] = sv;
          } else if (part == 1) {
            ks[t][i] = sv;
          } else {
            vs[t][i] = sv;
          }
        }
        for (int r = 0; r < CK - 1; r++) {
          conv_state_out[r * NP + gc] = win[TOK + r];
        }
      }

      // 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta.
      if (tid < uint(TOK * DK)) {
        const int t = int(tid) / DK;
        const int i = int(tid) % DK;
        float ea = metal::precise::exp(a_log[h]);
        float af = static_cast<float>(a_pre[t * QKV + h * DK + i]);
        float s1 = af + dt_bias[h * DK + i];
        float s2 = ea * s1;
        float s3 = glm_sigmoid<float>(s2);
        float s4 = lower * s3;
        gs[t][i] = metal::precise::exp(s4);
      }
      if (tid < uint(TOK)) {
        betas[tid] = glm_sigmoid<T>(b_in[tid * H + h]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 2. l2norm(q) * q_scale, l2norm(k): x * rsqrt(sum(x * x) + eps) in fp32.
      if (sg < uint(2 * TOK)) {
        const int t = int(sg) / 2;
        const bool is_q = (sg % 2) == 0;
        threadgroup T* row = is_q ? qs[t] : ks[t];
        float x[NPL];
        float tot = 0.0f;
        for (int e = 0; e < NPL; e++) {
          x[e] = static_cast<float>(row[NPL * lane + e]);
          float sq = x[e] * x[e];
          tot = sq + tot;
        }
        tot = simd_sum(tot);
        float u = tot + l2_eps;
        float r = metal::precise::rsqrt(u);
        for (int e = 0; e < NPL; e++) {
          float xn = x[e] * r;
          if (is_q) {
            float xs = xn * q_scale;
            row[NPL * lane + e] = static_cast<T>(xs);
          } else {
            row[NPL * lane + e] = static_cast<T>(xn);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 4. Per-channel gated delta rule (gated_delta_step_vec arithmetic).
      for (int j = 0; j < DK / 32; j++) {
        const int dv_idx = int(sg) + 32 * j;
        const int dk_idx = int(lane);
        float state[NPL];
        for (int i = 0; i < NPL; ++i) {
          auto s_idx = NPL * dk_idx + i;
          state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
        }
        for (int t = 0; t < TOK; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < NPL; ++i) {
            auto s_idx = NPL * dk_idx + i;
            state[i] = state[i] * gs[t][s_idx];
            kv_mem += state[i] * ks[t][s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

          float out = 0.0f;
          for (int i = 0; i < NPL; ++i) {
            auto s_idx = NPL * dk_idx + i;
            state[i] = state[i] + ks[t][s_idx] * delta;
            out += state[i] * qs[t][s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            ys[t][dv_idx] = static_cast<T>(out);
          }
        }
        for (int i = 0; i < NPL; ++i) {
          auto s_idx = NPL * dk_idx + i;
          state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = state[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 5. Gated RMSNorm: w * (x * rsqrt(mean(x * x) + eps)) * sigmoid(gate).
      if (sg < uint(TOK)) {
        const int t = int(sg);
        float x[NPL];
        float tot = 0.0f;
        for (int e = 0; e < NPL; e++) {
          x[e] = static_cast<float>(ys[t][NPL * lane + e]);
          float sq = x[e] * x[e];
          tot = sq + tot;
        }
        tot = simd_sum(tot);
        float var = tot * inv_n;
        float u = var + norm_eps;
        float r = metal::precise::rsqrt(u);
        for (int e = 0; e < NPL; e++) {
          const int c = NPL * lane + e;
          float xn = x[e] * r;
          float wf = static_cast<float>(norm_w[c]);
          float wx = wf * xn;
          float gf = static_cast<float>(gate_pre[t * QKV + h * DK + c]);
          float gsg = glm_sigmoid<float>(gf);
          float o = wx * gsg;
          y[t * QKV + h * DK + c] = static_cast<T>(o);
        }
      }
    """

private final class GLM5NextKDAKernel: Sendable {
    static let shared = GLM5NextKDAKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "glm5next_kda_decode",
            inputNames: [
                "q_in", "k_in", "v_in", "a_pre", "gate_pre", "b_in", "conv_state", "conv_w",
                "a_log", "dt_bias", "norm_w", "state_in", "consts",
            ],
            outputNames: ["y", "conv_state_out", "state_out"],
            source: glm5NextKDASource,
            header: glm5NextKernelHeader)
    }
}

/// Fused KDA layer body for one sequence and `T <= 8` tokens.
///
/// Inputs are the layer's projections of the normed hidden state, each
/// `[1, T, H * Dk]` (`bRaw`: `[1, T, H]`) in the activation dtype, the
/// `[1, 3, 3 * H * Dk]` conv state and the `[1, H, Dk, Dk]` float32 recurrent
/// state. Returns the gated-normed attention output `[1, T, H * Dk]` (before
/// `o_proj`) and both new states, or `nil` when the shapes are not covered
/// (the caller keeps the op path).
func glm5NextKDADecodeStep(
    q: MLXArray, k: MLXArray, v: MLXArray,
    aPre: MLXArray, gatePre: MLXArray, bRaw: MLXArray,
    convState: MLXArray, convWeight: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, normWeight: MLXArray, state: MLXArray,
    heads: Int, headDim: Int, lowerBound: Float, normEps: Float, l2Eps: Float = 1e-6
) -> (y: MLXArray, convState: MLXArray, state: MLXArray)? {
    let dtype = q.dtype
    guard dtype == .bfloat16 || dtype == .float16, headDim == 128, heads >= 1,
        q.ndim == 3, q.dim(0) == 1, (1 ... 8).contains(q.dim(1))
    else { return nil }
    let T = q.dim(1)
    let qkv = heads * headDim
    for part in [q, k, v, aPre, gatePre] where part.shape != [1, T, qkv] || part.dtype != dtype {
        return nil
    }
    guard bRaw.shape == [1, T, heads], bRaw.dtype == dtype,
        convState.shape == [1, 3, 3 * qkv], convState.dtype == dtype,
        convWeight.shape == [3 * qkv, 4, 1], convWeight.dtype == dtype,
        aLog.size == heads, aLog.dtype == .float32,
        dtBias.size == qkv, dtBias.dtype == .float32,
        normWeight.shape == [headDim],
        state.shape == [1, heads, headDim, headDim], state.dtype == .float32
    else { return nil }

    // Python float scalars, as the reference builds them (double, then float32).
    let consts = MLXArray(
        [Float(pow(Double(headDim), -0.5)), l2Eps, normEps, lowerBound,
         Float(1.0 / Double(headDim))])
    let outputs = GLM5NextKDAKernel.shared.kernel(
        [q, k, v, aPre, gatePre, bRaw, convState, convWeight, aLog.reshaped(-1),
         dtBias.reshaped(-1), normWeight, state, consts],
        template: [
            ("T", dtype), ("TOK", T), ("DK", headDim), ("QKV", qkv), ("H", heads),
        ],
        grid: (1024 * heads, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, T, qkv], [1, 3, 3 * qkv], [1, heads, headDim, headDim]],
        outputDTypes: [dtype, dtype, .float32])
    return (outputs[0], outputs[1], outputs[2])
}
