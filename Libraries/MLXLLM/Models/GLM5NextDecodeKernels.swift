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
//  `glm5NextHCNorm`'s source is mlx-vlm 0.7.8's `exact_hc_norm`
//  (`mlx_vlm/models/fast_ops.py`, https://github.com/Blaizzy/mlx-vlm),
//  Copyright (c) 2025 Prince Canuma, MIT License: permission is granted, free
//  of charge, to deal in the software without restriction, provided this
//  notice is included; the software is provided "as is", without warranty of
//  any kind.
//
//  The KDA, router and `glm5NextHCMix` kernels replay the arithmetic of the op graphs
//  they replace (rounding points, reduction order, MLX's Sigmoid functor);
//  `glm5NextHCNorm` is the reference model's own decode kernel (fast exp,
//  FMA collapse). `GLM5NextTests` checks each one.
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

// MARK: - Hyper-connection decode (mix GEMV, Sinkhorn + collapse + RMSNorm)

/// `matmul(rmsNorm(x.flatten(-2)), fn.T)` for one token per threadgroup row:
/// MLX's rms_looped reduction (1024 threads, 4 reads each) and its fp32 gemv
/// lane order (oMLX `hc_mix`).
private let glm5NextHCMixSource = """
      const uint lid = thread_position_in_threadgroup.x;
      const uint simd_lid = thread_index_in_simdgroup;
      const uint simd_gid = simdgroup_index_in_threadgroup;
      const int tok = int(threadgroup_position_in_grid.y);
      const int row = int(threadgroup_position_in_grid.x);
      constexpr int KSZ = HCD;
      const device T* xr = x + size_t(tok) * KSZ;

      threadgroup float local_inv_mean[1];
      threadgroup float local_sums[32];
      for (int v = 0; v < 4; v++) {
        const uint vt = uint(v) * 256 + lid;
        float acc = 0;
        for (uint r = 0; r < uint(KSZ); r += 1024 * 4) {
          for (int i = 0; i < 4; i++) {
            float xi = static_cast<float>(xr[r + vt * 4 + i]);
            acc += xi * xi;
          }
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) {
          local_sums[v * 8 + int(simd_gid)] = acc;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        float acc = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
          local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv = local_inv_mean[0];

      threadgroup float partial[8];
      const int sgN = int(simd_gid);
      float result = 0;
      const device float* mrow = fn + size_t(row) * KSZ;
      int bn = (32 * sgN + int(simd_lid)) * 4;
      for (int i = 0; i < KSZ / 1024; ++i) {
        float v_coeff[4];
        float inter[4];
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(xr[bn + tn]) * inv;
        }
        for (int tn = 0; tn < 4; tn++) {
          inter[tn] = mrow[bn + tn];
        }
        for (int tn = 0; tn < 4; tn++) {
          result += inter[tn] * v_coeff[tn];
        }
        bn += 1024;
      }
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
      }
      if (simd_lid == 0) {
        partial[sgN] = result;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sgN == 0 && simd_lid == 0) {
        float total = partial[0];
        for (int s = 1; s < 8; s++) {
          total += partial[s];
        }
        mixes[size_t(tok) * MIX + row] = total;
      }
    """

/// Sinkhorn (fast exp, as mlx-vlm's decode kernel), FMA collapse rounded to
/// the activation dtype, and the branch's RMSNorm, one token per threadgroup.
private let glm5NextHCNormSource = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float HC_EPS = HC_EPS_INT * 1e-9;
    constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
    constexpr int D4 = D / 4;

    const device float* mix = mixes + row * MIX;
    device float* post_out = post + row * HC;
    device float* comb_out = comb + row * HC * HC;
    threadgroup float pre_shared[HC];
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];

    if (sg == 0) {
      const float pre_scale = scale[0];
      const float post_scale = scale[1];
      const float comb_scale = scale[2];
      const float active = lane < (uint)HC ? 1.0f : 0.0f;
      const uint llane = metal::min(lane, (uint)(HC - 1));

      float pre_z = mix[llane] * pre_scale + base[llane];
      float post_z = mix[HC + llane] * post_scale + base[HC + llane];
      float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
      float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
      if (lane < (uint)HC) {
        pre_shared[lane] = pre_v;
        post_out[lane] = post_v;
      }

      float4 value =
          (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale +
           *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
      float row_max = metal::max(
          metal::max(value.x, value.y), metal::max(value.z, value.w));
      float4 exponent = metal::fast::exp(value - row_max) * active;
      float4 result = exponent *
              (1.0f /
               (exponent.x + exponent.y + exponent.z + exponent.w + HC_EPS)) +
          HC_EPS * active;
      float4 column_inv = 1.0f /
          (float4(
               simd_sum(result.x), simd_sum(result.y),
               simd_sum(result.z), simd_sum(result.w)) +
           HC_EPS);
      result *= column_inv;
      for (int iter = 1; iter < ITERS; ++iter) {
        result *=
            (1.0f / (result.x + result.y + result.z + result.w + HC_EPS)) *
            active;
        column_inv = 1.0f /
            (float4(
                 simd_sum(result.x), simd_sum(result.y),
                 simd_sum(result.z), simd_sum(result.w)) +
             HC_EPS);
        result *= column_inv;
      }
      if (lane < (uint)HC) {
        *(device float4*)(comb_out + lane * HC) = result;
      }
    }

    if (sg == 0) {
      local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device T* x_row = (const device T*)x_in + row * HC * D;
    using T4 = vec<T, 4>;
    const device T4* x0 = (const device T4*)(x_row);
    const device T4* x1 = (const device T4*)(x_row + D);
    const device T4* x2 = (const device T4*)(x_row + 2 * D);
    const device T4* x3 = (const device T4*)(x_row + 3 * D);

    T4 rounded = T4(0);
    float accum = 0.0f;
    if (tid < D4) {
      float4 collapsed = fma(
          float4(pre_shared[0]), float4(x0[tid]),
          fma(
              float4(pre_shared[1]), float4(x1[tid]),
              fma(
                  float4(pre_shared[2]), float4(x2[tid]),
                  float4(pre_shared[3]) * float4(x3[tid]))));
      rounded = T4(collapsed);
      float4 rounded_float = float4(rounded);
      accum += rounded_float.x * rounded_float.x;
      accum += rounded_float.y * rounded_float.y;
      accum += rounded_float.z * rounded_float.z;
      accum += rounded_float.w * rounded_float.w;
    }

    accum = simd_sum(accum);
    if (lane == 0) {
      local_sums[sg] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      accum = simd_sum(local_sums[lane]);
      if (lane == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(accum / D + NORM_EPS);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < D4) {
      float inv = local_inv_mean[0];
      const device T4* weights = (const device T4*)norm_weight;
      T4 scaled = T4(float4(rounded) * inv);
      T4 weight = weights[tid];
      ((device T4*)normalized)[row * D4 + tid] = T4(
          weight.x * scaled.x,
          weight.y * scaled.y,
          weight.z * scaled.z,
          weight.w * scaled.w);
    }
    """

private final class GLM5NextHCKernels: Sendable {
    static let shared = GLM5NextHCKernels()
    let mix: MLXFast.MLXFastKernel
    let norm: MLXFast.MLXFastKernel

    private init() {
        mix = MLXFast.metalKernel(
            name: "glm5next_hc_mix", inputNames: ["x", "fn", "eps"], outputNames: ["mixes"],
            source: glm5NextHCMixSource)
        norm = MLXFast.metalKernel(
            name: "glm5next_hc_norm",
            inputNames: ["x_in", "mixes", "scale", "base", "norm_weight"],
            outputNames: ["normalized", "post", "comb"],
            source: glm5NextHCNormSource)
    }
}

/// `matmul(rmsNorm(x.flatten(-2), eps), fn.T)` for `x: [B, L, HC, D]`:
/// `[B, L, MIX]` float32, or `nil` outside the replicated configuration.
func glm5NextHCMix(_ x: MLXArray, fn: MLXArray, eps: Float) -> MLXArray? {
    guard x.ndim == 4, fn.ndim == 2, fn.dtype == .float32,
        [.bfloat16, .float16, .float32].contains(x.dtype)
    else { return nil }
    let (B, L) = (x.dim(0), x.dim(1))
    let K = x.dim(2) * x.dim(3)
    let mix = fn.dim(0)
    guard fn.dim(1) == K, K % 4096 == 0, mix < 4096, K >= 16 * mix else { return nil }
    return GLM5NextHCKernels.shared.mix(
        [x, fn, MLXArray([eps])],
        template: [("T", x.dtype), ("HCD", K), ("MIX", mix)],
        grid: (256 * mix, B * L, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[B, L, mix]],
        outputDTypes: [.float32])[0]
}

/// Sinkhorn, collapse and the branch RMSNorm for `x: [B, L, 4, D]` and its
/// `mixes`: `(normalized [B, L, D], post [B, L, 4], comb [B, L, 4, 4])`, or
/// `nil` outside the kernel's configuration.
func glm5NextHCNorm(
    _ x: MLXArray, mixes: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray,
    sinkhornIterations: Int, hcEps: Float, normEps: Float
) -> (MLXArray, MLXArray, MLXArray)? {
    guard x.ndim == 4, [.bfloat16, .float16].contains(x.dtype), x.dim(2) == 4,
        x.dim(3) % 4 == 0, x.dim(3) <= 4096, normWeight.dtype == x.dtype,
        normWeight.shape == [x.dim(3)], mixes.dtype == .float32,
        mixes.shape == [x.dim(0), x.dim(1), 24],
        scale.dtype == .float32, scale.shape == [3], base.dtype == .float32, base.shape == [24]
    else { return nil }
    let (B, L, hc, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let threads = max(32, ((D / 4 + 31) / 32) * 32)
    let outputs = GLM5NextHCKernels.shared.norm(
        [x, mixes, scale, base, normWeight],
        template: [
            ("T", x.dtype), ("HC", hc), ("D", D), ("ITERS", sinkhornIterations),
            ("HC_EPS_INT", Int((Double(hcEps) / 1e-9).rounded())),
            ("NORM_EPS_INT", Int((Double(normEps) / 1e-9).rounded())),
        ],
        grid: (B * L * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[B, L, D], [B, L, hc], [B, L, hc, hc]],
        outputDTypes: [x.dtype, .float32, .float32])
    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - MoE router (one token per row)

/// `sigmoid(x @ W.T)` and `sigmoid + bias` per expert: MLX's fp32 gemv lane
/// order and shuffle ladder for one row per simdgroup (oMLX `moe_router`).
private let glm5NextRouterLogitsSource = """
      const uint lane = thread_index_in_simdgroup;
      const uint sg = simdgroup_index_in_threadgroup;
      const int tok = int(threadgroup_position_in_grid.y);
      const int out_row = int(threadgroup_position_in_grid.x) * 4 + int(sg);
      if (out_row >= E) {
        return;
      }
      const device float* mat = w + size_t(out_row) * K;
      const device T* xv = x + size_t(tok) * K;
      float result = 0.0f;
      int bn = int(lane) * 4;
      for (int i = 0; i < K / 128; ++i) {
        float v_coeff[4];
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(xv[bn + tn]);
        }
        float inter[4];
        for (int tn = 0; tn < 4; tn++) {
          inter[tn] = mat[bn + tn];
        }
        for (int tn = 0; tn < 4; tn++) {
          result += inter[tn] * v_coeff[tn];
        }
        bn += 128;
      }
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
      }
      if (lane == 0) {
        float sgm = glm_sigmoid<float>(result);
        float biased = sgm + bias[out_row];
        sig[size_t(tok) * E + out_row] = sgm;
        biased_out[size_t(tok) * E + out_row] = biased;
      }
    """

/// Top-k in argPartition's order (descending biased score, ties to the
/// lower expert), the gathered sigmoid scores, their sequential sum, the
/// normalization and the routed scaling factor.
private let glm5NextRouterSelectSource = """
      const uint lane = thread_index_in_simdgroup;
      const int tok = int(threadgroup_position_in_grid.x);
      constexpr int PER = (E + 31) / 32;
      const device float* bz = biased + size_t(tok) * E;
      const device float* sz = sig + size_t(tok) * E;
      float vals[PER];
      bool taken[PER];
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        vals[j] = e < E ? bz[e] : -INFINITY;
        taken[j] = e >= E;
      }
      int picked[TOPK];
      for (int r = 0; r < TOPK; r++) {
        float best = -INFINITY;
        int best_e = 0x7fffffff;
        for (int j = 0; j < PER; j++) {
          const int e = j * 32 + int(lane);
          if (!taken[j] && !isnan(vals[j]) &&
              (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
            best = vals[j];
            best_e = e;
          }
        }
        for (ushort off = 16; off >= 1; off >>= 1) {
          float ob = simd_shuffle_xor(best, off);
          int oe = simd_shuffle_xor(best_e, off);
          const bool other_better = oe != 0x7fffffff &&
              (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
          if (other_better) {
            best = ob;
            best_e = oe;
          }
        }
        if (best_e == 0x7fffffff) {
          // Only NaNs remain: argpartition's sort puts them last, lowest index first.
          for (int j = 0; j < PER; j++) {
            const int e = j * 32 + int(lane);
            if (!taken[j] && e < best_e) {
              best_e = e;
            }
          }
          for (ushort off = 16; off >= 1; off >>= 1) {
            best_e = min(best_e, simd_shuffle_xor(best_e, off));
          }
        }
        picked[r] = best_e;
        if ((best_e % 32) == int(lane)) {
          taken[best_e / 32] = true;
        }
      }
      if (lane == 0) {
        float total = 0.0f;
        float gathered[TOPK];
        for (int r = 0; r < TOPK; r++) {
          gathered[r] = sz[picked[r]];
          total = gathered[r] + total;
        }
        for (int r = 0; r < TOPK; r++) {
          float q = NORM ? gathered[r] / (total + 1e-20f) : gathered[r];
          float s = q * scaling[0];
          indices[tok * TOPK + r] = uint(picked[r]);
          scores[tok * TOPK + r] = s;
        }
      }
    """

private final class GLM5NextRouterKernels: Sendable {
    static let shared = GLM5NextRouterKernels()
    let logits: MLXFast.MLXFastKernel
    let select: MLXFast.MLXFastKernel

    private init() {
        logits = MLXFast.metalKernel(
            name: "glm5next_router_logits", inputNames: ["x", "w", "bias"],
            outputNames: ["sig", "biased_out"], source: glm5NextRouterLogitsSource,
            header: glm5NextKernelHeader)
        select = MLXFast.metalKernel(
            name: "glm5next_router_select", inputNames: ["sig", "biased", "scaling"],
            outputNames: ["indices", "scores"], source: glm5NextRouterSelectSource)
    }
}

/// The noaux_tc router for `n_group == 1` and rows `x: [T, K]`, each as a
/// one-token GEMV: `(indices uint32 [T, topK], scores fp32 [T, topK])`, or
/// `nil` outside the replicated gemv configuration.
func glm5NextMoERouter(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, scaling: Float, normalize: Bool
) -> (MLXArray, MLXArray)? {
    guard x.ndim == 2, weight.ndim == 2, bias.ndim == 1,
        weight.dtype == .float32, bias.dtype == .float32,
        [.bfloat16, .float16, .float32].contains(x.dtype)
    else { return nil }
    let (T, K) = (x.dim(0), x.dim(1))
    let E = weight.dim(0)
    // MLX's gemv for x @ W.T takes the bm = 4, bn = 1 layout for E < 4096 and
    // K < 16 * E; full 128-wide K blocks and whole 4-row tiles.
    guard weight.dim(1) == K, bias.dim(0) == E, E >= 16, E < 4096, E % 16 == 0, E <= 1024,
        K > 64, K % 128 == 0, K < 16 * E, (1 ... min(32, E)).contains(topK)
    else { return nil }
    let kernels = GLM5NextRouterKernels.shared
    let logits = kernels.logits(
        [x, weight, bias],
        template: [("T", x.dtype), ("K", K), ("E", E)],
        grid: (128 * (E / 4), T, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[T, E], [T, E]],
        outputDTypes: [.float32, .float32])
    let selected = kernels.select(
        [logits[0], logits[1], MLXArray([scaling])],
        template: [("E", E), ("TOPK", topK), ("NORM", normalize && topK > 1)],
        grid: (32 * T, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[T, topK], [T, topK]],
        outputDTypes: [.uint32, .float32])
    return (selected[0], selected[1])
}
