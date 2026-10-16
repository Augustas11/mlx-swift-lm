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
//  FMA collapse); the deferred `glm5NextHCPre` / `glm5NextHCPost` pair
//  reproduces expand + hc_mix + hc_norm bit for bit. `GLM5NextTests` checks
//  each one.
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
    #if GATE_ROWS
      threadgroup T as_[TOK][DK];
      threadgroup T gates[TOK][DK];
    #define A_PRE(t, i) as_[t][i]
    #define GATE_PRE(t, c) gates[t][c]
    #else
    #define A_PRE(t, i) a_pre[(t) * QKV + h * DK + (i)]
    #define GATE_PRE(t, c) gate_pre[(t) * QKV + h * DK + (c)]
    #endif

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

    #if GATE_ROWS && GATE5
      // One token, 5-bit K = 128 gate rows: MLX's qmv (qmv_impl): lanes 0..15
      // load 8 values each, lanes 16..31 add nothing, one simd_sum per row.
      {
        static_assert(TOK == 1 && DK == 128, "5-bit gate rows: one token");
        constexpr int WBYTES = 128 * 5 / 8;
        constexpr int G = 128 / GS;
        for (int rr = 0; rr < (2 * DK) / 32; rr++) {
          const int q = int(sg) * ((2 * DK) / 32) + rr;
          const int which = q / DK;
          const int i = q % DK;
          const int row = h * DK + i;
          float result = 0;
          if (lane < 16u) {
            const device T* xin = (which == 0 ? fa_in : ga_in) + int(lane) * 8;
            float x_thread[8];
            float sum = glm_load_vector<T, 8, 5>(xin, x_thread);
            const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
                + size_t(row) * WBYTES + int(lane) * 5;
            const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + int(lane) / (GS / 8);
            const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + int(lane) / (GS / 8);
            const float s = sl[0];
            const float b = bl[0];
            result += glm_qdot<8, 5>(wl, x_thread, s, b, sum);
          }
          result = simd_sum(result);
          if (lane == 0) {
            if (which == 0) {
              as_[0][i] = static_cast<T>(result);
            } else {
              gates[0][i] = static_cast<T>(result);
            }
          }
        }
      }
    #elif GATE_ROWS
      // 4/8-bit K = 128 gate rows: MLX's qmv_quad (one quad of lanes per row).
      {
        constexpr int VPT = 32;
        constexpr int WBYTES = 128 * BITS / 8;
        constexpr int G = 128 / GS;
        const int quad = int(tid) / 4;
        const int ql = int(tid) % 4;
        const int which = quad / DK;
        const int i = quad % DK;
        const int row = h * DK + i;
        const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
            + size_t(row) * WBYTES + ql * (VPT * BITS / 8);
        const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + ql / (GS / VPT);
        const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + ql / (GS / VPT);
        const float s = sl[0];
        const float b = bl[0];
        for (int t = 0; t < TOK; t++) {
          const device T* xin = (which == 0 ? fa_in : ga_in) + t * DK + ql * VPT;
          float x_thread[VPT];
          float sum = glm_load_vector<T, VPT, BITS>(xin, x_thread);
          float result = 0;
          result += glm_qdot<VPT, BITS>(wl, x_thread, s, b, sum);
          result = quad_sum(result);
          if (ql == 0) {
            if (which == 0) {
              as_[t][i] = static_cast<T>(result);
            } else {
              gates[t][i] = static_cast<T>(result);
            }
          }
        }
      }
    #endif
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta.
      if (tid < uint(TOK * DK)) {
        const int t = int(tid) / DK;
        const int i = int(tid) % DK;
        float ea = metal::precise::exp(a_log[h]);
        float af = static_cast<float>(A_PRE(t, i));
        float s1 = af + dt_bias[h * DK + i];
        float s2 = ea * s1;
        float s3 = glm_sigmoid<float>(s2);
        float s4 = lower * s3;
        gs[t][i] = metal::precise::exp(s4);
      }
      if (tid < uint(TOK)) {
        betas[tid] = glm_sigmoid<T>(b_in[tid * H + h]);
      }

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
          float gf = static_cast<float>(GATE_PRE(t, c));
          float gsg = glm_sigmoid<float>(gf);
          float o = wx * gsg;
          y[t * QKV + h * DK + c] = static_cast<T>(o);
        }
      }
    #undef A_PRE
    #undef GATE_PRE
    """

private final class GLM5NextKDAKernel: Sendable {
    static let shared = GLM5NextKDAKernel()
    /// Gate projections precomputed (`a_pre`, `gate_pre` inputs).
    let precomputed: MLXFast.MLXFastKernel
    /// Gate rows computed in the kernel: 4/8-bit (qmv_quad) and 5-bit (qmv).
    let quadGates: MLXFast.MLXFastKernel
    let fiveBitGates: MLXFast.MLXFastKernel

    private static let common = [
        "q_in", "k_in", "v_in", "b_in", "conv_state", "conv_w", "a_log", "dt_bias", "norm_w",
        "state_in", "consts",
    ]
    private static let gateRows = ["fa_in", "ga_in", "fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b"]

    private init() {
        func make(_ name: String, _ inputs: [String], _ defines: String) -> MLXFast.MLXFastKernel {
            MLXFast.metalKernel(
                name: name, inputNames: inputs,
                outputNames: ["y", "conv_state_out", "state_out"],
                source: defines + glm5NextKDASource,
                header: glm5NextQMVHeader)
        }
        precomputed = make(
            "glm5next_kda_decode", Self.common + ["a_pre", "gate_pre"],
            "#define GATE_ROWS 0\n#define GATE5 0\n")
        quadGates = make(
            "glm5next_kda_decode_gates", Self.common + Self.gateRows,
            "#define GATE_ROWS 1\n#define GATE5 0\n")
        fiveBitGates = make(
            "glm5next_kda_decode_gates5", Self.common + Self.gateRows,
            "#define GATE_ROWS 1\n#define GATE5 1\n")
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
    let outputs = GLM5NextKDAKernel.shared.precomputed(
        [q, k, v, bRaw, convState, convWeight, aLog.reshaped(-1), dtBias.reshaped(-1),
         normWeight, state, consts, aPre, gatePre],
        template: [
            ("T", dtype), ("TOK", T), ("DK", headDim), ("QKV", qkv), ("H", heads),
            ("BITS", 8), ("GS", 64),
        ],
        grid: (1024 * heads, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, T, qkv], [1, 3, 3 * qkv], [1, heads, headDim, headDim]],
        outputDTypes: [dtype, dtype, .float32])
    return (outputs[0], outputs[1], outputs[2])
}

/// `glm5NextKDADecodeStep` with the low-rank gate projections' second
/// halves (`f_b`, `g_b`, K = head_dim = 128) computed in the kernel from
/// `fa`/`ga` (`[1, T, 128]`): MLX's qmv_quad rows for 4/8 bits, its qmv rows
/// for 5 bits and one token. `nil` when not covered.
func glm5NextKDADecodeStep(
    q: MLXArray, k: MLXArray, v: MLXArray, fa: MLXArray, ga: MLXArray,
    fb: GLM5NextAffine, gb: GLM5NextAffine, bRaw: MLXArray,
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
    for part in [q, k, v] where part.shape != [1, T, qkv] || part.dtype != dtype {
        return nil
    }
    for part in [fa, ga] where part.shape != [1, T, headDim] || part.dtype != dtype {
        return nil
    }
    guard (fb.bits, fb.groupSize) == (gb.bits, gb.groupSize), [32, 64, 128].contains(fb.groupSize),
        fb.bits == 4 || fb.bits == 8 || (fb.bits == 5 && T == 1),
        fb.weight.shape == [qkv, 128 * fb.bits / 32], gb.weight.shape == fb.weight.shape,
        fb.scales.dtype == dtype, gb.scales.dtype == dtype,
        fb.biases.dtype == dtype, gb.biases.dtype == dtype,
        bRaw.shape == [1, T, heads], bRaw.dtype == dtype,
        convState.shape == [1, 3, 3 * qkv], convState.dtype == dtype,
        convWeight.shape == [3 * qkv, 4, 1], convWeight.dtype == dtype,
        aLog.size == heads, aLog.dtype == .float32,
        dtBias.size == qkv, dtBias.dtype == .float32,
        normWeight.shape == [headDim],
        state.shape == [1, heads, headDim, headDim], state.dtype == .float32
    else { return nil }
    let consts = MLXArray(
        [Float(pow(Double(headDim), -0.5)), l2Eps, normEps, lowerBound,
         Float(1.0 / Double(headDim))])
    let kernels = GLM5NextKDAKernel.shared
    let kernel = fb.bits == 5 ? kernels.fiveBitGates : kernels.quadGates
    let outputs = kernel(
        [q, k, v, bRaw, convState, convWeight, aLog.reshaped(-1), dtBias.reshaped(-1),
         normWeight, state, consts, fa, ga, fb.weight, fb.scales, fb.biases,
         gb.weight, gb.scales, gb.biases],
        template: [
            ("T", dtype), ("TOK", T), ("DK", headDim), ("QKV", qkv), ("H", heads),
            ("BITS", fb.bits), ("GS", fb.groupSize),
        ],
        grid: (1024 * heads, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, T, qkv], [1, 3, 3 * qkv], [1, heads, headDim, headDim]],
        outputDTypes: [dtype, dtype, .float32])
    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - Multi-projection qmv (one token)

/// `[linear(x) for linear in parts]` for up to six affine projections with
/// the same bits and group size, one dispatch: each output row with MLX's
/// qmv_fast arithmetic (oMLX `multi_qmv`, extended to six parts).
private let glm5NextMultiQMVSource = """
      const uint lane = thread_index_in_simdgroup;
      const int sg = int(simdgroup_index_in_threadgroup);
      const int row0 = int(threadgroup_position_in_grid.x) * 8;
      constexpr int WB = K * BITS / 8;
      constexpr int G = K / GS;
      const device uint8_t* w = (const device uint8_t*)w0;
      const device T* sc = s0;
      const device T* bi = b0;
      device T* y = y0;
      int local = row0;
    #if NP > 1
      if (row0 >= N0) { w = (const device uint8_t*)w1; sc = s1; bi = b1; y = y1; local = row0 - N0; }
    #endif
    #if NP > 2
      if (row0 >= N0 + N1) { w = (const device uint8_t*)w2; sc = s2; bi = b2; y = y2; local = row0 - N0 - N1; }
    #endif
    #if NP > 3
      if (row0 >= N0 + N1 + N2) { w = (const device uint8_t*)w3; sc = s3; bi = b3; y = y3; local = row0 - N0 - N1 - N2; }
    #endif
    #if NP > 4
      if (row0 >= N0 + N1 + N2 + N3) { w = (const device uint8_t*)w4; sc = s4; bi = b4; y = y4; local = row0 - N0 - N1 - N2 - N3; }
    #endif
    #if NP > 5
      if (row0 >= N0 + N1 + N2 + N3 + N4) { w = (const device uint8_t*)w5; sc = s5; bi = b5; y = y5; local = row0 - N0 - N1 - N2 - N3 - N4; }
    #endif
      const int r0 = local + sg * 4;
      float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      glm_qmv_rows<T, K, GS, BITS, 4>(
          w + size_t(r0) * WB, sc + r0 * G, bi + r0 * G, x, lane, result);
      for (int r = 0; r < 4; r++) {
        float v = simd_sum(result[r]);
        if (lane == 0) {
          y[r0 + r] = static_cast<T>(v);
        }
      }
    """

private final class GLM5NextMultiQMVKernels: @unchecked Sendable {
    static let shared = GLM5NextMultiQMVKernels()
    private let lock = NSLock()
    private var kernels: [Int: MLXFast.MLXFastKernel] = [:]

    func kernel(parts: Int) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let kernel = kernels[parts] { return kernel }
        let inputs = ["x"] + (0 ..< parts).flatMap { ["w\($0)", "s\($0)", "b\($0)"] }
        let kernel = MLXFast.metalKernel(
            name: "glm5next_multi_qmv_p\(parts)", inputNames: inputs,
            outputNames: (0 ..< parts).map { "y\($0)" },
            source: "#define NP \(parts)\n" + glm5NextMultiQMVSource + "\n#undef NP\n",
            header: glm5NextQMVHeader)
        kernels[parts] = kernel
        return kernel
    }
}

/// One-token outputs `[1, N_i]` of up to six projections sharing bits and
/// group size, or `nil` when not covered.
func glm5NextMultiQMV(_ x: MLXArray, _ parts: [GLM5NextAffine]) -> [MLXArray]? {
    guard x.ndim == 2, x.dim(0) == 1, [.bfloat16, .float16].contains(x.dtype),
        (1 ... 6).contains(parts.count)
    else { return nil }
    let K = x.dim(1)
    let (bits, groupSize) = (parts[0].bits, parts[0].groupSize)
    var rows = [Int]()
    for p in parts {
        guard (p.bits, p.groupSize) == (bits, groupSize), p.weight.ndim == 2,
            p.weight.dim(1) * 32 / bits == K, p.scales.dtype == x.dtype, p.biases.dtype == x.dtype,
            glm5NextQMVFastCovered(bits: bits, groupSize: groupSize, n: p.weight.dim(0), k: K)
        else { return nil }
        rows.append(p.weight.dim(0))
    }
    var template: [(String, any KernelTemplateArg)] = [
        ("T", x.dtype), ("K", K), ("BITS", bits), ("GS", groupSize),
    ]
    for i in 0 ..< 5 {
        template.append(("N\(i)", i < rows.count ? rows[i] : 0))
    }
    return GLM5NextMultiQMVKernels.shared.kernel(parts: parts.count)(
        [x] + parts.flatMap { [$0.weight, $0.scales, $0.biases] },
        template: template,
        grid: (64 * (rows.reduce(0, +) / 8), 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: rows.map { [1, $0] },
        outputDTypes: Array(repeating: x.dtype, count: rows.count))
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

/// `(sigmoid(x @ W.T), sigmoid + bias)` `[T, E]` fp32, one-token GEMV rows;
/// the caller checks the shapes (`glm5NextMoERouterCovered`).
func glm5NextRouterLogits(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> (MLXArray, MLXArray) {
    let (T, K, E) = (x.dim(0), x.dim(1), weight.dim(0))
    let logits = GLM5NextRouterKernels.shared.logits(
        [x, weight, bias],
        template: [("T", x.dtype), ("K", K), ("E", E)],
        grid: (128 * (E / 4), T, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[T, E], [T, E]],
        outputDTypes: [.float32, .float32])
    return (logits[0], logits[1])
}

/// Whether the router kernels replay MLX's gemv for `x: [T, K]` and `W: [E, K]`.
func glm5NextMoERouterCovered(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> Bool {
    guard x.ndim == 2, weight.ndim == 2, bias.ndim == 1,
        weight.dtype == .float32, bias.dtype == .float32,
        [.bfloat16, .float16, .float32].contains(x.dtype)
    else { return false }
    let (K, E) = (x.dim(1), weight.dim(0))
    // MLX's gemv for x @ W.T takes the bm = 4, bn = 1 layout for E < 4096 and
    // K < 16 * E; full 128-wide K blocks and whole 4-row tiles.
    return weight.dim(1) == K && bias.dim(0) == E && E >= 16 && E < 4096 && E % 16 == 0
        && E <= 1024 && K > 64 && K % 128 == 0 && K < 16 * E
}

/// The noaux_tc router for `n_group == 1` and rows `x: [T, K]`, each as a
/// one-token GEMV: `(indices uint32 [T, topK], scores fp32 [T, topK])`, or
/// `nil` outside the replicated gemv configuration.
func glm5NextMoERouter(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, scaling: Float, normalize: Bool
) -> (MLXArray, MLXArray)? {
    guard glm5NextMoERouterCovered(x, weight: weight, bias: bias),
        (1 ... min(32, weight.dim(0))).contains(topK)
    else { return nil }
    let (T, E) = (x.dim(0), weight.dim(0))
    let logits = glm5NextRouterLogits(x, weight: weight, bias: bias)
    let selected = GLM5NextRouterKernels.shared.select(
        [logits.0, logits.1, MLXArray([scaling])],
        template: [("E", E), ("TOPK", topK), ("NORM", normalize && topK > 1)],
        grid: (32 * T, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[T, topK], [T, topK]],
        outputDTypes: [.uint32, .float32])
    return (selected[0], selected[1])
}

// MARK: - Deferred hyper-connection decode (one token)
//
// A half-layer's chain is expand (previous branch) -> hc_mix -> hc_norm ->
// branch. `glm5NextHCPre` computes the mix GEMV (hc_mix's arithmetic), the
// pre weights, the FMA collapse and the branch RMSNorm (hc_norm's) in one
// dispatch, and in deferred form rebuilds h = bf16(post * y + mm) (the
// expand's epilogue) as it loads it, storing h for the next half-layer.
// `glm5NextHCPost` computes post, the Sinkhorn comb and the expand's comb
// product mm = comb^T @ float(h), which only the next half-layer reads.
// Sources adapted from oMLX `hc_pre_fused` / `hc_post_mm`; the comb product
// replays MLX's non-NAX fp32 steel GEMM (simdgroup MMA) instead of oMLX's
// NAX matmul2d, so these run where MLX runs that GEMM without NAX.

private let glm5NextHCPreHeader = """
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    // hc_expand_one's epilogue for element e = r * D + col of h.
    template <typename T, int D, typename YPtr, typename MPtr, typename PPtr>
    inline T glm_hc_expand_value(YPtr y, MPtr mm, PPtr post, int e) {
      const int r = e / D;
      const int col = e - r * D;
      volatile float prod = post[r] * static_cast<float>(y[col]);
      float sum = prod + mm[e];
      return static_cast<T>(sum);
    }
    """

private let glm5NextHCPreSource = """
      const uint lid = thread_position_in_threadgroup.x;
      const uint simd_lid = thread_index_in_simdgroup;
      const uint simd_gid = simdgroup_index_in_threadgroup;
      const int tile = int(threadgroup_position_in_grid.x);
      constexpr int KSZ = HC * D;
      constexpr int CH = KSZ / 4096;  // rms_looped reads per thread, 4 values each
      constexpr int D4 = D / 4;
      static_assert(KSZ % 4096 == 0 && D4 <= 1024 && D % 4 == 0, "hc_pre_fused shape");
      constexpr float HC_EPS = HC_EPS_INT * 1e-9;
      constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
      using T4 = vec<T, 4>;
    #if DEFERRED
    #define HVAL(e) glm_hc_expand_value<T, D>(y, mm, post, (e))
    #else
    #define HVAL(e) x[(e)]
    #endif

      // This thread's h values in rms_looped order.
      T hv[CH][4];
      for (int c = 0; c < CH; c++) {
        for (int i = 0; i < 4; i++) {
          hv[c][i] = HVAL(c * 4096 + int(lid) * 4 + i);
        }
      }
    #if DEFERRED
      if (tile == 0) {
        for (int c = 0; c < CH; c++) {
          for (int i = 0; i < 4; i++) {
            h_out[c * 4096 + int(lid) * 4 + i] = hv[c][i];
          }
        }
      }
    #endif

      // --- hc_mix: rms_looped (lsize = 1024, N_READS = 4) ---
      threadgroup float local_inv_mean[1];
      threadgroup float local_sums[32];
      float acc = 0;
      for (int c = 0; c < CH; c++) {
        for (int i = 0; i < 4; i++) {
          float xi = static_cast<float>(hv[c][i]);
          acc += xi * xi;
        }
      }
      acc = simd_sum(acc);
      if (simd_gid == 0) {
        local_sums[simd_lid] = 0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_lid == 0) {
        local_sums[simd_gid] = acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        acc = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
          local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv = local_inv_mean[0];

      // --- hc_mix gemv: 8 simdgroups per row, 4 rows per threadgroup ---
      const int slot = int(simd_gid) / 8;
      const int sgN = int(simd_gid) % 8;
      const int row = tile * 4 + slot;
      threadgroup float partial[4][8];
      threadgroup float row_total[4];
      float result = 0;
      {
        const device float* mrow = fn + size_t(row) * KSZ;
        int bn = (32 * sgN + int(simd_lid)) * 4;
        for (int i = 0; i < KSZ / 1024; ++i) {
          float v_coeff[4];
          float inter[4];
          for (int tn = 0; tn < 4; tn++) {
            v_coeff[tn] = static_cast<float>(HVAL(bn + tn)) * inv;
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
      }
      if (simd_lid == 0) {
        partial[slot][sgN] = result;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sgN == 0 && simd_lid == 0) {
        float total = partial[slot][0];
        for (int s = 1; s < 8; s++) {
          total += partial[slot][s];
        }
        mixes[row] = total;
        row_total[slot] = total;
      }
      if (tile != 0) {
        return;
      }

      // --- threadgroup 0 (the pre rows): exact_hc_norm's collapse + RMSNorm ---
      threadgroup float pre_shared[HC];
      threadgroup float norm_inv[1];
      threadgroup float norm_sums[32];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        const float pre_scale = scale[0];
        const uint llane = metal::min(simd_lid, (uint)(HC - 1));
        float pre_z = row_total[llane] * pre_scale + base[llane];
        float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
        if (simd_lid < (uint)HC) {
          pre_shared[simd_lid] = pre_v;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      T4 rounded = T4(0);
      float accum = 0.0f;
      if (int(lid) < D4) {
        T4 xs[HC];
        for (int r = 0; r < HC; r++) {
          if (D == 4096) {
            // Same elements as this thread's rms_looped reads.
            xs[r] = T4(hv[r][0], hv[r][1], hv[r][2], hv[r][3]);
          } else {
            const int e = r * D + int(lid) * 4;
            xs[r] = T4(HVAL(e), HVAL(e + 1), HVAL(e + 2), HVAL(e + 3));
          }
        }
        float4 collapsed = fma(
            float4(pre_shared[0]), float4(xs[0]),
            fma(
                float4(pre_shared[1]), float4(xs[1]),
                fma(
                    float4(pre_shared[2]), float4(xs[2]),
                    float4(pre_shared[3]) * float4(xs[3]))));
        rounded = T4(collapsed);
        float4 rounded_float = float4(rounded);
        accum += rounded_float.x * rounded_float.x;
        accum += rounded_float.y * rounded_float.y;
        accum += rounded_float.z * rounded_float.z;
        accum += rounded_float.w * rounded_float.w;
      }
      accum = simd_sum(accum);
      if (simd_lid == 0) {
        norm_sums[simd_gid] = accum;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        accum = simd_sum(norm_sums[simd_lid]);
        if (simd_lid == 0) {
          norm_inv[0] = metal::precise::rsqrt(accum / D + NORM_EPS);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (int(lid) < D4) {
        float ninv = norm_inv[0];
        const device T4* weights = (const device T4*)norm_weight;
        T4 scaled = T4(float4(rounded) * ninv);
        T4 weight = weights[lid];
        ((device T4*)normalized)[lid] = T4(
            weight.x * scaled.x,
            weight.y * scaled.y,
            weight.z * scaled.z,
            weight.w * scaled.w);
      }
    #undef HVAL
    """

private let glm5NextHCPostHeader = """
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    template <int HC, int ITERS, int HC_EPS_INT, typename MixPtr, typename BasePtr>
    inline void glm_hc_post_comb(
        MixPtr mix, const float post_scale, const float comb_scale, BasePtr base,
        uint lane, thread float& post_v, thread float4& result) {
      constexpr int BASE_OFF = 2 * HC;
      constexpr float HC_EPS = HC_EPS_INT * 1e-9;
      const float active = lane < (uint)HC ? 1.0f : 0.0f;
      const uint llane = metal::min(lane, (uint)(HC - 1));

      float post_z = mix[HC + llane] * post_scale + base[HC + llane];
      post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

      float4 value =
          (float4(mix[BASE_OFF + llane * HC], mix[BASE_OFF + llane * HC + 1],
                  mix[BASE_OFF + llane * HC + 2], mix[BASE_OFF + llane * HC + 3]) * comb_scale +
           float4(base[BASE_OFF + llane * HC], base[BASE_OFF + llane * HC + 1],
                  base[BASE_OFF + llane * HC + 2], base[BASE_OFF + llane * HC + 3])) * active;
      float row_max = metal::max(
          metal::max(value.x, value.y), metal::max(value.z, value.w));
      float4 exponent = metal::fast::exp(value - row_max) * active;
      result = exponent *
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
    }
    """

private let glm5NextHCPostSource = """
      const ushort lane = thread_index_in_simdgroup;
      const uint sg = simdgroup_index_in_threadgroup;
      static_assert(HC == 4, "one float4 comb row per lane");
      threadgroup float4 comb_rows[HC];
      threadgroup float a_tile[64];
      threadgroup float b_tiles[SIMDS * 64];
      if (sg == 0) {
        float post_v;
        float4 res;
        glm_hc_post_comb<HC, ITERS, HC_EPS_INT>(mixes, scale[1], scale[2], base, uint(lane), post_v, res);
        if (lane < HC) {
          comb_rows[lane] = res;
          if (threadgroup_position_in_grid.x == 0) {
            post_out[lane] = post_v;
            *(device float4*)(comb_out + lane * HC) = res;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // comb^T @ float(residual) as MLX's non-NAX steel GEMM computes it: one
      // 8x8x8 simdgroup MMA per 8 columns on zero-padded fragments
      // (A[r][k] = comb[k][r], B[k][c] = residual[k][c], K = HC <= 8).
      const threadgroup float* comb = (const threadgroup float*)comb_rows;
      if (sg == 0) {
        for (int e = lane; e < 64; e += 32) {
          const int r = e / 8, k = e % 8;
          a_tile[e] = (r < HC && k < HC) ? comb[k * HC + r] : 0.0f;
        }
      }
      const int col0 = (int(threadgroup_position_in_grid.x) * SIMDS + int(sg)) * 8;
      threadgroup float* bt = b_tiles + sg * 64;
      for (int e = lane; e < 64; e += 32) {
        const int k = e / 8, c = e % 8;
        bt[e] = (k < HC && col0 + c < D) ? static_cast<float>(residual[k * D + col0 + c]) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (col0 < D) {
        simdgroup_matrix<float, 8, 8> A;
        simdgroup_matrix<float, 8, 8> B;
        simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_load(A, a_tile, 8);
        simdgroup_load(B, bt, 8);
        simdgroup_multiply_accumulate(C, A, B, C);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(C, bt, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lane; e < HC * 8; e += 32) {
          const int r = e / 8, c = e % 8;
          if (col0 + c < D) {
            mm[r * D + col0 + c] = bt[r * 8 + c];
          }
        }
      }
    """

private final class GLM5NextHCDeferredKernels: Sendable {
    static let shared = GLM5NextHCDeferredKernels()
    let pre: MLXFast.MLXFastKernel
    let preDeferred: MLXFast.MLXFastKernel
    let post: MLXFast.MLXFastKernel

    private init() {
        let common = ["fn", "eps", "scale", "base", "norm_weight"]
        pre = MLXFast.metalKernel(
            name: "glm5next_hc_pre", inputNames: ["x"] + common,
            outputNames: ["normalized", "mixes"],
            source: "#define DEFERRED 0\n" + glm5NextHCPreSource + "\n#undef DEFERRED\n",
            header: glm5NextHCPreHeader)
        preDeferred = MLXFast.metalKernel(
            name: "glm5next_hc_pre_deferred", inputNames: ["y", "mm", "post"] + common,
            outputNames: ["normalized", "mixes", "h_out"],
            source: "#define DEFERRED 1\n" + glm5NextHCPreSource + "\n#undef DEFERRED\n",
            header: glm5NextHCPreHeader)
        post = MLXFast.metalKernel(
            name: "glm5next_hc_post", inputNames: ["mixes", "scale", "base", "residual"],
            outputNames: ["post_out", "comb_out", "mm"],
            source: glm5NextHCPostSource, header: glm5NextHCPostHeader)
    }
}

/// Whether MLX computes the expand's fp32 comb product without NAX here
/// (`metal::is_nax_available()` and `MLX_ENABLE_TF32`), the GEMM
/// `glm5NextHCPost` replays. False on NAX GPUs with TF32 enabled.
let glm5NextHCDeferredExact: Bool = {
    let architecture = GPU.deviceInfo().architecture
    guard let range = architecture.range(of: #"applegpu_g(\d+)"#, options: .regularExpression)
    else { return true }
    let gen = Int(architecture[range].dropFirst("applegpu_g".count)) ?? 0
    let kind = architecture.last
    var naxOS = false
    if #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) { naxOS = true }
    let nax = naxOS && gen >= (kind == "p" ? 18 : 17)
    let tf32 = (ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"].flatMap { Int($0) } ?? 1) != 0
    return !(nax && tf32)
}()

/// Shapes the deferred kernels cover: one token, HC = 4, mix = 24,
/// HC * D a multiple of 4096 and D <= 4096, bf16/fp16 activations.
func glm5NextHCDeferredSupported(
    dtype: DType, width: Int, fn: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray
) -> Bool {
    [.bfloat16, .float16].contains(dtype) && width % 4 == 0 && width <= 4096
        && (4 * width) % 4096 == 0 && fn.dtype == .float32 && fn.shape == [24, 4 * width]
        && scale.dtype == .float32 && scale.shape == [3] && base.dtype == .float32
        && base.shape == [24] && normWeight.dtype == dtype && normWeight.shape == [width]
}

/// One-token branch input from `x: [1, 1, 4, D]` or from the previous
/// half-layer `(y [1, 1, D], mm [4 * D] fp32, post [1, 1, 4] fp32)`.
/// Returns `(normalized [1, 1, D], mixes [1, 1, 24], h [1, 1, 4, D])`.
func glm5NextHCPre(
    x: MLXArray?, deferred: (y: MLXArray, mm: MLXArray, post: MLXArray)?,
    fn: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray,
    normEps: Float, hcEps: Float, branchNormEps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let kernels = GLM5NextHCDeferredKernels.shared
    let dtype = x?.dtype ?? deferred!.y.dtype
    let D = normWeight.dim(0)
    let rest: [MLXArray] = [fn, MLXArray([normEps]), scale, base, normWeight]
    let template: [(String, any KernelTemplateArg)] = [
        ("T", dtype), ("HC", 4), ("D", D),
        ("HC_EPS_INT", Int((Double(hcEps) / 1e-9).rounded())),
        ("NORM_EPS_INT", Int((Double(branchNormEps) / 1e-9).rounded())),
    ]
    if let deferred {
        let out = kernels.preDeferred(
            [deferred.y, deferred.mm, deferred.post] + rest, template: template,
            grid: (1024 * 6, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, 1, D], [1, 1, 24], [1, 1, 4, D]],
            outputDTypes: [dtype, .float32, dtype])
        return (out[0], out[1], out[2])
    }
    let out = kernels.pre(
        [x!] + rest, template: template,
        grid: (1024 * 6, 1, 1), threadGroup: (1024, 1, 1),
        outputShapes: [[1, 1, D], [1, 1, 24]], outputDTypes: [dtype, .float32])
    return (out[0], out[1], x!)
}

/// `(post [1, 1, 4], comb [1, 1, 4, 4], mm [4 * D])` for one token.
func glm5NextHCPost(
    mixes: MLXArray, scale: MLXArray, base: MLXArray, residual: MLXArray,
    sinkhornIterations: Int, hcEps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let D = residual.dim(-1)
    let simds = 8
    let groups = (D / 8 + simds - 1) / simds
    let out = GLM5NextHCDeferredKernels.shared.post(
        [mixes, scale, base, residual],
        template: [
            ("T", residual.dtype), ("HC", 4), ("D", D), ("ITERS", sinkhornIterations),
            ("HC_EPS_INT", Int((Double(hcEps) / 1e-9).rounded())), ("SIMDS", simds),
        ],
        grid: (32 * simds * groups, 1, 1), threadGroup: (32 * simds, 1, 1),
        outputShapes: [[1, 1, 4], [1, 1, 4, 4], [4 * D]],
        outputDTypes: [.float32, .float32, .float32])
    return (out[0], out[1], out[2])
}

// MARK: - MoE experts (one token)
//
// Adapted from oMLX v0.7.1rc1 `moe_gate_up_swiglu` (SELECT form) and
// `moe_down_combine` (Apache-2.0). The gate/up kernel replays the router's
// top-k in every routed threadgroup (publishing the routes and routing
// weights), and computes each routed and the shared expert's gate/up rows
// with MLX's qmv_fast arithmetic plus the clamped SwiGLU; the down kernel
// computes the routed down rows, the routing-weighted sum (bf16-rounded
// expert outputs times fp32 scores, unfused, summed in route order) and the
// shared expert's down rows, as gather_qmv + weightedExpertSum + the shared
// MLP + add do. The selection normalizes with `total + 1e-20`, the op path's
// form.

private let glm5NextQMVHeader = """
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    template <int bits>
    constexpr int glm_pack_factor() {
      return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
    }

    template <int bits>
    constexpr int glm_bytes_per_pack() {
      return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
    }

    // Verbatim copy of MLX quantized.h load_vector (U = float).
    template <typename T, int values_per_thread, int bits>
    inline float glm_load_vector(const device T* x, thread float* x_thread) {
      float sum = 0;
      if (bits == 3) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
              float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 8.0f;
          x_thread[i + 2] = x[i + 2] / 64.0f;
          x_thread[i + 3] = x[i + 3] / 2.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 128.0f;
          x_thread[i + 6] = x[i + 6] / 4.0f;
          x_thread[i + 7] = x[i + 7] / 32.0f;
        }
      } else if (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
          sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      } else if (bits == 5) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
              float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      } else if (bits == 6) {
        for (int i = 0; i < values_per_thread; i += 4) {
          sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 64.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 4.0f;
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
          sum += x[i];
          x_thread[i] = x[i];
        }
      }
      return sum;
    }

    // Verbatim copy of MLX quantized.h qdot (U = float).
    template <int values_per_thread, int bits>
    inline float glm_qdot(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0;
      if (bits == 3) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;
          w += 3 * i;

          accum += (w[0] & 0x07) * x_thread[0];
          accum += (w[0] & 0x38) * x_thread[1];
          accum += (w[0] & 0xc0) * x_thread[2];
          accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

          accum += (w[1] & 0x0e) * x_thread[3];
          accum += (w[1] & 0x70) * x_thread[4];
          accum += (w[1] & 0x80) * x_thread[5];
          accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

          accum += (w[2] & 0x1c) * x_thread[6];
          accum += (w[2] & 0xe0) * x_thread[7];
        }
      } else if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      } else if (bits == 5) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;
          w += 5 * i;
          accum += (w[0] & 0x1f) * x_thread[0];
          accum += (w[0] & 0xe0) * x_thread[1];
          accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0x7c) * x_thread[2];
          accum += (w[1] & 0x80) * x_thread[3];
          accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
          accum += (w[2] & 0xf0) * x_thread[4];
          accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
          accum += (w[3] & 0x3e) * x_thread[5];
          accum += (w[3] & 0xc0) * x_thread[6];
          accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
          accum += (w[4] & 0xf8) * x_thread[7];
        }
      } else if (bits == 6) {
        for (int i = 0; i < (values_per_thread / 4); i++) {
          x_thread += 4 * i;
          w += 3 * i;
          accum += (w[0] & 0x3f) * x_thread[0];
          accum += (w[0] & 0xc0) * x_thread[1];
          accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0xf0) * x_thread[2];
          accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);
          accum += (w[2] & 0xfc) * x_thread[3];
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
          accum += x_thread[i] * w[i];
        }
      }
      return scale * accum + sum * bias;
    }

    // qmv_fast_impl for RPS consecutive rows of one [N, K] affine matrix
    // (row pointers already offset to the first row), one simdgroup.  Leaves the
    // per-lane partial sums in `result`; the caller simd_sums them.
    template <typename T, int K, int group_size, int bits, int RPS>
    inline void glm_qmv_rows(
        const device uint8_t* ws,
        const device T* scales,
        const device T* biases,
        const device T* x,
        uint simd_lid,
        thread float* result) {
      constexpr int packs_per_thread = bits == 2 ? 1 : 2;
      constexpr int pack_factor = glm_pack_factor<bits>();
      constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
      constexpr int values_per_thread = pack_factor * packs_per_thread;
      constexpr int block_size = values_per_thread * 32;
      constexpr int scale_step_per_thread = group_size / values_per_thread;
      constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
      constexpr int in_vec_size_g = K / group_size;

      thread float x_thread[values_per_thread];
      ws += simd_lid * packs_per_thread * bytes_per_pack;
      scales += simd_lid / scale_step_per_thread;
      biases += simd_lid / scale_step_per_thread;
      x += simd_lid * values_per_thread;

      for (int k = 0; k < K; k += block_size) {
        float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
        for (int row = 0; row < RPS; row++) {
          const device uint8_t* wl = ws + row * in_vec_size_w;
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          float s = sl[0];
          float b = bl[0];
          result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
        }
        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        biases += block_size / group_size;
        x += block_size;
      }
    }

    // Verbatim copy of MLX quantized.h dequantize (U = float) for 4/5/6/8 bits.
    template <int N, int bits>
    inline void glm_dequantize(const device uint8_t* w, float scale, float bias, thread float* w_local) {
      const float s = float(scale);
      const float b = float(bias);
      if (bits == 4) {
        float sc[2] = {s, s / 16.0f};
        for (int i = 0; i < (N / 2); i++) {
          w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
          w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
        }
      } else if (bits == 5) {
        for (int i = 0; i < (N / 8); i++) {
          w_local += 8 * i;
          w += 5 * i;
          w_local[0] = static_cast<float>((w[0] & 0x1f) * s + b);
          w_local[1] =
              static_cast<float>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
          w_local[2] = static_cast<float>(((w[1] & 0x7c) >> 2) * s + b);
          w_local[3] =
              static_cast<float>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
          w_local[4] =
              static_cast<float>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
          w_local[5] = static_cast<float>(((w[3] & 0x3e) >> 1) * s + b);
          w_local[6] =
              static_cast<float>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
          w_local[7] = static_cast<float>(((w[4] & 0xf8) >> 3) * s + b);
        }
      } else if (bits == 6) {
        for (int i = 0; i < (N / 4); i++) {
          w_local += 4 * i;
          w += 3 * i;
          w_local[0] = static_cast<float>((w[0] & 0x3f) * s + b);
          w_local[1] =
              static_cast<float>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
          w_local[2] =
              static_cast<float>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
          w_local[3] = static_cast<float>(((w[2] >> 2) & 0x3f) * s + b);
        }
      } else if (bits == 8) {
        for (int i = 0; i < N; i++) {
          w_local[i] = static_cast<float>(s * w[i] + b);
        }
      }
    }

    // MLX qmv_wide_impl (affine, k_lanes = 8) for one weight row and NV input
    // vectors: each lane reduces groups k_lane, k_lane + 8, ... in 8-value
    // sub-chunks; the caller applies the 4/2/1 shuffle-down ladder.
    template <typename T, int K, int GS, int BITS, int NV>
    inline void glm_qmv_wide_row(
        const device uint8_t* wrow,
        const device T* srow,
        const device T* brow,
        const device T* x,
        int nv,
        int k_lane,
        thread float* result) {
      constexpr int sub = 8;
      constexpr int G = K / GS;
      for (int g = k_lane; g < G; g += 8) {
        float scale = srow[g];
        float bias = brow[g];
        for (int sc = 0; sc < GS / sub; sc++) {
          const int k0 = g * GS + sc * sub;
          const device uint8_t* wc = wrow + k0 * BITS / 8;
          float w_dq[sub];
          glm_dequantize<sub, BITS>(wc, scale, bias, w_dq);
          for (int v = 0; v < NV; v++) {
            if (v < nv) {
              const device T* xc = x + v * K + k0;
              float acc = 0;
              for (int i = 0; i < sub; i++) {
                acc += static_cast<float>(xc[i]) * w_dq[i];
              }
              result[v] += acc;
            }
          }
        }
      }
    }

    // Same expressions as MLX's Sigmoid / Minimum / Maximum functors.
    template <typename T>
    inline T glm_sigmoid(T x) {
      auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
      return (x < 0) ? y : 1 - y;
    }
    template <typename T>
    inline T glm_minimum(T x, T y) {
      if (metal::isnan(x)) {
        return x;
      }
      return x < y ? x : y;
    }
    template <typename T>
    inline T glm_maximum(T x, T y) {
      if (metal::isnan(x)) {
        return x;
      }
      return x > y ? x : y;
    }

    // The router's top-k selection (the select kernel's loop, one simdgroup):
    // argpartition order of the biased scores, i.e. descending values with ties
    // to the lower expert index and NaNs last (lowest index first). Every lane
    // ends with the same picked[].
    template <int E, int TOPK, typename P>
    inline void glm_router_topk(P bz, uint lane, thread int* picked) {
      constexpr int PER = (E + 31) / 32;
      float vals[PER];
      bool taken[PER];
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        vals[j] = e < E ? bz[e] : -INFINITY;
        taken[j] = e >= E;
      }
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
    }

    // Glm5NextClampedSwiGLU / Glm5NextMLP epilogue on bfloat16 projections:
    //   silu(minimum(gate, limit)) * minimum(maximum(up, -limit), limit)
    template <typename T>
    inline T glm_clamped_swiglu(T gate, T up, T limit, T neg_limit) {
      T g = glm_minimum(gate, limit);
      T s = g * glm_sigmoid(g);
      T u = glm_minimum(glm_maximum(up, neg_limit), limit);
      return s * u;
    }
    """

private let glm5NextMoEGateUpSource = """
      const uint simd_lid = thread_index_in_simdgroup;
      const uint simd_gid = simdgroup_index_in_threadgroup;
    #if SLOT_MAJOR
      // Route slots vary fastest, so the slots of an expert that several
      // tokens share read its row block back to back (cache hits).
      const int tile = int(threadgroup_position_in_grid.z);
      const int z = int(threadgroup_position_in_grid.y);
    #else
      const int tile = int(threadgroup_position_in_grid.y);
      const int z = int(threadgroup_position_in_grid.z);
    #endif
      const T lim = T(limit[0]);
      const T neg_lim = T(-limit[0]);
    #if SHARED_WIDE
      // Last z slice: the shared expert for all NTOK tokens with MLX's
      // multi-row qmv_wide arithmetic (8 lanes per row, 4 rows per simdgroup).
      if (z == NTOK * TOPK) {
        const int k_lane = int(simd_lid) % 8;
        const int row = (tile * NSG + int(simd_gid)) * 4 + int(simd_lid) / 8;
        constexpr int WB = K * SBITS / 8;
        constexpr int G = K / SGS;
        float g_res[NTOK];
        float u_res[NTOK];
        for (int v = 0; v < NTOK; v++) {
          g_res[v] = 0.0f;
          u_res[v] = 0.0f;
        }
        glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
            (const device uint8_t*)sh_gate_w + size_t(row) * WB, sh_gate_s + row * G,
            sh_gate_b + row * G, x, NTOK, k_lane, g_res);
        glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
            (const device uint8_t*)sh_up_w + size_t(row) * WB, sh_up_s + row * G,
            sh_up_b + row * G, x, NTOK, k_lane, u_res);
        for (int v = 0; v < NTOK; v++) {
          g_res[v] += simd_shuffle_down(g_res[v], 4);
          g_res[v] += simd_shuffle_down(g_res[v], 2);
          g_res[v] += simd_shuffle_down(g_res[v], 1);
          u_res[v] += simd_shuffle_down(u_res[v], 4);
          u_res[v] += simd_shuffle_down(u_res[v], 2);
          u_res[v] += simd_shuffle_down(u_res[v], 1);
        }
        if (k_lane == 0) {
          for (int v = 0; v < NTOK; v++) {
            shared_out[size_t(v) * N + row] = glm_clamped_swiglu<T>(
                static_cast<T>(g_res[v]), static_cast<T>(u_res[v]), lim, neg_lim);
          }
        }
        return;
      }
      constexpr int RT = TOPK;
    #else
      constexpr int RT = TOPK + HAS_SHARED;
    #endif
      const int token = z / RT;
      const int r = z - token * RT;
      const int out_row = (tile * NSG + int(simd_gid)) * RPS;
      const device T* xr = x + token * K;

      float g_res[RPS] = {0};
      float u_res[RPS] = {0};
      if (r < TOPK) {
    #if SELECT
        // One token: this simdgroup replays the router's selection on the
        // biased sigmoid scores (no separate select dispatch); slot 0 / tile 0
        // publishes the routes and routing weights for the down kernel.
        int picked[TOPK];
        glm_router_topk<NE, TOPK>(sel_biased, simd_lid, picked);
        const int expert = picked[r];
        if (z == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {
          float total = 0.0f;
          float gathered[TOPK];
          for (int q = 0; q < TOPK; q++) {
            gathered[q] = sel_sig[picked[q]];
            total = gathered[q] + total;
          }
          for (int q = 0; q < TOPK; q++) {
            float qv = SEL_NORM ? gathered[q] / (total + 1e-20f) : gathered[q];
            float sv = qv * sel_scaling[0];
            sel_indices[q] = uint(picked[q]);
            sel_scores[q] = sv;
          }
        }
    #else
        const int expert = int(indices[token * TOPK + r]);
    #endif
        constexpr int WB = K * RBITS / 8;   // bytes per weight row
        constexpr int G = K / RGS;          // groups per row
        // ESTRIDE rows per expert; a fused [gate; up] tensor (ESTRIDE = 2N) is
        // passed as both gate and up with the up rows UP_OFF = N further on.
        const size_t row0 = size_t(expert) * ESTRIDE + out_row;
        const size_t urow0 = row0 + UP_OFF;
        glm_qmv_rows<T, K, RGS, RBITS, RPS>(
            (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
            gate_b + row0 * G, xr, simd_lid, g_res);
        glm_qmv_rows<T, K, RGS, RBITS, RPS>(
            (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
            up_b + urow0 * G, xr, simd_lid, u_res);
      } else {
    #if HAS_SHARED
        constexpr int WB = K * SBITS / 8;
        constexpr int G = K / SGS;
        const size_t row0 = size_t(out_row);
        glm_qmv_rows<T, K, SGS, SBITS, RPS>(
            (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
            sh_gate_b + row0 * G, xr, simd_lid, g_res);
        glm_qmv_rows<T, K, SGS, SBITS, RPS>(
            (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
            sh_up_b + row0 * G, xr, simd_lid, u_res);
    #endif
      }
      device T* o = out + size_t(z) * N + out_row;
      for (int row = 0; row < RPS; row++) {
        float gv = simd_sum(g_res[row]);
        float uv = simd_sum(u_res[row]);
        if (simd_lid == 0) {
          o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
        }
      }
    """

private let glm5NextMoEDownSource = """
      const uint simd_lid = thread_index_in_simdgroup;
      const uint simd_gid = simdgroup_index_in_threadgroup;
    #if SLOT_MAJOR
      // Tokens vary fastest: experts they share are read back to back.
      const int tile = int(threadgroup_position_in_grid.z);
      const int token = int(threadgroup_position_in_grid.y);
    #else
      const int tile = int(threadgroup_position_in_grid.y);
      const int token = int(threadgroup_position_in_grid.z);
    #endif
      // Activation slots per token: the routed ones, then the shared expert's.
      constexpr int RT = TOPK + HAS_SHARED;
      const int out_row = (tile * NSG + int(simd_gid)) * RPS;

      float acc[RPS] = {0};
      constexpr int WB = K * RBITS / 8;
      constexpr int G = K / RGS;
      for (int r = 0; r < TOPK; r++) {
        const int expert = int(indices[token * TOPK + r]);
        const size_t row0 = size_t(expert) * N + out_row;
        float res[RPS] = {0};
        glm_qmv_rows<T, K, RGS, RBITS, RPS>(
            (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
            down_b + row0 * G, act + (size_t(token) * RT + r) * K, simd_lid, res);
        const float score = scores[token * TOPK + r];
        for (int row = 0; row < RPS; row++) {
          float v = simd_sum(res[row]);
          // The reference rounds the fp32 product in its own Multiply kernel
          // before the Sum; keep the compiler from contracting it into an FMA.
          volatile float weighted = static_cast<float>(static_cast<T>(v)) * score;
          acc[row] += weighted;
        }
      }
    #if HAS_SHARED
      float sres[RPS] = {0};
      {
        constexpr int SWB = K * SBITS / 8;
        constexpr int SG = K / SGS;
        const size_t row0 = size_t(out_row);
        const device T* sh_x = act + (size_t(token) * RT + TOPK) * K;
        glm_qmv_rows<T, K, SGS, SBITS, RPS>(
            (const device uint8_t*)sh_down_w + row0 * SWB, sh_down_s + row0 * SG,
            sh_down_b + row0 * SG, sh_x, simd_lid, sres);
      }
    #endif
    #if SHARED_WIDE_DOWN
      // Shared expert down projection of this token with the multi-row qmv_wide
      // arithmetic its own [T, K] matmul uses (8 lanes per row; each token's
      // accumulation is independent of the others).
      {
        static_assert(RPS == 4, "qmv_wide rows per simdgroup");
        constexpr int SWB = K * SBITS / 8;
        constexpr int SG = K / SGS;
        const int k_lane = int(simd_lid) % 8;
        const int srow = out_row + int(simd_lid) / 8;
        float sv[1] = {0.0f};
        glm_qmv_wide_row<T, K, SGS, SBITS, 1>(
            (const device uint8_t*)sh_down_w + size_t(srow) * SWB, sh_down_s + srow * SG,
            sh_down_b + srow * SG, sh_act + size_t(token) * K, 1, k_lane, sv);
        sv[0] += simd_shuffle_down(sv[0], 4);
        sv[0] += simd_shuffle_down(sv[0], 2);
        sv[0] += simd_shuffle_down(sv[0], 1);
        if (k_lane == 0) {
          const int r = int(simd_lid) / 8;
          float a = acc[0];
          for (int row = 1; row < RPS; row++) {
            a = row == r ? acc[row] : a;
          }
          out[size_t(token) * N + srow] = static_cast<T>(a) + static_cast<T>(sv[0]);
        }
      }
      return;
    #endif
      device T* o = out + size_t(token) * N + out_row;
      for (int row = 0; row < RPS; row++) {
    #if HAS_SHARED
        float sv = simd_sum(sres[row]);
    #endif
        if (simd_lid == 0) {
    #if HAS_SHARED
          o[row] = static_cast<T>(acc[row]) + static_cast<T>(sv);
    #elif ADD_SHARED_Y
          o[row] = static_cast<T>(acc[row]) + shared_y[size_t(token) * N + out_row + row];
    #else
          o[row] = static_cast<T>(acc[row]);
    #endif
        }
      }
    """

private final class GLM5NextMoEKernels: Sendable {
    static let shared = GLM5NextMoEKernels()
    let gateUp: MLXFast.MLXFastKernel
    let down: MLXFast.MLXFastKernel

    private init() {
        gateUp = MLXFast.metalKernel(
            name: "glm5next_moe_gate_up_select",
            inputNames: [
                "x", "sel_sig", "sel_biased", "sel_scaling", "limit", "gate_w", "gate_s",
                "gate_b", "up_w", "up_s", "up_b", "sh_gate_w", "sh_gate_s", "sh_gate_b",
                "sh_up_w", "sh_up_s", "sh_up_b",
            ],
            outputNames: ["out", "sel_indices", "sel_scores"],
            source: "#define HAS_SHARED 1\n#define SHARED_WIDE 0\n#define SLOT_MAJOR 0\n#define SELECT 1\n" + glm5NextMoEGateUpSource,
            header: glm5NextQMVHeader)
        down = MLXFast.metalKernel(
            name: "glm5next_moe_down_combine",
            inputNames: [
                "act", "indices", "scores", "down_w", "down_s", "down_b", "sh_down_w",
                "sh_down_s", "sh_down_b",
            ],
            outputNames: ["out"],
            source: "#define HAS_SHARED 1\n#define ADD_SHARED_Y 0\n#define SLOT_MAJOR 0\n#define SHARED_WIDE_DOWN 0\n" + glm5NextMoEDownSource,
            header: glm5NextQMVHeader)
    }
}

/// An affine quantized projection's `(weight, scales, biases, bits, groupSize)`.
struct GLM5NextAffine {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let bits: Int
    let groupSize: Int
}

/// MLX's qmv_fast tiles for these kernels: full 8-row output tiles and whole
/// 32-lane K blocks.
private func glm5NextQMVFastCovered(bits: Int, groupSize: Int, n: Int, k: Int) -> Bool {
    guard [3, 4, 5, 6, 8].contains(bits), [32, 64, 128].contains(groupSize) else { return false }
    let packFactor = (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits)
    let valuesPerThread = packFactor * 2
    return groupSize % valuesPerThread == 0 && n % 8 == 0 && k % (valuesPerThread * 32) == 0
}

/// One token's routed + shared experts: router logits (`glm5NextRouterLogits`
/// outputs `sig`, `biased`) to the combined output `[1, K]`, or `nil` when
/// the shapes are not covered.
func glm5NextMoEExperts(
    _ x: MLXArray, sig: MLXArray, biased: MLXArray, topK: Int, scaling: Float,
    normalize: Bool, limit: Float,
    gate: GLM5NextAffine, up: GLM5NextAffine, down: GLM5NextAffine,
    sharedGate: GLM5NextAffine, sharedUp: GLM5NextAffine, sharedDown: GLM5NextAffine
) -> MLXArray? {
    let rps = 4
    let nsg = 2
    guard x.ndim == 2, x.dim(0) == 1, [.bfloat16, .float16].contains(x.dtype),
        sig.dtype == .float32, biased.dtype == .float32, sig.ndim == 2, sig.dim(0) == 1,
        biased.shape == sig.shape, sig.dim(1) <= 1024, (1 ... 32).contains(topK),
        gate.weight.ndim == 3, gate.weight.shape == up.weight.shape,
        (gate.bits, gate.groupSize) == (up.bits, up.groupSize),
        (sharedGate.bits, sharedGate.groupSize) == (sharedUp.bits, sharedUp.groupSize),
        down.weight.ndim == 3
    else { return nil }
    let K = x.dim(1)
    let N = gate.weight.dim(1)
    let experts = sig.dim(1)
    let dtype = x.dtype
    for p in [gate, up, down, sharedGate, sharedUp, sharedDown]
    where p.scales.dtype != dtype || p.biases.dtype != dtype {
        return nil
    }
    guard gate.weight.dim(0) >= experts, down.weight.dim(1) == K, down.weight.dim(0) >= experts,
        sharedGate.weight.dim(0) == N, sharedUp.weight.dim(0) == N, sharedDown.weight.dim(0) == K,
        gate.weight.dim(2) * 32 / gate.bits == K, sharedGate.weight.dim(1) * 32 / sharedGate.bits == K,
        down.weight.dim(2) * 32 / down.bits == N, sharedDown.weight.dim(1) * 32 / sharedDown.bits == N,
        glm5NextQMVFastCovered(bits: gate.bits, groupSize: gate.groupSize, n: N, k: K),
        glm5NextQMVFastCovered(bits: sharedGate.bits, groupSize: sharedGate.groupSize, n: N, k: K),
        glm5NextQMVFastCovered(bits: down.bits, groupSize: down.groupSize, n: K, k: N),
        glm5NextQMVFastCovered(bits: sharedDown.bits, groupSize: sharedDown.groupSize, n: K, k: N),
        N % (rps * nsg) == 0, K % (rps * nsg) == 0
    else { return nil }

    let kernels = GLM5NextMoEKernels.shared
    let slots = topK + 1
    let selected = kernels.gateUp(
        [x, sig, biased, MLXArray([scaling]), MLXArray([limit]),
         gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases,
         sharedGate.weight, sharedGate.scales, sharedGate.biases,
         sharedUp.weight, sharedUp.scales, sharedUp.biases],
        template: [
            ("T", dtype), ("K", K), ("N", N), ("TOPK", topK), ("RBITS", gate.bits),
            ("RGS", gate.groupSize), ("RPS", rps), ("NSG", nsg), ("ESTRIDE", N), ("UP_OFF", 0),
            ("NE", experts), ("SEL_NORM", normalize && topK > 1),
            ("SBITS", sharedGate.bits), ("SGS", sharedGate.groupSize),
        ],
        grid: (32, (N / (rps * nsg)) * nsg, slots),
        threadGroup: (32, nsg, 1),
        outputShapes: [[1, slots, N], [1, topK], [1, topK]],
        outputDTypes: [dtype, .uint32, .float32])
    return kernels.down(
        [selected[0], selected[1], selected[2], down.weight, down.scales, down.biases,
         sharedDown.weight, sharedDown.scales, sharedDown.biases],
        template: [
            ("T", dtype), ("K", N), ("N", K), ("TOPK", topK), ("RBITS", down.bits),
            ("RGS", down.groupSize), ("RPS", rps), ("NSG", nsg),
            ("SBITS", sharedDown.bits), ("SGS", sharedDown.groupSize),
        ],
        grid: (32, (K / (rps * nsg)) * nsg, 1),
        threadGroup: (32, nsg, 1),
        outputShapes: [[1, K]],
        outputDTypes: [dtype])[0]
}
