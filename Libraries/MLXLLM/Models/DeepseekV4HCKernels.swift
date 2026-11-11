// Copyright © 2026 Apple Inc.
// Reduction orders replayed from MLX's row_reduce_looped, col_reduce_small,
// softmax_single_row and fp32 gemv kernels, under the repository's MIT license.

import MLX

// DeepSeek's canonical hyper-connection decode arithmetic is a graph of about
// a hundred small ops per half-layer (mix, a 20-step Sinkhorn, collapse and
// the branch RMSNorm). These kernels evaluate that graph per token, with the
// graph's rounding points and reduction orders, so a token's result does not
// depend on how many tokens or rows share the dispatch.
//
// Products the graph materializes are written as fma(a, b, 0), which rounds
// once exactly like a stored product and cannot be contracted into a later
// addition. Products the graph fuses (the compiled Sinkhorn) stay plain
// separate statements, as MLX's compiled kernels emit them.

private let deepseekV4HCHeader = """
    // MLX's Sigmoid functor.
    inline float dsv4_sigmoid(float x) {
      auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
      return (x < 0) ? y : 1 - y;
    }
    """

/// `matmul(flat, fn.T) * rsqrt(mean(square(flat)) + eps)` for one token per
/// `y` row of the grid: row_reduce_looped's 1024-thread order for the mean
/// of the materialized squares, MLX's fp32 gemv lane order for the product.
private let deepseekV4HCMixSource = """
      const uint lid = thread_position_in_threadgroup.x;
      const uint simd_lid = thread_index_in_simdgroup;
      const uint simd_gid = simdgroup_index_in_threadgroup;
      const int tok = int(threadgroup_position_in_grid.y);
      const int row = int(threadgroup_position_in_grid.x);
      constexpr int KSZ = HCD;
      const device T* xr = x + size_t(tok) * KSZ;

      threadgroup float local_inv[1];
      threadgroup float local_sums[32];
      for (int v = 0; v < 4; v++) {
        const uint vt = uint(v) * 256 + lid;
        float acc = 0;
        for (uint r = 0; r < uint(KSZ); r += 1024 * 4) {
          for (int i = 0; i < 4; i++) {
            float xi = static_cast<float>(xr[r + vt * 4 + i]);
            float sq = metal::fma(xi, xi, 0.0f);
            acc = sq + acc;
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
          float mean = acc * (1.0f / float(KSZ));
          float shifted = mean + eps[0];
          local_inv[0] = metal::precise::rsqrt(shifted);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv = local_inv[0];

      threadgroup float partial[8];
      const int sgN = int(simd_gid);
      float result = 0;
      const device float* mrow = fn + size_t(row) * KSZ;
      int bn = (32 * sgN + int(simd_lid)) * 4;
      for (int i = 0; i < KSZ / 1024; ++i) {
        float v_coeff[4];
        float inter[4];
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(xr[bn + tn]);
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
        mixes[size_t(tok) * MIX + row] = total * inv;
      }
    """

/// The compiled Sinkhorn (MLX Sigmoid, softmax_single_row with fast exp,
/// sequential four-element sums), the materialized collapse rounded to the
/// activation dtype, and DeepSeek's float32-weighted RMSNorm with
/// row_reduce_looped's 1024-thread order. One token per threadgroup.
private let deepseekV4HCNormSource = """
    const uint tok = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int MIX = (2 + HC) * HC;
    constexpr int CB = 2 * HC;
    const float hc_eps = hc_eps_in[0];
    const float norm_eps = norm_eps_in[0];

    threadgroup float pre_shared[HC];
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];

    if (tid == 0) {
      const device float* m = mixes + size_t(tok) * MIX;
      const float s0 = scale[0];
      const float s1 = scale[1];
      const float s2 = scale[2];
      for (int i = 0; i < HC; i++) {
        float t = m[i] * s0;
        float z = t + base[i];
        float g = dsv4_sigmoid(z);
        pre_shared[i] = g + hc_eps;
      }
      for (int i = 0; i < HC; i++) {
        float t = m[HC + i] * s1;
        float z = t + base[HC + i];
        float g = dsv4_sigmoid(z);
        post[size_t(tok) * HC + i] = 2.0f * g;
      }
      float c[HC][HC];
      for (int i = 0; i < HC; i++) {
        for (int j = 0; j < HC; j++) {
          float t = m[CB + i * HC + j] * s2;
          c[i][j] = t + base[CB + i * HC + j];
        }
      }
      // softmax(axis: -1, precise: true)
      for (int i = 0; i < HC; i++) {
        float maxval = -FLT_MAX;
        for (int j = 0; j < HC; j++) {
          maxval = (maxval < c[i][j]) ? c[i][j] : maxval;
        }
        float normalizer = 0;
        float e[HC];
        for (int j = 0; j < HC; j++) {
          float d = c[i][j] - maxval;
          e[j] = metal::fast::exp(d);
          normalizer += e[j];
        }
        normalizer = 1 / normalizer;
        for (int j = 0; j < HC; j++) {
          float p = e[j] * normalizer;
          c[i][j] = p + hc_eps;
        }
      }
      for (int j = 0; j < HC; j++) {
        float s = 0;
        for (int i = 0; i < HC; i++) { s = c[i][j] + s; }
        float d = s + hc_eps;
        for (int i = 0; i < HC; i++) { c[i][j] = c[i][j] / d; }
      }
      for (int it = 1; it < ITERS; it++) {
        for (int i = 0; i < HC; i++) {
          float s = 0;
          for (int j = 0; j < HC; j++) { s = c[i][j] + s; }
          float d = s + hc_eps;
          for (int j = 0; j < HC; j++) { c[i][j] = c[i][j] / d; }
        }
        for (int j = 0; j < HC; j++) {
          float s = 0;
          for (int i = 0; i < HC; i++) { s = c[i][j] + s; }
          float d = s + hc_eps;
          for (int i = 0; i < HC; i++) { c[i][j] = c[i][j] / d; }
        }
      }
      device float* comb_out = comb + size_t(tok) * HC * HC;
      for (int i = 0; i < HC; i++) {
        for (int j = 0; j < HC; j++) { comb_out[i * HC + j] = c[i][j]; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device T* xt = x + size_t(tok) * HC * D;
    device T* out = normalized + size_t(tok) * D;
    float acc = 0;
    for (int b = 0; b < D / 4096; b++) {
      for (int k = 0; k < 4; k++) {
        const int d = b * 4096 + int(tid) * 4 + k;
        float s = 0;
        for (int i = 0; i < HC; i++) {
          float p = metal::fma(pre_shared[i], static_cast<float>(xt[i * D + d]), 0.0f);
          s = p + s;
        }
        float v = static_cast<float>(static_cast<T>(s));
        float sq = metal::fma(v, v, 0.0f);
        acc = sq + acc;
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float total = simd_sum(local_sums[lane]);
      if (lane == 0) {
        float mean = total * (1.0f / float(D));
        float shifted = mean + norm_eps;
        local_inv[0] = metal::precise::rsqrt(shifted);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv[0];
    for (int b = 0; b < D / 4096; b++) {
      for (int k = 0; k < 4; k++) {
        const int d = b * 4096 + int(tid) * 4 + k;
        float s = 0;
        for (int i = 0; i < HC; i++) {
          float p = metal::fma(pre_shared[i], static_cast<float>(xt[i * D + d]), 0.0f);
          s = p + s;
        }
        float v = static_cast<float>(static_cast<T>(s));
        float scaled = v * inv;
        float weighted = scaled * static_cast<float>(norm_weight[d]);
        out[d] = static_cast<T>(weighted);
      }
    }
    """

/// `post * x + sum_i(comb[i, j] * residual[i])` with every product and the
/// sequential four-term sum rounded as the materialized graph rounds them.
private let deepseekV4HCExpandSource = """
    const uint gid = thread_position_in_grid.x;
    const uint tok = gid / uint(HC * D);
    const uint rem = gid % uint(HC * D);
    const uint j = rem / uint(D);
    const uint d = rem % uint(D);
    const device float* c = comb + size_t(tok) * HC * HC;
    const device T* r = residual + size_t(tok) * HC * D;
    float s = 0;
    for (int i = 0; i < HC; i++) {
      float p = metal::fma(c[i * HC + j], static_cast<float>(r[i * D + d]), 0.0f);
      s = p + s;
    }
    float e = metal::fma(post[size_t(tok) * HC + j], static_cast<float>(x[size_t(tok) * D + d]), 0.0f);
    float total = e + s;
    out[gid] = static_cast<T>(total);
    """

private final class DeepseekV4HCKernels: Sendable {
    static let shared = DeepseekV4HCKernels()
    let mix: MLXFast.MLXFastKernel
    let norm: MLXFast.MLXFastKernel
    let expand: MLXFast.MLXFastKernel

    private init() {
        mix = MLXFast.metalKernel(
            name: "deepseek_v4_hc_mix", inputNames: ["x", "fn", "eps"], outputNames: ["mixes"],
            source: deepseekV4HCMixSource)
        norm = MLXFast.metalKernel(
            name: "deepseek_v4_hc_norm",
            inputNames: ["x", "mixes", "scale", "base", "norm_weight", "hc_eps_in", "norm_eps_in"],
            outputNames: ["normalized", "post", "comb"],
            source: deepseekV4HCNormSource, header: deepseekV4HCHeader)
        expand = MLXFast.metalKernel(
            name: "deepseek_v4_hc_expand",
            inputNames: ["x", "residual", "post", "comb"], outputNames: ["out"],
            source: deepseekV4HCExpandSource)
    }
}

/// Canonical DeepSeek branch input for `x: [B, L, 4, D]` on the GPU:
/// `(norm(collapse(x)), post, comb)`, or `nil` outside the replayed shapes.
func deepseekV4HCBranchInput(
    _ x: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray,
    sinkhornIterations: Int, mixEps: Float, hcEps: Float, normEps: Float
) -> (MLXArray, MLXArray, MLXArray)? {
    guard Device.defaultDevice().deviceType == .gpu, x.ndim == 4, x.dtype == .bfloat16,
        x.dim(2) == 4, x.dim(3) >= 4096, x.dim(3) % 4096 == 0,
        x.dim(3) & (x.dim(3) - 1) == 0,
        fn.dtype == .float32, fn.shape == [24, 4 * x.dim(3)],
        scale.dtype == .float32, scale.shape == [3], base.dtype == .float32, base.shape == [24],
        normWeight.shape == [x.dim(3)]
    else { return nil }
    let (B, L, hc, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let kernels = DeepseekV4HCKernels.shared
    let mixes = kernels.mix(
        [x, fn, MLXArray([mixEps])],
        template: [("T", x.dtype), ("HCD", hc * D), ("MIX", 24)],
        grid: (256 * 24, B * L, 1), threadGroup: (256, 1, 1),
        outputShapes: [[B, L, 24]], outputDTypes: [.float32])[0]
    let outputs = kernels.norm(
        [x, mixes, scale, base, normWeight, MLXArray([hcEps]), MLXArray([normEps])],
        template: [
            ("T", x.dtype), ("HC", hc), ("D", D),
            ("ITERS", sinkhornIterations),
        ],
        grid: (B * L * 1024, 1, 1), threadGroup: (1024, 1, 1),
        outputShapes: [[B, L, D], [B, L, hc], [B, L, hc, hc]],
        outputDTypes: [x.dtype, .float32, .float32])
    return (outputs[0], outputs[1], outputs[2])
}

/// `deepseekV4HCExpand` as one GPU kernel, or `nil` outside its shapes.
func deepseekV4HCExpandKernel(
    _ x: MLXArray, _ residual: MLXArray, _ post: MLXArray, _ comb: MLXArray
) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu, x.dtype == .bfloat16,
        residual.dtype == x.dtype, residual.ndim == 4, residual.dim(2) == 4,
        residual.dim(3) >= 64,
        x.shape == [residual.dim(0), residual.dim(1), residual.dim(3)],
        post.dtype == .float32, post.shape == Array(residual.shape.dropLast()),
        comb.dtype == .float32, comb.shape == Array(residual.shape.dropLast()) + [4]
    else { return nil }
    let (B, L, hc, D) = (residual.dim(0), residual.dim(1), residual.dim(2), residual.dim(3))
    return DeepseekV4HCKernels.shared.expand(
        [x, residual, post, comb],
        template: [("T", x.dtype), ("HC", hc), ("D", D)],
        grid: (B * L * hc * D, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [residual.shape], outputDTypes: [x.dtype])[0]
}

/// DeepSeek's float32 RMSNorm graphs, one row per threadgroup with MLX's
/// row reduction order for `D` (row_reduce_looped and row_reduce_simple
/// agree for these widths): `MODE 0` multiplies the float32 weight before the
/// output cast (`DeepseekV4RMSNorm`); `MODE 1` rounds the inverse to the
/// input dtype and multiplies in it (`deepseekV4QueryHeadNorm`).
private let deepseekV4RMSNormSource = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = TG / 32;
    const device T* xr = x + size_t(row) * D;
    device T* out_row = out + size_t(row) * D;
    threadgroup float shared_sums[32];
    threadgroup float shared_inv[1];

    float acc = 0;
    for (int b = 0; b < D / (TG * 4); b++) {
      for (int i = 0; i < 4; i++) {
        float v = static_cast<float>(xr[b * TG * 4 + int(lid) * 4 + i]);
        float sq = metal::fma(v, v, 0.0f);
        acc = sq + acc;
      }
    }
    acc = simd_sum(acc);
    if (NSG > 1) {
      if (lane == 0) { shared_sums[sg] = acc; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float value = (lid < uint(NSG)) ? shared_sums[lid] : 0.0f;
      acc = simd_sum(value);
    }
    if (lid == 0) {
      float mean = acc * (1.0f / float(D));
      float shifted = mean + eps_in[0];
      shared_inv[0] = metal::precise::rsqrt(shifted);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = shared_inv[0];
    for (int b = 0; b < D / (TG * 4); b++) {
      for (int i = 0; i < 4; i++) {
        const int d = b * TG * 4 + int(lid) * 4 + i;
        if (MODE == 0) {
          float v = static_cast<float>(xr[d]);
          float scaled = v * inv;
          float weighted = scaled * static_cast<float>(weight[d]);
          out_row[d] = static_cast<T>(weighted);
        } else {
          T rounded = static_cast<T>(inv);
          out_row[d] = static_cast<T>(static_cast<float>(xr[d]) * static_cast<float>(rounded));
        }
      }
    }
    """

private final class DeepseekV4NormKernels: Sendable {
    static let shared = DeepseekV4NormKernels()
    let norm = MLXFast.metalKernel(
        name: "deepseek_v4_rms_norm", inputNames: ["x", "weight", "eps_in"],
        outputNames: ["out"], source: deepseekV4RMSNormSource)
}

/// MLX's reduction threadgroup width for a row (`threadgroup_size_from_row_size`).
private func deepseekV4ReduceThreads(_ width: Int) -> Int {
    if width <= 512 { return 32 }
    if width <= 1024 { return 128 }
    return min(1024, ((width + 3) / 4 + 31) / 32 * 32)
}

private func deepseekV4NormKernel(
    _ x: MLXArray, weight: MLXArray?, eps: Float, mode: Int
) -> MLXArray? {
    let width = x.dim(-1)
    let threads = deepseekV4ReduceThreads(width)
    guard Device.defaultDevice().deviceType == .gpu, x.ndim >= 1, x.size > 0,
        [.bfloat16, .float16, .float32].contains(x.dtype),
        width > 64, width % (threads * 4) == 0, width & (width - 1) == 0
    else { return nil }
    if mode == 0 { guard let weight, weight.shape == [width] else { return nil } }
    let w = weight ?? MLXArray.zeros([1], dtype: .float32)
    return DeepseekV4NormKernels.shared.norm(
        [x, w, MLXArray([eps])],
        template: [("T", x.dtype), ("D", width), ("TG", threads), ("MODE", mode)],
        grid: (x.size / width * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

/// `DeepseekV4RMSNorm` as one GPU kernel, or `nil` outside its shapes.
func deepseekV4RMSNormKernel(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray? {
    deepseekV4NormKernel(x, weight: weight, eps: eps, mode: 0)
}

/// `deepseekV4QueryHeadNorm` as one GPU kernel, or `nil` outside its shapes.
func deepseekV4QueryHeadNormKernel(_ q: MLXArray, eps: Float) -> MLXArray? {
    deepseekV4NormKernel(q, weight: nil, eps: eps, mode: 1)
}
