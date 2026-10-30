// Copyright © 2025-2026 Apple Inc.
// MXFP4 qmv_wide arithmetic adapted from MLX v0.32.4 fp_quantized.h,
// fp4.h and fp8.h, under the MIT license:
// https://github.com/ml-explore/mlx/blob/v0.32.4/LICENSE
// Changes: fixed MXFP4/group32/non-batched layout and runtime Metal templates.

import MLX

private let deepseekV4ReferencePrefillWideSource = """
    const uint lane = thread_index_in_simdgroup;
    const int k_lane = lane % 16;
    const int sg_row = lane / 16;
    const int out_row = int(threadgroup_position_in_grid.y) * 4
        + 2 * int(simdgroup_index_in_threadgroup) + sg_row;
    const int vec0 = int(threadgroup_position_in_grid.x) * NV;
    const int row = min(out_row, N - 1);
    const device uchar* wrow = reinterpret_cast<const device uchar*>(w)
        + size_t(row) * (K / 2);
    const device uchar* srow = s + size_t(row) * (K / 32);
    const device T* xv[NV];
    for (int v = 0; v < NV; ++v) {
        xv[v] = x + size_t(min(vec0 + v, M - 1)) * K;
    }
    float result[NV] = {0};
    for (int g = k_lane; g < K / 32; g += 16) {
        const int k0 = g * 32;
        const uchar sb = srow[g];
        const float scale = as_type<float>(sb == 0 ? 0x400000u : uint(sb) << 23);
        const device ushort* wq = reinterpret_cast<const device ushort*>(wrow + k0 / 2);
        float4 w4[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const ushort packed = wq[i];
            float decoded[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uchar b = uchar(packed >> (4 * j));
                half value = as_type<half>(ushort((b & 7) << 9));
                value *= 16384.0f;
                decoded[j] = float((b & 8) ? -value : value);
            }
            w4[i] = float4(decoded[0], decoded[1], decoded[2], decoded[3]);
        }
        #pragma unroll
        for (int v = 0; v < NV; ++v) {
            const device vec<T, 4>* xv4 = reinterpret_cast<const device vec<T, 4>*>(xv[v] + k0);
            float acc = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += dot(w4[j], float4(xv4[j]));
            }
            result[v] += scale * acc;
        }
    }
    for (int v = 0; v < NV; ++v) {
        result[v] += simd_shuffle_down(result[v], 8);
        result[v] += simd_shuffle_down(result[v], 4);
        result[v] += simd_shuffle_down(result[v], 2);
        result[v] += simd_shuffle_down(result[v], 1);
    }
    if (k_lane == 0 && out_row < N) {
        for (int v = 0; v < NV; ++v) {
            if (vec0 + v < M) {
                y[size_t(vec0 + v) * N + out_row] = static_cast<T>(result[v]);
            }
        }
    }
    """

private final class DeepseekV4ReferencePrefillWideKernel: Sendable {
    static let shared = DeepseekV4ReferencePrefillWideKernel()
    let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_reference_prefill_mxfp4_wide", inputNames: ["x", "w", "s"],
        outputNames: ["y"], source: deepseekV4ReferencePrefillWideSource,
        ensureRowContiguous: true)
}

/// Per-expert prefill projection with the frozen MLX 0.32.4 reference's
/// qmv_wide/split-K arithmetic. Call only from prefill; ordinary decode and
/// continuous batching retain their qualified one-vector reduction order.
/// Other architectures and formats return nil to preserve the existing path.
func deepseekV4ReferencePrefillQMM(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray
) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu,
        GPU.deviceInfo().architecture == "applegpu_g15d",
        x.ndim == 2, x.dtype == .bfloat16,
        weight.ndim == 2, weight.dtype == .uint32,
        scales.ndim == 2, scales.dtype == .uint8
    else { return nil }
    let (m, k, n) = (x.dim(0), x.dim(1), weight.dim(0))
    guard m > 0, m <= Int(Int32.max), k > 0, k % 32 == 0, n > 0,
        k <= Int(Int32.max), n <= Int(Int32.max) / 16,
        weight.dim(1) == k / 8, scales.shape == [n, k / 32]
    else { return nil }
    let vectorLimit = k <= 2048 && n <= 2048 ? 32 : (k <= 4096 && n <= 4096 ? 18 : 12)
    if m == 1 {
        return quantizedMM(
            x, weight, scales: scales, biases: nil, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
    }
    if m < vectorLimit {
        let tiles = (m + 4) / 5
        let vectors = (m + tiles - 1) / tiles
        return DeepseekV4ReferencePrefillWideKernel.shared.kernel(
            [x, weight, scales],
            template: [("T", x.dtype), ("M", m), ("K", k), ("N", n), ("NV", vectors)],
            grid: (tiles * 64, (n + 3) / 4, 1), threadGroup: (64, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
    }
    let threadgroups = ((n + 31) / 32) * ((m + 31) / 32)
    var splits = min(max(1, 512 / threadgroups), k / 32)
    while splits > 1 && k % (splits * 32) != 0 { splits -= 1 }
    guard splits > 1 else {
        return quantizedMM(
            x, weight, scales: scales, biases: nil, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
    }
    let length = k / splits
    var total: MLXArray?
    for start in stride(from: 0, to: k, by: length) {
        var part = x[0..., start ..< (start + length)].asType(.float32)
        // All Ultra vector limits are <=32. Extra zero rows select qmm for
        // these float32 partial products, without changing any retained row.
        if m < 33 {
            part = concatenated([part, MLXArray.zeros([33 - m, length], dtype: .float32)], axis: 0)
        }
        let partial = quantizedMM(
            part, weight[0..., (start / 8) ..< ((start + length) / 8)],
            scales: scales[0..., (start / 32) ..< ((start + length) / 32)], biases: nil,
            transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)[..<m]
        if let previous = total {
            total = previous + partial
        } else {
            total = partial
        }
    }
    return total?.asType(x.dtype)
}
