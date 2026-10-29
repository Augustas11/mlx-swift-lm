// Copyright © 2025-2026 Apple Inc.
// MXFP8 decoding and accumulation adapted from MLX's fp8.h and
// fp_quantized.h (fp_qmv_fast_impl), under the repository's MIT license.

import MLX
import MLXNN

/// Several activation vectors share each MXFP8 weight tile. Each vector keeps
/// the pinned MLX one-vector qmv_fast reduction order, including the final
/// output conversion; this deliberately does not use multi-vector GEMM order.
private let deepseekV4QMVSource = """
    const uint lane = thread_index_in_simdgroup;
    const int r0 = (int(threadgroup_position_in_grid.x) * 2
        + int(simdgroup_index_in_threadgroup)) * 4;
    const device uchar* ws = reinterpret_cast<const device uchar*>(w)
        + size_t(r0) * K + lane * 8;
    const device uchar* ss = s + size_t(r0) * (K / 32) + lane / 4;
    float result[NV][4];
    for (int v = 0; v < NV; ++v) {
        for (int r = 0; r < 4; ++r) { result[v][r] = 0.0f; }
    }
    for (int k = 0; k < K; k += 256) {
        float xv[NV][8];
        for (int v = 0; v < NV; ++v) {
            for (int i = 0; i < 8; ++i) {
                xv[v][i] = x[size_t(v) * K + k + lane * 8 + i];
            }
        }
        for (int r = 0; r < 4; ++r) {
            const uchar sb = ss[r * (K / 32)];
            // fp8_e8m0::operator float(), including the subnormal scale.
            const float scale = as_type<float>(sb == 0 ? 0x400000u : uint(sb) << 23);
            float weight[8];
            for (int i = 0; i < 8; ++i) {
                const uchar b = ws[size_t(r) * K + i];
                // fp8_e4m3 widens via its half conversion in pinned MLX.
                const ushort a = b & 127;
                const ushort u = (a << 7) | (((a + 1) >> 7) << 14)
                    | (ushort((b >> 7) & 1) << 15);
                const half decoded = as_type<half>(u) * 256.0f;
                weight[i] = float(decoded);
            }
            for (int v = 0; v < NV; ++v) {
                float accum = 0.0f;
                for (int i = 0; i < 8; ++i) { accum += xv[v][i] * weight[i]; }
                result[v][r] += scale * accum;
            }
        }
        ws += 256;
        ss += 8;
    }
    for (int v = 0; v < NV; ++v) {
        for (int r = 0; r < 4; ++r) {
            const float total = simd_sum(result[v][r]);
            if (lane == 0) { y[size_t(v) * N + r0 + r] = static_cast<T>(total); }
        }
    }
    """

private final class DeepseekV4QMVKernel: Sendable {
    static let shared = DeepseekV4QMVKernel()
    let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_mxfp8_qmv_batch", inputNames: ["x", "w", "s"],
        outputNames: ["y"], source: deepseekV4QMVSource, ensureRowContiguous: true)
}

/// Only complete pinned qmv_fast tiles and the artifact's MXFP8 layout are
/// covered. Returning nil leaves all other formats/devices on the MLX path.
func deepseekV4QMVBatch(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu,
        x.ndim == 2, (2 ... 8).contains(x.dim(0)),
        [.bfloat16, .float16].contains(x.dtype),
        layer.mode == .mxfp8, layer.bits == 8, layer.groupSize == 32,
        layer.bias == nil, layer.biases == nil, layer.globalScale == nil,
        layer.weight.ndim == 2, layer.weight.dtype == .uint32,
        layer.scales.ndim == 2, layer.scales.dtype == .uint8
    else { return nil }
    let (vectors, k, n) = (x.dim(0), x.dim(1), layer.weight.dim(0))
    // quantized.cpp:qmv_fast_k_alignment(8) = (32 / 8) * 2 * 32.
    guard k > 0, k % 256 == 0, n > 0, n % 8 == 0,
        layer.weight.dim(1) == k / 4, layer.scales.shape == [n, k / 32],
        n <= Int(Int32.max) / 8, k <= Int(Int32.max)
    else { return nil }
    return DeepseekV4QMVKernel.shared.kernel(
        [x, layer.weight, layer.scales],
        template: [("T", x.dtype), ("NV", vectors), ("K", k), ("N", n)],
        grid: (64 * (n / 8), 1, 1), threadGroup: (64, 1, 1),
        outputShapes: [[vectors, n]], outputDTypes: [x.dtype])[0]
}

/// Shared projection path for serial decode, continuous batch and packed
/// verification. Unsupported inputs preserve the original layer call.
func deepseekV4Project(_ layer: Linear, _ x: MLXArray) -> MLXArray {
    guard x.ndim >= 2, x.dim(-1) > 0 else { return layer(x) }
    let k = x.dim(-1)
    let vectors = x.size / k
    guard (2 ... 8).contains(vectors), let quantized = layer as? QuantizedLinear,
        let y = deepseekV4QMVBatch(x.reshaped(vectors, k), quantized)
    else { return layer(x) }
    return y.reshaped(Array(x.shape.dropLast()) + [y.dim(-1)])
}
