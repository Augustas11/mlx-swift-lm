// Decode short convolutions in one Metal dispatch each, writing the next float32 state as well:
//
// - a static depthwise causal convolution (mlx-vlm's `inkling2_virtual_short_convolution`, with
//   the residual added in the kernel): `out = residual + round((sum_j ctx_j w_j + x) / divisor)`;
// - a dynamic one whose taps are the static taps plus taps projected from the input, one set per
//   head: `out = round((0 + sum_lag ctx_lag (s_lag + d_lag) + x) / divisor)`.
//
// Ported from Ollama's Inkling 2 short convolution (Ollama pull request 19843, `mlx/short_conv.go`;
// MIT licence, Copyright (c) Ollama), which adds the residual in the convolution's launch and
// handles the dynamic taps in the same kernel. This port keeps the reference's arithmetic and
// order: the static kernel's convolution is mlx-vlm's kernel text unchanged (the residual is only
// an addition after its rounding), and the dynamic kernel computes each lag product, then adds
// them from zero in lag order, with floating-point contraction off so no product is fused into an
// addition. Both are bit-identical to the op-by-op graph they replace.

import Foundation
import MLX

/// mlx-vlm `models/inkling2/kernels.py` `_short_convolution` (math mode safe, MLX's default), its
/// convolution unchanged, with the next state (the last `W - 1` rows of the state/input
/// concatenation, float32) and an optional residual added in float32 after the rounding.
private let staticConvolutionSource = """
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        T convolved = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                         + c * x_strides[2]])) / divisor);
        out[i] = HAS_RESIDUAL
            ? OutT(float(residual[b * residual_strides[0] + t * residual_strides[1]
                                  + c * residual_strides[2]]) + float(convolved))
            : OutT(convolved);
        if (t == L - 1) {
            for (int s = 0; s < W - 1; ++s) {
                int p = L + s - (W - 1);
                nstate[(b * (W - 1) + s) * C + c] = p < 0
                    ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                                  + c * state_strides[2]])
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
            }
        }
    """

/// The dynamic short convolution: lag `l` reads context row `W - 1 - l + t` (the state below
/// row `W - 1`, the input above) and the tap `static_w[l, c] + taps[b, t, head * W + l]`.
private let dynamicConvolutionSource = """
        #pragma clang fp contract(off)
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        int head = c / (C / Heads);
        float sum = 0.0f;
        for (int lag = 0; lag < W; ++lag) {
            int p = t - lag;
            float value = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
            float tap = float(static_w[lag * static_w_strides[0] + c * static_w_strides[1]])
                + float(taps[b * taps_strides[0] + t * taps_strides[1]
                             + (head * W + lag) * taps_strides[2]]);
            float product = value * tap;
            sum = sum + product;
        }
        float xf = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
        out[i] = T((sum + xf) / divisor);
        if (t == L - 1) {
            for (int s = 0; s < W - 1; ++s) {
                int p = L + s - (W - 1);
                nstate[(b * (W - 1) + s) * C + c] = p < 0
                    ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                                  + c * state_strides[2]])
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
            }
        }
    """

/// Decode short convolutions (see the file header).
package enum ShortConvolutionKernels {
    private static let staticKernel = MLXFast.metalKernel(
        name: "short_convolution_residual",
        inputNames: ["state", "weight", "x", "mask", "divisor", "residual"],
        outputNames: ["out", "nstate"], source: staticConvolutionSource,
        ensureRowContiguous: false)
    private static let dynamicKernel = MLXFast.metalKernel(
        name: "short_convolution_dynamic",
        inputNames: ["state", "x", "taps", "static_w", "divisor"],
        outputNames: ["out", "nstate"], source: dynamicConvolutionSource,
        ensureRowContiguous: false)

    /// The unread mask (`HAS_MASK` false) and the divisors for width 4, held once as host data so
    /// no fill kernel runs per call.
    nonisolated(unsafe) private static let noMask = MLXArray([false], [1, 1])
    nonisolated(unsafe) private static let staticDivisor4 = MLXArray(Float(Double(5).squareRoot()))
    nonisolated(unsafe) private static let dynamicDivisor4 = MLXArray(Float(Double(9).squareRoot()))

    /// mlx-vlm's GPU condition for its decode kernel: bfloat16 input `[B, L, C]` with `B <= 4`,
    /// `L <= 8`, `C` a multiple of 128, width-4 taps, on the GPU.
    private static func decodeShape(_ x: MLXArray) -> Bool {
        Device.defaultDevice().deviceType == .gpu && x.dtype == .bfloat16 && x.ndim == 3
            && x.dim(0) <= 4 && x.dim(1) <= 8 && x.dim(2) % 128 == 0
    }

    /// Whether ``convolve(_:taps:state:residual:outputDType:)`` runs: taps `[C, 4]`.
    package static func supportsStatic(_ x: MLXArray, taps: MLXArray) -> Bool {
        decodeShape(x) && taps.ndim == 2 && taps.dim(0) == x.dim(2) && taps.dim(1) == 4
    }

    /// The static convolution of `x` over `state` (`[B, 3, C]` float32), plus `residual`
    /// (`[B, L, C]`, added in float32 after the convolution's rounding) when given, in
    /// `outputDType`; and the next state.
    package static func convolve(
        _ x: MLXArray, taps: MLXArray, state: MLXArray, residual: MLXArray?, outputDType: DType
    ) -> (out: MLXArray, state: MLXArray) {
        precondition(supportsStatic(x, taps: taps), "unsupported static short convolution")
        if let residual { precondition(residual.shape == x.shape, "residual must match x") }
        let (batch, channels) = (x.dim(0), x.dim(2))
        let results = staticKernel(
            [state, taps, x, noMask, staticDivisor4, residual ?? x],
            template: [
                ("T", x.dtype), ("W", 4), ("HAS_MASK", false), ("HAS_RESIDUAL", residual != nil),
                ("OutT", outputDType),
            ],
            grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [x.shape, [batch, 3, channels]], outputDTypes: [outputDType, .float32])
        return (results[0], results[1])
    }

    /// Whether ``convolveDynamic(_:dynamicTaps:staticTaps:heads:state:)`` runs: static taps
    /// `[4, C]`, dynamic taps `[B, L, heads * 4]`, `C` a multiple of `heads`.
    package static func supportsDynamic(
        _ x: MLXArray, dynamicTaps: MLXArray, staticTaps: MLXArray, heads: Int
    ) -> Bool {
        decodeShape(x) && heads > 0 && x.dim(2) % heads == 0 && staticTaps.ndim == 2
            && staticTaps.dim(0) == 4 && staticTaps.dim(1) == x.dim(2)
            && dynamicTaps.shape == [x.dim(0), x.dim(1), heads * 4]
    }

    /// The dynamic convolution of `x` over `state` (`[B, 3, C]` float32) in the dtype of `x`, and
    /// the next state.
    package static func convolveDynamic(
        _ x: MLXArray, dynamicTaps: MLXArray, staticTaps: MLXArray, heads: Int, state: MLXArray
    ) -> (out: MLXArray, state: MLXArray) {
        precondition(
            supportsDynamic(x, dynamicTaps: dynamicTaps, staticTaps: staticTaps, heads: heads),
            "unsupported dynamic short convolution")
        let (batch, channels) = (x.dim(0), x.dim(2))
        let results = dynamicKernel(
            [state, x, dynamicTaps, staticTaps, dynamicDivisor4],
            template: [("T", x.dtype), ("W", 4), ("Heads", heads)],
            grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [x.shape, [batch, 3, channels]], outputDTypes: [x.dtype, .float32])
        return (results[0], results[1])
    }
}
