// Fused low-rank gated blends ("message controllers"): two Metal dispatches for
//
//     h        = [rms(x), rms(m)] . down          (down [2H, rank], float32)
//     gain     = tanh(h . gainForget[:, :H])
//     forget   = tanh(h . gainForget[:, H:])
//     out      = (1 - forget) * x + gain * m
//
// for several controllers at once (Inkling 2's attn, k and v controllers share one launch).
//
// Ported from Ollama's Inkling 2 loop controllers (Ollama pull request 19843,
// `mlx/loop_controller.go`; MIT licence, Copyright (c) Ollama): a down kernel that folds the
// two unit RMS norms into the low-rank product and a gate kernel that finishes the blend, where
// the graph runs two norms, a concatenation, three matmuls and the elementwise tail per
// controller. This port keeps mlx-vlm's `MessageController` arithmetic (the norms are applied to
// the inputs before the product, float32 weights as the oQ layout stores them) and takes its own
// layout: the down product is split over the 2H rows into partial sums that the gate kernel adds
// in a fixed order. The reduction order is not MLX's, so it is not bit-identical to the graph;
// callers gate it on greedy token parity.

import Foundation
import MLX

/// The down kernel: one threadgroup of 256 threads per (row block, controller, row). Every
/// threadgroup computes the two RMS scales itself, then its `Span` rows of `down`; lane `l` of a
/// SIMD group owns rank columns `l, l + 32, ...`.
private let lowRankDownSource = """
        constexpr uint Threads = 256, SIMDs = 8;
        constexpr uint Span = 2 * Width / Parts;
        constexpr uint PerLane = Rank / 32;
        const uint part = threadgroup_position_in_grid.x;
        const uint c = threadgroup_position_in_grid.y;
        const uint row = threadgroup_position_in_grid.z;
        const uint rows = threadgroups_per_grid.z;
        const uint t = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const size_t xb = (size_t(row) * Inputs + (Inputs == 1 ? 0u : c)) * Width;
        const size_t mb = (size_t(row) * Messages + First + c) * Width;
        threadgroup float sums[2][SIMDs];
        threadgroup float dots[SIMDs][Rank];

        float sx = 0.0f, sm = 0.0f;
        for (uint k = t; k < Width; k += Threads) {
            const float a = x[xb + k];
            const float b = message[mb + k];
            sx += a * a;
            sm += b * b;
        }
        sx = simd_sum(sx);
        sm = simd_sum(sm);
        if (lane == 0) {
            sums[0][sg] = sx;
            sums[1][sg] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tx = 0.0f, tm = 0.0f;
        for (uint g = 0; g < SIMDs; ++g) {
            tx += sums[0][g];
            tm += sums[1][g];
        }
        const float eps = as_type<float>(EpsBits);
        const float ix = metal::precise::rsqrt(tx / float(Width) + eps);
        const float im = metal::precise::rsqrt(tm / float(Width) + eps);

        float acc[PerLane];
        for (uint p = 0; p < PerLane; ++p) acc[p] = 0.0f;
        const uint begin = part * Span;
        const size_t wb = size_t(First + c) * (2 * Width) * Rank;
        for (uint i = sg; i < Span; i += SIMDs) {
            const uint k = begin + i;
            const float v = k < Width ? x[xb + k] * ix : message[mb + (k - Width)] * im;
            for (uint p = 0; p < PerLane; ++p) {
                acc[p] += v * down[wb + size_t(k) * Rank + lane + 32 * p];
            }
        }
        for (uint p = 0; p < PerLane; ++p) dots[sg][lane + 32 * p] = acc[p];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < Rank) {
            float s = 0.0f;
            for (uint g = 0; g < SIMDs; ++g) s += dots[g][t];
            partial[((size_t(c) * rows + row) * Parts + part) * Rank + t] = s;
        }
    """

/// The gate kernel: one thread per output channel; each threadgroup first adds the down
/// kernel's partial sums into the controller's `rank` values, in part order.
private let lowRankGateSource = """
        const uint j = thread_position_in_grid.x;
        const uint c = threadgroup_position_in_grid.y;
        const uint row = threadgroup_position_in_grid.z;
        const uint rows = threadgroups_per_grid.z;
        const uint t = thread_index_in_threadgroup;
        threadgroup float h[Rank];
        for (uint r = t; r < Rank; r += threads_per_threadgroup.x) {
            float s = 0.0f;
            for (uint p = 0; p < Parts; ++p) s += partial[((size_t(c) * rows + row) * Parts + p) * Rank + r];
            h[r] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j >= Width) return;
        const size_t gb = size_t(First + c) * Rank * (2 * Width);
        float gain = 0.0f, forget = 0.0f;
        for (uint r = 0; r < Rank; ++r) {
            gain += h[r] * gain_forget[gb + size_t(r) * (2 * Width) + j];
            forget += h[r] * gain_forget[gb + size_t(r) * (2 * Width) + Width + j];
        }
        const float xv = x[(size_t(row) * Inputs + (Inputs == 1 ? 0u : c)) * Width + j];
        const float mv = message[(size_t(row) * Messages + First + c) * Width + j];
        const float kept = (1.0f - metal::precise::tanh(forget)) * xv;
        const float added = metal::precise::tanh(gain) * mv;
        out[(size_t(c) * rows + row) * Width + j] = kept + added;
    """

/// Fused low-rank gated blends (see the file header).
package enum LowRankGateKernels {
    private static let down = MLXFast.metalKernel(
        name: "low_rank_gate_down", inputNames: ["x", "message", "down"],
        outputNames: ["partial"], source: lowRankDownSource)
    private static let gate = MLXFast.metalKernel(
        name: "low_rank_gate_blend", inputNames: ["x", "message", "partial", "gain_forget"],
        outputNames: ["out"], source: lowRankGateSource)

    /// Rows of `down` per down threadgroup.
    private static let span = 128

    /// Whether ``blend(inputs:inputCount:messages:first:count:down:gainForget:eps:)`` runs:
    /// float32 inputs `[..., inputCount, H]` (or `[..., H]` for one input) and messages
    /// `[..., n, H]` over the same leading rows, float32 `down [C, 2, H, rank]` and
    /// `gainForget [C, rank, 2H]`, `H` a multiple of 64, `rank` a multiple of 32 up to 256, on
    /// the GPU.
    package static func supports(
        inputs x: MLXArray, inputCount: Int, messages: MLXArray, first: Int, count: Int,
        down: MLXArray, gainForget: MLXArray
    ) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu, x.dtype == .float32,
            messages.dtype == .float32, down.dtype == .float32, gainForget.dtype == .float32,
            messages.ndim >= 2, down.ndim == 4, gainForget.ndim == 3, count >= 1, first >= 0,
            inputCount == 1 || inputCount == count
        else { return false }
        let width = messages.dim(-1)
        let rank = down.dim(-1)
        let controllers = down.dim(0)
        let rows = messages.size / (messages.dim(-2) * width)
        return width % 64 == 0 && rank % 32 == 0 && rank <= 256 && rank > 0
            && first + count <= controllers && first + count <= messages.dim(-2)
            && down.shape == [controllers, 2, width, rank]
            && gainForget.shape == [controllers, rank, 2 * width]
            && x.size == rows * inputCount * width && x.dim(-1) == width && rows > 0
    }

    /// Controllers `first ..< first + count` applied to their inputs: `[count, rows, H]` float32,
    /// controller `first + i` blending input `i` (or the one input) with message `first + i`.
    package static func blend(
        inputs x: MLXArray, inputCount: Int, messages: MLXArray, first: Int, count: Int,
        down weights: MLXArray, gainForget: MLXArray, eps: Float
    ) -> MLXArray {
        precondition(
            supports(
                inputs: x, inputCount: inputCount, messages: messages, first: first, count: count,
                down: weights, gainForget: gainForget),
            "unsupported low-rank gate")
        let width = messages.dim(-1)
        let rank = weights.dim(-1)
        let rows = messages.size / (messages.dim(-2) * width)
        let parts = 2 * width / span
        let template: [(String, any KernelTemplateArg)] = [
            ("Width", width), ("Rank", rank), ("Inputs", inputCount),
            ("Messages", messages.dim(-2)), ("First", first), ("Parts", parts),
            ("EpsBits", Int(Int32(bitPattern: eps.bitPattern))),
        ]
        let partial = down(
            [x, messages, weights], template: template,
            grid: (256 * parts, count, rows), threadGroup: (256, 1, 1),
            outputShapes: [[count, rows, parts, rank]], outputDTypes: [.float32])[0]
        return gate(
            [x, messages, partial, gainForget], template: template,
            grid: ((width + 255) / 256 * 256, count, rows), threadGroup: (256, 1, 1),
            outputShapes: [[count, rows, width]], outputDTypes: [.float32])[0]
    }
}
