// Fused attention-residual mix: one Metal dispatch for `softmax(rms(s) . q / sqrt(H)) . rms(s)`
// over a few residual sources, optionally followed by the RMS norm of each mixture.
//
// Ported from Ollama's Inkling 2 block residual mixer (Ollama pull requests 19808 and 19843,
// `mlx/block_residual_mix.go`; MIT licence, Copyright (c) Ollama). Ollama keeps the normalised
// sources and the softmax weights in one threadgroup instead of storing and rereading them
// between kernels. This port follows the arithmetic of mlx-vlm's `_attention_residual` (the
// score is the dot product times the scale; float32 throughout) rather than Ollama's graph, and
// takes its own threadgroup layout: one threadgroup per (row, query), four channels per lane.
// Its reduction order is not MLX's, so it is not bit-identical to the op-by-op graph; callers
// gate it on greedy token parity.

import Foundation
import MLX

private let residualMixSource = """
        // One threadgroup per (row, query): Width / 4 lanes, four adjacent channels each.
        constexpr uint Groups = Width / 128;
        const uint row = threadgroup_position_in_grid.z;
        const uint q = threadgroup_position_in_grid.y;
        const uint t = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint first = t * 4;
        const size_t at = size_t(row) * Width + first;
        threadgroup float partial[Sources][Groups];
        threadgroup float reduced[Groups];

        // rms_norm(source, None, eps) of every source, in float32.
        float values[Sources][4];
        for (uint s = 0; s < Sources; ++s) {
            float total = 0.0f;
            for (uint j = 0; j < 4; ++j) {
                float v;
                switch (s) {
                    case 0: v = float(x0[at + j]); break;
                    case 1: v = float(x1[at + j]); break;
                    case 2: v = float(x2[at + j]); break;
                    case 3: v = float(x3[at + j]); break;
                    default: v = float(x4[at + j]); break;
                }
                values[s][j] = v;
                total += v * v;
            }
            float p = simd_sum(total);
            if (lane == 0) partial[s][sg] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float eps = as_type<float>(EpsBits);
        for (uint s = 0; s < Sources; ++s) {
            float total = 0.0f;
            for (uint g = 0; g < Groups; ++g) total += partial[s][g];
            const float inv = metal::precise::rsqrt(total / float(Width) + eps);
            for (uint j = 0; j < 4; ++j) values[s][j] = values[s][j] * inv;
        }
        // Every lane has read the sums of squares before they are overwritten.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores: (key . query) * scale.
        for (uint s = 0; s < Sources; ++s) {
            float dot = 0.0f;
            for (uint j = 0; j < 4; ++j) dot += values[s][j] * float(query[q * Width + first + j]);
            float p = simd_sum(dot);
            if (lane == 0) partial[s][sg] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float scale = as_type<float>(ScaleBits);
        float weights[Sources];
        float peak = -INFINITY;
        for (uint s = 0; s < Sources; ++s) {
            float total = 0.0f;
            for (uint g = 0; g < Groups; ++g) total += partial[s][g];
            weights[s] = total * scale;
            peak = max(peak, weights[s]);
        }
        float denominator = 0.0f;
        for (uint s = 0; s < Sources; ++s) {
            weights[s] = metal::precise::exp(weights[s] - peak);
            denominator += weights[s];
        }
        for (uint s = 0; s < Sources; ++s) weights[s] = weights[s] / denominator;

        // The mixture of the normalised sources.
        float mix[4];
        for (uint j = 0; j < 4; ++j) {
            float value = 0.0f;
            for (uint s = 0; s < Sources; ++s) value += weights[s] * values[s][j];
            mix[j] = value;
        }
        const size_t out = (size_t(row) * Queries + q) * Width + first;
        if (!HasNorm) {
            for (uint j = 0; j < 4; ++j) y[out + j] = OutT(mix[j]);
            return;
        }
        // RMSNorm(weight) of the float32 mixture as MLX computes it, weight * (x * inv),
        // rounded once to OutT.
        float squares = 0.0f;
        for (uint j = 0; j < 4; ++j) squares += mix[j] * mix[j];
        float p = simd_sum(squares);
        if (lane == 0) reduced[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint g = 0; g < Groups; ++g) total += reduced[g];
        const float inv = metal::precise::rsqrt(total / float(Width) + as_type<float>(NormEpsBits));
        for (uint j = 0; j < 4; ++j) {
            y[out + j] = OutT(float(norm[q * Width + first + j]) * (mix[j] * inv));
        }
    """

/// The fused attention-residual mix (see the file header).
package enum ResidualMixKernels {
    /// Sources one dispatch mixes.
    package static let maxSources = 5

    private static let kernel = MLXFast.metalKernel(
        name: "fused_attention_residual_mix",
        inputNames: ["x0", "x1", "x2", "x3", "x4", "query", "norm"],
        outputNames: ["y"],
        source: residualMixSource)

    private static func bits(_ value: Float) -> Int { Int(Int32(bitPattern: value.bitPattern)) }

    /// Whether ``mix(_:query:eps:scale:norm:normEps:outputDType:)`` runs for these inputs: 1 to 5
    /// sources of one `[B, L, H]` shape in bfloat16 or float32, a float32 `[queries, H]` query,
    /// `H` a multiple of 128 and at most 4096, on the GPU.
    package static func supports(_ sources: [MLXArray], query: MLXArray) -> Bool {
        guard (1 ... maxSources).contains(sources.count),
            Device.defaultDevice().deviceType == .gpu
        else { return false }
        let shape = sources[0].shape
        guard shape.count == 3, shape[0] > 0, shape[1] > 0 else { return false }
        let width = shape[2]
        guard width % 128 == 0, width <= 4096,
            sources.allSatisfy({ $0.shape == shape && ($0.dtype == .bfloat16 || $0.dtype == .float32) }),
            query.ndim == 2, query.dim(1) == width, query.dtype == .float32,
            (1 ... 8).contains(query.dim(0))
        else { return false }
        return true
    }

    /// `softmax_n(rms(s_n) . q * scale) . rms(s_n)` for each query: `[B, L, queries, H]`.
    ///
    /// - Parameters:
    ///   - sources: the residual sources in order, as ``supports(_:query:)`` requires.
    ///   - eps: the sources' RMS epsilon.
    ///   - scale: the score scale, `H ** -0.5` as the caller rounds it.
    ///   - norm: optional `[queries, H]` RMS norm weights; each mixture is then normalised with
    ///     its query's weight and `normEps` and rounded once to `outputDType`. Without it the
    ///     result is float32.
    package static func mix(
        _ sources: [MLXArray], query: MLXArray, eps: Float, scale: Float,
        norm: MLXArray? = nil, normEps: Float = 0, outputDType: DType = .float32
    ) -> MLXArray {
        precondition(supports(sources, query: query), "unsupported fused residual mix")
        let shape = sources[0].shape
        let (rows, width, queries) = (shape[0] * shape[1], shape[2], query.dim(0))
        if let norm {
            precondition(norm.shape == [queries, width], "norm weights must be [queries, H]")
        }
        // The kernel binds five sources; unused slots alias the last one and are never read.
        var inputs = [any ScalarOrArray]()
        for i in 0 ..< maxSources { inputs.append(sources[min(i, sources.count - 1)]) }
        inputs.append(query)
        inputs.append(norm ?? query)
        let dtype: DType = norm == nil ? .float32 : outputDType
        return kernel(
            inputs,
            template: [
                ("Width", width), ("Sources", sources.count), ("Queries", queries),
                ("HasNorm", norm != nil), ("EpsBits", bits(eps)), ("ScaleBits", bits(scale)),
                ("NormEpsBits", bits(normEps)), ("OutT", dtype),
            ],
            grid: (width / 4, queries, rows), threadGroup: (width / 4, 1, 1),
            outputShapes: [[shape[0], shape[1], queries, width]], outputDTypes: [dtype])[0]
    }
}
