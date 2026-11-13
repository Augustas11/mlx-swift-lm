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


/// The wide kernel for several experts that each take the same `M` rows, one
/// threadgroup layer per expert. The header only moves the four buffers to
/// the expert's rows, weights and scales; the arithmetic is the per-expert
/// source above, unchanged, so every output row rounds the same.
private let deepseekV4ReferencePrefillGroupedWideSource = """
    const int expert_group = int(threadgroup_position_in_grid.z);
    const size_t expert_id = size_t(experts[expert_group]);
    const auto x_group = x + size_t(expert_group) * M * K;
    const auto w_group = w + expert_id * N * (K / 8);
    const auto s_group = s + expert_id * N * (K / 32);
    const auto y_group = y + size_t(expert_group) * M * N;
    {
    const auto x = x_group;
    const auto w = w_group;
    const auto s = s_group;
    const auto y = y_group;
    """ + deepseekV4ReferencePrefillWideSource + """
    }
    """

private final class DeepseekV4ReferencePrefillGroupedWideKernel: Sendable {
    static let shared = DeepseekV4ReferencePrefillGroupedWideKernel()
    let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_reference_prefill_mxfp4_wide_grouped",
        inputNames: ["x", "w", "s", "experts"], outputNames: ["y"],
        source: deepseekV4ReferencePrefillGroupedWideSource, ensureRowContiguous: true)
}

/// The arithmetic `deepseekV4ReferencePrefillQMM` uses for `m` rows of a
/// K x N projection.
enum DeepseekV4PrefillQMMRoute: Hashable {
    /// One row: `quantizedMM` (qmv).
    case vector
    /// The wide Metal kernel with template `M = m`.
    case wide
    /// Float32 partial `quantizedMM` per K slice, summed in slice order.
    case splitK(Int)
    /// One `quantizedMM` (qmm).
    case plain
}

func deepseekV4ReferencePrefillRoute(m: Int, k: Int, n: Int) -> DeepseekV4PrefillQMMRoute {
    let vectorLimit = k <= 2048 && n <= 2048 ? 32 : (k <= 4096 && n <= 4096 ? 18 : 12)
    if m == 1 { return .vector }
    if m < vectorLimit { return .wide }
    let threadgroups = ((n + 31) / 32) * ((m + 31) / 32)
    var splits = min(max(1, 512 / threadgroups), k / 32)
    while splits > 1 && k % (splits * 32) != 0 { splits -= 1 }
    return splits > 1 ? .splitK(splits) : .plain
}

/// Tests only: run the reference prefill kernels on any GPU so the grouped
/// layout can be checked off the M3 Ultra. Never set in serving code.
nonisolated(unsafe) var deepseekV4ReferencePrefillOnAnyGPU = false

/// Whether `deepseekV4ReferencePrefillQMM` serves a bf16 input of width `k`
/// for an expert of `weight` and `scales` (2-D, or 3-D stacked by expert).
func deepseekV4ReferencePrefillAvailable(
    k: Int, weight: MLXArray, scales: MLXArray, dtype: DType
) -> Bool {
    guard Device.defaultDevice().deviceType == .gpu,
        deepseekV4ReferencePrefillOnAnyGPU || GPU.deviceInfo().architecture == "applegpu_g15d",
        dtype == .bfloat16, weight.dtype == .uint32, scales.dtype == .uint8,
        weight.ndim == scales.ndim, weight.ndim >= 2
    else { return false }
    let n = weight.dim(-2)
    return k > 0 && k % 32 == 0 && n > 0 && k <= Int(Int32.max) && n <= Int(Int32.max) / 16
        && weight.dim(-1) == k / 8 && scales.dim(-2) == n && scales.dim(-1) == k / 32
}

/// Per-expert prefill projection with the frozen MLX 0.32.4 reference's
/// qmv_wide/split-K arithmetic. Call only from prefill; ordinary decode and
/// continuous batching retain their qualified one-vector reduction order.
/// Other architectures and formats return nil to preserve the existing path.
func deepseekV4ReferencePrefillQMM(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray
) -> MLXArray? {
    guard x.ndim == 2, weight.ndim == 2, scales.ndim == 2,
        deepseekV4ReferencePrefillAvailable(
            k: x.dim(1), weight: weight, scales: scales, dtype: x.dtype)
    else { return nil }
    let (m, k, n) = (x.dim(0), x.dim(1), weight.dim(0))
    guard m > 0, m <= Int(Int32.max) else { return nil }
    switch deepseekV4ReferencePrefillRoute(m: m, k: k, n: n) {
    case .vector, .plain:
        return quantizedMM(
            x, weight, scales: scales, biases: nil, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
    case .wide:
        let tiles = (m + 4) / 5
        let vectors = (m + tiles - 1) / tiles
        return DeepseekV4ReferencePrefillWideKernel.shared.kernel(
            [x, weight, scales],
            template: [("T", x.dtype), ("M", m), ("K", k), ("N", n), ("NV", vectors)],
            grid: (tiles * 64, (n + 3) / 4, 1), threadGroup: (64, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
    case .splitK(let splits):
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
}

/// The wide route for `experts.count` experts of `m` rows each, in one
/// launch. `x` is `[experts.count * m, K]` with expert `i`'s rows at
/// `i * m ..< (i + 1) * m`; `weight` and `scales` are stacked by expert.
/// Each row equals `deepseekV4ReferencePrefillQMM` for its expert.
func deepseekV4ReferencePrefillGroupedWide(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray, experts: [Int32], m: Int
) -> MLXArray {
    let (k, n) = (x.dim(1), weight.dim(1))
    precondition(deepseekV4ReferencePrefillRoute(m: m, k: k, n: n) == .wide)
    precondition(x.dim(0) == experts.count * m)
    let tiles = (m + 4) / 5
    let vectors = (m + tiles - 1) / tiles
    return DeepseekV4ReferencePrefillGroupedWideKernel.shared.kernel(
        [x, weight, scales, MLXArray(experts)],
        template: [("T", x.dtype), ("M", m), ("K", k), ("N", n), ("NV", vectors)],
        grid: (tiles * 64, (n + 3) / 4, experts.count), threadGroup: (64, 1, 1),
        outputShapes: [[experts.count * m, n]], outputDTypes: [x.dtype])[0]
}

/// Serve the `.plain` and `.splitK` routes of a whole run of experts with
/// MLX's sorted gather_qmm_rhs instead of one `quantizedMM` per expert (and
/// per K slice). On: the M3 Ultra (applegpu_g15d) run of
/// `DeepseekV4GroupedPrefillTests` showed it bit-identical to per-expert
/// `quantizedMM` at 9, 64, 1K and 8K tokens. Setting it false restores the
/// per-expert products inside the grouped path.
nonisolated(unsafe) var deepseekV4PrefillGroupedGatherQMM = true

/// MLX takes gather_qmm_rhs only for one-row batches, at least 16 rows and
/// at least four rows per stacked expert; anything else would be qmv.
private func deepseekV4GatherTakesRHS(rows: Int, experts: Int) -> Bool {
    rows >= 16 && rows / max(1, experts) >= 4
}

/// `.plain` route for a run of experts, rows grouped by expert (`counts[i]`
/// rows for `experts[i]`, in any expert order). MLX finds each expert's rows
/// by binary search, so rows are reordered to ascending expert id for the
/// gather and restored after; each row's arithmetic does not depend on its
/// position. Nil when MLX would not take the sorted gather_qmm_rhs kernel.
func deepseekV4PrefillGatherPlain(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray, experts: [Int32], counts: [Int]
) -> MLXArray? {
    let (rows, k, n) = (x.dim(0), x.dim(1), weight.dim(1))
    guard deepseekV4GatherTakesRHS(rows: rows, experts: weight.dim(0)) else { return nil }
    var starts = [Int]()
    var offset = 0
    for count in counts {
        starts.append(offset)
        offset += count
    }
    let ascending = experts.indices.sorted { experts[$0] < experts[$1] }
    var rowExperts = [UInt32]()
    var permutation = [Int32]()
    rowExperts.reserveCapacity(rows)
    permutation.reserveCapacity(rows)
    for i in ascending {
        rowExperts.append(contentsOf: repeatElement(UInt32(experts[i]), count: counts[i]))
        permutation.append(contentsOf: (starts[i] ..< (starts[i] + counts[i])).map { Int32($0) })
    }
    let reordered = ascending != Array(experts.indices)
    let input = reordered ? x[MLXArray(permutation)] : x
    let output = gatherQuantizedMM(
        input.reshaped(rows, 1, k), weight, scales: scales, biases: nil,
        rhsIndices: MLXArray(rowExperts), transpose: true, groupSize: 32, bits: 4,
        mode: .mxfp4, sortedIndices: true
    ).reshaped(rows, n)
    guard reordered else { return output }
    var inverse = [Int32](repeating: 0, count: rows)
    for (position, row) in permutation.enumerated() { inverse[Int(row)] = Int32(position) }
    return output[MLXArray(inverse)]
}

/// `.splitK(splits)` route for a run of experts, rows grouped by expert
/// (`counts[i]` rows for `experts[i]`; the run's weights are gathered, so
/// local expert ids ascend with the rows): the float32 partial product of each
/// K slice for every row in one gather, summed in slice order, then cast.
/// Nil when MLX would not take the sorted gather_qmm_rhs kernel.
func deepseekV4PrefillGatherSplitK(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray, experts: [Int32], counts: [Int],
    splits: Int
) -> MLXArray? {
    let (rows, k, n) = (x.dim(0), x.dim(1), weight.dim(1))
    guard splits > 1, k % (splits * 32) == 0,
        deepseekV4GatherTakesRHS(rows: rows, experts: experts.count)
    else { return nil }
    let selected = MLXArray(experts)
    let runWeight = weight[selected]
    let runScales = scales[selected]
    var local = [UInt32]()
    local.reserveCapacity(rows)
    for (i, count) in counts.enumerated() {
        local.append(contentsOf: repeatElement(UInt32(i), count: count))
    }
    let rowExperts = MLXArray(local)
    let length = k / splits
    var total: MLXArray?
    for start in stride(from: 0, to: k, by: length) {
        let part = x[0..., start ..< (start + length)].asType(.float32).reshaped(rows, 1, length)
        let partial = gatherQuantizedMM(
            part, runWeight[0..., 0..., (start / 8) ..< ((start + length) / 8)],
            scales: runScales[0..., 0..., (start / 32) ..< ((start + length) / 32)], biases: nil,
            rhsIndices: rowExperts, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
            sortedIndices: true
        ).reshaped(rows, n)
        if let previous = total {
            total = previous + partial
        } else {
            total = partial
        }
    }
    return total?.asType(x.dtype)
}
