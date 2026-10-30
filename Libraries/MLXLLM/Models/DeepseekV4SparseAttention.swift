// Copyright © 2026 Apple Inc.

import MLX

/// Canonical float32 attention graph, with bf16-rounded block weights.
func deepseekV4GraphAttention(
    _ q: MLXArray, keys: MLXArray, valid: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray {
    let (B, H, L) = (q.dim(0), q.dim(1), q.dim(2))
    let K = keys.dim(2)
    let block = 64
    let blocks = (K + block - 1) / block
    let keys32 = keys.asType(.float32)
    var scores = matmul(
        q.transposed(0, 2, 1, 3).asType(.float32), keys32.swappedAxes(-1, -2)) * scale
    scores = MLX.where(expandedDimensions(valid, axis: 2), scores, MLXArray(-Float.infinity))
    if blocks * block > K {
        scores = concatenated(
            [scores, MLXArray.full([B, L, H, blocks * block - K], values: MLXArray(-Float.infinity))],
            axis: -1)
    }
    let blocked = scores.reshaped(B, L, H, blocks, block)
    let running = cummax(blocked.max(axis: -1), axis: -1)
    let peak = running[.ellipsis, (blocks - 1)...]
    // Empty leading blocks contribute zero, including an entirely empty list.
    // Finite score maxima retain the reference graph's exact arithmetic.
    let safeRunning = MLX.where(running .== MLXArray(-Float.infinity), MLXArray(Float(0)), running)
    let safePeak = MLX.where(peak .== MLXArray(-Float.infinity), MLXArray(Float(0)), peak)
    let weights = exp(blocked - expandedDimensions(safeRunning, axis: -1))
    let rescale = exp(running - safePeak)
    let normalizer = (weights.sum(axis: -1) * rescale).sum(axis: -1, keepDims: true)
        + exp(sink.asType(.float32).reshaped(1, 1, H, 1) - safePeak)
    let valueWeights = (weights.asType(.bfloat16).asType(.float32)
        * expandedDimensions(rescale, axis: -1))
        .reshaped(B, L, H, blocks * block)[.ellipsis, ..<K]
    return (matmul(valueWeights, keys32) / normalizer).transposed(0, 2, 1, 3)
}

/// Keep each decode query's GEMM geometry identical to ordinary single-row decode.
/// Large prefill and unsupported inputs retain their existing fallback.
func deepseekV4SparseAttention(
    _ q: MLXArray, keys: MLXArray, valid: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu,
        q.ndim == 4, keys.ndim == 4, valid.ndim == 3,
        [DType.float32, .float16, .bfloat16].contains(q.dtype),
        [DType.float32, .float16, .bfloat16].contains(keys.dtype), valid.dtype == .bool
    else { return nil }
    let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let K = keys.dim(2)
    guard (1 ... 8).contains(B), (1 ... 128).contains(H), (1 ... 8).contains(L),
        (32 ... 1024).contains(D), D % 32 == 0, (1 ... 65536).contains(K),
        keys.shape == [B, L, K, D], valid.shape == [B, L, K], sink.size == H
    else { return nil }

    var rows = [MLXArray]()
    for row in 0 ..< B {
        var queries = [MLXArray]()
        for position in 0 ..< L {
            queries.append(deepseekV4GraphAttention(
                contiguous(q[row ..< (row + 1), 0..., position ..< (position + 1), 0...]),
                keys: contiguous(keys[row ..< (row + 1), position ..< (position + 1), 0..., 0...]),
                valid: contiguous(valid[row ..< (row + 1), position ..< (position + 1), 0...]),
                sink: sink, scale: scale))
        }
        rows.append(queries.count == 1 ? queries[0] : concatenated(queries, axis: 2))
    }
    return rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
}
