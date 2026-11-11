// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// One-position decode kernels for DeepSeek V4 attention. Every row is
// computed by its own threadgroups with the same code and reads only its
// own window, pool and selection, so a row's result does not depend on how
// many rows share the dispatch: serial decode (one row) and continuous
// batching (up to eight rows) produce the same bits.

/// Largest key list (window plus compressed entries) one decode row may hold.
let deepseekV4DecodeKeyCapacity = 2048

private func deepseekV4RowSwitch(_ rows: Int, _ assign: (Int) -> String) -> String {
    (0 ..< rows).map { "case \($0): \(assign($0)) break;" }.joined(separator: " ")
}

/// Indexer scores for one query per row, as a sort key: the descending
/// score `sum_h relu(q_h . k_p) * w_h` in the high word and the pool index in
/// the low word. Keys are unique, so the selection and its order are the
/// same for any sort and any padding. Positions past a row's pool sort last.
private func deepseekV4SelectSource(rows: Int) -> String {
    """
    const uint p = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    const int pmax = meta[rows_meta * 8];
    if (int(p) >= pmax) { return; }
    const int count = meta[b * 8];
    if (int(p) >= count) { keys[size_t(b) * pmax + p] = ~0ul; return; }
    const device PT* pool = nullptr;
    switch (b) { \(deepseekV4RowSwitch(rows) { "pool = ip\($0);" }) }
    const device PT* key = pool + size_t(p) * DK;
    float score = 0.0f;
    for (int h = 0; h < HEADS; ++h) {
      const size_t qrow = (size_t(b) * HEADS + h) * DK;
      float dot = 0.0f;
      for (int d = 0; d < DK; ++d) { dot += float(q[qrow + d]) * float(key[d]); }
      score += metal::max(dot, 0.0f) * float(wts[size_t(b) * HEADS + h]);
    }
    const uint bits = as_type<uint>(score);
    const uint ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    keys[size_t(b) * pmax + p] = (ulong(~ordered) << 32) | ulong(p);
    """
}

/// The reference sparse attention for one query per row: float32 scores
/// over the row's window list then compressed list, 64-key blocks with a
/// running maximum, bf16-rounded block weights rescaled to the final peak,
/// and the float32 sink in the normalizer. One threadgroup per row and head.
private func deepseekV4AttentionSource(rows: Int) -> String {
    """
    const uint h = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int ring = meta[b * 8 + 0];
    const int start = meta[b * 8 + 1];
    const int base = meta[b * 8 + 2];
    const int valid_window = meta[b * 8 + 3];
    const int listed = meta[b * 8 + 4];
    const int pooled = meta[b * 8 + 5];
    const int visible = meta[b * 8 + 6];
    const int has_topk = meta[b * 8 + 7];
    const float scale = as_type<float>(meta[rows_meta * 8 + 0]);
    const int H = meta[rows_meta * 8 + 1];
    const int kmax = meta[rows_meta * 8 + 2];
    const int K = WINDOW + listed;
    const int nb = (K + 63) / 64;
    const device KT* win = nullptr;
    const device KT* pool = nullptr;
    switch (b) { \(deepseekV4RowSwitch(rows) { "win = w\($0); pool = p\($0);" }) }

    threadgroup float qs[D];
    threadgroup float sc[CAP];
    threadgroup int rowof[CAP];
    threadgroup float run[CAP / 64];
    threadgroup float product[CAP / 64];
    threadgroup float shared[2];

    for (uint d = t; d < uint(D); d += 256) { qs[d] = float(q[(size_t(b) * H + h) * D + d]); }
    // rowof: window rows as -(row + 2), pool rows as row, invalid as -1.
    for (int j = int(t); j < K; j += 256) {
      int r = -1;
      if (j < WINDOW) {
        if (j < valid_window) { r = -(((start + base + j) % ring) + 2); }
      } else {
        const int i = j - WINDOW;
        const int idx = has_topk ? int(topk[size_t(b) * kmax + i]) : i;
        if (idx >= 0 && idx < pooled && idx < visible) { r = idx; }
      }
      rowof[j] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j = int(sg); j < K; j += 8) {
      const int r = rowof[j];
      if (r == -1) {
        if (lane == 0) { sc[j] = -INFINITY; }
        continue;
      }
      const device KT* key = r >= 0 ? pool + size_t(r) * D : win + size_t(-r - 2) * D;
      float acc = 0.0f;
      for (int d = int(lane); d < D; d += 32) { acc += qs[d] * float(key[d]); }
      acc = simd_sum(acc);
      if (lane == 0) { sc[j] = acc * scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < uint(nb)) {
      float m = -INFINITY;
      for (int j = int(t) * 64; j < min(K, int(t) * 64 + 64); ++j) { m = metal::max(m, sc[j]); }
      run[t] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      for (int i = 1; i < nb; ++i) { run[i] = metal::max(run[i - 1], run[i]); }
      const float p = run[nb - 1];
      shared[0] = p == -INFINITY ? 0.0f : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float safe_peak = shared[0];
    if (t < uint(nb)) {
      const float r = run[t];
      const float safe = r == -INFINITY ? 0.0f : r;
      const float rescale = metal::precise::exp(r - safe_peak);
      float total = 0.0f;
      for (int j = int(t) * 64; j < min(K, int(t) * 64 + 64); ++j) {
        const float w = metal::precise::exp(sc[j] - safe);
        total += w;
        sc[j] = float(bfloat16_t(w)) * rescale;
      }
      product[t] = total * rescale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float total = 0.0f;
      for (int i = 0; i < nb; ++i) { total += product[i]; }
      shared[1] = total + metal::precise::exp(float(sink[h]) - safe_peak);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float normalizer = shared[1];
    for (uint d = t; d < uint(D); d += 256) {
      float acc = 0.0f;
      for (int j = 0; j < K; ++j) {
        const int r = rowof[j];
        if (r == -1) { continue; }
        const device KT* key = r >= 0 ? pool + size_t(r) * D : win + size_t(-r - 2) * D;
        acc += sc[j] * float(key[d]);
      }
      out[(size_t(b) * H + h) * D + d] = acc / normalizer;
    }
    """
}

private final class DeepseekV4DecodeKernels: @unchecked Sendable {
    static let shared = DeepseekV4DecodeKernels()
    private let lock = NSLock()
    private var select = [Int: MLXFast.MLXFastKernel]()
    private var attention = [Int: MLXFast.MLXFastKernel]()

    func select(rows: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = select[rows] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: "deepseek_v4_decode_select_r\(rows)",
                inputNames: ["q", "wts", "meta"] + (0 ..< rows).map { "ip\($0)" },
                outputNames: ["keys"],
                source: deepseekV4SelectSource(rows: rows))
            select[rows] = kernel
            return kernel
        }
    }

    func attention(rows: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = attention[rows] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: "deepseek_v4_decode_attention_r\(rows)",
                inputNames: ["q", "sink", "meta", "topk"]
                    + (0 ..< rows).map { "w\($0)" } + (0 ..< rows).map { "p\($0)" },
                outputNames: ["out"],
                source: deepseekV4AttentionSource(rows: rows))
            attention[rows] = kernel
            return kernel
        }
    }
}

/// `[B, kmax]` indexer selections for one query per row: `q` `[B, H, 1, Dk]`,
/// `weights` `[B, 1, H]` (already scaled), `pools[b]` `[1, P_b, Dk]`. Row
/// `b` holds `min(topK, P_b)` pool indices in descending score order (ties
/// in ascending index order) followed by padding. `nil` outside the kernel's
/// configuration.
func deepseekV4DecodeSelect(
    q: MLXArray, weights: MLXArray, pools: [MLXArray], topK: Int
) -> MLXArray? {
    let rows = pools.count
    guard Device.defaultDevice().deviceType == .gpu, (1 ... 8).contains(rows),
        q.ndim == 4, q.dim(0) == rows, q.dim(2) == 1, weights.size == rows * q.dim(1),
        let dtype = pools.first?.dtype, pools.allSatisfy({ $0.dtype == dtype && $0.ndim == 3 }),
        [DType.bfloat16, .float16, .float32].contains(dtype),
        [DType.bfloat16, .float16, .float32].contains(q.dtype)
    else { return nil }
    let (heads, dk) = (q.dim(1), q.dim(3))
    let counts = pools.map { $0.dim(1) }
    let pmax = counts.max()!
    guard pmax > 0, pools.allSatisfy({ $0.dim(2) == dk }) else { return nil }
    var meta = [Int32](repeating: 0, count: (rows + 1) * 8)
    for (b, count) in counts.enumerated() { meta[b * 8] = Int32(count) }
    meta[rows * 8] = Int32(pmax)
    let placeholder = MLXArray.zeros([1, 1, dk], dtype: dtype)
    let inputs = [contiguous(q.reshaped(rows, heads, dk)),
                  weights.asType(.float32).reshaped(rows, heads),
                  MLXArray(meta).reshaped(rows + 1, 8)]
        + pools.map { $0.dim(1) > 0 ? $0 : placeholder }
    let keys = DeepseekV4DecodeKernels.shared.select(rows: rows)(
        inputs,
        template: [("PT", dtype), ("DK", dk), ("HEADS", heads), ("rows_meta", rows)],
        grid: (pmax, rows, 1), threadGroup: (min(256, pmax), 1, 1),
        outputShapes: [[rows, pmax]], outputDTypes: [.uint64])[0]
    let k = min(topK, pmax)
    return argSort(keys, axis: -1)[0..., ..<k].asType(.int32)
}

/// One decode row's attention inputs: its window storage, where the window
/// list starts in it, and its compressed list.
struct DeepseekV4DecodeRow {
    /// `[1, 1, S, D]` window storage as the rotating cache returned it.
    let window: MLXArray
    /// Storage index of the window's oldest kept position.
    let start: Int
    /// Temporal index of the first window-list key.
    let base: Int
    /// Valid window-list keys (`min(position + 1, window)`).
    let validWindow: Int
    /// `[1, P, D]` compressed pool, or `nil` without one.
    let pool: MLXArray?
    /// Compressed-list length (`min(topK, P)` with an indexer, else `P`).
    let listed: Int
    /// Pool entries visible to this position.
    let visible: Int
}

/// `[B, H, 1, D]` float32 attention for one query per row (`q` `[B, H, 1,
/// D]`), or `nil` outside the kernel's configuration. `topk` is `[B, kmax]`
/// pool indices for indexed layers.
func deepseekV4DecodeAttention(
    q: MLXArray, rows: [DeepseekV4DecodeRow], topk: MLXArray?, sink: MLXArray, scale: Float,
    window: Int
) -> MLXArray? {
    let B = rows.count
    let (H, D) = (q.dim(1), q.dim(3))
    guard Device.defaultDevice().deviceType == .gpu, (1 ... 8).contains(B),
        q.ndim == 4, q.dim(0) == B, q.dim(2) == 1, D % 32 == 0, (32 ... 1024).contains(D),
        H > 1, sink.size == H, window > 0, window <= deepseekV4DecodeKeyCapacity,
        let dtype = rows.first?.window.dtype,
        [DType.bfloat16, .float16, .float32].contains(dtype),
        [DType.bfloat16, .float16, .float32].contains(q.dtype),
        [DType.bfloat16, .float16, .float32].contains(sink.dtype),
        rows.allSatisfy({ row in
            row.window.dtype == dtype && row.window.ndim == 4 && row.window.dim(3) == D
                && row.window.dim(2) > 0 && (row.pool.map { $0.dtype == dtype && $0.dim(2) == D } ?? true)
                && window + row.listed <= deepseekV4DecodeKeyCapacity
                && row.base >= 0 && row.base + row.validWindow <= row.window.dim(2)
        })
    else { return nil }
    let kmax = topk?.dim(1) ?? 0
    var meta = [Int32](repeating: 0, count: (B + 1) * 8)
    for (b, row) in rows.enumerated() {
        let values = [row.window.dim(2), row.start, row.base, row.validWindow, row.listed,
                      row.pool?.dim(1) ?? 0, row.visible, topk == nil ? 0 : 1]
        for (i, value) in values.enumerated() { meta[b * 8 + i] = Int32(value) }
    }
    meta[B * 8] = Int32(bitPattern: scale.bitPattern)
    meta[B * 8 + 1] = Int32(H)
    meta[B * 8 + 2] = Int32(kmax)
    let placeholder = MLXArray.zeros([1, 1, D], dtype: dtype)
    let inputs = [contiguous(q.reshaped(B, H, D)), sink, MLXArray(meta).reshaped(B + 1, 8),
                  topk ?? MLXArray.zeros([1, 8], dtype: .int32)]
        + rows.map { $0.window }
        + rows.map { row in row.pool.flatMap { $0.dim(1) > 0 ? $0 : nil } ?? placeholder }
    let out = DeepseekV4DecodeKernels.shared.attention(rows: B)(
        inputs,
        template: [("KT", dtype), ("D", D), ("WINDOW", window),
                   ("CAP", deepseekV4DecodeKeyCapacity), ("rows_meta", B)],
        grid: (256 * H, B, 1), threadGroup: (256, 1, 1),
        outputShapes: [[B, H, D]], outputDTypes: [.float32])[0]
    return out.reshaped(B, H, 1, D)
}
