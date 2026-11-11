// Copyright © 2026 Apple Inc.
// Accumulation orders replayed from MLX's steel_gemm_fused, steel_gemm_splitk
// and col_reduce_looped kernels, under the repository's MIT license.

import Foundation
import MLX

// One-position decode kernels for DeepSeek V4 attention. Every row is
// computed by its own simdgroups with the same code and reads only its own
// window, pool and selection, so a row's result does not depend on how many
// rows share the dispatch: serial decode (one row) and continuous batching
// (up to eight rows) produce the same bits.
//
// The products are the graph's float32 matmuls computed with the same 8x8
// simdgroup multiply-accumulates in the same k order as MLX's steel GEMM
// (an unaligned last BK tile first, then the aligned tiles; or split-K
// partitions summed in order), so on GPUs that run float32 matmuls on steel
// the kernels reproduce the graph's bits. Bf16 value weights turn any score
// rounding difference into a whole bf16 ulp, which compounds over layers.

/// Largest key list (window plus compressed entries) one decode row may hold.
let deepseekV4DecodeKeyCapacity = 2048

/// The device architecture's class letter, which MLX's GEMM tiling keys on.
private let deepseekV4DeviceClass: Character = {
    GPU.deviceInfo().architecture.last ?? "g"
}()

/// MLX's float32 GEMM k order for one output element of an `M x N` product
/// over `K` (`steel_matmul_axpby`, `GEMM_TPARAM_MACRO`, non-NAX, one batch):
/// `[0, bk, 1, 0]` for one pass, `[1, 16, parts, partitionSize]` for split-K.
func deepseekV4SteelOrder(M: Int, N: Int, K: Int, transposeB: Bool) -> [Int32] {
    let devc = deepseekV4DeviceClass
    let tm16 = (M + 15) / 16
    let tn16 = (N + 15) / 16
    let tk16 = K / 16
    let threshold = devc == "s" || devc == "d" ? 2048 : 1024
    if tm16 * tn16 <= threshold, tk16 >= 8, K >= max(M, N) {
        let tm32 = (M + 31) / 32
        let tn32 = (N + 31) / 32
        let ratio = tk16 / (tm32 * tn32)
        let next = ratio <= 0 ? 0 : (ratio & (ratio - 1)) == 0 ? ratio : 1 << (Int.bitWidth - ratio.leadingZeroBitCount)
        let parts = min(max(2, next), 32)
        return [1, 16, Int32(parts), Int32((tk16 / parts) * 16)]
    }
    var bk = 16
    if devc == "g" || devc == "p" {
        if transposeB { bk = 32 }
    } else if devc == "d", M * N < 1 << 20, !transposeB {
        bk = 32
    }
    return [0, Int32(bk), 1, 0]
}

private let deepseekV4DecodeHeader = """
    #include <metal_simdgroup_matrix>

    // A lane's two elements of an 8x8 simdgroup matrix: row fm, columns fn
    // and fn + 1 (MLX's BaseMMAFrag<T, 8, 8>::get_coord).
    inline short2 dsv4_coord(ushort lane) {
      const short qid = lane / 4;
      const short fm = (qid & 4) + ((lane / 2) % 4);
      const short fn = (qid & 2) * 2 + (lane % 2) * 2;
      return short2(fn, fm);
    }

    // The k ranges of partition `p` in MLX's order: one pass takes an
    // unaligned last BK tile first (zero padded) and then [0, aligned);
    // split-K partition p takes [p * size, (p + 1) * size) and the last one
    // runs on to the 16-padded end.
    inline void dsv4_ranges(
        int mode, int bk, int parts, int psize, int kr, int p,
        thread int& a0, thread int& a1, thread int& b0, thread int& b1) {
      if (mode == 0) {
        const int kl = (kr / bk) * bk;
        a0 = kl;
        a1 = kr > kl ? kl + bk : kl;
        b0 = 0;
        b1 = kl;
      } else {
        a0 = p * psize;
        a1 = (p + 1 == parts) ? ((kr + 15) / 16) * 16 : a0 + psize;
        b0 = 0;
        b1 = 0;
      }
    }

    // Key j of a decode row: window rows as -(storage row + 2), pool rows as
    // the row, invalid as -1. `m` is the row's meta.
    template <typename M, typename P>
    inline int dsv4_row_of(int j, int window, M m, P topk, int kmax, uint b) {
      if (j < window) {
        return j < m[3] ? -(((m[1] + m[2] + j) % m[0]) + 2) : -1;
      }
      const int i = j - window;
      const int idx = m[7] ? int(topk[size_t(b) * kmax + i]) : i;
      return (idx >= 0 && idx < m[5] && idx < m[6]) ? idx : -1;
    }
    """

private func deepseekV4RowSwitch(_ rows: Int, _ assign: (Int) -> String) -> String {
    (0 ..< rows).map { "case \($0): \(assign($0)) break;" }.joined(separator: " ")
}

/// Indexer scores for one query per row, as a sort key: the descending
/// score in the high word and the pool index in the low word, so the
/// selection and its order are the same for any sort and any padding.
/// The score is the graph's `sum_h(maximum(q_h . k_p, 0) * w_h)`: the dot
/// products in the steel GEMM order of `[H, Dk] x [Dk, P]`, the head sum in
/// col_reduce_looped's order (lane y adds heads y, y + 32, ..., then a
/// simd_sum). One threadgroup per eight pool entries of one row, one
/// simdgroup per eight heads.
private func deepseekV4SelectSource(rows: Int) -> String {
    """
    const uint pb = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int pmax = meta[rows_meta * 8];
    auto m = meta + b * 8;
    const int count = m[0];
    threadgroup float dots[HEADS * 8];
    const short2 c = dsv4_coord(lane);
    const device PT* pool = nullptr;
    switch (b) { \(deepseekV4RowSwitch(rows) { "pool = ip\($0);" }) }
    if (int(pb) * 8 < count) {
      const int e0 = int(pb) * 8 + c.x;
      const device PT* k0p = e0 < count ? pool + size_t(e0) * DK : nullptr;
      const device PT* k1p = e0 + 1 < count ? pool + size_t(e0 + 1) * DK : nullptr;
      const device QT* qrow = q + (size_t(b) * HEADS + sg * 8 + c.y) * DK;
      simdgroup_matrix<float, 8, 8> total = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      const int parts = m[1] ? m[3] : 1;
      for (int p = 0; p < parts; ++p) {
        int a0, a1, b0, b1;
        dsv4_ranges(m[1], m[2], m[3], m[4], DK, p, a0, a1, b0, b1);
        simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int r = 0; r < 2; ++r) {
          const int lo = r == 0 ? a0 : b0;
          const int hi = r == 0 ? a1 : b1;
          for (int k = lo; k < hi; k += 8) {
            simdgroup_matrix<float, 8, 8> A;
            simdgroup_matrix<float, 8, 8> B;
            const int ka = k + c.x;
            const int kb = k + c.y;
            A.thread_elements()[0] = ka < DK ? float(qrow[ka]) : 0.0f;
            A.thread_elements()[1] = ka + 1 < DK ? float(qrow[ka + 1]) : 0.0f;
            B.thread_elements()[0] = (k0p && kb < DK) ? float(k0p[kb]) : 0.0f;
            B.thread_elements()[1] = (k1p && kb < DK) ? float(k1p[kb]) : 0.0f;
            simdgroup_multiply_accumulate(acc, A, B, acc);
          }
        }
        if (m[1]) {
          for (int i = 0; i < 2; ++i) {
            total.thread_elements()[i] = total.thread_elements()[i] + acc.thread_elements()[i];
          }
        } else {
          total = acc;
        }
      }
      dots[(sg * 8 + c.y) * 8 + c.x] = total.thread_elements()[0];
      dots[(sg * 8 + c.y) * 8 + c.x + 1] = total.thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = sg; s < 8; s += HEADS / 8) {
      const int e = int(pb) * 8 + int(s);
      if (e >= pmax) { continue; }
      if (e >= count) {
        if (lane == 0) { keys[size_t(b) * pmax + e] = ~0ul; }
        continue;
      }
      float t = 0.0f;
      for (int h = int(lane); h < HEADS; h += 32) {
        float v = dots[h * 8 + int(s)];
        v = metal::isnan(v) ? v : (v > 0.0f ? v : 0.0f);
        t = metal::fma(v, float(wts[size_t(b) * HEADS + h]), 0.0f) + t;
      }
      const float score = simd_sum(t);
      if (lane == 0) {
        const uint bits = as_type<uint>(score);
        const uint ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        keys[size_t(b) * pmax + e] = (ulong(~ordered) << 32) | ulong(e);
      }
    }
    """
}

/// Raw attention scores `q . k` for every valid key of every row, in the
/// steel GEMM order of the graph's `[H, D] x [D, K]`. One simdgroup per
/// eight heads and sixteen keys of one row.
private func deepseekV4ScoresSource(rows: Int) -> String {
    """
    const uint b = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int kmax = meta[rows_meta * 16 + 1];
    const int stride = meta[rows_meta * 16 + 2];
    auto m = meta + b * 16;
    const int K = WINDOW + m[4];
    const int task = int(threadgroup_position_in_grid.x) * 4 + int(sg);
    const int hb = task % (H / 8);
    const int kb = (task / (H / 8)) * 16;
    if (kb >= K) { return; }
    const device KT* win = nullptr;
    const device KT* pool = nullptr;
    switch (b) { \(deepseekV4RowSwitch(rows) { "win = w\($0); pool = p\($0);" }) }
    const short2 c = dsv4_coord(lane);
    const device KT* kp[4];
    for (int i = 0; i < 4; ++i) {
      const int j = kb + (i / 2) * 8 + c.x + (i % 2);
      const int r = j < K ? dsv4_row_of(j, WINDOW, m, topk, kmax, b) : -1;
      kp[i] = r == -1 ? nullptr : (r >= 0 ? pool + size_t(r) * D : win + size_t(-r - 2) * D);
    }
    const device QT* qrow = q + (size_t(b) * H + hb * 8 + c.y) * D;
    simdgroup_matrix<float, 8, 8> total0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> total1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int parts = m[8] ? m[10] : 1;
    for (int p = 0; p < parts; ++p) {
      int a0, a1, b0, b1;
      dsv4_ranges(m[8], m[9], m[10], m[11], D, p, a0, a1, b0, b1);
      simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int r = 0; r < 2; ++r) {
        const int lo = r == 0 ? a0 : b0;
        const int hi = r == 0 ? a1 : b1;
        for (int k = lo; k < hi; k += 8) {
          simdgroup_matrix<float, 8, 8> A;
          simdgroup_matrix<float, 8, 8> B0;
          simdgroup_matrix<float, 8, 8> B1;
          const int ka = k + c.x;
          const int kd = k + c.y;
          A.thread_elements()[0] = ka < D ? float(qrow[ka]) : 0.0f;
          A.thread_elements()[1] = ka + 1 < D ? float(qrow[ka + 1]) : 0.0f;
          const bool in = kd < D;
          B0.thread_elements()[0] = (in && kp[0]) ? float(kp[0][kd]) : 0.0f;
          B0.thread_elements()[1] = (in && kp[1]) ? float(kp[1][kd]) : 0.0f;
          B1.thread_elements()[0] = (in && kp[2]) ? float(kp[2][kd]) : 0.0f;
          B1.thread_elements()[1] = (in && kp[3]) ? float(kp[3][kd]) : 0.0f;
          simdgroup_multiply_accumulate(acc0, A, B0, acc0);
          simdgroup_multiply_accumulate(acc1, A, B1, acc1);
        }
      }
      if (m[8]) {
        for (int i = 0; i < 2; ++i) {
          total0.thread_elements()[i] = total0.thread_elements()[i] + acc0.thread_elements()[i];
          total1.thread_elements()[i] = total1.thread_elements()[i] + acc1.thread_elements()[i];
        }
      } else {
        total0 = acc0;
        total1 = acc1;
      }
    }
    device float* dst = raw + (size_t(b) * H + hb * 8 + c.y) * stride;
    const float result[4] = {
        total0.thread_elements()[0], total0.thread_elements()[1],
        total1.thread_elements()[0], total1.thread_elements()[1]};
    for (int i = 0; i < 4; ++i) {
      const int j = kb + (i / 2) * 8 + c.x + (i % 2);
      if (j < K) { dst[j] = result[i]; }
    }
    """
}

/// `deepseekV4QueryAttention`'s fused weights for each row and head: scaled
/// and masked scores, 64-key block maxima, the running maximum, block
/// weights and sums, the rescale, the normalizer with the sink, and the
/// bf16-rounded rescaled value weights, in the same order and rounding.
private func deepseekV4WeightsSource() -> String {
    """
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint t = thread_position_in_threadgroup.x;
    const float sc = as_type<float>(meta[rows_meta * 16 + 0]);
    const int kmax = meta[rows_meta * 16 + 1];
    const int stride = meta[rows_meta * 16 + 2];
    auto m = meta + b * 16;
    const int K = WINDOW + m[4];
    const int nb = (K + 63) / 64;
    threadgroup float run[64];
    threadgroup float product[64];
    threadgroup float safe_peak;
    auto s = raw + (size_t(b) * H + h) * stride;
    device float* wout = weights + (size_t(b) * H + h) * stride;
    const float ninf = -INFINITY;
    if (t < uint(nb)) {
      float mx = ninf;
      for (int j = int(t) * 64; j < min(K, int(t) * 64 + 64); ++j) {
        const bool ok = dsv4_row_of(j, WINDOW, m, topk, kmax, b) != -1;
        float v = ok ? metal::fma(s[j], sc, 0.0f) : ninf;
        mx = metal::max(mx, v);
      }
      run[t] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      for (int i = 1; i < nb; ++i) { run[i] = metal::max(run[i - 1], run[i]); }
      float p = run[nb - 1];
      safe_peak = p == ninf ? 0.0f : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < uint(nb)) {
      const float r = run[t];
      const float safe = r == ninf ? 0.0f : r;
      const float scale_b = metal::precise::exp(r - safe_peak);
      float total = 0.0f;
      for (int j = int(t) * 64; j < int(t) * 64 + 64; ++j) {
        float w = 0.0f;
        if (j < K) {
          const bool ok = dsv4_row_of(j, WINDOW, m, topk, kmax, b) != -1;
          float v = ok ? metal::fma(s[j], sc, 0.0f) : ninf;
          w = metal::precise::exp(v - safe);
          wout[j] = metal::fma(static_cast<float>(static_cast<bfloat16_t>(w)), scale_b, 0.0f);
        }
        total = w + total;
      }
      product[t] = metal::fma(total, scale_b, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float total = 0.0f;
      for (int i = 0; i < nb; ++i) { total = product[i] + total; }
      normalizer[size_t(b) * H + h] =
          total + metal::precise::exp(static_cast<float>(sink[h]) - safe_peak);
    }
    """
}

/// `(weights x keys) / normalizer` for every row, in the steel GEMM order
/// of the graph's `[H, K] x [K, D]`. One simdgroup per eight heads and
/// sixteen dimensions of one row.
private func deepseekV4ValuesSource(rows: Int) -> String {
    """
    const uint b = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int kmax = meta[rows_meta * 16 + 1];
    const int stride = meta[rows_meta * 16 + 2];
    auto m = meta + b * 16;
    const int K = WINDOW + m[4];
    const int task = int(threadgroup_position_in_grid.x) * 4 + int(sg);
    const int hb = task % (H / 8);
    const int db = (task / (H / 8)) * 16;
    if (db >= D) { return; }
    const device KT* win = nullptr;
    const device KT* pool = nullptr;
    switch (b) { \(deepseekV4RowSwitch(rows) { "win = w\($0); pool = p\($0);" }) }
    const short2 c = dsv4_coord(lane);
    auto wrow = weights + (size_t(b) * H + hb * 8 + c.y) * stride;
    simdgroup_matrix<float, 8, 8> total0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> total1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int parts = m[12] ? m[14] : 1;
    for (int p = 0; p < parts; ++p) {
      int a0, a1, b0, b1;
      dsv4_ranges(m[12], m[13], m[14], m[15], K, p, a0, a1, b0, b1);
      simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int r = 0; r < 2; ++r) {
        const int lo = r == 0 ? a0 : b0;
        const int hi = r == 0 ? a1 : b1;
        for (int k = lo; k < hi; k += 8) {
          simdgroup_matrix<float, 8, 8> A;
          simdgroup_matrix<float, 8, 8> B0;
          simdgroup_matrix<float, 8, 8> B1;
          const int ka = k + c.x;
          A.thread_elements()[0] = ka < K ? wrow[ka] : 0.0f;
          A.thread_elements()[1] = ka + 1 < K ? wrow[ka + 1] : 0.0f;
          const int j = k + c.y;
          const int rr = j < K ? dsv4_row_of(j, WINDOW, m, topk, kmax, b) : -1;
          if (rr == -1) {
            B0.thread_elements()[0] = 0.0f;
            B0.thread_elements()[1] = 0.0f;
            B1.thread_elements()[0] = 0.0f;
            B1.thread_elements()[1] = 0.0f;
          } else {
            auto key = rr >= 0 ? pool + size_t(rr) * D : win + size_t(-rr - 2) * D;
            B0.thread_elements()[0] = float(key[db + c.x]);
            B0.thread_elements()[1] = float(key[db + c.x + 1]);
            B1.thread_elements()[0] = float(key[db + 8 + c.x]);
            B1.thread_elements()[1] = float(key[db + 8 + c.x + 1]);
          }
          simdgroup_multiply_accumulate(acc0, A, B0, acc0);
          simdgroup_multiply_accumulate(acc1, A, B1, acc1);
        }
      }
      if (m[12]) {
        for (int i = 0; i < 2; ++i) {
          total0.thread_elements()[i] = total0.thread_elements()[i] + acc0.thread_elements()[i];
          total1.thread_elements()[i] = total1.thread_elements()[i] + acc1.thread_elements()[i];
        }
      } else {
        total0 = acc0;
        total1 = acc1;
      }
    }
    const float norm = normalizer[size_t(b) * H + hb * 8 + c.y];
    device float* dst = out + (size_t(b) * H + hb * 8 + c.y) * D + db;
    dst[c.x] = metal::precise::divide(total0.thread_elements()[0], norm);
    dst[c.x + 1] = metal::precise::divide(total0.thread_elements()[1], norm);
    dst[8 + c.x] = metal::precise::divide(total1.thread_elements()[0], norm);
    dst[8 + c.x + 1] = metal::precise::divide(total1.thread_elements()[1], norm);
    """
}

private final class DeepseekV4DecodeKernels: @unchecked Sendable {
    static let shared = DeepseekV4DecodeKernels()
    private let lock = NSLock()
    private var cache = [String: MLXFast.MLXFastKernel]()

    private func kernel(
        _ name: String, inputs: [String], outputs: [String], source: () -> String
    ) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = cache[name] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: name, inputNames: inputs, outputNames: outputs, source: source(),
                header: deepseekV4DecodeHeader)
            cache[name] = kernel
            return kernel
        }
    }

    func select(rows: Int) -> MLXFast.MLXFastKernel {
        kernel(
            "deepseek_v4_decode_select_r\(rows)",
            inputs: ["q", "wts", "meta"] + (0 ..< rows).map { "ip\($0)" }, outputs: ["keys"]
        ) { deepseekV4SelectSource(rows: rows) }
    }

    func scores(rows: Int) -> MLXFast.MLXFastKernel {
        kernel(
            "deepseek_v4_decode_scores_r\(rows)",
            inputs: ["q", "meta", "topk"] + (0 ..< rows).map { "w\($0)" } + (0 ..< rows).map { "p\($0)" },
            outputs: ["raw"]
        ) { deepseekV4ScoresSource(rows: rows) }
    }

    func weights() -> MLXFast.MLXFastKernel {
        kernel(
            "deepseek_v4_decode_weights", inputs: ["raw", "sink", "meta", "topk"],
            outputs: ["weights", "normalizer"]
        ) { deepseekV4WeightsSource() }
    }

    func values(rows: Int) -> MLXFast.MLXFastKernel {
        kernel(
            "deepseek_v4_decode_values_r\(rows)",
            inputs: ["weights", "normalizer", "meta", "topk"]
                + (0 ..< rows).map { "w\($0)" } + (0 ..< rows).map { "p\($0)" },
            outputs: ["out"]
        ) { deepseekV4ValuesSource(rows: rows) }
    }
}

/// `[B, kmax]` indexer selections for one query per row: `q` `[B, H, 1, Dk]`,
/// `weights` `[B, 1, H]` (float32, already scaled), `pools[b]` `[1, P_b, Dk]`.
/// Row `b` holds `min(topK, P_b)` pool indices in descending score order
/// (ties in ascending index order) followed by padding. `nil` outside the
/// kernel's configuration.
func deepseekV4DecodeSelect(
    q: MLXArray, weights: MLXArray, pools: [MLXArray], topK: Int
) -> MLXArray? {
    guard let keys = deepseekV4DecodeSelectKeys(q: q, weights: weights, pools: pools) else { return nil }
    let k = min(topK, keys.dim(1))
    return argSort(keys, axis: -1)[0..., ..<k].asType(.int32)
}

/// `[B, Pmax]` uint64 sort keys behind `deepseekV4DecodeSelect`: the high
/// word is the bitwise complement of the score's order-preserving bits.
func deepseekV4DecodeSelectKeys(q: MLXArray, weights: MLXArray, pools: [MLXArray]) -> MLXArray? {
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
    guard pmax > 0, heads % 8 == 0, (8 ... 256).contains(heads), dk % 8 == 0,
        pools.allSatisfy({ $0.dim(2) == dk })
    else { return nil }
    var meta = [Int32](repeating: 0, count: (rows + 1) * 8)
    for (b, count) in counts.enumerated() {
        meta[b * 8] = Int32(count)
        let order = deepseekV4SteelOrder(M: heads, N: max(count, 1), K: dk, transposeB: true)
        for i in 0 ..< 4 { meta[b * 8 + 1 + i] = order[i] }
    }
    meta[rows * 8] = Int32(pmax)
    let placeholder = MLXArray.zeros([1, 1, dk], dtype: dtype)
    let inputs = [contiguous(q.reshaped(rows, heads, dk)),
                  weights.asType(.float32).reshaped(rows, heads),
                  MLXArray(meta).reshaped(rows + 1, 8)]
        + pools.map { $0.dim(1) > 0 ? $0 : placeholder }
    return DeepseekV4DecodeKernels.shared.select(rows: rows)(
        inputs,
        template: [("PT", dtype), ("QT", q.dtype), ("DK", dk), ("HEADS", heads), ("rows_meta", rows)],
        grid: (heads * 4 * ((pmax + 7) / 8), rows, 1), threadGroup: (heads * 4, 1, 1),
        outputShapes: [[rows, pmax]], outputDTypes: [.uint64])[0]
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
/// D]`), or `nil` outside the kernels' configuration. `topk` is `[B, kmax]`
/// pool indices for indexed layers.
func deepseekV4DecodeAttention(
    q: MLXArray, rows: [DeepseekV4DecodeRow], topk: MLXArray?, sink: MLXArray, scale: Float,
    window: Int
) -> MLXArray? {
    let B = rows.count
    let (H, D) = (q.dim(1), q.dim(3))
    guard Device.defaultDevice().deviceType == .gpu, (1 ... 8).contains(B),
        q.ndim == 4, q.dim(0) == B, q.dim(2) == 1, D % 16 == 0, (32 ... 1024).contains(D),
        H % 8 == 0, (8 ... 256).contains(H), sink.size == H, window > 0,
        window <= deepseekV4DecodeKeyCapacity,
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
    let maxK = window + rows.map(\.listed).max()!
    var meta = [Int32](repeating: 0, count: (B + 1) * 16)
    for (b, row) in rows.enumerated() {
        let K = window + row.listed
        let values = [row.window.dim(2), row.start, row.base, row.validWindow, row.listed,
                      row.pool?.dim(1) ?? 0, row.visible, topk == nil ? 0 : 1]
        for (i, value) in values.enumerated() { meta[b * 16 + i] = Int32(value) }
        let scores = deepseekV4SteelOrder(M: H, N: K, K: D, transposeB: true)
        let products = deepseekV4SteelOrder(M: H, N: D, K: K, transposeB: false)
        for i in 0 ..< 4 {
            meta[b * 16 + 8 + i] = scores[i]
            meta[b * 16 + 12 + i] = products[i]
        }
    }
    meta[B * 16] = Int32(bitPattern: scale.bitPattern)
    meta[B * 16 + 1] = Int32(kmax)
    meta[B * 16 + 2] = Int32(maxK)
    let metaArray = MLXArray(meta).reshaped(B + 1, 16)
    let topkArray = topk ?? MLXArray.zeros([1, 8], dtype: .int32)
    let placeholder = MLXArray.zeros([1, 1, D], dtype: dtype)
    let storage = rows.map { $0.window }
        + rows.map { row in row.pool.flatMap { $0.dim(1) > 0 ? $0 : nil } ?? placeholder }
    let kernels = DeepseekV4DecodeKernels.shared
    let template: [(String, any KernelTemplateArg)] = [
        ("QT", q.dtype), ("KT", dtype), ("D", D), ("H", H), ("WINDOW", window), ("rows_meta", B),
    ]
    let q3 = contiguous(q.reshaped(B, H, D))
    let scoreTasks = (H / 8) * ((maxK + 15) / 16)
    let raw = kernels.scores(rows: B)(
        [q3, metaArray, topkArray] + storage, template: template,
        grid: (((scoreTasks + 3) / 4) * 128, B, 1), threadGroup: (128, 1, 1),
        outputShapes: [[B, H, maxK]], outputDTypes: [.float32])[0]
    let fused = kernels.weights()(
        [raw, sink, metaArray, topkArray],
        template: [("H", H), ("WINDOW", window), ("rows_meta", B)],
        grid: (64, H, B), threadGroup: (64, 1, 1),
        outputShapes: [[B, H, maxK], [B, H]], outputDTypes: [.float32, .float32])
    let valueTasks = (H / 8) * (D / 16)
    let out = kernels.values(rows: B)(
        [fused[0], fused[1], metaArray, topkArray] + storage, template: template,
        grid: (((valueTasks + 3) / 4) * 128, B, 1), threadGroup: (128, 1, 1),
        outputShapes: [[B, H, D]], outputDTypes: [.float32])[0]
    return out.reshaped(B, H, 1, D)
}
