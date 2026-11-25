// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// Sparse-MLA prefill GEMMs that read each query's selected latent rows in
// place (macprovider #2524).
//
// The op path gathers every query's top-k latent rows into a new
// `[queries, 1, K, R]` array (`take`) and runs MLX's SDPA, which for head dim
// 512 with an array mask is the unfused fallback: `scale * q`, `q @ kᵀ`,
// `where(mask, scores, finfo.min)`, precise softmax, `p @ v`. The gathered
// copy is written to memory and read back by both matmuls; the latent itself
// (T rows of R values) stays cache-resident.
//
// These kernels replace the two matmuls and the gather and leave the scale,
// mask and softmax as the same MLX ops. Each is MLX's steel GEMM arithmetic
// with the B operand's rows indexed through the selection instead of read
// from a gathered copy: bfloat16 operands widened to float, one 8x8 float
// simdgroup accumulator per output fragment, zero-initialised, fed in the
// same K order as `steel_gemm_fused` (an unaligned K tail first, then the
// aligned BK=16 blocks in order, 8 columns per `simdgroup_multiply_accumulate`),
// and rounded to bfloat16 at the store. An output element's value depends only
// on its A row, its B column and that order, not on the tile shape, so the
// results equal the op path's bit for bit.
//
// Not used on NAX GPUs, whose MLX GEMMs accumulate differently; there the op
// path runs unchanged.

/// The threadgroup tile: BM x BN outputs over BK columns of K, 2 x 2
/// simdgroups of 32 x 32 each, as steel's 64x64x16 fused GEMM.
private let glm5NextSparseMLAGemmSource = """
        constexpr int BM = 64, BN = 64, BK = 16;
        constexpr int TM = 4, TN = 4;  // 8x8 fragments per simdgroup
        threadgroup T As[BM * BK];
        threadgroup T Bs[BK * BN];

        const uint lid = thread_index_in_threadgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int n0 = int(threadgroup_position_in_grid.x) * BN;
        const int m0 = int(threadgroup_position_in_grid.y) * BM;
        const int query = int(threadgroup_position_in_grid.z);

        const device T* aq = a + size_t(query) * M_ * KD_;
        const device int* iq = idx + size_t(query) * IDX_S;

        // steel's BaseMMAFrag<float, 8, 8> lane layout: row fm, columns fn, fn + 1.
        const short qid = short(lane / 4);
        const short fm = (qid & 4) + short((lane / 2) % 4);
        const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
        const short sm = short(sg / 2) * 32, sn = short(sg % 2) * 32;

        simdgroup_float8x8 C[TM * TN];
        for (int f = 0; f < TM * TN; ++f) {
            C[f] = simdgroup_float8x8(0);
        }

        const int kLast = (KD_ / BK) * BK;
        const int blocks = KD_ / BK + (kLast < KD_ ? 1 : 0);
        for (int step = 0; step < blocks; ++step) {
            // steel_gemm_fused: the unaligned tail block first, then 0, BK, ...
            const int k0 = kLast < KD_ ? (step == 0 ? kLast : (step - 1) * BK) : step * BK;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            {   // A tile [BM][BK]: 8 consecutive k per thread, zero outside.
                const int r = int(lid) / 2, c = (int(lid) % 2) * 8;
                const int m = m0 + r;
                for (int e = 0; e < 8; ++e) {
                    const int k = k0 + c + e;
                    As[r * BK + c + e] = (m < M_ && k < KD_) ? aq[size_t(m) * KD_ + k] : T(0);
                }
            }
            if (GATHER_N) {
                // B[k][n] = latent[idx[n]][k]: 8 consecutive k of one selected row.
                const int n = int(lid) / 2, c = (int(lid) % 2) * 8;
                const int col = n0 + n;
                const bool ok = col < NN_;
                const device T* row = lat + (ok ? size_t(iq[col]) * LAT_R : 0);
                for (int e = 0; e < 8; ++e) {
                    const int k = k0 + c + e;
                    Bs[(c + e) * BN + n] = (ok && k < KD_) ? row[k] : T(0);
                }
            } else {
                // B[k][n] = latent[idx[k]][n]: 8 consecutive n of one selected row.
                const int kk = int(lid) / 8, c = (int(lid) % 8) * 8;
                const int k = k0 + kk;
                const bool ok = k < KD_;
                const device T* row = lat + (ok ? size_t(iq[k]) * LAT_R : 0);
                for (int e = 0; e < 8; ++e) {
                    const int col = n0 + c + e;
                    Bs[kk * BN + c + e] = (ok && col < NN_) ? row[col] : T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 Af[TM];
                simdgroup_float8x8 Bf[TN];
                for (short i = 0; i < TM; ++i) {
                    const short r = sm + i * 8 + fm;
                    Af[i].thread_elements()[0] = float(As[r * BK + kk + fn]);
                    Af[i].thread_elements()[1] = float(As[r * BK + kk + fn + 1]);
                }
                for (short j = 0; j < TN; ++j) {
                    const short c = sn + j * 8 + fn;
                    Bf[j].thread_elements()[0] = float(Bs[(kk + fm) * BN + c]);
                    Bf[j].thread_elements()[1] = float(Bs[(kk + fm) * BN + c + 1]);
                }
                for (short i = 0; i < TM; ++i) {
                    for (short j = 0; j < TN; ++j) {
                        simdgroup_multiply_accumulate(C[i * TN + j], Af[i], Bf[j], C[i * TN + j]);
                    }
                }
            }
        }

        device T* oq = out + size_t(query) * M_ * NN_;
        for (short i = 0; i < TM; ++i) {
            const int m = m0 + sm + i * 8 + fm;
            if (m >= M_) continue;
            for (short j = 0; j < TN; ++j) {
                const int n = n0 + sn + j * 8 + fn;
                const auto v = C[i * TN + j].thread_elements();
                if (n < NN_) oq[size_t(m) * NN_ + n] = static_cast<T>(v[0]);
                if (n + 1 < NN_) oq[size_t(m) * NN_ + n + 1] = static_cast<T>(v[1]);
            }
        }
    """

private final class GLM5NextSparseMLAKernels: @unchecked Sendable {
    static let shared = GLM5NextSparseMLAKernels()

    let gemm = MLXFast.metalKernel(
        name: "glm5_next_sparse_mla_gemm",
        inputNames: ["a", "lat", "idx"],
        outputNames: ["out"],
        source: glm5NextSparseMLAGemmSource,
        header: "#include <metal_simdgroup>\nusing namespace metal;\n")
}

/// Calls of `glm5NextSparseMLAAttention` in debug builds, so tests can tell
/// the kernel path ran (unsynchronized; release builds never touch it).
nonisolated(unsafe) var glm5NextSparseMLAAttentionCalls = 0

/// Whether the in-place sparse-MLA GEMMs reproduce MLX's matmuls for these
/// operands: bfloat16 or float16 (steel's 16-wide K blocks for the `p @ v`
/// shape), a latent width that is a whole number of steel's widest K block
/// (so the `q @ kᵀ` K order has no tail), at least 16 heads (below that MLX
/// runs `q @ kᵀ` as `gemv_wide`, whose K order differs), and no NAX. A block
/// of one query (MLX's split-K for `p @ v`) is the caller's to keep off them.
func glm5NextSparseMLAKernelsApply(dtype: DType, heads: Int, latentWidth: Int) -> Bool {
    !glm5NextNAXAvailable && (dtype == .bfloat16 || dtype == .float16)
        && heads >= 16 && latentWidth > 0 && latentWidth % 32 == 0
}

/// `a[q] @ B(q)` for every query `q`, where `B(q)` is built from the latent
/// rows `index[q]` names: `[Q, M, KD] x [KD, NN] -> [Q, M, NN]`.
/// `gatherColumns` (`q @ kᵀ`): `B[k][n] = latent[index[q][n]][k]`, `NN` the
/// selection width, `KD` the latent width. Otherwise (`p @ v`):
/// `B[k][n] = latent[index[q][k]][n]`, `KD` the selection width.
/// `index` holds valid absolute rows of `latent` (`[rows, R]`).
private func glm5NextSparseMLAGemm(
    _ a: MLXArray, latent: MLXArray, index: MLXArray, gatherColumns: Bool
) -> MLXArray {
    let (queries, m, kd) = (a.dim(0), a.dim(1), a.dim(2))
    let (latentWidth, selection) = (latent.dim(1), index.dim(1))
    let nn = gatherColumns ? selection : latentWidth
    return GLM5NextSparseMLAKernels.shared.gemm(
        [a, latent, index],
        template: [
            ("T", a.dtype), ("M_", m), ("KD_", kd), ("NN_", nn), ("LAT_R", latentWidth),
            ("IDX_S", selection), ("GATHER_N", gatherColumns),
        ],
        grid: (((nn + 63) / 64) * 128, (m + 63) / 64, queries),
        threadGroup: (128, 1, 1),
        outputShapes: [[queries, m, nn]],
        outputDTypes: [a.dtype])[0]
}

/// Absorbed-MLA attention of `queries` (`[Q, H, R]`, heads on the query
/// axis) over the latent rows `index` (`[Q, K]`, clipped absolute rows of
/// `latent: [rows, R]`) with the selection mask `valid` (`[Q, K]`): the
/// values of MLX's SDPA fallback over the gathered rows, bit for bit, without
/// the gathered copy. Returns `[Q, H, R]`.
func glm5NextSparseMLAAttention(
    queries: MLXArray, latent: MLXArray, index: MLXArray, valid: MLXArray, scale: Float
) -> MLXArray {
    #if DEBUG
        glm5NextSparseMLAAttentionCalls &+= 1
    #endif
    let (q, h) = (queries.dim(0), queries.dim(1))
    let k = index.dim(1)
    // mlx/fast.cpp's fallback: the scale in the query dtype, times the query.
    let scaled = MLXArray(scale).asType(queries.dtype) * queries
    let scores = glm5NextSparseMLAGemm(scaled, latent: latent, index: index, gatherColumns: true)
        .reshaped(q, 1, h, k)
    // `finfo(scores.dtype).min`, as the fallback masks.
    let lowest: Float = scores.dtype == .float16 ? -65504 : -3.3895313892515355e38
    let masked = which(
        valid.reshaped(q, 1, 1, k), scores, MLXArray(lowest).asType(scores.dtype))
    let probs = softmax(masked, axis: -1, precise: true).reshaped(q, h, k)
    return glm5NextSparseMLAGemm(probs, latent: latent, index: index, gatherColumns: false)
}
