// A relative-position band mask over a sliding-window ring read as stored: `[B, H, L, S]`
// additive logits where key column `j` holds the entry at chronological position
// `(j - rotation) mod S`, so attention can read the ring without first copying it into order.
//
// Ported from Ollama's ordered window cache for Inkling 2 (Ollama pull request 19843,
// `mlxrunner/cache/rotating.go` and `mlx/relative_attention.go`; MIT licence, Copyright (c)
// Ollama), which keeps the ring in place and moves the ordering into the mask. The logits per
// (query, chronological key) are mlx-vlm's `_mask_v2_kernel` values unchanged; only the column a
// key lands in differs. Attention then reduces the keys in storage order, so the result is not
// bit-identical to attention over the ordered copy; callers gate it on greedy token parity.

import Foundation
import MLX

private let ringBandMaskSource = """
        const int B  = rel_shape[0];
        const int LQ = rel_shape[1];
        const int H  = rel_shape[2];
        const int S  = kshape_shape[2];
        const int rotation = rot[0];
        uint j  = thread_position_in_grid.x;
        uint i  = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;
        if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
        uint b = bh / H, h = bh % H;
        int chronological = ((int)j - rotation + S) % S;
        int dist = ((int)i + (S - LQ)) - chronological;
        T val;
        if (dist < 0) {
            val = (T)(-1e30f);
        } else if (SLIDING > 0 && dist >= (int)SLIDING) {
            val = (T)(-1e30f);
        } else if (dist < (int)REL_EXTENT) {
            float acc = 0.0f;
            const size_t rbase = (size_t)b * rel_strides[0]
                + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
            for (uint d = 0; d < D_REL; ++d)
                acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                     * (float)proj[(size_t)d * proj_strides[0]
                                   + (size_t)dist * proj_strides[1]];
            val = (T)acc;
        } else {
            val = (T)0;
        }
        out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
    """

/// The ring-aware relative band mask (see the file header).
package enum RingBandMaskKernels {
    private static let kernel = MLXFast.metalKernel(
        name: "ring_band_mask", inputNames: ["rel", "proj", "kshape", "rot"],
        outputNames: ["out"], source: ringBandMaskSource, ensureRowContiguous: false)

    /// `[B, H, L, S]` logits in the dtype of `rel` (`[B, L, H, d_rel]`) for queries that are the
    /// last `L` of the `S` keys of `keys` (`[B, kvHeads, S, D]`), the oldest stored at column
    /// `rotation`. `proj` is `[d_rel, extent]`; keys at a distance of `sliding` or more
    /// (`sliding > 0`) and future keys get -1e30, keys at `extent` or more get 0.
    package static func mask(
        _ rel: MLXArray, _ proj: MLXArray, keys: MLXArray, rotation: Int, sliding: Int,
        extent: Int
    ) -> MLXArray {
        let (batch, length, heads, relativeDim) = (rel.dim(0), rel.dim(1), rel.dim(2), rel.dim(3))
        let size = keys.dim(2)
        precondition(rotation >= 0 && rotation < max(size, 1), "ring rotation out of range")
        func roundUp(_ a: Int) -> Int { (a + 7) / 8 * 8 }
        return kernel(
            [rel, proj, keys, MLXArray([Int32(rotation)])],
            template: [
                ("T", rel.dtype), ("D_REL", relativeDim), ("REL_EXTENT", extent),
                ("SLIDING", sliding),
            ],
            grid: (roundUp(size), roundUp(length), batch * heads), threadGroup: (8, 8, 1),
            outputShapes: [[batch, heads, length, size]], outputDTypes: [rel.dtype])[0]
    }
}
