//
//  Qwen35FusedMoE.swift
//  mlx-swift-lm
//
//  Fused small-T MoE path for Qwen3.5/3.6 sparse MoE blocks.
//  Default-on for its exact qualified layout. MLX_LM_QWEN35_FUSED_MOE=0 is
//  the process-level kill switch; the stock path is unchanged when disabled.
//
//  Per MoE layer the stock graph dispatches about 24 kernels at T <= 7 (router
//  qmv, softmax, argpartition, gathers, sum, divide, three gather_qmv, silu
//  product, weighted sum, four shared-expert qmv, sigmoid, products, adds).
//  The fused path (kernel version 3) dispatches four:
//
//   1. router logits: 8-bit gate GEMV for the 256 router rows and the
//      shared-expert gate row, spread over 33 threadgroups.
//   2. router top-k: softmax + top-k + renormalize + sigmoid(shared gate), one
//      simdgroup per token.
//   3. gate/up: for every distinct routed expert (inline de-duplication) and
//      for the shared expert, gate and up GEMV with the weight tile held in
//      registers for all tokens routed to it, then silu(gate) * up.
//   4. down: per output row tile and token, walk the token's experts in
//      ascending id, down-project, weight and accumulate, then add the gated
//      shared expert.
//
//  No kernel keeps device state across threadgroups or calls: every scratch
//  buffer is a fresh output of the call, so calls may overlap on any streams.
//
//  Rounding mirrors the stock bf16 graph at each op boundary (logits, softmax
//  probabilities, scores, gate/up, silu product, expert outputs, weighted
//  products, shared gate). Reductions inside a row use a fixed order, and the
//  expert sum order is ascending expert id, so a token's output does not depend
//  on which other tokens share the call (batch invariant).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen35FusedMoE {
    static let enabled: Bool = {
        switch ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE"] {
        case "0", "false", "FALSE", "no", "NO", "off", "OFF":
            return false
        default:
            return true
        }
    }()

    // M3 Ultra, A3B (v3 kernels): fused wins per layer at T = 1...7 and loses
    // from T = 8, where stock switches to its sorted gather path.
    private static let maximumFusedTokens = 7
    private static let gateUpRows = 1
    private static let gateUpSimdgroups = 4
    private static let gateUpTokensPerPass = 2
    private static let downRows = 2
    private static let downSimdgroups = 4
    private static let downTokensPerBlock = 1
    private static let routerRows = 1

    // MARK: - Weights

    final class Weights {
        let gateW: MLXArray, gateS: MLXArray, gateB: MLXArray
        let sgateW: MLXArray, sgateS: MLXArray, sgateB: MLXArray
        let upW: MLXArray, upS: MLXArray, upB: MLXArray
        let gpW: MLXArray, gpS: MLXArray, gpB: MLXArray
        let dnW: MLXArray, dnS: MLXArray, dnB: MLXArray
        let shUpW: MLXArray, shUpS: MLXArray, shUpB: MLXArray
        let shGpW: MLXArray, shGpS: MLXArray, shGpB: MLXArray
        let shDnW: MLXArray, shDnS: MLXArray, shDnB: MLXArray
        let hidden: Int
        let inter: Int

        init(
            gate: (MLXArray, MLXArray, MLXArray), sgate: (MLXArray, MLXArray, MLXArray),
            up: (MLXArray, MLXArray, MLXArray), gp: (MLXArray, MLXArray, MLXArray),
            dn: (MLXArray, MLXArray, MLXArray), shUp: (MLXArray, MLXArray, MLXArray),
            shGp: (MLXArray, MLXArray, MLXArray), shDn: (MLXArray, MLXArray, MLXArray),
            hidden: Int, inter: Int
        ) {
            (gateW, gateS, gateB) = gate
            (sgateW, sgateS, sgateB) = sgate
            (upW, upS, upB) = up
            (gpW, gpS, gpB) = gp
            (dnW, dnS, dnB) = dn
            (shUpW, shUpS, shUpB) = shUp
            (shGpW, shGpS, shGpB) = shGp
            (shDnW, shDnS, shDnB) = shDn
            self.hidden = hidden
            self.inter = inter
        }
    }

    /// Per-block cache. A plain class (not a Module) so module reflection
    /// ignores it.
    final class Cache {
        var resolved = false
        var weights: Weights?
    }

    private static func q(_ layer: Linear, bits: Int) -> (MLXArray, MLXArray, MLXArray)? {
        guard let ql = layer as? QuantizedLinear, ql.bits == bits, ql.groupSize == 64,
            ql.mode == .affine, let b = ql.biases, ql.bias == nil, ql.scales.dtype == .bfloat16
        else { return nil }
        return (ql.weight, ql.scales, b)
    }

    private static func q(_ layer: SwitchLinear) -> (MLXArray, MLXArray, MLXArray)? {
        guard let ql = layer as? QuantizedSwitchLinear, ql.bits == 4, ql.groupSize == 64,
            ql.mode == .affine
        else { return nil }
        let parts = ql.quantizedParts
        guard let b = parts.biases, parts.bias == nil, parts.scales.dtype == .bfloat16 else {
            return nil
        }
        return (parts.weight, parts.scales, b)
    }

    private static let resolveLock = NSLock()

    static func resolve(_ block: Qwen35SparseMoeBlock) -> Weights? {
        resolveLock.lock()
        defer { resolveLock.unlock() }
        let cache = block.fusedCache
        if cache.resolved { return cache.weights }
        cache.resolved = true
        guard block.numExperts == 256, block.topK == 8, block.normTopkProb,
            let gate = q(block.gate, bits: 8), let sgate = q(block.sharedExpertGate, bits: 8),
            let up = q(block.switchMLP.upProjection), let gp = q(block.switchMLP.gateProjection),
            let dn = q(block.switchMLP.downProjection),
            let shUp = q(block.sharedExpert.upProj, bits: 4),
            let shGp = q(block.sharedExpert.gateProj, bits: 4),
            let shDn = q(block.sharedExpert.downProj, bits: 4)
        else { return nil }
        // up/gate packed [E, I, H/8]; down packed [E, H, I/8].
        let inter = up.0.dim(-2)
        let hidden = up.0.dim(-1) * 8
        guard hidden % 2048 == 0, [256, 512, 1024, 2048].contains(inter),
            inter % (4 * gateUpRows) == 0, hidden % (downRows * downSimdgroups) == 0,
            gate.0.dim(-1) * 4 == hidden, dn.0.dim(-2) == hidden,
            shUp.0.dim(0) == inter, shDn.0.dim(0) == hidden, sgate.0.dim(0) == 1
        else { return nil }
        cache.weights = Weights(
            gate: gate, sgate: sgate, up: up, gp: gp, dn: dn, shUp: shUp, shGp: shGp,
            shDn: shDn, hidden: hidden, inter: inter)
        return cache.weights
    }

    // MARK: - Kernels

    private struct KernelKey: Hashable {
        let kind: Int
        let tokens: Int
        let hidden: Int
        let inter: Int
        let rows: Int
        let sgs: Int
        var tpb: Int = 0
        var extra: Int = 0
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [KernelKey: MLXFast.MLXFastKernel] = [:]

    private static let header = """
        inline float fm_bf(float v) { return float(bfloat16_t(v)); }
        inline float fm_sigmoid_bf(float x) {
          // MLX Sigmoid on bf16: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y.
          float e = fm_bf(metal::exp(metal::abs(x)));
          float y = fm_bf(1.0f / fm_bf(1.0f + e));
          return x < 0.0f ? y : fm_bf(1.0f - y);
        }
        """

    private static func kernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = kernels[key] { return k }
        let k: MLXFast.MLXFastKernel
        switch key.kind {
        case 0: k = routerLogitsKernel(key)
        case 1: k = routerTopKKernel(key)
        case 2: k = gateUpKernel(key)
        default: k = downKernel(key)
        }
        kernels[key] = k
        return k
    }

    // MARK: - Router

    /// Router logits: the 8-bit 257-row GEMV (256 router rows + the shared-expert
    /// gate row) over ceil(257 / (8 * rows)) threadgroups; each threadgroup does its
    /// rows for every token, so the weights stream once. Writes `flog [T, 257]`
    /// (bf16-rounded, stored as float). No cross-threadgroup state: the top-k runs
    /// in its own dispatch, so concurrent or overlapping calls are independent.
    private static func routerLogitsKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, nt, rps) = (key.hidden, key.tokens, key.rows)
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_T = \(nt);
            constexpr int FM_RPS = \(rps);
            constexpr int FM_GPL = FM_H / 2048;
            constexpr int FM_HW8 = FM_H / 4;
            constexpr int FM_NG = FM_H / 64;
            const uint tgi = threadgroup_position_in_grid.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int rowb = int(tgi) * (8 * FM_RPS) + int(sg) * FM_RPS;
            if (rowb > 256) { return; }
            for (int t = 0; t < FM_T; t++) {
              float xv[FM_GPL][64];
              float xs[FM_GPL];
              for (int j = 0; j < FM_GPL; j++) {
                const device bfloat16_t* xp = fx + t * FM_H + (int(lane) + 32 * j) * 64;
                float s = 0.0f;
                for (int i = 0; i < 64; i++) { float v = float(xp[i]); xv[j][i] = v; s += v; }
                xs[j] = s;
              }
              for (int r = 0; r < FM_RPS; r++) {
                const int row = rowb + r;
                if (row > 256) { break; }
                const bool shared = (row == 256);
                const int wr = shared ? 0 : row;
                const device uint32_t* W = shared ? fsw : fgw;
                const device bfloat16_t* S = shared ? fss : fgs;
                const device bfloat16_t* B = shared ? fsb : fgb;
                float acc = 0.0f;
                for (int j = 0; j < FM_GPL; j++) {
                  const int g = int(lane) + 32 * j;
                  const device uint4* wp = (const device uint4*)(W + wr * FM_HW8 + g * 16);
                  float d = 0.0f;
                  for (int v = 0; v < 4; v++) {
                    uint4 w = wp[v];
                    for (int c = 0; c < 4; c++) {
                      for (int b = 0; b < 4; b++) {
                        d += float((w[c] >> (8 * b)) & 0xffu) * xv[j][v * 16 + c * 4 + b];
                      }
                    }
                  }
                  acc += float(S[wr * FM_NG + g]) * d + float(B[wr * FM_NG + g]) * xs[j];
                }
                acc = simd_sum(acc);
                if (lane == 0) { flog[t * 257 + row] = fm_bf(acc); }
              }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_rlogits_h\(h)_t\(nt)_r\(rps)",
            inputNames: ["fx", "fgw", "fgs", "fgb", "fsw", "fss", "fsb"],
            outputNames: ["flog"],
            source: source, header: header)
    }

    /// Router top-k: one simdgroup (threadgroup) per token. Softmax over the bf16
    /// logits, top-8 by `(bf16 prob bits | expert id)` keys (the stock last-k
    /// stable-sort tie rule), bf16 renormalization, sigmoid(shared gate).
    private static func routerTopKKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let source = """
            const uint t = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const device float* lg = flog + t * 257;
            float p[8];
            float m = -INFINITY;
            for (int i = 0; i < 8; i++) { p[i] = lg[lane * 8 + i]; m = metal::max(m, p[i]); }
            m = simd_max(m);
            float s = 0.0f;
            for (int i = 0; i < 8; i++) { p[i] = metal::fast::exp(p[i] - m); s += p[i]; }
            s = simd_sum(s);
            const float inv = 1.0f / s;
            uint key[8];
            for (int i = 0; i < 8; i++) {
              key[i] = (as_type<uint>(fm_bf(p[i] * inv)) | (lane * 8 + i)) + 1u;
            }
            uint sel[8];
            float selp[8];
            for (int k = 0; k < 8; k++) {
              uint best = 0u;
              for (int i = 0; i < 8; i++) { best = metal::max(best, key[i]); }
              best = simd_max(best) - 1u;
              const uint idx = best & 0xffffu;
              for (int i = 0; i < 8; i++) { if (lane * 8 + i == idx) { key[i] = 0u; } }
              sel[k] = idx;
              selp[k] = as_type<float>(best & 0xffff0000u);
            }
            float tot = 0.0f;
            for (int k = 7; k >= 0; k--) { tot += selp[k]; }
            const float denom = fm_bf(tot);
            if (lane == 0) {
              for (int k = 0; k < 8; k++) {
                finds[t * 8 + k] = sel[7 - k];
                fwts[t * 9 + k] = fm_bf(selp[7 - k] / denom);
              }
              fwts[t * 9 + 8] = fm_sigmoid_bf(lg[256]);
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_rtopk",
            inputNames: ["flog"],
            outputNames: ["finds", "fwts"],
            source: source, header: header)
    }

    /// Gate/up: grid `(I / (SG * R) row tiles) x (S + T*8 slots)`. The first
    /// `S = ceil(T / TT)` slot columns are the shared expert, `TT` tokens each, so
    /// its longer work is dispatched first and split; the remaining columns are
    /// the routed slots with inline de-duplication. Lane `l` reads 16-byte words
    /// `l + 32 v` of a row (each instruction covers 512 contiguous bytes), the
    /// `R x 2` row tile stays in registers, and up to `TT` tokens routed to the
    /// expert are processed together so each weight word is dequantized once per
    /// pass. A token's arithmetic does not depend on the tokens it is grouped
    /// with (batch invariant).
    private static func gateUpKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, inter, nt, rows, sgs, tt, stage) = (
            key.hidden, key.inter, key.tokens, key.rows, key.sgs, key.tpb, key.extra
        )
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_I = \(inter);
            constexpr int FM_T = \(nt);
            constexpr int FM_R = \(rows);
            constexpr int FM_SG = \(sgs);
            constexpr int FM_TT = \(tt);
            constexpr bool FM_STAGE = \(stage != 0 ? "true" : "false");
            constexpr int FM_P = FM_T * 8;
            constexpr int FM_S = (FM_T + FM_TT - 1) / FM_TT;
            constexpr int FM_NV = FM_H / 1024;
            constexpr int FM_HW4 = FM_H / 8;
            constexpr int FM_NG = FM_H / 64;
            constexpr int FM_XV = FM_H / 8;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint tid = thread_index_in_threadgroup;
            const uint col = threadgroup_position_in_grid.y;
            const bool shared = col < uint(FM_S);
            uint e = 0u;
            int ntok = 0;
            int tok[FM_T];
            int outi[FM_T];
            if (shared) {
              for (int t = int(col) * FM_TT; t < metal::min(FM_T, int(col + 1) * FM_TT); t++) {
                tok[ntok] = t;
                outi[ntok] = FM_P + t;
                ntok++;
              }
            } else {
              const int slot = int(col) - FM_S;
              e = finds[slot];
              int firstp = -1;
              for (int j = 0; j < (FM_P + 31) / 32; j++) {
                const int p = j * 32 + int(lane);
                const bool hit = (p < FM_P) && (finds[p] == e);
                uint bits = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)) & 0xffffffffull);
                while (bits != 0u) {
                  const int pp = j * 32 + int(ctz(bits));
                  bits &= bits - 1u;
                  if (firstp < 0) { firstp = pp; }
                  tok[ntok] = pp / 8;
                  outi[ntok] = pp;
                  ntok++;
                }
              }
              if (firstp != slot) { return; }
            }
            const int row0 = int(threadgroup_position_in_grid.x) * (FM_SG * FM_R) + int(sg) * FM_R;
            const size_t eoff = size_t(e) * size_t(FM_I) * size_t(FM_HW4);
            const size_t soff = size_t(e) * size_t(FM_I) * size_t(FM_NG);
            const device uint32_t* Wg = shared ? fsgw : (fgw + eoff);
            const device uint32_t* Wu = shared ? fsuw : (fuw + eoff);
            const device bfloat16_t* Sg = shared ? fsgs : (fgs + soff);
            const device bfloat16_t* Bg = shared ? fsgb : (fgb + soff);
            const device bfloat16_t* Su = shared ? fsus : (fus + soff);
            const device bfloat16_t* Bu = shared ? fsub : (fub + soff);
            uint4 wg[FM_R][FM_NV];
            uint4 wu[FM_R][FM_NV];
            float sgv[FM_R][FM_NV];
            float bgv[FM_R][FM_NV];
            float suv[FM_R][FM_NV];
            float buv[FM_R][FM_NV];
            for (int r = 0; r < FM_R; r++) {
              const int row = row0 + r;
              const device uint4* pg = (const device uint4*)(Wg + row * FM_HW4);
              const device uint4* pu = (const device uint4*)(Wu + row * FM_HW4);
              for (int v = 0; v < FM_NV; v++) {
                const int u = int(lane) + 32 * v;
                wg[r][v] = pg[u];
                wu[r][v] = pu[u];
                const int gi = row * FM_NG + u / 2;
                sgv[r][v] = float(Sg[gi]);
                bgv[r][v] = float(Bg[gi]);
                suv[r][v] = float(Su[gi]);
                buv[r][v] = float(Bu[gi]);
              }
            }
            threadgroup uint4 xst[FM_STAGE ? FM_TT * FM_XV : 1];
            for (int c0 = 0; c0 < ntok; c0 += FM_TT) {
              const int nc = metal::min(FM_TT, ntok - c0);
              if (FM_STAGE) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = int(tid); i < nc * FM_XV; i += FM_SG * 32) {
                  const int q = i / FM_XV;
                  xst[i] = ((const device uint4*)(fx + size_t(tok[c0 + q]) * FM_H))[i - q * FM_XV];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }
              float ag[FM_R][FM_TT];
              float au[FM_R][FM_TT];
              for (int r = 0; r < FM_R; r++) {
                for (int q = 0; q < FM_TT; q++) { ag[r][q] = 0.0f; au[r][q] = 0.0f; }
              }
              for (int v = 0; v < FM_NV; v++) {
                const int u = int(lane) + 32 * v;
                float dg[FM_R][FM_TT];
                float du[FM_R][FM_TT];
                float xs[FM_TT];
                for (int q = 0; q < FM_TT; q++) {
                  xs[q] = 0.0f;
                  for (int r = 0; r < FM_R; r++) { dg[r][q] = 0.0f; du[r][q] = 0.0f; }
                }
                for (int c = 0; c < 4; c++) {
                  float xv[FM_TT][8];
                  for (int q = 0; q < FM_TT; q++) {
                    const int qq = q < nc ? q : 0;
                    const uint4 xr = FM_STAGE
                      ? xst[qq * FM_XV + u * 4 + c]
                      : ((const device uint4*)(fx + size_t(tok[c0 + qq]) * FM_H))[u * 4 + c];
                    for (int i = 0; i < 4; i++) {
                      xv[q][2 * i] = as_type<float>(xr[i] << 16);
                      xv[q][2 * i + 1] = as_type<float>(xr[i] & 0xffff0000u);
                    }
                    for (int i = 0; i < 8; i++) { xs[q] += xv[q][i]; }
                  }
                  for (int r = 0; r < FM_R; r++) {
                    const uint a = wg[r][v][c];
                    const uint b = wu[r][v][c];
                    for (int n = 0; n < 8; n++) {
                      const float qa = float((a >> (4 * n)) & 0xfu);
                      const float qb = float((b >> (4 * n)) & 0xfu);
                      for (int q = 0; q < FM_TT; q++) {
                        dg[r][q] += qa * xv[q][n];
                        du[r][q] += qb * xv[q][n];
                      }
                    }
                  }
                }
                for (int r = 0; r < FM_R; r++) {
                  for (int q = 0; q < FM_TT; q++) {
                    ag[r][q] += sgv[r][v] * dg[r][q] + bgv[r][v] * xs[q];
                    au[r][q] += suv[r][v] * du[r][q] + buv[r][v] * xs[q];
                  }
                }
              }
              for (int r = 0; r < FM_R; r++) {
                for (int q = 0; q < FM_TT; q++) {
                  const float gsum = simd_sum(ag[r][q]);
                  const float usum = simd_sum(au[r][q]);
                  if (q < nc && lane == uint(r * FM_TT + q) % 32u) {
                    const float gv = fm_bf(gsum);
                    const float uv = fm_bf(usum);
                    const float act = fm_bf(fm_bf(gv * fm_sigmoid_bf(gv)) * uv);
                    fh[size_t(outi[c0 + q]) * FM_I + row0 + r] = bfloat16_t(act);
                  }
                }
              }
            }
            """
        return MLXFast.metalKernel(
            name:
                "qwen35_fused_gateup3_h\(h)_i\(inter)_t\(nt)_r\(rows)_s\(sgs)_tt\(tt)_x\(stage)",
            inputNames: [
                "fx", "finds", "fgw", "fgs", "fgb", "fuw", "fus", "fub",
                "fsgw", "fsgs", "fsgb", "fsuw", "fsus", "fsub",
            ],
            outputNames: ["fh"],
            source: source, header: header)
    }

    /// Down: threadgroups split over token blocks of `tpb` tokens, so each
    /// threadgroup walks only its tokens' experts (ascending id) instead of the
    /// union over all tokens.
    private static func downKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, inter, nt, rows, sgs, tpb) = (
            key.hidden, key.inter, key.tokens, key.rows, key.sgs, key.tpb
        )
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_I = \(inter);
            constexpr int FM_R = \(rows);
            constexpr int FM_SGS = \(sgs);
            constexpr int FM_TPB = \(tpb);
            constexpr int FM_P = \(nt) * 8;
            constexpr int FM_IW4 = FM_I / 8;
            constexpr int FM_NGI = FM_I / 64;
            constexpr int FM_LPV = FM_I / 32;
            constexpr int FM_LW = FM_LPV / 8;
            const uint tid = thread_index_in_threadgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int t0 = int(threadgroup_position_in_grid.y) * FM_TPB;
            threadgroup atomic_uint emask[8];
            threadgroup uchar kslot[256 * FM_TPB];
            if (tid < 8u) { atomic_store_explicit(&emask[tid], 0u, memory_order_relaxed); }
            for (uint i = tid; i < uint(256 * FM_TPB); i += uint(FM_SGS * 32)) { kslot[i] = uchar(255); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(FM_TPB * 8)) {
              const uint pi = uint(t0) * 8u + tid;
              if (pi < uint(FM_P)) {
                const uint e = finds[pi];
                atomic_fetch_or_explicit(&emask[e >> 5], 1u << (e & 31u), memory_order_relaxed);
                kslot[e * FM_TPB + tid / 8u] = uchar(tid % 8u);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int row0 = int(threadgroup_position_in_grid.x) * (FM_SGS * FM_R) + int(sg) * FM_R;
            const int g = int(lane) * FM_LPV / 64;
            float acc[FM_R][FM_TPB];
            for (int r = 0; r < FM_R; r++) { for (int q = 0; q < FM_TPB; q++) { acc[r][q] = 0.0f; } }
            for (int w = 0; w < 8; w++) {
              uint bits = atomic_load_explicit(&emask[w], memory_order_relaxed);
              while (bits != 0u) {
                const uint e = uint(w) * 32u + ctz(bits);
                bits &= bits - 1u;
                uint wq[FM_R][FM_LW];
                float sc[FM_R];
                float bi[FM_R];
                for (int r = 0; r < FM_R; r++) {
                  const size_t rowi = size_t(e) * FM_H + size_t(row0 + r);
                  const device uint32_t* wr = fdw + rowi * FM_IW4 + lane * FM_LW;
                  for (int q = 0; q < FM_LW; q++) { wq[r][q] = wr[q]; }
                  sc[r] = float(fds[rowi * FM_NGI + g]);
                  bi[r] = float(fdb[rowi * FM_NGI + g]);
                }
                for (int q = 0; q < FM_TPB; q++) {
                  const uint k = uint(kslot[e * FM_TPB + q]);
                  if (k < 8u) {
                    const int t = t0 + q;
                    const device bfloat16_t* hp = fh + size_t(t * 8 + int(k)) * FM_I + lane * FM_LPV;
                    float hv[FM_LPV];
                    float hs = 0.0f;
                    for (int i = 0; i < FM_LPV; i++) { hv[i] = float(hp[i]); hs += hv[i]; }
                    const float wt = fwts[t * 9 + int(k)];
                    for (int r = 0; r < FM_R; r++) {
                      float d = 0.0f;
                      for (int c = 0; c < FM_LW; c++) {
                        for (int n = 0; n < 8; n++) {
                          d += float((wq[r][c] >> (4 * n)) & 0xfu) * hv[c * 8 + n];
                        }
                      }
                      d = sc[r] * d + bi[r] * hs;
                      const float y = fm_bf(simd_sum(d));
                      acc[r][q] += fm_bf(y * wt);
                    }
                  }
                }
              }
            }
            for (int r = 0; r < FM_R; r++) {
              const int row = row0 + r;
              const device uint32_t* wr = fsdw + size_t(row) * FM_IW4 + lane * FM_LW;
              uint wq[FM_LW];
              for (int c = 0; c < FM_LW; c++) { wq[c] = wr[c]; }
              const float sc = float(fsds[row * FM_NGI + g]);
              const float bi = float(fsdb[row * FM_NGI + g]);
              for (int q = 0; q < FM_TPB; q++) {
                const int t = t0 + q;
                if (t * 8 >= FM_P) { break; }
                const device bfloat16_t* hp = fh + size_t(FM_P + t) * FM_I + lane * FM_LPV;
                float d = 0.0f;
                float hs = 0.0f;
                for (int c = 0; c < FM_LW; c++) {
                  for (int n = 0; n < 8; n++) {
                    const float hval = float(hp[c * 8 + n]);
                    hs += hval;
                    d += float((wq[c] >> (4 * n)) & 0xfu) * hval;
                  }
                }
                d = sc * d + bi * hs;
                const float y = fm_bf(simd_sum(d));
                const float gated = fm_bf(fwts[t * 9 + 8] * y);
                if (lane == 0) {
                  fy[size_t(t) * FM_H + row] = bfloat16_t(fm_bf(acc[r][q]) + gated);
                }
              }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_down2_h\(h)_i\(inter)_t\(nt)_r\(rows)_s\(sgs)_b\(tpb)",
            inputNames: [
                "fh", "finds", "fwts", "fdw", "fds", "fdb", "fsdw", "fsds", "fsdb",
            ],
            outputNames: ["fy"],
            source: source, header: header)
    }

    // MARK: - Forward

    /// True when `module` is an eligible Qwen3.5 sparse-MoE block.
    static func isFusable(_ module: Module) -> Bool {
        guard let block = module as? Qwen35SparseMoeBlock else { return false }
        return resolve(block) != nil
    }

    /// Router as two dispatches: logits, then top-k. Returns (indices [T, 8]
    /// uint32, weights [T, 9] float32 holding bf16-rounded values; column 8 is the
    /// shared-expert gate). All scratch is per-call output, so any number of calls
    /// may be in flight on any streams.
    static func routerSplit(_ w: Weights, _ x: MLXArray, tokens: Int) -> (MLXArray, MLXArray) {
        let rps = routerRows
        let lk = kernel(
            KernelKey(kind: 0, tokens: tokens, hidden: w.hidden, inter: 0, rows: rps, sgs: 8))
        let tgs = (257 + 8 * rps - 1) / (8 * rps)
        let logits = lk(
            [x, w.gateW, w.gateS, w.gateB, w.sgateW, w.sgateS, w.sgateB],
            grid: (tgs * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[tokens, 257]],
            outputDTypes: [.float32])[0]
        let tk = kernel(KernelKey(kind: 1, tokens: 0, hidden: 0, inter: 0, rows: 0, sgs: 0))
        let out = tk(
            [logits],
            grid: (tokens * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[tokens, 8], [tokens, 9]],
            outputDTypes: [.uint32, .float32])
        return (out[0], out[1])
    }

    static func forward(_ block: Qwen35SparseMoeBlock, _ x: MLXArray) -> MLXArray? {
        guard x.dtype == .bfloat16, x.ndim >= 2 else { return nil }
        let tokens = x.size / x.dim(-1)
        guard tokens >= 1, tokens <= maximumFusedTokens, let w = resolve(block), x.dim(-1) == w.hidden
        else { return nil }
        let (guR, guSG) = (gateUpRows, gateUpSimdgroups)
        let guTT = min(gateUpTokensPerPass, tokens)
        guard w.inter % (guR * guSG) == 0 else { return nil }
        let flat = x.reshaped([tokens, w.hidden])
        let (inds, wts) = routerSplit(w, flat, tokens: tokens)
        let slots = tokens * 8 + 1
        let sharedColumns = (tokens + guTT - 1) / guTT
        let gateUpInputs = [
            flat, inds, w.gpW, w.gpS, w.gpB, w.upW, w.upS, w.upB,
            w.shGpW, w.shGpS, w.shGpB, w.shUpW, w.shUpS, w.shUpB,
        ]
        let gu = kernel(
            KernelKey(
                kind: 2, tokens: tokens, hidden: w.hidden, inter: w.inter, rows: guR,
                sgs: guSG, tpb: guTT, extra: 0))
        let h = gu(
            gateUpInputs,
            grid: ((w.inter / (guSG * guR)) * guSG * 32, sharedColumns + slots - 1, 1),
            threadGroup: (guSG * 32, 1, 1),
            outputShapes: [[slots - 1 + tokens, w.inter]],
            outputDTypes: [.bfloat16])[0]
        let tg = downSimdgroups * 32
        let tpb = min(downTokensPerBlock, tokens)
        let blocks = (tokens + tpb - 1) / tpb
        let dk = kernel(
            KernelKey(
                kind: 3, tokens: tokens, hidden: w.hidden, inter: w.inter,
                rows: downRows, sgs: downSimdgroups, tpb: tpb))
        let y = dk(
            [h, inds, wts, w.dnW, w.dnS, w.dnB, w.shDnW, w.shDnS, w.shDnB],
            grid: ((w.hidden / (downSimdgroups * downRows)) * tg, blocks, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[tokens, w.hidden]],
            outputDTypes: [.bfloat16])[0]
        return y.reshaped(x.shape)
    }
}
