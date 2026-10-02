//
//  Qwen35FusedMoE.swift
//  mlx-swift-lm
//
//  Lab fused small-T MoE path for Qwen3.5/3.6 sparse MoE blocks
//  (MacProvider #1770 follow-up). Opt-in via MLX_LM_QWEN35_FUSED_MOE or
//  `Qwen35FusedMoE.mode`; the stock path is unchanged when off.
//
//  Per MoE layer the stock graph dispatches about 20 kernels at T <= 7 (router
//  qmv, softmax, argpartition, gathers, sum, divide, three gather_qmv, silu
//  product, weighted sum, four shared-expert qmv, sigmoid, products, adds).
//  The fused path dispatches three:
//
//   1. router: 8-bit gate GEMV + softmax + top-k + renormalize, plus the
//      shared-expert gate GEMV + sigmoid. One threadgroup per token.
//   2. gate/up: for every distinct routed expert (inline de-duplication) and
//      for the shared expert, gate and up GEMV for all tokens routed to it,
//      then silu(gate) * up. Weights stream once per expert per dispatch.
//   3. down: per output row tile, walk the distinct experts in ascending id
//      order, down-project each token's activation, weight and accumulate,
//      then add the gated shared expert.
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

public enum Qwen35FusedMoE {
    public enum Mode: String, Sendable {
        case off
        /// Fused router kernel only; routed and shared experts stay stock.
        case router
        /// Router + routed + shared expert in three dispatches.
        case full
    }

    nonisolated(unsafe) public static var mode: Mode = {
        switch ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE"] {
        case "1", "full": return .full
        case "router": return .router
        default: return .off
        }
    }()

    /// Largest token count (batch * width) routed through the fused kernels.
    nonisolated(unsafe) public static var maxTokens: Int = {
        if let raw = ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE_MAX_T"],
            let value = Int(raw), value > 0
        {
            return min(value, 32)
        }
        return 16
    }()

    /// Rows per simdgroup in the gate/up kernel (4 simdgroups per threadgroup).
    nonisolated(unsafe) public static var gateUpRows: Int = 4
    /// Rows per simdgroup and simdgroups per threadgroup in the down kernel.
    nonisolated(unsafe) public static var downRows: Int = 2
    nonisolated(unsafe) public static var downSimdgroups: Int = 4

    /// Kernel generation (1 = first cut, 2 = split router + token-shared gate/up
    /// + token-blocked down).
    nonisolated(unsafe) public static var kernelVersion: Int =
        Int(ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE_V"] ?? "") ?? 2
    /// v2 router rows per simdgroup (8 simdgroups per threadgroup).
    nonisolated(unsafe) public static var routerRows: Int = 1
    /// v2 gate/up tokens per weight pass.
    nonisolated(unsafe) public static var gateUpTokens: Int = 4
    /// v2 down tokens per threadgroup (0 = all tokens in one block).
    nonisolated(unsafe) public static var downTokensPerBlock: Int = 1
    /// Lab attribution: 1 = stop after the router, 2 = stop after gate/up (the
    /// block returns x plus a zero-weighted dependency on the partial result).
    nonisolated(unsafe) public static var labStopAfter: Int = 0

    /// Lab hook: receives every sparse-MoE block input (stock or fused).
    nonisolated(unsafe) public static var inputTap: ((MLXArray) -> Void)?

    /// Count of MoE blocks that took the fused path (lab observability).
    nonisolated(unsafe) public static var fusedCalls: Int = 0

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
        /// v2 router last-threadgroup counter (8 x uint32 so MLX binds it as
        /// device memory; the kernel resets it to zero).
        let counter: MLXArray

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
            self.counter = MLXArray.zeros([8], type: UInt32.self)
            eval(self.counter)
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

    static func resolve(_ block: Qwen35SparseMoeBlock) -> Weights? {
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
        case 0: k = routerKernel(key)
        case 1: k = gateUpKernel(key)
        case 2: k = downKernel(key)
        case 3: k = routerKernelV2(key)
        case 4: k = gateUpKernelV2(key)
        default: k = downKernelV2(key)
        }
        kernels[key] = k
        return k
    }

    private static func routerKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let h = key.hidden
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_GPL = FM_H / 2048;
            constexpr int FM_HW8 = FM_H / 4;
            constexpr int FM_NG = FM_H / 64;
            const uint t = threadgroup_position_in_grid.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            threadgroup float logits[256];
            threadgroup float shlogit[1];
            float xv[FM_GPL][64];
            float xs[FM_GPL];
            for (int j = 0; j < FM_GPL; j++) {
              const device bfloat16_t* xp = fx + t * FM_H + (lane + 32 * j) * 64;
              float s = 0.0f;
              for (int i = 0; i < 64; i++) { float v = float(xp[i]); xv[j][i] = v; s += v; }
              xs[j] = s;
            }
            const int nrows = (sg == 0) ? 9 : 8;
            for (int r = 0; r < nrows; r++) {
              const bool shared = (r == 8);
              const int row = shared ? 0 : int(sg) * 8 + r;
              const device uint32_t* W = shared ? fsw : fgw;
              const device bfloat16_t* S = shared ? fss : fgs;
              const device bfloat16_t* B = shared ? fsb : fgb;
              float acc = 0.0f;
              for (int j = 0; j < FM_GPL; j++) {
                const int g = int(lane) + 32 * j;
                const device uint4* wp = (const device uint4*)(W + row * FM_HW8 + g * 16);
                float d = 0.0f;
                for (int v = 0; v < 4; v++) {
                  uint4 w = wp[v];
                  for (int c = 0; c < 4; c++) {
                    uint word = w[c];
                    for (int b = 0; b < 4; b++) {
                      d += float((word >> (8 * b)) & 0xffu) * xv[j][v * 16 + c * 4 + b];
                    }
                  }
                }
                acc += float(S[row * FM_NG + g]) * d + float(B[row * FM_NG + g]) * xs[j];
              }
              acc = simd_sum(acc);
              if (lane == 0) {
                if (shared) { shlogit[0] = fm_bf(acc); } else { logits[row] = fm_bf(acc); }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg != 0) { return; }
            float p[8];
            float m = -INFINITY;
            for (int i = 0; i < 8; i++) { p[i] = logits[lane * 8 + i]; m = metal::max(m, p[i]); }
            m = simd_max(m);
            float s = 0.0f;
            for (int i = 0; i < 8; i++) { p[i] = metal::fast::exp(p[i] - m); s += p[i]; }
            s = simd_sum(s);
            const float inv = 1.0f / s;
            uint key[8];
            for (int i = 0; i < 8; i++) {
              // bf16 probabilities leave the low 16 bits free for the expert id;
              // ties pick the higher id, like the last-k slice of a stable sort.
              key[i] = (as_type<uint>(fm_bf(p[i] * inv)) | (lane * 8 + i)) + 1u;
            }
            uint sel[8];
            float selp[8];
            float tot = 0.0f;
            for (int k = 0; k < 8; k++) {
              uint best = 0u;
              for (int i = 0; i < 8; i++) { best = metal::max(best, key[i]); }
              best = simd_max(best) - 1u;
              const uint idx = best & 0xffffu;
              for (int i = 0; i < 8; i++) { if (lane * 8 + i == idx) { key[i] = 0u; } }
              sel[k] = idx;
              selp[k] = as_type<float>(best & 0xffff0000u);
            }
            // Stock sums the ascending top-k slice.
            for (int k = 7; k >= 0; k--) { tot += selp[k]; }
            const float denom = fm_bf(tot);
            if (lane == 0) {
              for (int k = 0; k < 8; k++) {
                finds[t * 8 + k] = sel[7 - k];
                fwts[t * 9 + k] = fm_bf(selp[7 - k] / denom);
              }
              fwts[t * 9 + 8] = fm_sigmoid_bf(shlogit[0]);
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_router_h\(h)",
            inputNames: ["fx", "fgw", "fgs", "fgb", "fsw", "fss", "fsb"],
            outputNames: ["finds", "fwts"],
            source: source, header: header)
    }

    private static func gateUpKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, inter, nt, rows) = (key.hidden, key.inter, key.tokens, key.rows)
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_I = \(inter);
            constexpr int FM_T = \(nt);
            constexpr int FM_R = \(rows);
            constexpr int FM_P = FM_T * 8;
            constexpr int FM_GPL = FM_H / 2048;
            constexpr int FM_HW4 = FM_H / 8;
            constexpr int FM_NG = FM_H / 64;
            const uint slot = threadgroup_position_in_grid.y;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int row0 = int(threadgroup_position_in_grid.x) * (4 * FM_R) + int(sg) * FM_R;
            const bool shared = (slot == uint(FM_P));
            uint e = 0u;
            if (!shared) {
              e = finds[slot];
              bool dup = false;
              for (uint i = lane; i < slot; i += 32u) { dup = dup || (finds[i] == e); }
              if (simd_any(dup)) { return; }
            }
            const size_t eoff = size_t(e) * size_t(FM_I) * size_t(FM_HW4);
            const size_t soff = size_t(e) * size_t(FM_I) * size_t(FM_NG);
            const device uint32_t* Wg = shared ? fsgw : (fgw + eoff);
            const device uint32_t* Wu = shared ? fsuw : (fuw + eoff);
            const device bfloat16_t* Sg = shared ? fsgs : (fgs + soff);
            const device bfloat16_t* Bg = shared ? fsgb : (fgb + soff);
            const device bfloat16_t* Su = shared ? fsus : (fus + soff);
            const device bfloat16_t* Bu = shared ? fsub : (fub + soff);
            for (int t = 0; t < FM_T; t++) {
              int out = FM_P + t;
              if (!shared) {
                int k = -1;
                for (int kk = 0; kk < 8; kk++) { if (finds[t * 8 + kk] == e) { k = kk; } }
                if (k < 0) { continue; }
                out = t * 8 + k;
              }
              float xv[FM_GPL][64];
              float xs[FM_GPL];
              for (int j = 0; j < FM_GPL; j++) {
                const device bfloat16_t* xp = fx + t * FM_H + (int(lane) + 32 * j) * 64;
                float s = 0.0f;
                for (int i = 0; i < 64; i++) { float v = float(xp[i]); xv[j][i] = v; s += v; }
                xs[j] = s;
              }
              for (int r = 0; r < FM_R; r++) {
                const int row = row0 + r;
                float dg = 0.0f;
                float du = 0.0f;
                for (int j = 0; j < FM_GPL; j++) {
                  const int g = int(lane) + 32 * j;
                  const device uint4* wg = (const device uint4*)(Wg + row * FM_HW4 + g * 8);
                  const device uint4* wu = (const device uint4*)(Wu + row * FM_HW4 + g * 8);
                  float ag = 0.0f;
                  float au = 0.0f;
                  for (int v = 0; v < 2; v++) {
                    uint4 a = wg[v];
                    uint4 b = wu[v];
                    for (int c = 0; c < 4; c++) {
                      for (int n = 0; n < 8; n++) {
                        const float xval = xv[j][v * 32 + c * 8 + n];
                        ag += float((a[c] >> (4 * n)) & 0xfu) * xval;
                        au += float((b[c] >> (4 * n)) & 0xfu) * xval;
                      }
                    }
                  }
                  dg += float(Sg[row * FM_NG + g]) * ag + float(Bg[row * FM_NG + g]) * xs[j];
                  du += float(Su[row * FM_NG + g]) * au + float(Bu[row * FM_NG + g]) * xs[j];
                }
                dg = simd_sum(dg);
                du = simd_sum(du);
                if (lane == 0) {
                  const float gv = fm_bf(dg);
                  const float uv = fm_bf(du);
                  const float act = fm_bf(fm_bf(gv * fm_sigmoid_bf(gv)) * uv);
                  fh[size_t(out) * FM_I + row] = bfloat16_t(act);
                }
              }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_gateup_h\(h)_i\(inter)_t\(nt)_r\(rows)",
            inputNames: [
                "fx", "finds", "fgw", "fgs", "fgb", "fuw", "fus", "fub",
                "fsgw", "fsgs", "fsgb", "fsuw", "fsus", "fsub",
            ],
            outputNames: ["fh"],
            source: source, header: header)
    }

    private static func downKernel(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, inter, nt, rows, sgs) = (key.hidden, key.inter, key.tokens, key.rows, key.sgs)
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_I = \(inter);
            constexpr int FM_T = \(nt);
            constexpr int FM_R = \(rows);
            constexpr int FM_SGS = \(sgs);
            constexpr int FM_P = FM_T * 8;
            constexpr int FM_IW4 = FM_I / 8;
            constexpr int FM_NGI = FM_I / 64;
            constexpr int FM_LPV = FM_I / 32;
            constexpr int FM_LW = FM_LPV / 8;
            const uint tid = thread_index_in_threadgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            threadgroup atomic_uint emask[8];
            threadgroup uchar kslot[256 * FM_T];
            if (tid < 8u) { atomic_store_explicit(&emask[tid], 0u, memory_order_relaxed); }
            for (uint i = tid; i < uint(256 * FM_T); i += uint(FM_SGS * 32)) { kslot[i] = uchar(255); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < uint(FM_P); i += uint(FM_SGS * 32)) {
              const uint e = finds[i];
              atomic_fetch_or_explicit(&emask[e >> 5], 1u << (e & 31u), memory_order_relaxed);
              kslot[e * FM_T + i / 8u] = uchar(i % 8u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int row0 = int(threadgroup_position_in_grid.x) * (FM_SGS * FM_R) + int(sg) * FM_R;
            const int g = int(lane) * FM_LPV / 64;
            float acc[FM_R][FM_T];
            for (int r = 0; r < FM_R; r++) { for (int t = 0; t < FM_T; t++) { acc[r][t] = 0.0f; } }
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
                for (int t = 0; t < FM_T; t++) {
                  const uint k = uint(kslot[e * FM_T + t]);
                  if (k < 8u) {
                    const device bfloat16_t* hp = fh + size_t(t * 8 + int(k)) * FM_I + lane * FM_LPV;
                    float hv[FM_LPV];
                    float hs = 0.0f;
                    for (int i = 0; i < FM_LPV; i++) { hv[i] = float(hp[i]); hs += hv[i]; }
                    const float wt = fwts[t * 9 + int(k)];
                    for (int r = 0; r < FM_R; r++) {
                      float d = 0.0f;
                      for (int q = 0; q < FM_LW; q++) {
                        for (int n = 0; n < 8; n++) {
                          d += float((wq[r][q] >> (4 * n)) & 0xfu) * hv[q * 8 + n];
                        }
                      }
                      d = sc[r] * d + bi[r] * hs;
                      const float y = fm_bf(simd_sum(d));
                      acc[r][t] += fm_bf(y * wt);
                    }
                  }
                }
              }
            }
            for (int r = 0; r < FM_R; r++) {
              const int row = row0 + r;
              const device uint32_t* wr = fsdw + size_t(row) * FM_IW4 + lane * FM_LW;
              uint wq[FM_LW];
              for (int q = 0; q < FM_LW; q++) { wq[q] = wr[q]; }
              const float sc = float(fsds[row * FM_NGI + g]);
              const float bi = float(fsdb[row * FM_NGI + g]);
              for (int t = 0; t < FM_T; t++) {
                const device bfloat16_t* hp = fh + size_t(FM_P + t) * FM_I + lane * FM_LPV;
                float d = 0.0f;
                float hs = 0.0f;
                for (int q = 0; q < FM_LW; q++) {
                  for (int n = 0; n < 8; n++) {
                    const float hval = float(hp[q * 8 + n]);
                    hs += hval;
                    d += float((wq[q] >> (4 * n)) & 0xfu) * hval;
                  }
                }
                d = sc * d + bi * hs;
                const float y = fm_bf(simd_sum(d));
                const float gated = fm_bf(fwts[t * 9 + 8] * y);
                if (lane == 0) {
                  fy[size_t(t) * FM_H + row] = bfloat16_t(fm_bf(acc[r][t]) + gated);
                }
              }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_down_h\(h)_i\(inter)_t\(nt)_r\(rows)_s\(sgs)",
            inputNames: [
                "fh", "finds", "fwts", "fdw", "fds", "fdb", "fsdw", "fsds", "fsdb",
            ],
            outputNames: ["fy"],
            source: source, header: header)
    }

    // MARK: - v2 kernels

    /// Router v2: the logits GEMV is spread over ceil(257 / (8 * rows)) threadgroups
    /// (each threadgroup does its rows for every token, so weights stream once); the
    /// last threadgroup to finish (device atomic counter, reset by that threadgroup)
    /// runs softmax / top-k / renormalize, one simdgroup per token.
    private static func routerKernelV2(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, nt, rps) = (key.hidden, key.tokens, key.rows)
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_T = \(nt);
            constexpr int FM_RPS = \(rps);
            constexpr int FM_GPL = FM_H / 2048;
            constexpr int FM_HW8 = FM_H / 4;
            constexpr int FM_NG = FM_H / 64;
            const uint tgi = threadgroup_position_in_grid.x;
            const uint ntg = threadgroups_per_grid.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint tid = thread_index_in_threadgroup;
            const int rowb = int(tgi) * (8 * FM_RPS) + int(sg) * FM_RPS;
            for (int t = 0; t < FM_T; t++) {
              if (rowb > 256) { break; }
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
            threadgroup uint lastflag[1];
            threadgroup_barrier(mem_flags::mem_device);
            device atomic_uint* cnt = (device atomic_uint*)(fcnt);
            if (tid == 0) {
              atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
              const uint prev = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
              lastflag[0] = (prev == ntg - 1u) ? 1u : 0u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lastflag[0] == 0u) { return; }
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            for (int t = int(sg); t < FM_T; t += 8) {
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
            }
            if (tid == 0) { atomic_store_explicit(cnt, 0u, memory_order_relaxed); }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_router2_h\(h)_t\(nt)_r\(rps)",
            inputNames: ["fx", "fgw", "fgs", "fgb", "fsw", "fss", "fsb", "fcnt"],
            outputNames: ["finds", "fwts", "flog"],
            source: source, header: header)
    }

    /// Gate/up v2: lanes cover 16 consecutive k per step (4 steps for H = 2048) so
    /// up to TT tokens routed to the same expert share one weight load and dequant.
    private static func gateUpKernelV2(_ key: KernelKey) -> MLXFast.MLXFastKernel {
        let (h, inter, nt, rows) = (key.hidden, key.inter, key.tokens, key.rows)
        let tt = min(nt, key.sgs)  // tokens per weight pass
        let source = """
            constexpr int FM_H = \(h);
            constexpr int FM_I = \(inter);
            constexpr int FM_T = \(nt);
            constexpr int FM_R = \(rows);
            constexpr int FM_TT = \(tt);
            constexpr int FM_P = FM_T * 8;
            constexpr int FM_STEPS = FM_H / 512;
            constexpr int FM_HW4 = FM_H / 8;
            constexpr int FM_NG = FM_H / 64;
            const uint slot = threadgroup_position_in_grid.y;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int row0 = int(threadgroup_position_in_grid.x) * (4 * FM_R) + int(sg) * FM_R;
            const bool shared = (slot == uint(FM_P));
            uint e = 0u;
            if (!shared) {
              e = finds[slot];
              bool dup = false;
              for (uint i = lane; i < slot; i += 32u) { dup = dup || (finds[i] == e); }
              if (simd_any(dup)) { return; }
            }
            int tok[FM_T];
            int outi[FM_T];
            int ntok = 0;
            for (int t = 0; t < FM_T; t++) {
              int k = -1;
              if (shared) { k = 8; }
              else { for (int kk = 0; kk < 8; kk++) { if (finds[t * 8 + kk] == e) { k = kk; } } }
              if (k >= 0) { tok[ntok] = t; outi[ntok] = shared ? (FM_P + t) : (t * 8 + k); ntok++; }
            }
            const size_t eoff = size_t(e) * size_t(FM_I) * size_t(FM_HW4);
            const size_t soff = size_t(e) * size_t(FM_I) * size_t(FM_NG);
            const device uint32_t* Wg = shared ? fsgw : (fgw + eoff);
            const device uint32_t* Wu = shared ? fsuw : (fuw + eoff);
            const device bfloat16_t* Sg = shared ? fsgs : (fgs + soff);
            const device bfloat16_t* Bg = shared ? fsgb : (fgb + soff);
            const device bfloat16_t* Su = shared ? fsus : (fus + soff);
            const device bfloat16_t* Bu = shared ? fsub : (fub + soff);
            for (int p0 = 0; p0 < ntok; p0 += FM_TT) {
              float ag[FM_R][FM_TT];
              float au[FM_R][FM_TT];
              for (int r = 0; r < FM_R; r++) { for (int q = 0; q < FM_TT; q++) { ag[r][q] = 0.0f; au[r][q] = 0.0f; } }
              for (int s = 0; s < FM_STEPS; s++) {
                const int kb = s * 512 + int(lane) * 16;
                const int g = kb / 64;
                float xv[FM_TT][16];
                float xs[FM_TT];
                for (int q = 0; q < FM_TT; q++) {
                  xs[q] = 0.0f;
                  const int t = (p0 + q < ntok) ? tok[p0 + q] : tok[p0];
                  const device bfloat16_t* xp = fx + t * FM_H + kb;
                  for (int i = 0; i < 16; i++) { xv[q][i] = float(xp[i]); xs[q] += xv[q][i]; }
                }
                for (int r = 0; r < FM_R; r++) {
                  const int row = row0 + r;
                  const uint2 wg = *(const device uint2*)(Wg + row * FM_HW4 + kb / 8);
                  const uint2 wu = *(const device uint2*)(Wu + row * FM_HW4 + kb / 8);
                  float dg[FM_TT];
                  float du[FM_TT];
                  for (int q = 0; q < FM_TT; q++) { dg[q] = 0.0f; du[q] = 0.0f; }
                  for (int c = 0; c < 2; c++) {
                    for (int n = 0; n < 8; n++) {
                      const float qg = float((wg[c] >> (4 * n)) & 0xfu);
                      const float qu = float((wu[c] >> (4 * n)) & 0xfu);
                      for (int q = 0; q < FM_TT; q++) {
                        dg[q] += qg * xv[q][c * 8 + n];
                        du[q] += qu * xv[q][c * 8 + n];
                      }
                    }
                  }
                  const float sgv = float(Sg[row * FM_NG + g]);
                  const float bgv = float(Bg[row * FM_NG + g]);
                  const float suv = float(Su[row * FM_NG + g]);
                  const float buv = float(Bu[row * FM_NG + g]);
                  for (int q = 0; q < FM_TT; q++) {
                    ag[r][q] += sgv * dg[q] + bgv * xs[q];
                    au[r][q] += suv * du[q] + buv * xs[q];
                  }
                }
              }
              for (int r = 0; r < FM_R; r++) {
                for (int q = 0; q < FM_TT; q++) {
                  const float gsum = simd_sum(ag[r][q]);
                  const float usum = simd_sum(au[r][q]);
                  if (lane == 0 && p0 + q < ntok) {
                    const float gv = fm_bf(gsum);
                    const float uv = fm_bf(usum);
                    const float act = fm_bf(fm_bf(gv * fm_sigmoid_bf(gv)) * uv);
                    fh[size_t(outi[p0 + q]) * FM_I + row0 + r] = bfloat16_t(act);
                  }
                }
              }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen35_fused_gateup2_h\(h)_i\(inter)_t\(nt)_r\(rows)_tt\(tt)",
            inputNames: [
                "fx", "finds", "fgw", "fgs", "fgb", "fuw", "fus", "fub",
                "fsgw", "fsgs", "fsgb", "fsuw", "fsus", "fsub",
            ],
            outputNames: ["fh"],
            source: source, header: header)
    }

    /// Down v2: threadgroups also split over token blocks of `tpb` tokens, so each
    /// threadgroup walks only its tokens' experts (ascending id) instead of the
    /// union over all tokens.
    private static func downKernelV2(_ key: KernelKey) -> MLXFast.MLXFastKernel {
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

    /// Fused router. Returns (indices [T, 8] uint32, weights [T, 9] float32
    /// holding bf16-rounded values; column 8 is the shared-expert gate).
    static func router(_ w: Weights, _ x: MLXArray, tokens: Int) -> (MLXArray, MLXArray) {
        let k = kernel(
            KernelKey(kind: 0, tokens: 0, hidden: w.hidden, inter: 0, rows: 0, sgs: 0))
        let out = k(
            [x, w.gateW, w.gateS, w.gateB, w.sgateW, w.sgateS, w.sgateB],
            grid: (tokens * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[tokens, 8], [tokens, 9]],
            outputDTypes: [.uint32, .float32])
        return (out[0], out[1])
    }

    /// Lab: run only the fused router on a sparse-MoE block's input `[T, H]`.
    /// Returns nil when `module` is not an eligible Qwen3.5 sparse-MoE block.
    public static func labRouter(_ module: Module, _ x: MLXArray) -> (MLXArray, MLXArray)? {
        guard let block = module as? Qwen35SparseMoeBlock, let w = resolve(block) else {
            return nil
        }
        let tokens = x.size / x.dim(-1)
        let flat = x.reshaped([tokens, w.hidden])
        return kernelVersion >= 2
            ? routerV2(w, flat, tokens: tokens) : router(w, flat, tokens: tokens)
    }

    /// Lab: true when `module` is an eligible Qwen3.5 sparse-MoE block.
    public static func labIsFusable(_ module: Module) -> Bool {
        guard let block = module as? Qwen35SparseMoeBlock else { return false }
        return resolve(block) != nil
    }

    /// v2 router (split logits + last-threadgroup top-k).
    static func routerV2(_ w: Weights, _ x: MLXArray, tokens: Int) -> (MLXArray, MLXArray) {
        let rps = routerRows
        let k = kernel(
            KernelKey(kind: 3, tokens: tokens, hidden: w.hidden, inter: 0, rows: rps, sgs: 8))
        let tgs = (257 + 8 * rps - 1) / (8 * rps)
        let out = k(
            [x, w.gateW, w.gateS, w.gateB, w.sgateW, w.sgateS, w.sgateB, w.counter],
            grid: (tgs * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[tokens, 8], [tokens, 9], [tokens, 257]],
            outputDTypes: [.uint32, .float32, .float32])
        return (out[0], out[1])
    }

    static func forward(_ block: Qwen35SparseMoeBlock, _ x: MLXArray) -> MLXArray? {
        let m = mode
        guard m != .off, x.dtype == .bfloat16, x.ndim >= 2 else { return nil }
        let tokens = x.size / x.dim(-1)
        guard tokens >= 1, tokens <= maxTokens, let w = resolve(block), x.dim(-1) == w.hidden
        else { return nil }
        let flat = x.reshaped([tokens, w.hidden])
        let v2 = kernelVersion >= 2
        let (inds, wts) = v2 ? routerV2(w, flat, tokens: tokens) : router(w, flat, tokens: tokens)
        fusedCalls += 1
        if m == .router {
            let scores = wts[0..., ..<8].asType(.bfloat16)
            let y = block.switchMLP(x, inds.reshaped(Array(x.shape.dropLast()) + [8]))
            let combined = weightedExpertSum(
                y, scores.reshaped(Array(x.shape.dropLast()) + [8]))
            var sharedY = block.sharedExpert(x)
            sharedY = sigmoid(block.sharedExpertGate(x)) * sharedY
            return combined + sharedY
        }
        if labStopAfter == 1 {
            return x + (wts[0, 0] * 0).asType(.bfloat16)
        }
        let slots = tokens * 8 + 1
        let gu = kernel(
            KernelKey(
                kind: v2 ? 4 : 1, tokens: tokens, hidden: w.hidden, inter: w.inter,
                rows: gateUpRows, sgs: v2 ? gateUpTokens : 4))
        let h = gu(
            [
                flat, inds, w.gpW, w.gpS, w.gpB, w.upW, w.upS, w.upB,
                w.shGpW, w.shGpS, w.shGpB, w.shUpW, w.shUpS, w.shUpB,
            ],
            grid: ((w.inter / (4 * gateUpRows)) * 128, slots, 1), threadGroup: (128, 1, 1),
            outputShapes: [[slots - 1 + tokens, w.inter]],
            outputDTypes: [.bfloat16])[0]
        if labStopAfter == 2 {
            return x + h[0, 0] * 0
        }
        let tg = downSimdgroups * 32
        let tpb = (downTokensPerBlock <= 0 || !v2) ? tokens : min(downTokensPerBlock, tokens)
        let blocks = (tokens + tpb - 1) / tpb
        let dk = kernel(
            KernelKey(
                kind: v2 ? 5 : 2, tokens: tokens, hidden: w.hidden, inter: w.inter,
                rows: downRows, sgs: downSimdgroups, tpb: tpb))
        let y = dk(
            [h, inds, wts, w.dnW, w.dnS, w.dnB, w.shDnW, w.shDnS, w.shDnB],
            grid: ((w.hidden / (downSimdgroups * downRows)) * tg, v2 ? blocks : 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[tokens, w.hidden]],
            outputDTypes: [.bfloat16])[0]
        return y.reshaped(x.shape)
    }
}
