import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// The grouped MoE prefill must be bit-identical to the per-expert reference
/// loop: CB parity and serial/batched token identity depend on it. Every
/// comparison here is on raw bit patterns, never a tolerance.
final class DeepseekV4GroupedPrefillTests: XCTestCase {
    private var isUltra: Bool {
        Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g15d"
    }

    private struct SplitMix {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func below(_ n: Int) -> Int { Int(next() % UInt64(n)) }
        mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
    }

    /// V4-Flash routed experts (the configuration defaults: hidden 4096,
    /// moe_intermediate 2048, 256 experts, 6 per token) with random MXFP4
    /// codes and scales, written directly so no float expert stack is built.
    private func makeMoE(seed: UInt64) throws -> (DeepseekV4MoE, DeepseekV4Configuration) {
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: Data(#"{"model_type": "deepseek_v4"}"#.utf8))
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.moeIntermediateSize, 2048)
        XCTAssertEqual(config.nRoutedExperts, 256)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        MLXRandom.seed(seed)
        let moe = DeepseekV4MoE(config, layerIdx: config.numHashLayers)
        quantize(model: moe.switchMLP, groupSize: 32, bits: 4, mode: .mxfp4)
        var parameters = [String: MLXArray]()
        let (experts, hidden, inter) = (
            config.nRoutedExperts, config.hiddenSize, config.moeIntermediateSize
        )
        for (name, n, k) in [
            ("gate_proj", inter, hidden), ("up_proj", inter, hidden), ("down_proj", hidden, inter),
        ] {
            let (weight, scales) = randomMXFP4(experts: experts, n: n, k: k)
            parameters["\(name).weight"] = weight
            parameters["\(name).scales"] = scales
        }
        moe.switchMLP.update(parameters: ModuleParameters.unflattened(parameters))
        eval(moe)
        for projection in [
            moe.switchMLP.gateProjection, moe.switchMLP.upProjection, moe.switchMLP.downProjection,
        ] {
            let quantized = try XCTUnwrap(projection as? QuantizedSwitchLinear)
            XCTAssertEqual(quantized.mode, .mxfp4)
            XCTAssertNil(quantized.quantizedParts.biases)
        }
        return (moe, config)
    }

    /// Random MXFP4 codes (every nibble random, distinct per expert and row)
    /// and e8m0 scales of 2^-8 ... 2^-3.
    private func randomMXFP4(experts: Int, n: Int, k: Int) -> (MLXArray, MLXArray) {
        let block = 64
        let bytes = MLXRandom.randInt(0 ..< 256, [experts, n, block]).asType(.uint8)
        let weight = tiled(bytes, repetitions: [1, 1, k / 2 / block]).view(dtype: .uint32)
        let scales = MLXRandom.randInt(119 ..< 125, [experts, n, k / 32]).asType(.uint8)
        eval(weight, scales)
        return (weight, scales)
    }

    /// Routing with every token on `k` distinct experts and a skewed load:
    /// some experts idle, some on exactly 1 ... 31, 33, 64, 200 routes where
    /// the token count allows, the rest drawn from a heavy-tailed load. Slot
    /// order within a token is shuffled so slot order is not expert order.
    private func makeRouting(tokens: Int, experts: Int, k: Int, seed: UInt64) -> [Int32] {
        var rng = SplitMix(state: seed)
        let total = tokens * k
        var perm = Array(0 ..< experts)
        for i in stride(from: experts - 1, to: 0, by: -1) { perm.swapAt(i, rng.below(i + 1)) }
        let idle = 16
        var target = [Int](repeating: 0, count: experts)
        var remaining = total
        var used = 0
        for count in Array(1 ... 31) + [33, 64, 200] where count <= tokens && count <= remaining {
            target[perm[used]] = count
            remaining -= count
            used += 1
        }
        let free = Array(perm[used ..< (experts - idle)])
        let weights = free.map { _ in pow(rng.unit(), 3) + 1e-3 }
        var cumulative = [Double]()
        var sum = 0.0
        for w in weights {
            sum += w
            cumulative.append(sum)
        }
        while remaining > 0 {
            let u = rng.unit() * sum
            let i = cumulative.firstIndex { $0 >= u } ?? (free.count - 1)
            guard target[free[i]] < tokens else { continue }
            target[free[i]] += 1
            remaining -= 1
        }
        var tokenOrder = Array(0 ..< tokens)
        for i in stride(from: tokens - 1, to: 0, by: -1) { tokenOrder.swapAt(i, rng.below(i + 1)) }
        var perToken = [[Int32]](repeating: [], count: tokens)
        var i = 0
        for expert in 0 ..< experts {
            for _ in 0 ..< target[expert] {
                perToken[tokenOrder[i % tokens]].append(Int32(expert))
                i += 1
            }
        }
        var flat = [Int32]()
        for var list in perToken {
            precondition(list.count == k && Set(list).count == k)
            for j in stride(from: k - 1, to: 0, by: -1) { list.swapAt(j, rng.below(j + 1)) }
            flat += list
        }
        return flat
    }

    private func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertEqual(a.dtype, b.dtype)
        let bitType: DType = a.dtype.size == 2 ? .uint16 : .uint32
        let lhs = a.view(dtype: bitType).asType(.uint32).asArray(UInt32.self)
        let rhs = b.view(dtype: bitType).asType(.uint32).asArray(UInt32.self)
        return zip(lhs, rhs).filter { $0 != $1 }.count + abs(lhs.count - rhs.count)
    }

    private func routeSummary(_ routed: [Int32], experts: Int, k: Int, n: Int) -> String {
        var counts = [Int](repeating: 0, count: experts)
        for e in routed { counts[Int(e)] += 1 }
        var histogram = [String: Int]()
        for m in counts where m > 0 {
            let key: String
            switch deepseekV4ReferencePrefillRoute(m: m, k: k, n: n) {
            case .vector: key = "vector"
            case .wide: key = "wide"
            case .splitK(let s): key = "splitK\(s)"
            case .plain: key = "plain"
            }
            histogram[key, default: 0] += 1
        }
        let idle = counts.filter { $0 == 0 }.count
        return "idle=\(idle) " + histogram.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
    }

    /// Grouped prefill vs the per-expert reference loop, bit for bit, at
    /// 9, 64, 1024, 8192 and 32768 (the serving chunk, several batches) tokens, with the gather flag off and on. On the
    /// M3 Ultra this is the serving arithmetic. Elsewhere it runs twice: as
    /// served there (per-expert `quantizedMM`: layout, SwiGLU and combine
    /// order), and with the reference kernels forced on (grouped wide
    /// launches, route runs and gathers against this GPU's own per-expert
    /// kernels).
    func testGroupedPrefillMatchesReferenceBitwise() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let (moe, config) = try makeMoE(seed: 2305)
        let k = config.numExpertsPerTok
        let saved = (deepseekV4PrefillGroupedGatherQMM, deepseekV4ReferencePrefillOnAnyGPU)
        defer {
            deepseekV4PrefillGroupedGatherQMM = saved.0
            deepseekV4ReferencePrefillOnAnyGPU = saved.1
        }
        for forced in isUltra ? [false] : [false, true] {
            deepseekV4ReferencePrefillOnAnyGPU = forced
            try checkGroupedPrefill(moe, config: config, k: k, label: "forced=\(forced)")
        }
    }

    private func checkGroupedPrefill(
        _ moe: DeepseekV4MoE, config: DeepseekV4Configuration, k: Int, label: String
    ) throws {
        for (n, tokens) in [9, 64, 1024, 8192, 32768].enumerated() {
            let routed = makeRouting(
                tokens: tokens, experts: config.nRoutedExperts, k: k, seed: UInt64(17 + n))
            let indices = MLXArray(routed).reshaped(1, tokens, k)
            let weights = MLXRandom.uniform(low: 0.05, high: 1.0, [1, tokens, k])
            let x = MLXRandom.normal([1, tokens, config.hiddenSize]).asType(.bfloat16)
            eval(x, weights)
            print("grouped-prefill tokens=\(tokens) gate/up "
                + routeSummary(routed, experts: config.nRoutedExperts, k: 4096, n: 2048)
                + " | down " + routeSummary(routed, experts: config.nRoutedExperts, k: 2048, n: 4096))
            var started = Date()
            let reference = try XCTUnwrap(
                moe.quantizedPrefillReference(x, indices: indices, weights: weights))
            eval(reference)
            let referenceSeconds = Date().timeIntervalSince(started)
            for gather in [false, true] {
                deepseekV4PrefillGroupedGatherQMM = gather
                started = Date()
                let grouped = try XCTUnwrap(moe.quantizedPrefill(x, indices: indices, weights: weights))
                eval(grouped)
                let groupedSeconds = Date().timeIntervalSince(started)
                let different = mismatches(grouped, reference)
                print("grouped-prefill \(label) tokens=\(tokens) gather=\(gather) ultra=\(isUltra) "
                    + "mismatches=\(different)/\(reference.size) "
                    + String(format: "reference=%.3fs grouped=%.3fs", referenceSeconds, groupedSeconds))
                XCTAssertEqual(different, 0, "\(label) tokens=\(tokens) gather=\(gather)")
            }
        }
    }

    /// One grouped wide launch for several experts of one row count vs the
    /// per-expert wide kernel,
    /// for both V4-Flash projection shapes and every wide row count. Off the
    /// M3 Ultra the kernels are forced on and compared on this GPU.
    func testGroupedWideMatchesPerExpertWide() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let saved = deepseekV4ReferencePrefillOnAnyGPU
        defer { deepseekV4ReferencePrefillOnAnyGPU = saved }
        deepseekV4ReferencePrefillOnAnyGPU = true
        let experts: [Int32] = [0, 3, 117, 255]
        for (n, k) in [(2048, 4096), (4096, 2048)] {
            let (weight, scales) = randomMXFP4(experts: 256, n: n, k: k)
            var m = 2
            while deepseekV4ReferencePrefillRoute(m: m, k: k, n: n) == .wide {
                let x = MLXRandom.normal([experts.count * m, k]).asType(.bfloat16)
                let grouped = deepseekV4ReferencePrefillGroupedWide(
                    x, weight: weight, scales: scales, experts: experts, m: m)
                var parts = [MLXArray]()
                for (i, e) in experts.enumerated() {
                    parts.append(try XCTUnwrap(deepseekV4ReferencePrefillQMM(
                        x[(i * m) ..< ((i + 1) * m)], weight: weight[Int(e)],
                        scales: scales[Int(e)])))
                }
                let different = mismatches(grouped, concatenated(parts, axis: 0))
                XCTAssertEqual(different, 0, "n=\(n) k=\(k) m=\(m)")
                m += 1
            }
            print("grouped-wide n=\(n) k=\(k) checked m=2..<\(m)")
        }
    }

    /// MLX's sorted gather (gather_qmm_rhs) vs one `quantizedMM` per expert
    /// on the same MXFP4 weights: bf16 full-K (the plain route) and float32
    /// K slices padded to 33 rows (the split-K partials). The gather flag may
    /// be turned on only where this holds on the serving GPU; it is asserted
    /// on the M3 Ultra and reported elsewhere.
    func testGatherMatchesPerExpertQuantizedMM() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let rowCounts = [32, 33, 64, 65, 100, 128, 129, 200, 512, 1024, 2048, 4096]
        let architecture = GPU.deviceInfo().architecture
        var failures = [String]()
        for (n, k) in [(2048, 4096), (4096, 2048)] {
            let (weight, scales) = randomMXFP4(experts: rowCounts.count, n: n, k: k)
            let total = rowCounts.reduce(0, +)
            let x = MLXRandom.normal([total, k]).asType(.bfloat16)
            var rowExperts = [Int32]()
            for (e, m) in rowCounts.enumerated() {
                rowExperts += repeatElement(Int32(e), count: m)
            }
            func perExpert(_ input: MLXArray, _ w: MLXArray, _ s: MLXArray) -> MLXArray {
                var start = 0
                var parts = [MLXArray]()
                for (e, m) in rowCounts.enumerated() {
                    var rows = input[start ..< (start + m)]
                    if rows.dtype == .float32, m < 33 {
                        rows = concatenated(
                            [rows, MLXArray.zeros([33 - m, rows.dim(1)], dtype: .float32)], axis: 0)
                    }
                    parts.append(quantizedMM(
                        rows, w[e], scales: s[e], biases: nil, transpose: true, groupSize: 32,
                        bits: 4, mode: .mxfp4)[..<m])
                    start += m
                }
                return concatenated(parts, axis: 0)
            }
            func gathered(_ input: MLXArray, _ w: MLXArray, _ s: MLXArray) -> MLXArray {
                gatherQuantizedMM(
                    input.reshaped(total, 1, input.dim(1)), w, scales: s, biases: nil,
                    rhsIndices: MLXArray(rowExperts).asType(.uint32), transpose: true,
                    groupSize: 32, bits: 4, mode: .mxfp4, sortedIndices: true
                ).reshaped(total, w.dim(1))
            }
            func check(_ label: String, _ lhs: MLXArray, _ rhs: MLXArray) {
                var start = 0
                for m in rowCounts {
                    let different = mismatches(lhs[start ..< (start + m)], rhs[start ..< (start + m)])
                    let line = "gather-vs-qmm arch=\(architecture) \(label) n=\(n) k=\(k) m=\(m) "
                        + "mismatches=\(different)/\(m * n)"
                    print(line)
                    if different != 0 { failures.append(line) }
                    start += m
                }
            }
            check("bf16-fullK", gathered(x, weight, scales), perExpert(x, weight, scales))
            for splits in [2, 4, 8] where k % (splits * 32) == 0 {
                let length = k / splits
                for start in [0, k - length] {
                    let slice = x[0..., start ..< (start + length)].asType(.float32)
                    let w = weight[0..., 0..., (start / 8) ..< ((start + length) / 8)]
                    let s = scales[0..., 0..., (start / 32) ..< ((start + length) / 32)]
                    check("f32-slice\(start / length)of\(splits)",
                        gathered(slice, w, s), perExpert(slice, w, s))
                }
            }
        }
        if isUltra {
            XCTAssertTrue(failures.isEmpty, failures.joined(separator: "\n"))
        } else {
            print("gather-vs-qmm arch=\(architecture) not asserted off the M3 Ultra; "
                + "\(failures.count) row groups differ")
        }
    }

    /// Grouped vs reference for one input, both gather settings, bitwise.
    private func assertMatchesReference(
        _ moe: DeepseekV4MoE, x: MLXArray, indices: MLXArray, weights: MLXArray, label: String
    ) throws {
        let reference = try XCTUnwrap(
            moe.quantizedPrefillReference(x, indices: indices, weights: weights))
        for gather in [false, true] {
            deepseekV4PrefillGroupedGatherQMM = gather
            let grouped = try XCTUnwrap(moe.quantizedPrefill(x, indices: indices, weights: weights))
            let different = mismatches(grouped, reference)
            print("grouped-prefill \(label) gather=\(gather) mismatches=\(different)/\(reference.size)")
            XCTAssertEqual(different, 0, "\(label) gather=\(gather)")
        }
    }

    /// A small tests-only batch budget splits the experts into many batches
    /// (some experts alone, larger than the budget): each token's float32
    /// combine chain must still match the reference bit for bit.
    func testBatchedPrefillMatchesReferenceBitwise() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let (moe, config) = try makeMoE(seed: 2306)
        let k = config.numExpertsPerTok
        let saved = (
            deepseekV4PrefillGroupedGatherQMM, deepseekV4ReferencePrefillOnAnyGPU,
            deepseekV4PrefillBatchRows
        )
        defer {
            deepseekV4PrefillGroupedGatherQMM = saved.0
            deepseekV4ReferencePrefillOnAnyGPU = saved.1
            deepseekV4PrefillBatchRows = saved.2
        }
        deepseekV4ReferencePrefillOnAnyGPU = true
        let tokens = 1024
        let routed = makeRouting(tokens: tokens, experts: config.nRoutedExperts, k: k, seed: 41)
        var counts = [Int](repeating: 0, count: config.nRoutedExperts)
        for e in routed { counts[Int(e)] += 1 }
        let x = MLXRandom.normal([1, tokens, config.hiddenSize]).asType(.bfloat16)
        let weights = MLXRandom.uniform(low: 0.05, high: 1.0, [1, tokens, k])
        eval(x, weights)
        for budget in [1500, 150] {
            deepseekV4PrefillBatchRows = budget
            let batches = deepseekV4PrefillExpertBatches(counts: counts, budget: budget)
            XCTAssertGreaterThanOrEqual(batches.count, 3, "budget=\(budget)")
            XCTAssertEqual(batches.flatMap { $0 }, counts.indices.filter { counts[$0] > 0 })
            print("grouped-prefill budget=\(budget) batches=\(batches.count)")
            try assertMatchesReference(
                moe, x: x, indices: MLXArray(routed).reshaped(1, tokens, k), weights: weights,
                label: "budget=\(budget)")
        }
    }

    /// Packed verify input `[4, 5, H]` (20 tokens over four rows) vs the
    /// reference, bitwise, as served here and with the kernels forced on.
    func testPackedRowsPrefillMatchesReferenceBitwise() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let (moe, config) = try makeMoE(seed: 2307)
        let k = config.numExpertsPerTok
        let saved = (deepseekV4PrefillGroupedGatherQMM, deepseekV4ReferencePrefillOnAnyGPU)
        defer {
            deepseekV4PrefillGroupedGatherQMM = saved.0
            deepseekV4ReferencePrefillOnAnyGPU = saved.1
        }
        let routed = makeRouting(tokens: 20, experts: config.nRoutedExperts, k: k, seed: 43)
        let x = MLXRandom.normal([4, 5, config.hiddenSize]).asType(.bfloat16)
        let weights = MLXRandom.uniform(low: 0.05, high: 1.0, [4, 5, k])
        eval(x, weights)
        for forced in isUltra ? [false] : [false, true] {
            deepseekV4ReferencePrefillOnAnyGPU = forced
            try assertMatchesReference(
                moe, x: x, indices: MLXArray(routed).reshaped(4, 5, k), weights: weights,
                label: "packed[4,5] forced=\(forced)")
        }
    }

    /// Runs built to reach the sorted gathers must take them (non-nil), so
    /// the grouped tests cannot pass through the per-expert fallback alone.
    /// Each gathered row is checked bitwise against the per-expert reference
    /// projection; asserted on the M3 Ultra, reported elsewhere.
    func testGatherPathsTaken() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let saved = deepseekV4ReferencePrefillOnAnyGPU
        defer { deepseekV4ReferencePrefillOnAnyGPU = saved }
        deepseekV4ReferencePrefillOnAnyGPU = true
        let (n, k) = (2048, 4096)
        let (weight, scales) = randomMXFP4(experts: 256, n: n, k: k)
        // .plain: > 256 rows per expert; >= 4 rows per stacked expert overall.
        // .splitK: 40 and 50 rows; experts given out of id order.
        for (experts, counts) in [([200, 7, 31], [300, 400, 500]), ([90, 12], [40, 50])] {
            let total = counts.reduce(0, +)
            let x = MLXRandom.normal([total, k]).asType(.bfloat16)
            let route = deepseekV4ReferencePrefillRoute(m: counts[0], k: k, n: n)
            XCTAssertTrue(counts.allSatisfy {
                deepseekV4ReferencePrefillRoute(m: $0, k: k, n: n) == route
            })
            let gathered: MLXArray?
            switch route {
            case .plain:
                gathered = deepseekV4PrefillGatherPlain(
                    x, weight: weight, scales: scales, experts: experts.map { Int32($0) },
                    counts: counts)
            case .splitK(let splits):
                gathered = deepseekV4PrefillGatherSplitK(
                    x, weight: weight, scales: scales, experts: experts.map { Int32($0) },
                    counts: counts, splits: splits)
            default:
                XCTFail("route \(route) for counts \(counts)")
                continue
            }
            let output = try XCTUnwrap(gathered, "gather declined for \(route)")
            var parts = [MLXArray]()
            var start = 0
            for (e, m) in zip(experts, counts) {
                parts.append(try XCTUnwrap(deepseekV4ReferencePrefillQMM(
                    x[start ..< (start + m)], weight: weight[e], scales: scales[e])))
                start += m
            }
            let different = mismatches(output, concatenated(parts, axis: 0))
            print("gather-taken route=\(route) counts=\(counts) mismatches=\(different)")
            if isUltra { XCTAssertEqual(different, 0, "route=\(route)") }
        }
    }
}
