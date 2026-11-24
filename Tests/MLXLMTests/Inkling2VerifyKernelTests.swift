import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The one-dispatch speculative verify of Inkling 2 Mini at its real widths (hidden 2304, 24
/// query and 8 key heads of 128, 128 experts): every verify row must equal its one-row decode
/// step bit for bit. The kernels MLX picks depend on these shapes, so the toy-width model test
/// (``Inkling2SpeculativeRoundTests``) does not cover them. GPU only: set `INKLING2_GPU_TESTS=1`.
final class Inkling2VerifyKernelTests: XCTestCase {

    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_GPU_TESTS"] == "1" else {
            throw XCTSkip("INKLING2_GPU_TESTS is not set")
        }
    }

    private func equal(_ a: MLXArray, _ b: MLXArray, _ note: String) {
        XCTAssertEqual(a.shape, b.shape, note)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), note)
    }

    /// The projection spans of a verify: `in_proj` (local, 8600 rows), `q_proj` (global, 6552
    /// rows) and the output projection, whole and in the row ranges attention slices, over a
    /// strided row batch, against each row projected alone.
    func testRowBatchProjectionsMatchOneRowProductsAtMiniShapes() throws {
        try requireGPU()
        try Device.withDefaultDevice(.gpu) {
            MLXRandom.seed(11)
            for (bits, group) in [(8, 64), (4, 64)] {
                for (inputs, outputs) in [(2304, 8600), (2304, 6552), (3072, 2304), (2304, 1024)] {
                    let linear = QuantizedLinear(
                        weight: (MLXRandom.normal([outputs, inputs]) * 0.05).asType(.bfloat16),
                        bias: nil, groupSize: group, bits: bits)
                    let ranges = [0 ..< outputs, 0 ..< 3072, 3072 ..< outputs]
                        .filter { $0.upperBound <= outputs && !$0.isEmpty }
                    for rows in [2, 5, 8] {
                        let x = (MLXRandom.normal([1, rows, inputs])).asType(.bfloat16)
                        for range in ranges {
                            let batched = inkling2Project(linear, range, inkling2RowBatch(x))
                                .reshaped(1, rows, range.count)
                            for row in 0 ..< rows {
                                let alone = inkling2Project(
                                    linear, range, x[0..., row ..< (row + 1)])
                                equal(
                                    batched[0..., row ..< (row + 1)], alone,
                                    "\(bits)-bit [\(outputs), \(inputs)] rows \(range) of \(rows): row \(row)")
                            }
                        }
                    }
                }
            }
        }
    }

    /// The router of a verify (float32 scores over a strided row batch and the fused top-k)
    /// against each row routed alone, at 128 experts over 2304.
    func testVerifyRoutingMatchesOneRowRoutingAtMiniShapes() throws {
        try requireGPU()
        try Device.withDefaultDevice(.gpu) {
            let directory = try XCTUnwrap(
                Bundle.module.url(forResource: "inkling2", withExtension: nil))
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: Data(contentsOf: directory.appendingPathComponent("oq4e-config.json")))
                    as? [String: Any])
            json["mtp_config"] = nil
            let config = try JSONDecoder().decode(
                Inkling2Configuration.self, from: JSONSerialization.data(withJSONObject: json))
            XCTAssertEqual(config.text.hiddenSize, 2304)
            XCTAssertEqual(config.text.experts, 128)
            let format = Inkling2ExpertFormat(mode: .affine, bits: 8, groupSize: 32)
            let moe = Inkling2SparseMoE(config.text, gateUp: format, down: format)
            MLXRandom.seed(13)
            moe.update(
                parameters: ModuleParameters.unflattened([
                    "router_weight": MLXRandom.normal([128, 2304]) * 0.02,
                    "router_bias": MLXRandom.normal([128]) * 0.01,
                ]))
            for rows in [2, 5, 8] {
                let x = MLXRandom.normal([1, rows, 2304]).asType(.bfloat16)
                let verify = moe.route(x, verify: true)
                for row in 0 ..< rows {
                    let alone = moe.route(x[0..., row ..< (row + 1)])
                    equal(verify.indices[0..., row ..< (row + 1)], alone.indices, "indices \(row) of \(rows)")
                    equal(verify.weights[0..., row ..< (row + 1)], alone.weights, "weights \(row) of \(rows)")
                }
            }
        }
    }

    /// Attention of a verify with the rows folded into the head axis against each row's one-row
    /// call over its own keys, at 24 query and 8 key heads, for key counts on both sides of
    /// MLX's one-pass/two-pass switch. Where MLX would route a row differently from the fold
    /// (``Inkling2VerifyAttention``) the model keeps the per-row calls; this checks that the
    /// fold is exact wherever the check allows it, and that the check allows it at short keys.
    func testFoldedVerifyAttentionMatchesOneRowCallsAtMiniShapes() throws {
        try requireGPU()
        try Device.withDefaultDevice(.gpu) {
            MLXRandom.seed(17)
            let (heads, kvHeads, dim) = (24, 8, 128)
            let scale = 1 / Float(dim).squareRoot()
            var folds = 0
            for size in [24, 300, 1030, 4100, 9000] {
                for length in [2, 8] where length <= size {
                    let q = MLXRandom.normal([1, heads, length, dim]).asType(.bfloat16)
                    let k = MLXRandom.normal([1, kvHeads, size, dim]).asType(.bfloat16)
                    let v = MLXRandom.normal([1, kvHeads, size, dim]).asType(.bfloat16)
                    // A band bias with the causal edge the model's mask kernel writes (-1e30).
                    let bias = (MLXRandom.normal([1, heads, length, size]) * 0.5)
                    let key = MLXArray((0 ..< size).map(Int32.init)).reshaped(1, 1, 1, size)
                    let last = MLXArray((0 ..< length).map { Int32(size - length + $0) })
                        .reshaped(1, 1, length, 1)
                    let mask = which(key .> last, MLXArray(Float(-1e30)), bias).asType(.bfloat16)
                    let sinks = MLXRandom.normal([1, length, heads]).asType(.bfloat16)

                    let group = heads / kvHeads
                    let folded = Inkling2VerifyAttention.route(
                        keys: size, simds: group * length, grouped: true)
                    let foldable = (0 ..< length).allSatisfy {
                        Inkling2VerifyAttention.route(
                            keys: size - length + $0 + 1, simds: group, grouped: true) == folded
                    }
                    if size < 1000 {
                        XCTAssertTrue(foldable, "\(length) rows over \(size) keys should fold")
                    }
                    guard foldable else { continue }
                    folds += 1
                    let rows = heads * length
                    let out = MLXFast.scaledDotProductAttention(
                        queries: q.reshaped(1, rows, 1, dim), keys: k, values: v, scale: scale,
                        mask: .array(mask.reshaped(1, rows, 1, size)),
                        sinks: sinks.reshaped(length, heads).transposed().reshaped(rows)
                    ).reshaped(1, heads, length, dim)
                    for row in 0 ..< length {
                        let count = size - length + row + 1
                        let alone = MLXFast.scaledDotProductAttention(
                            queries: q[0..., 0..., row ..< (row + 1), 0...],
                            keys: k[0..., 0..., ..<count, 0...], values: v[0..., 0..., ..<count, 0...],
                            scale: scale,
                            mask: .array(mask[0..., 0..., row ..< (row + 1), ..<count]),
                            sinks: sinks[0..., row, 0...].reshaped(heads))
                        equal(
                            out[0..., 0..., row ..< (row + 1), 0...], alone,
                            "\(length) rows over \(size) keys: row \(row)")
                    }
                }
            }
            XCTAssertGreaterThan(folds, 0)
            print("INKLING2_VERIFY_FOLDS device=\(Inkling2VerifyAttention.device) folds=\(folds)")
        }
    }
}
