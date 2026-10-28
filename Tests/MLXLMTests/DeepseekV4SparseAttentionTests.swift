import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4SparseAttentionTests: XCTestCase {
    private func attention(dim: Int, heads: Int) throws -> DeepseekV4Attention {
        let json = """
            {"model_type":"deepseek_v4","num_hidden_layers":1,"hidden_size":64,
             "num_attention_heads":\(heads),"o_groups":1,"head_dim":\(dim),
             "qk_rope_head_dim":16,"q_lora_rank":32,"o_lora_rank":8,
             "sliding_window":128,"compress_ratios":[0]}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        return DeepseekV4Attention(config, layerIdx: 0)
    }

    private func requireGPU() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
    }

    /// Tiny and production head dimensions; tails, holes, multiple blocks,
    /// bf16 inputs and a nonzero, per-head sink all use the graph oracle.
    func testFusedMatchesGraph() throws {
        try requireGPU()
        for dim in [64, 512] {
            let B = 2
            let H = 3
            let L = 3
            let attn = try attention(dim: dim, heads: H)
            try attn.update(parameters: ModuleParameters.unflattened([
                ("attn_sink", MLXArray([Float(-2), 0.5, 3]))
            ]), verify: .none)
            for count in [1, 63, 64, 65, 137] {
                let q = MLXArray(
                    (0 ..< B * H * L * dim).map {
                        Float(sin(Double($0 * 7 + 3))) * 0.7
                    }
                ).reshaped(B, H, L, dim).asType(.bfloat16)
                let keys = MLXArray(
                    (0 ..< B * L * count * dim).map {
                        Float(cos(Double($0 * 3 + 1)))
                    }
                ).reshaped(B, L, count, dim).asType(.bfloat16)
                let valid = MLXArray(
                    (0 ..< B * L * count).map {
                        $0 % count % 64 == 0 || $0 % 5 != 0
                    }
                ).reshaped(B, L, count)
                let fused = try XCTUnwrap(
                    deepseekV4SparseAttention(
                        q, keys: keys, valid: valid, sink: attn.attnSink, scale: attn.scale))
                let expected = attn.graphAttention(q, keys: keys, valid: valid)
                XCTAssertEqual(fused.shape, q.shape)
                XCTAssertEqual(fused.dtype, .float32)
                XCTAssertLessThan(
                    abs(fused - expected).max().item(Float.self), 2e-5,
                    "D=\(dim), K=\(count)")
                // Exercise the shared production entry point as well.
                XCTAssertEqual(
                    attn.referenceAttention(q, keys: keys, valid: valid).asArray(Float.self),
                    fused.asArray(Float.self))
            }
        }
    }

    /// A new block maximum must rescale the float32 accumulator AFTER the
    /// previous block's bf16 weights have been consumed.
    func testRunningMaximumAndBF16Rounding() throws {
        try requireGPU()
        let dim = 512
        let attn = try attention(dim: dim, heads: 1)
        var query = [Float](repeating: 0, count: dim)
        query[0] = sqrt(Float(dim))
        var keys = [Float](repeating: 0, count: 128 * dim)
        keys[1] = 128
        keys[64 * dim] = 1
        keys[64 * dim + 1] = -47
        let q = MLXArray(query).reshaped(1, 1, 1, dim)
        let k = MLXArray(keys).reshaped(1, 1, 128, dim)
        let out = try XCTUnwrap(
            deepseekV4SparseAttention(
                q, keys: k, valid: MLXArray.ones([1, 1, 128], dtype: .bool),
                sink: attn.attnSink, scale: attn.scale))
        let e = Float(Foundation.exp(-1.0))
        let expected = (128 * e - 47) / (1 + 128 * e)
        XCTAssertEqual(out[0, 0, 0, 1].item(Float.self), expected, accuracy: 1e-6)
        XCTAssertGreaterThan(out[0, 0, 0, 1].item(Float.self), 1e-3)
    }

    func testEmptyBlocksAndSink() throws {
        try requireGPU()
        let dim = 64
        let count = 130
        let q = MLXArray.zeros([1, 2, 1, dim])
        let keys = MLXArray.ones([1, 1, count, dim])
        let sink = MLXArray([Float(0), Float(log(3.0))])
        for active in [[], [129]] as [[Int]] {
            let valid = MLXArray((0 ..< count).map { active.contains($0) }).reshaped(1, 1, count)
            let out = try XCTUnwrap(
                deepseekV4SparseAttention(
                    q, keys: keys, valid: valid, sink: sink, scale: 0.125))
            let expectedValues: [Float] = active.isEmpty ? [0, 0] : [0.5, 0.25]
            for (head, expected) in expectedValues.enumerated() {
                XCTAssertLessThan(abs(out[0, head, 0] - expected).max().item(Float.self), 1e-6)
            }
        }
    }

    /// Prefix rows and packed query slices must be bit-identical, even though
    /// their contiguous strides and threadgroup counts differ.
    func testRowsAndPackedQueriesAreExactlyIndependent() throws {
        try requireGPU()
        let B = 8
        let H = 4
        let L = 8
        let D = 512
        let K = 65
        let queryCount = B * H * L * D
        let queryValues: [Float] = (0 ..< queryCount).map { index in
            let value = (index * 7 % 23) - 11
            return Float(value) / 16
        }
        let q = MLXArray(queryValues).reshaped(B, H, L, D).asType(.bfloat16)
        let keyCount = B * L * K * D
        let keyValues: [Float] = (0 ..< keyCount).map { index in
            let value = (index * 3 % 19) - 9
            return Float(value) / 16
        }
        let keys = MLXArray(keyValues).reshaped(B, L, K, D).asType(.bfloat16)
        let valid = MLXArray.ones([B, L, K], dtype: .bool)
        let sink = MLXArray.zeros([H])
        let full = try XCTUnwrap(
            deepseekV4SparseAttention(
                q, keys: keys, valid: valid, sink: sink, scale: 1 / sqrt(Float(D))))
        for rows in [1, 2, 4, 8] {
            let prefix = try XCTUnwrap(
                deepseekV4SparseAttention(
                    q[..<rows], keys: keys[..<rows], valid: valid[..<rows], sink: sink,
                    scale: 1 / sqrt(Float(D))))
            XCTAssertEqual(prefix.asArray(Float.self), full[..<rows].asArray(Float.self))
        }
        for row in 0 ..< B {
            for position in 0 ..< L {
                let single = try XCTUnwrap(
                    deepseekV4SparseAttention(
                        q[row ..< row + 1, 0..., position ..< position + 1],
                        keys: keys[row ..< row + 1, position ..< position + 1],
                        valid: valid[row ..< row + 1, position ..< position + 1], sink: sink,
                        scale: 1 / sqrt(Float(D))))
                XCTAssertEqual(
                    single.asArray(Float.self),
                    full[row ..< row + 1, 0..., position ..< position + 1].asArray(Float.self))
            }
        }
    }

    func testCPUUsesGraphFallback() throws {
        try Device.withDefaultDevice(.cpu) {
            let attn = try attention(dim: 64, heads: 2)
            let q = MLXArray.ones([1, 2, 1, 64])
            let keys = MLXArray.ones([1, 1, 3, 64])
            let valid = MLXArray.ones([1, 1, 3], dtype: .bool)
            XCTAssertNil(
                deepseekV4SparseAttention(
                    q, keys: keys, valid: valid, sink: attn.attnSink, scale: attn.scale))
            XCTAssertEqual(
                attn.referenceAttention(q, keys: keys, valid: valid).asArray(Float.self),
                attn.graphAttention(q, keys: keys, valid: valid).asArray(Float.self))
        }
    }

    func testLargePrefillUsesGraphFallback() throws {
        try requireGPU()
        let attn = try attention(dim: 64, heads: 2)
        let q = MLXArray.ones([1, 2, 9, 64])
        let keys = MLXArray.ones([1, 9, 3, 64])
        let valid = MLXArray.ones([1, 9, 3], dtype: .bool)
        XCTAssertNil(
            deepseekV4SparseAttention(
                q, keys: keys, valid: valid, sink: attn.attnSink, scale: attn.scale))
        XCTAssertEqual(
            attn.referenceAttention(q, keys: keys, valid: valid).asArray(Float.self),
            attn.graphAttention(q, keys: keys, valid: valid).asArray(Float.self))
    }
}
