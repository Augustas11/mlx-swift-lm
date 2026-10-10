import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Fused small-T Qwen3.5 MoE path (`Qwen35FusedMoE`): calls that overlap on
/// one stream or across streams must match the same calls evaluated alone, and
/// a token's output must not depend on the tokens batched with it.
final class Qwen35FusedMoETests: XCTestCase {
    /// A256-expert, top-8 block with the A3B quantization layout (router and
    /// shared-expert gate 8-bit, experts 4-bit, group 64, bf16 scales), small
    /// expert width to keep the test light.
    private func makeBlock(
        seed: UInt64, expectFusable: Bool = true,
        replaceModules: ((Qwen35SparseMoeBlock) throws -> Void)? = nil
    ) throws -> Qwen35SparseMoeBlock {
        let json = """
            {"hidden_size": 2048, "num_experts": 256, "num_experts_per_tok": 8,
             "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
             "norm_topk_prob": true}
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(seed)
        let block = Qwen35SparseMoeBlock(config)
        try replaceModules?(block)
        // Sharper router logits so the top-8 is not a near-tie everywhere.
        let params = block.parameters().flattened().map { key, value -> (String, MLXArray) in
            let scaled = key == "gate.weight" ? value * 16 : value
            return (key, value.dtype == .float32 ? scaled.asType(.bfloat16) : scaled)
        }
        block.update(parameters: ModuleParameters.unflattened(params))
        quantize(model: block) { path, _ in
            path == "gate" || path == "shared_expert_gate" ? (64, 8) : (64, 4)
        }
        eval(block)
        if expectFusable { XCTAssertTrue(Qwen35FusedMoE.isFusable(block)) }
        return block
    }

    private func input(_ tokens: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        let x = MLXRandom.normal([1, tokens, 2048]).asType(.bfloat16)
        eval(x)
        return x
    }

    private func assertBitEqual(
        _ a: [MLXArray], _ b: [MLXArray], _ label: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        for (i, (x, y)) in zip(a, b).enumerated() {
            let diff = (x .!= y).asType(.int32).sum().item(Int.self)
            XCTAssertEqual(diff, 0, "\(label) call \(i): \(diff) elements differ", file: file, line: line)
        }
    }

    private func fused(_ block: Qwen35SparseMoeBlock, _ x: MLXArray) throws -> MLXArray {
        try XCTUnwrap(Qwen35FusedMoE.forward(block, x), "expected production fused path")
    }

    func testProductionGateRejectsRowsAboveMeasuredTokenBound() throws {
        let block = try makeBlock(seed: 4)
        let x = input(8, seed: 48)
        XCTAssertNil(Qwen35FusedMoE.forward(block, x))
        let automatic = block(x)
        let stock = block.stockForward(x)
        eval(automatic, stock)
        assertBitEqual([automatic], [stock], "8-token row automatic stock fallback")
    }

    /// Decode (1 token per row) and verify (2 tokens per row) batches above the
    /// single-call bound stay on the fused path, and every token matches the
    /// same token evaluated alone, so crossing 7 flattened tokens never changes
    /// a row's output.
    func testDecodeBatchesAboveSingleCallBoundAreChunkedAndBatchInvariant() throws {
        let disabled = ["0", "false", "FALSE", "no", "NO", "off", "OFF"].contains(
            ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE"] ?? "")
        try XCTSkipIf(disabled, "fused MoE kill switch is set")
        let block = try makeBlock(seed: 6)
        for (rows, rowTokens) in [(8, 1), (9, 1), (16, 1), (4, 2), (5, 2), (8, 2)] {
            MLXRandom.seed(UInt64(1000 + 10 * rows + rowTokens))
            let x = MLXRandom.normal([rows, rowTokens, 2048]).asType(.bfloat16)
            eval(x)
            let batched = try fused(block, x)
            let automatic = block(x)
            var singles: [MLXArray] = []
            for r in 0 ..< rows {
                for t in 0 ..< rowTokens {
                    singles.append(try fused(block, x[r ..< (r + 1), t ..< (t + 1)]))
                }
            }
            let expected = concatenated(singles, axis: 0).reshaped(x.shape)
            eval(batched, automatic, expected)
            let label = "rows=\(rows) rowTokens=\(rowTokens)"
            XCTAssertEqual(batched.shape, x.shape, label)
            assertBitEqual([batched], [expected], "\(label) batch invariance")
            assertBitEqual([automatic], [batched], "\(label) automatic fused path")
        }
    }

    /// The kernels index every scale and bias from the weight dimensions, so a
    /// block whose companion tensors do not have the exact layout must not
    /// take the fused path.
    func testMismatchedQuantizedLayoutIsNotFusable() throws {
        for key in ["shared_expert.down_proj.scales", "switch_mlp.gate_proj.biases"] {
            // Not resolved yet: resolution is cached per block.
            let block = try makeBlock(seed: 8, expectFusable: false)
            let params = Dictionary(uniqueKeysWithValues: block.parameters().flattened())
            let original = try XCTUnwrap(params[key], key)
            let truncated = original[.ellipsis, 0 ..< (original.dim(-1) - 1)]
            _ = block.update(parameters: ModuleParameters.unflattened([(key, truncated)]))
            eval(block)
            XCTAssertFalse(Qwen35FusedMoE.isFusable(block), key)
            XCTAssertNil(Qwen35FusedMoE.forward(block, input(1, seed: 81)), key)
        }
    }

    /// A ParoQuant `RotateSwitchGLU` keeps the stock quantized expert layout
    /// but rotates the expert input and hidden state, which the fused kernels
    /// would skip. Only the exact stock `SwitchGLU` is fusable.
    func testRotatedSwitchGLUIsNotFusable() throws {
        let block = try makeBlock(seed: 9, expectFusable: false) { block in
            let rotated = RotateSwitchGLU(
                inputDims: 2048, hiddenDims: 256, numExperts: 256, groupSize: 128, krot: 1)
            _ = try block.update(
                modules: ModuleChildren.unflattened([("switch_mlp", rotated)]), verify: .none)
        }
        XCTAssertTrue(block.switchMLP is RotateSwitchGLU)
        // The expert projections still have the eligible quantized layout.
        let up = try XCTUnwrap(block.switchMLP.upProjection as? QuantizedSwitchLinear)
        XCTAssertEqual(up.bits, 4)
        XCTAssertEqual(up.quantizedParts.scales.dtype, .bfloat16)
        XCTAssertFalse(Qwen35FusedMoE.isFusable(block))
        XCTAssertNil(Qwen35FusedMoE.forward(block, input(1, seed: 91)))
    }

    /// A LoRA adapter subclasses `QuantizedLinear` and keeps its tensors, but
    /// adds a low-rank term the fused kernels would omit. Only the exact stock
    /// `QuantizedLinear` is fusable, for the shared expert and the router.
    func testAdapterBackedProjectionIsNotFusable() throws {
        for path in ["shared_expert.up_proj", "shared_expert.down_proj", "gate"] {
            let block = try makeBlock(seed: 10, expectFusable: false)
            let modules = Dictionary(uniqueKeysWithValues: block.leafModules().flattened())
            let stock = try XCTUnwrap(modules[path] as? QuantizedLinear, path)
            XCTAssertTrue(type(of: stock) == QuantizedLinear.self, path)
            let adapter = try XCTUnwrap(QLoRALinear.from(linear: stock, rank: 4) as? Module, path)
            _ = try block.update(
                modules: ModuleChildren.unflattened([(path, adapter)]), verify: .none)
            eval(block)
            XCTAssertFalse(Qwen35FusedMoE.isFusable(block), path)
            XCTAssertNil(Qwen35FusedMoE.forward(block, input(1, seed: 92)), path)
        }
        // Control: the same block without an adapter is fusable.
        XCTAssertTrue(Qwen35FusedMoE.isFusable(try makeBlock(seed: 10)))
    }

    func testEligibleBlockUsesFusedPathByDefault() throws {
        let disabled = ["0", "false", "FALSE", "no", "NO", "off", "OFF"].contains(
            ProcessInfo.processInfo.environment["MLX_LM_QWEN35_FUSED_MOE"] ?? "")
        try XCTSkipIf(disabled, "fused MoE kill switch is set")
        let block = try makeBlock(seed: 5)
        let x = input(1, seed: 51)
        let automatic = block(x)
        let fused = try fused(block, x)
        eval(automatic, fused)
        assertBitEqual([automatic], [fused], "T=1 automatic fused path")
    }

    func testOverlappingCallsMatchSerialEvaluation() throws {
        let blocks = [try makeBlock(seed: 1), try makeBlock(seed: 2)]
        for tokens in [1, 2, 4, 7] {
            // The same block on two inputs, and a second block.
            var calls: [(Qwen35SparseMoeBlock, MLXArray)] = []
            for (b, block) in blocks.enumerated() {
                for v in 0 ..< 2 {
                    calls.append((block, input(tokens, seed: UInt64(100 * tokens + 10 * b + v))))
                }
            }
            var serial: [MLXArray] = []
            for (block, x) in calls {
                let y = try fused(block, x)
                eval(y)
                serial.append(y)
            }

            for _ in 0 ..< 3 {
                // One stream, no eval between the calls.
                let together = try calls.map { try fused($0.0, $0.1) }
                eval(together)
                assertBitEqual(together, serial, "T=\(tokens) one stream")

                // Two streams, no eval between the calls.
                let half = calls.count / 2
                let a = try Stream.withNewDefaultStream {
                    try calls[..<half].map { try fused($0.0, $0.1) }
                }
                let b = try Stream.withNewDefaultStream {
                    try calls[half...].map { try fused($0.0, $0.1) }
                }
                eval(a + b)
                assertBitEqual(a + b, serial, "T=\(tokens) two streams")
            }
        }
    }

    func testBatchInvarianceAndAgreementWithStock() throws {
        let block = try makeBlock(seed: 3)
        for tokens in [2, 4, 7] {
            let x = input(tokens, seed: UInt64(7 + tokens))
            let fused = try fused(block, x)
            let singles = concatenated(
                try (0 ..< tokens).map { try self.fused(block, x[0..., $0 ..< ($0 + 1)]) },
                axis: 1)
            let stock = block.stockForward(x)
            eval(fused, singles, stock)
            assertBitEqual([fused], [singles], "T=\(tokens) batch invariance")
            let f = fused.asType(.float32)
            let s = stock.asType(.float32)
            let rel = (sqrt(((f - s) * (f - s)).sum()) / sqrt((s * s).sum())).item(Float.self)
            XCTAssertLessThan(rel, 0.02, "T=\(tokens) fused vs stock relative error \(rel)")
        }
    }
}
