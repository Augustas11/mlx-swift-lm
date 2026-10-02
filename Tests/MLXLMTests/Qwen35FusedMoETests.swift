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
    private var savedMode = Qwen35FusedMoE.Mode.off
    private var savedMaxTokens = 4

    override func setUp() {
        super.setUp()
        savedMode = Qwen35FusedMoE.mode
        savedMaxTokens = Qwen35FusedMoE.maxTokens
    }

    override func tearDown() {
        Qwen35FusedMoE.mode = savedMode
        Qwen35FusedMoE.maxTokens = savedMaxTokens
        super.tearDown()
    }

    /// A256-expert, top-8 block with the A3B quantization layout (router and
    /// shared-expert gate 8-bit, experts 4-bit, group 64, bf16 scales), small
    /// expert width to keep the test light.
    private func makeBlock(seed: UInt64) throws -> Qwen35SparseMoeBlock {
        let json = """
            {"hidden_size": 2048, "num_experts": 256, "num_experts_per_tok": 8,
             "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
             "norm_topk_prob": true}
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(seed)
        let block = Qwen35SparseMoeBlock(config)
        // Sharper router logits so the top-8 is not a near-tie everywhere.
        let params = block.parameters().flattened().map { key, value -> (String, MLXArray) in
            let scaled = key == "gate.weight" ? value * 16 : value
            return (key, scaled.asType(.bfloat16))
        }
        block.update(parameters: ModuleParameters.unflattened(params))
        quantize(model: block) { path, _ in
            path == "gate" || path == "shared_expert_gate" ? (64, 8) : (64, 4)
        }
        eval(block)
        XCTAssertTrue(Qwen35FusedMoE.labIsFusable(block))
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

    func testOverlappingCallsMatchSerialEvaluation() throws {
        let blocks = [try makeBlock(seed: 1), try makeBlock(seed: 2)]
        Qwen35FusedMoE.mode = .full
        Qwen35FusedMoE.maxTokens = 16
        for tokens in [1, 2, 4, 8, 16] {
            // The same block on two inputs, and a second block.
            var calls: [(Qwen35SparseMoeBlock, MLXArray)] = []
            for (b, block) in blocks.enumerated() {
                for v in 0 ..< 2 {
                    calls.append((block, input(tokens, seed: UInt64(100 * tokens + 10 * b + v))))
                }
            }
            let before = Qwen35FusedMoE.fusedCalls
            var serial: [MLXArray] = []
            for (block, x) in calls {
                let y = block(x)
                eval(y)
                serial.append(y)
            }
            XCTAssertEqual(Qwen35FusedMoE.fusedCalls - before, calls.count, "fused path taken")

            for _ in 0 ..< 3 {
                // One stream, no eval between the calls.
                let together = calls.map { $0.0($0.1) }
                eval(together)
                assertBitEqual(together, serial, "T=\(tokens) one stream")

                // Two streams, no eval between the calls.
                let half = calls.count / 2
                let a = Stream.withNewDefaultStream { calls[..<half].map { $0.0($0.1) } }
                let b = Stream.withNewDefaultStream { calls[half...].map { $0.0($0.1) } }
                eval(a + b)
                assertBitEqual(a + b, serial, "T=\(tokens) two streams")
            }
        }
    }

    func testBatchInvarianceAndAgreementWithStock() throws {
        let block = try makeBlock(seed: 3)
        Qwen35FusedMoE.maxTokens = 16
        for tokens in [2, 4, 8, 16] {
            let x = input(tokens, seed: UInt64(7 + tokens))
            Qwen35FusedMoE.mode = .full
            let fused = block(x)
            let singles = concatenated(
                (0 ..< tokens).map { block(x[0..., $0 ..< ($0 + 1)]) }, axis: 1)
            Qwen35FusedMoE.mode = .off
            let stock = block(x)
            eval(fused, singles, stock)
            assertBitEqual([fused], [singles], "T=\(tokens) batch invariance")
            let f = fused.asType(.float32)
            let s = stock.asType(.float32)
            let rel = (sqrt(((f - s) * (f - s)).sum()) / sqrt((s * s).sum())).item(Float.self)
            XCTAssertLessThan(rel, 0.02, "T=\(tokens) fused vs stock relative error \(rel)")
        }
    }
}
