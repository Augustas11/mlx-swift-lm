// Copyright © 2026 Apple Inc.

import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import XCTest

/// #2373: the Llama and Qwen3 MLP run `silu(gate) * up` as one compiled kernel.
/// The fused result must be bit-equal to the two-kernel form it replaced.
final class SwiGLUFusionTests: XCTestCase {

    private let dtypes: [DType] = [.float16, .bfloat16, .float32]

    /// Largest absolute difference and count of unequal elements, for the failure message.
    private func mismatch(_ a: MLXArray, _ b: MLXArray) -> String {
        let diff = abs(a.asType(.float32) - b.asType(.float32))
        let count = (a .!= b).asType(.int32).sum().item(Int.self)
        return "unequal=\(count)/\(a.size) maxAbs=\(diff.max().item(Float.self))"
    }

    func testFusedActivationIsBitEqual() {
        MLXRandom.seed(2373)
        for dtype in dtypes {
            for shape in [[1, 1, 14336], [1, 7, 1024], [4, 1, 3072]] {
                // Wide range: covers saturation of sigmoid at both ends.
                let gate = (MLXRandom.normal(shape) * 8).asType(dtype)
                let up = (MLXRandom.normal(shape) * 4).asType(dtype)
                let fused = swiGLU(gate, up)
                let unfused = silu(gate) * up
                eval(fused, unfused)
                XCTAssertEqual(fused.dtype, unfused.dtype)
                XCTAssertTrue(
                    arrayEqual(fused, unfused).item(Bool.self),
                    "\(dtype) \(shape): \(mismatch(fused, unfused))")
            }
        }
    }

    func testLlamaMLPIsBitEqualToUnfused() {
        MLXRandom.seed(2373)
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 1, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        for dtype in dtypes {
            let mlp = LlamaMLP(config)
            mlp.update(parameters: mlp.parameters().mapValues { $0.asType(dtype) })
            quantize(model: mlp, groupSize: 64, bits: 4)
            let x = MLXRandom.normal([1, 3, 64]).asType(dtype)
            let fused = mlp(x)
            let unfused = mlp.down(silu(mlp.gate(x)) * mlp.up(x))
            eval(fused, unfused)
            XCTAssertTrue(
                arrayEqual(fused, unfused).item(Bool.self), "\(dtype): \(mismatch(fused, unfused))")
        }
    }

    func testQwen3MLPIsBitEqualToUnfused() {
        MLXRandom.seed(2373)
        for dtype in dtypes {
            let mlp = Qwen3MLP(dimensions: 64, hiddenDimensions: 128)
            mlp.update(parameters: mlp.parameters().mapValues { $0.asType(dtype) })
            quantize(model: mlp, groupSize: 64, bits: 4)
            let x = MLXRandom.normal([1, 3, 64]).asType(dtype)
            let fused = mlp(x)
            let unfused = mlp.down(silu(mlp.gate(x)) * mlp.up(x))
            eval(fused, unfused)
            XCTAssertTrue(
                arrayEqual(fused, unfused).item(Bool.self), "\(dtype): \(mismatch(fused, unfused))")
        }
    }
}
