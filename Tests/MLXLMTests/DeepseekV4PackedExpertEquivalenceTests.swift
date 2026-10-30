import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4PackedExpertEquivalenceTests: XCTestCase {
    private enum FixtureError: Error { case unsupportedGeometry }

    private func assertExact(
        _ actual: MLXArray, _ expected: MLXArray, width: Int, stage: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "width=\(width) stage=\(stage)", file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, "width=\(width) stage=\(stage)", file: file, line: line)
        guard actual.shape == expected.shape, actual.dtype == expected.dtype else { return }
        let a = actual.asType(.float32).asArray(Float.self)
        let b = expected.asType(.float32).asArray(Float.self)
        var differences = 0
        var maxAbs: Float = 0
        for (lhs, rhs) in zip(a, b) where lhs != rhs {
            differences += 1
            let delta = abs(lhs - rhs)
            maxAbs = delta.isFinite ? max(maxAbs, delta) : .infinity
        }
        XCTAssertEqual(differences, 0,
            "width=\(width) stage=\(stage) different=\(differences) maxAbs=\(maxAbs)",
            file: file, line: line)
    }

    /// Compare expert primitives with identical corrected routing, without
    /// changing the production packed implementation or its exactness gate.
    func testGPUArtifactGatherMatchesGroupedPackedExperts() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_PACKED_ARTIFACT_DIR"],
            let inputPath = env["DSV4_PACKED_MOE_INPUT"],
            let tokenPath = env["DSV4_DUMP_TOKENS"]
        else { throw XCTSkip("Set DSV4_PACKED_ARTIFACT_DIR, DSV4_PACKED_MOE_INPUT and DSV4_DUMP_TOKENS") }
        guard Device.defaultDevice().deviceType == .gpu,
            GPU.deviceInfo().architecture == "applegpu_g15d"
        else { throw XCTSkip("This exactness probe requires the qualified M3 Ultra GPU") }

        let directory = URL(fileURLWithPath: artifact, isDirectory: true)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
        let fixture = try XCTUnwrap(arrays["ffn_in.0"])
        XCTAssertEqual(fixture.shape, [8192, 4096])
        guard fixture.shape == [8192, 4096] else { throw FixtureError.unsupportedGeometry }
        struct Request: Decodable { let tokens: [Int] }
        let request = try JSONDecoder().decode(Request.self,
            from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
        XCTAssertGreaterThanOrEqual(request.tokens.count, fixture.dim(0))
        guard request.tokens.count >= fixture.dim(0) else { throw FixtureError.unsupportedGeometry }

        let ffn = model.layers[0].ffn
        let gate = try XCTUnwrap(ffn.switchMLP.gateProjection as? QuantizedSwitchLinear)
        let up = try XCTUnwrap(ffn.switchMLP.upProjection as? QuantizedSwitchLinear)
        let down = try XCTUnwrap(ffn.switchMLP.downProjection as? QuantizedSwitchLinear)
        for (name, projection) in [("gate", gate), ("up", up), ("down", down)] {
            let parts = projection.quantizedParts
            let k = parts.weight.dim(-1) * 8
            let n = parts.weight.dim(-2)
            print("packed-expert-geometry projection=\(name) K=\(k) N=\(n) experts=\(parts.weight.dim(0))")
            XCTAssertTrue(projection.mode == .mxfp4 && projection.bits == 4 && projection.groupSize == 32)
            XCTAssertTrue(parts.bias == nil && parts.biases == nil)
            XCTAssertEqual(k % 512, 0, "\(name) must select qmv_fast")
            XCTAssertEqual(n % 8, 0, "\(name) must select gather_qmv_fast")
            guard projection.mode == .mxfp4, projection.bits == 4, projection.groupSize == 32,
                parts.bias == nil, parts.biases == nil, k % 512 == 0, n % 8 == 0
            else { throw FixtureError.unsupportedGeometry }
        }
        let activation = makeLimitedSwiGLU(limit: config.swigluLimit > 0 ? config.swigluLimit : nil)
        let selectedRows: [Int32] = [0, 31, 87, 183, 511, 1024, 2047, 4095]
        for width in [1, 2, 8] {
            let rows = Array(selectedRows.prefix(width))
            let x = fixture[MLXArray(rows)].asType(.bfloat16).reshaped(width, fixture.dim(-1))
            let ids = MLXArray(rows.map { Int32(request.tokens[Int($0)]) })
            let (indices, weights) = ffn.packedGate(x, inputIDs: ids)
            let topK = indices.dim(-1)
            // Keep this probe on the original unsorted gathered-vector route.
            XCTAssertLessThan(indices.size, 64)
            guard indices.size < 64 else { throw FixtureError.unsupportedGeometry }
            let inputs = expandedDimensions(x, axes: [-2, -3])
            let gatherGate = gate(inputs, indices)
            let gatherUp = up(inputs, indices)
            let gatherHidden = (activation(gatherGate.asType(.float32), gatherUp.asType(.float32))
                * expandedDimensions(weights, axes: [-1, -2])).asType(.bfloat16)
            let gatherDown = down(gatherHidden, indices)
            let shared = ffn.sharedExperts(x).asType(.float32)
            let gatherSum = (gatherDown.squeezed(axis: -2).asType(.float32).sum(axis: -2)
                + shared).asType(.bfloat16)

            var assignments = [[Int32]](repeating: [], count: gate.quantizedParts.weight.dim(0))
            for (slot, expert) in indices.asType(.int32).asArray(Int32.self).enumerated() {
                guard expert >= 0, Int(expert) < assignments.count else { throw FixtureError.unsupportedGeometry }
                assignments[Int(expert)].append(Int32(slot))
            }
            var groupedGate = MLXArray.zeros([width * topK, gate.quantizedParts.weight.dim(-2)], dtype: .bfloat16)
            var groupedUp = MLXArray.zeros(groupedGate.shape, dtype: .bfloat16)
            var groupedHidden = MLXArray.zeros(groupedGate.shape, dtype: .bfloat16)
            var groupedDown = MLXArray.zeros([width * topK, x.dim(-1)], dtype: .bfloat16)
            func project(_ projection: QuantizedSwitchLinear, _ input: MLXArray, expert: Int) -> MLXArray {
                let parts = projection.quantizedParts
                return quantizedMM(input, parts.weight[expert], scales: parts.scales[expert], biases: nil,
                    transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
            }
            for (expert, slots) in assignments.enumerated() where !slots.isEmpty {
                XCTAssertLessThanOrEqual(slots.count, 8)
                guard slots.count <= 8 else { throw FixtureError.unsupportedGeometry }
                let routes = MLXArray(slots)
                let input = x.reshaped(width, x.dim(-1))[routes.floorDivide(Int32(topK))]
                let g = project(gate, input, expert: expert)
                let u = project(up, input, expert: expert)
                let h = (activation(g.asType(.float32), u.asType(.float32))
                    * weights.flattened()[routes].expandedDimensions(axis: -1)).asType(.bfloat16)
                let d = project(down, h, expert: expert)
                groupedGate = groupedGate.at[routes].add(g)
                groupedUp = groupedUp.at[routes].add(u)
                groupedHidden = groupedHidden.at[routes].add(h)
                groupedDown = groupedDown.at[routes].add(d)
            }
            let groupedSum = (groupedDown.reshaped(indices.shape + [x.dim(-1)])
                .asType(.float32).sum(axis: -2) + shared).asType(.bfloat16)
            assertExact(gatherGate.reshaped(groupedGate.shape), groupedGate, width: width, stage: "gate")
            assertExact(gatherUp.reshaped(groupedUp.shape), groupedUp, width: width, stage: "up")
            assertExact(gatherHidden.reshaped(groupedHidden.shape), groupedHidden, width: width, stage: "weighted_bf16_activation")
            assertExact(gatherDown.reshaped(groupedDown.shape), groupedDown, width: width, stage: "down")
            assertExact(gatherSum, groupedSum, width: width, stage: "final_sum")
            assertExact(gatherSum, ffn.packed(x, inputIDs: ids), width: width, stage: "production_packed")
        }
    }
}
