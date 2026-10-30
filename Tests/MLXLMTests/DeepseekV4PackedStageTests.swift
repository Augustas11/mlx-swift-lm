import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4PackedStageTests: XCTestCase {
    private enum DiagnosticError: Error {
        case firstDifference(layer: Int, stage: String)
    }

    private func loadArtifact() throws -> DeepseekV4Model {
        guard let path = ProcessInfo.processInfo.environment["DSV4_PACKED_ARTIFACT_DIR"] else {
            throw XCTSkip("Set DSV4_PACKED_ARTIFACT_DIR to the DeepSeek V4 model artifact")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        return model
    }

    private func compareFirstPosition(
        _ packed: MLXArray, _ serial: MLXArray, layer: Int, stage: String,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let actual = packed[0 ..< 1]
        guard actual.shape == serial.shape, actual.dtype == serial.dtype else {
            XCTFail("layer=\(layer) stage=\(stage) shape/dtype mismatch", file: file, line: line)
            throw DiagnosticError.firstDifference(layer: layer, stage: stage)
        }
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = serial.asType(.float32).asArray(Float.self)
        var different = 0
        var maxAbs: Float = 0
        for (actual, expected) in zip(actualValues, expectedValues) where actual != expected {
            different += 1
            let delta = abs(actual - expected)
            maxAbs = delta.isFinite ? max(maxAbs, delta) : .infinity
        }
        guard different == 0 else {
            let message = "layer=\(layer) stage=\(stage) different=\(different) maxAbs=\(maxAbs)"
            print("packed-stage-first-difference \(message)")
            XCTFail(message, file: file, line: line)
            throw DiagnosticError.firstDifference(layer: layer, stage: stage)
        }
    }

    /// Follow the actual bf16 packed branch and its scalar counterpart until
    /// their first differing intermediate. This diagnostic does not change
    /// projection selection or substitute a scalar packed implementation.
    func testGPUArtifactPackedFirstDifference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadArtifact()
            let original = try model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(11), 18, 25]).reshaped(1, 3)
            eval(model(prompt, cache: original))
            eval(original)
            let packedCaches = original.map { $0.copy() }
            let serialCaches = original.map { $0.copy() }
            let packedIDs = MLXArray([Int32(3), 16, 29, 42, 55, 68, 81, 94]).reshaped(8, 1)
            let serialIDs = MLXArray([Int32(3)]).reshaped(1, 1)
            var packedHidden = repeated(
                expandedDimensions(model.embed(packedIDs), axis: 2),
                count: model.configuration.hcMult, axis: 2)
            var serialHidden = repeated(
                expandedDimensions(model.embed(serialIDs), axis: 2),
                count: model.configuration.hcMult, axis: 2)
            for (index, layer) in model.layers.enumerated() {
                let packedAttention = layer.attnHC.normalizedBranchInput(
                    packedHidden, norm: layer.attnNorm)
                let serialAttention = layer.attnHC.normalizedBranchInput(
                    serialHidden, norm: layer.attnNorm)
                try compareFirstPosition(
                    packedAttention.0, serialAttention.0, layer: index, stage: "attention.input")
                try compareFirstPosition(
                    packedAttention.1, serialAttention.1, layer: index, stage: "attention.post")
                try compareFirstPosition(
                    packedAttention.2, serialAttention.2, layer: index, stage: "attention.comb")
                let packedOutput = layer.attn.verifyPacked(
                    packedAttention.0, caches: [packedCaches[index]], rowLengths: [8]).output
                let serialOutput = layer.attn(
                    serialAttention.0, mask: nil,
                    cache: DeepseekV4LayerCache(serialCaches[index], ratio: layer.attn.ratio))
                try compareFirstPosition(
                    packedOutput, serialOutput, layer: index, stage: "attention.output")
                packedHidden = deepseekV4HCExpand(
                    packedOutput, packedHidden, packedAttention.1, packedAttention.2)
                serialHidden = deepseekV4HCExpand(
                    serialOutput, serialHidden, serialAttention.1, serialAttention.2)
                try compareFirstPosition(
                    packedHidden, serialHidden, layer: index, stage: "attention.residual")
                let packedFFN = layer.ffnHC.normalizedBranchInput(packedHidden, norm: layer.ffnNorm)
                let serialFFN = layer.ffnHC.normalizedBranchInput(serialHidden, norm: layer.ffnNorm)
                try compareFirstPosition(
                    packedFFN.0, serialFFN.0, layer: index, stage: "ffn.input")
                try compareFirstPosition(
                    packedFFN.1, serialFFN.1, layer: index, stage: "ffn.post")
                try compareFirstPosition(
                    packedFFN.2, serialFFN.2, layer: index, stage: "ffn.comb")
                let packedGate = layer.ffn.packedGate(packedFFN.0, inputIDs: packedIDs)
                let serialGate = layer.ffn.gate(serialFFN.0, inputIDs: serialIDs)
                try compareFirstPosition(
                    packedGate.0, serialGate.0, layer: index, stage: "ffn.gate.indices")
                try compareFirstPosition(
                    packedGate.1, serialGate.1, layer: index, stage: "ffn.gate.weights")
                let packedFFNOutput = layer.ffn.packed(packedFFN.0, inputIDs: packedIDs)
                let serialFFNOutput = layer.ffn(serialFFN.0, inputIDs: serialIDs)
                try compareFirstPosition(
                    packedFFNOutput, serialFFNOutput, layer: index, stage: "ffn.output")
                packedHidden = deepseekV4HCExpand(
                    packedFFNOutput, packedHidden, packedFFN.1, packedFFN.2)
                serialHidden = deepseekV4HCExpand(
                    serialFFNOutput, serialHidden, serialFFN.1, serialFFN.2)
                try compareFirstPosition(
                    packedHidden, serialHidden, layer: index, stage: "ffn.residual")
            }
            print("packed-stage-all-exact layers=\(model.layers.count) positions=1")
        }
    }
}
