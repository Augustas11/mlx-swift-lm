import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4DSparkLoadingTests: XCTestCase {
    private func configurationObject() throws -> [String: Any] {
        let directory = try XCTUnwrap(
            Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["compress_ratios"] = [0, 4, 128, 4, 0, 0, 0]
        object["index_block"] = 0
        object["index_keep"] = 0
        object["dspark_target_layer_ids"] = [1, 2, 3]
        object["dspark_noise_token_id"] = 127
        object["dspark_markov_rank"] = 8
        object["dspark_block_size"] = 5
        return object
    }

    private func decode(_ object: [String: Any]) throws -> DeepseekV4Configuration {
        try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: JSONSerialization.data(withJSONObject: object))
    }

    func testConfigurationKeepsThreeDSparkStagesThroughRoundTrip() throws {
        let configuration = try decode(configurationObject())
        try configuration.validateModelConfiguration()
        XCTAssertEqual(configuration.dsparkStageCount, 3)
        XCTAssertEqual(configuration.compressRatios, [0, 4, 128, 4])
        let restored = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: JSONEncoder().encode(configuration))
        try restored.validateModelConfiguration()
        XCTAssertEqual(restored.dsparkStageCount, 3)
        XCTAssertEqual(restored.dsparkTargetLayerIDs, [1, 2, 3])
        XCTAssertEqual(restored.dsparkBlockSize, 5)
    }

    func testInvalidStageRatiosAndTargetLayerIDsAreRejected() throws {
        var object = try configurationObject()
        object["compress_ratios"] = [0, 4, 128, 4, 0, 4, 0]
        XCTAssertThrowsError(try decode(object))
        object = try configurationObject()
        object["dspark_target_layer_ids"] = [1, 1, 3]
        XCTAssertThrowsError(try decode(object).validateModelConfiguration())
        object["dspark_target_layer_ids"] = [1, 2, 4]
        XCTAssertThrowsError(try decode(object).validateModelConfiguration())
    }

    func testSanitizerRetainsEveryStageAndConfidenceHead() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try decode(configurationObject())
            var weights = [String: MLXArray]()
            for stage in 0 ..< 3 {
                for expert in 0 ..< configuration.nRoutedExperts {
                    weights["mtp.\(stage).ffn.experts.\(expert).w1.weight"] = MLXArray.full(
                        [2, 4], values: MLXArray(Float(expert)))
                }
                weights["mtp.\(stage).hc_attn_fn"] = MLXArray.ones([4, 8])
                weights["mtp.\(stage).ffn.gate.weight"] = MLXArray.ones([8, 64], dtype: .bfloat16)
            }
            weights["mtp.0.main_proj.weight"] = MLXArray.ones([64, 192])
            weights["mtp.2.confidence_head.proj.weight"] = MLXArray.ones([1, 72])
            weights["mtp.2.markov_head.markov_w1.weight"] = MLXArray.ones([128, 8])
            weights["mtp.2.hc_head_fn"] = MLXArray.ones([4, 256])
            let sanitized = DeepseekV4Weights.sanitize(weights, configuration: configuration)
            for stage in 0 ..< 3 {
                XCTAssertEqual(
                    sanitized["mtp.\(stage).ffn.switch_mlp.gate_proj.weight"]?.shape, [8, 2, 4])
                XCTAssertNotNil(sanitized["mtp.\(stage).attn_hc.fn"])
                XCTAssertEqual(sanitized["mtp.\(stage).ffn.gate.weight"]?.dtype, .float32)
            }
            XCTAssertNotNil(sanitized["mtp.0.main_proj.weight"])
            XCTAssertNotNil(sanitized["mtp.2.confidence_head.proj.weight"])
            XCTAssertNotNil(sanitized["mtp.2.markov_head.markov_w1.weight"])
            XCTAssertNotNil(sanitized["mtp.2.hc_head.fn"])
            XCTAssertFalse(sanitized.keys.contains { $0.contains(".experts.") })
        }
    }

    func testHiddenCaptureUsesTheMeanOfEachConfiguredHyperConnectionLayer() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try decode(configurationObject())
            let model = DeepseekV4Model(configuration)
            XCTAssertEqual(model.mtp.count, 3)
            let tokens = MLXArray([Int32(17)]).reshaped(1, 1)
            let expectedCaches = try model.newCache(parameters: nil)
            var hidden = repeated(
                expandedDimensions(model.embed(tokens), axis: 2),
                count: configuration.hcMult, axis: 2)
            var expected = [MLXArray]()
            for (index, layer) in model.layers.enumerated() {
                hidden = layer(
                    hidden, mask: nil,
                    cache: DeepseekV4LayerCache(expectedCaches[index], ratio: layer.attn.ratio),
                    inputIDs: tokens)
                if configuration.dsparkTargetLayerIDs.contains(index) {
                    expected.append(hidden.mean(axis: 2))
                }
            }
            let result = model.decodeBatchOutput(
                tokens, rowCaches: [try model.newCache(parameters: nil)], emitMTPState: true)
            let captured = try XCTUnwrap(result.state?[mtpLastHiddenStatesKey])
            XCTAssertEqual(captured.shape, [1, 1, 192])
            XCTAssertEqual(
                captured.asArray(Float.self), concatenated(expected, axis: -1).asArray(Float.self))
        }
    }


}
