import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4ContinuousBatchTests: XCTestCase {
    func testWrappedDecodeWindowReusesTheReferenceKeyOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let data = Data("""
                {"num_hidden_layers":1,"compress_ratios":[0],"sliding_window":8}
                """.utf8)
            let configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            let attention = DeepseekV4Attention(configuration, layerIdx: 0)
            for position in [0, 1, 6, 7, 8, 15, 127, 128] {
                let count = min(position + 1, 8)
                let first = position + 1 - count
                let local = MLXArray.arange(count * 4, dtype: .float32).reshaped(count, 4)
                let (keys, validity) = attention.windowList(
                    local, first: first, positions: [position])
                let indices = (0 ..< 8).map { slot -> Int32 in
                    let absolute = position >= 7 ? position - 7 + slot : slot
                    return Int32(absolute <= position ? absolute - first : 0)
                }
                let expected = take(local, MLXArray(indices), axis: 0).reshaped(1, 8, 4)
                XCTAssertEqual(keys.asArray(Float.self), expected.asArray(Float.self))
                XCTAssertEqual(validity.asArray(Bool.self), (0 ..< 8).map { $0 < count })
            }
        }
    }

    func testOneRowSchedulerForwardKeepsSerialLogitsAndEveryCacheTensor() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = try XCTUnwrap(
                Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            var configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            configuration.indexBlock = 0
            configuration.indexKeep = 0
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let model = DeepseekV4Model(configuration)
            try loadWeights(
                modelDirectory: directory, model: model,
                perLayerQuantization: base.perLayerQuantization)
            let serial = try model.newCache(parameters: nil)
            let scheduled = try model.newCache(parameters: nil)
            for position in 0 ..< 137 {
                let input = MLXArray([Int32(position % configuration.vocabularySize)]).reshaped(1, 1)
                let expected = model(input, cache: serial)
                let actual = model.decodeBatch(input, rowCaches: [scheduled])
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "position=\(position)")
                for (lhs, rhs) in zip(scheduled, serial) {
                    XCTAssertEqual(lhs.metaState, rhs.metaState)
                    XCTAssertEqual(lhs.state.count, rhs.state.count)
                    for (actual, expected) in zip(lhs.state, rhs.state) {
                        XCTAssertEqual(actual.shape, expected.shape)
                        XCTAssertEqual(
                            actual.asType(.float32).asArray(Float.self),
                            expected.asType(.float32).asArray(Float.self), "position=\(position)")
                    }
                }
            }
        }
    }
}
