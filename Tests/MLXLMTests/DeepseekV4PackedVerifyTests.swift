import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4PackedVerifyTests: XCTestCase {
    private func loadTiny(indexTopK: Int = 4) throws -> DeepseekV4Model {
        let directory = try XCTUnwrap(
            Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        configuration.indexBlock = 0
        configuration.indexKeep = 0
        configuration.indexTopk = indexTopK
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    private func prefill(_ model: DeepseekV4Model, count: Int) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        for start in stride(from: 0, to: count, by: 128) {
            let end = min(start + 128, count)
            let tokens = MLXArray((start ..< end).map { Int32(($0 * 7 + 11) % 128) })
                .reshaped(1, end - start)
            eval(model(tokens, cache: cache))
            eval(cache)
        }
        return cache
    }

    private func assertCachesEqual(
        _ actual: [KVCache], _ expected: [KVCache], label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
        for (actual, expected) in zip(actual, expected) {
            XCTAssertEqual(actual.metaState, expected.metaState, label, file: file, line: line)
            XCTAssertEqual(actual.state.count, expected.state.count, label, file: file, line: line)
            for (actual, expected) in zip(actual.state, expected.state) {
                XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
                XCTAssertEqual(
                    actual.asType(.float32).asArray(Float.self),
                    expected.asType(.float32).asArray(Float.self), label, file: file, line: line)
            }
        }
    }

    private func checkEveryPrefix(
        _ model: DeepseekV4Model, starts: [Int], widths: [Int]
    ) throws {
        let original = try starts.map { try prefill(model, count: $0) }
        let untouched = original.map { $0.map { $0.copy() } }
        let inputs = widths.enumerated().map { row, width in
            (0 ..< width).map { (row * 17 + $0 * 13 + 3) % 128 }
        }
        let verified = model.verifyBatchOutput(inputs, rowCaches: original, emitMTPState: false)
        XCTAssertEqual(verified.output.logits.shape, [1, widths.reduce(0, +), 128])
        var packed = 0
        for row in inputs.indices {
            XCTAssertEqual(verified.rowOffsets[row], packed)
            assertCachesEqual(original[row], untouched[row], label: "input remains committed row=\(row)")
            let serial = original[row].map { $0.copy() }
            var expectedPrefixes = [[KVCache]]()
            for column in inputs[row].indices {
                let token = MLXArray([Int32(inputs[row][column])]).reshaped(1, 1)
                let expected = model(token, cache: serial)
                let actual = verified.output.logits[0..., (packed + column) ..< (packed + column + 1)]
                XCTAssertEqual(
                    actual.asArray(Float.self), expected.asArray(Float.self),
                    "row=\(row) column=\(column) starts=\(starts)")
                let prefix = serial.map { $0.copy() }
                expectedPrefixes.append(prefix)
                assertCachesEqual(
                    verified.cachePrefixes[row][column], prefix,
                    label: "row=\(row) retained=\(column + 1) starts=\(starts)")
            }
            // All keep counts, including zero and full acceptance, must retain
            // exactly the serial prefix and its next-token computation.
            for keep in 0 ... inputs[row].count {
                let actual = (keep == 0 ? original[row] : verified.cachePrefixes[row][keep - 1])
                    .map { $0.copy() }
                let expected = (keep == 0 ? untouched[row] : expectedPrefixes[keep - 1])
                    .map { $0.copy() }
                let next = MLXArray([Int32(41)]).reshaped(1, 1)
                XCTAssertEqual(
                    model(next, cache: actual).asArray(Float.self),
                    model(next, cache: expected).asArray(Float.self),
                    "continuation row=\(row) retained=\(keep) starts=\(starts)")
                assertCachesEqual(actual, expected, label: "continued row=\(row) retained=\(keep)")
            }
            packed += inputs[row].count
        }
    }

    func testEveryAcceptPrefixAcrossRatioFourAndRatio128Boundaries() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            for start in [0, 3, 7, 127, 128] {
                try checkEveryPrefix(model, starts: [start], widths: [8])
            }
        }
    }

    func testRaggedRowsKeepTheirOwnOverlapAndWrappedWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny()
            try checkEveryPrefix(model, starts: [3, 127], widths: [5, 3])
            try checkEveryPrefix(model, starts: [128, 7, 0], widths: [3, 4, 1])
            try checkEveryPrefix(model, starts: [0, 3, 7, 127], widths: [2, 2, 2, 2])
        }
    }

    func testIndexerPoolCrosses512EntriesInsideVerify() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexTopK: 512)
            try checkEveryPrefix(model, starts: [2047], widths: [8])
        }
    }

    func testGPUEveryAcceptPrefixAcrossRatioFourAndRatio128Boundaries() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadTiny()
            for start in [0, 3, 7, 127, 128] {
                try checkEveryPrefix(model, starts: [start], widths: [8])
            }
        }
    }

    func testGPURaggedRowsKeepTheirOwnOverlapAndWrappedWindows() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadTiny()
            try checkEveryPrefix(model, starts: [3, 127], widths: [5, 3])
            try checkEveryPrefix(model, starts: [128, 7, 0], widths: [3, 4, 1])
            try checkEveryPrefix(model, starts: [0, 3, 7, 127], widths: [2, 2, 2, 2])
        }
    }

    func testGPUIndexerPoolCrosses512EntriesInsideVerify() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadTiny(indexTopK: 512)
            try checkEveryPrefix(model, starts: [2047], widths: [8])
        }
    }
}
