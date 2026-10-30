import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4PackedVerifyTests: XCTestCase {
    private enum FixtureError: Error { case unexpectedPrepareResult }
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
        if count > 0 {
            let tokens = MLXArray((0 ..< count).map { Int32(($0 * 7 + 11) % 128) })
            guard case .logits(let prepared) = try model.prepare(
                LMInput(tokens: tokens), cache: cache, state: nil, prefill: .init())
            else { throw FixtureError.unexpectedPrepareResult }
            eval(prepared.logits)
            eval(cache)
        }
        return cache
    }

    private func assertCachesEqual(
        _ actual: [KVCache], _ expected: [KVCache], label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
        for (layer, pair) in zip(actual, expected).enumerated() {
            let (actual, expected) = pair
            let layerLabel = "\(label) layer=\(layer)"
            XCTAssertEqual(actual.metaState, expected.metaState, layerLabel, file: file, line: line)
            XCTAssertEqual(actual.state.count, expected.state.count, layerLabel, file: file, line: line)
            for (part, pair) in zip(actual.state, expected.state).enumerated() {
                let (actual, expected) = pair
                let partLabel = "\(layerLabel) part=\(part)"
                XCTAssertEqual(actual.shape, expected.shape, partLabel, file: file, line: line)
                XCTAssertTrue(
                    actual.asType(.float32).asArray(Float.self)
                        == expected.asType(.float32).asArray(Float.self),
                    partLabel, file: file, line: line)
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
        // Exercise the public serving path against every serial prefix,
        // captured hidden state, rollback choice and next-token continuation.
        let verified = model.verifyBatchOutput(
            inputs, rowCaches: original, emitMTPState: true)
        XCTAssertEqual(verified.output.logits.shape, [1, widths.reduce(0, +), model.vocabularySize])
        var packed = 0
        for row in inputs.indices {
            XCTAssertEqual(verified.rowOffsets[row], packed)
            assertCachesEqual(original[row], untouched[row], label: "input remains committed row=\(row)")
            let serial = original[row].map { $0.copy() }
            var expectedPrefixes = [[KVCache]]()
            for column in inputs[row].indices {
                let token = MLXArray([Int32(inputs[row][column])]).reshaped(1, 1)
                var emit = LMOutput.State()
                emit[mtpEmitFlagKey] = true
                let expected = model(LMInput.Text(tokens: token), cache: serial, state: emit)
                let actual = verified.output.logits[0..., (packed + column) ..< (packed + column + 1)]
                XCTAssertTrue(
                    actual.asArray(Float.self) == expected.logits.asArray(Float.self),
                    "row=\(row) column=\(column) starts=\(starts)")
                if let hidden = expected.state?[mtpLastHiddenStatesKey] {
                    let actualHidden = try XCTUnwrap(verified.output.state?[mtpLastHiddenStatesKey])
                    XCTAssertTrue(
                        actualHidden[0..., (packed + column) ..< (packed + column + 1)]
                            .asArray(Float.self) == hidden.asArray(Float.self),
                        "hidden row=\(row) column=\(column) starts=\(starts)")
                } else {
                    XCTAssertNil(verified.output.state?[mtpLastHiddenStatesKey])
                }
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
                XCTAssertTrue(
                    model(next, cache: actual).asArray(Float.self)
                        == model(next, cache: expected).asArray(Float.self),
                    "continuation row=\(row) retained=\(keep) starts=\(starts)")
                assertCachesEqual(actual, expected, label: "continued row=\(row) retained=\(keep)")
            }
            packed += inputs[row].count
        }
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

    func testGPUArtifactPackedPrefixes() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadArtifact()
            try checkEveryPrefix(model, starts: [3, 127], widths: [5, 3])
            try checkEveryPrefix(model, starts: [2047], widths: [8])
        }
    }

    private func assertExactWithDiagnostics(
        _ actual: MLXArray, _ expected: MLXArray, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        var different = abs(actualValues.count - expectedValues.count)
        var maxAbs: Float = 0
        for (actual, expected) in zip(actualValues, expectedValues) {
            if actual != expected { different += 1 }
            maxAbs = max(maxAbs, abs(actual - expected))
        }
        print("packed-first-position \(label): different=\(different) maxAbs=\(maxAbs)")
        XCTAssertTrue(actualValues == expectedValues, label, file: file, line: line)
    }

    func testGPUArtifactPackedFirstPosition() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadArtifact()
            let original = try prefill(model, count: 3)
            let serial = original.map { $0.copy() }
            let verified = model.verifyBatchOutput(
                [[3, 16, 29, 42, 55, 68, 81, 94]], rowCaches: [original], emitMTPState: false)
            let expectedLogits = model(MLXArray([Int32(3)]).reshaped(1, 1), cache: serial)
            let prefix = verified.cachePrefixes[0][0]
            XCTAssertEqual(prefix.count, serial.count)
            for (layer, pair) in zip(prefix, serial).enumerated() {
                let (actual, expected) = pair
                XCTAssertEqual(actual.metaState, expected.metaState, "layer=\(layer)")
                XCTAssertEqual(actual.state.count, expected.state.count, "layer=\(layer)")
                for (part, pair) in zip(actual.state, expected.state).enumerated() {
                    assertExactWithDiagnostics(
                        pair.0, pair.1, label: "layer=\(layer) part=\(part)")
                }
            }
            assertExactWithDiagnostics(
                verified.output.logits[0..., ..<1], expectedLogits, label: "logits position=0")
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
