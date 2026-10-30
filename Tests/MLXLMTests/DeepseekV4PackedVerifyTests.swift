import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4PackedVerifyTests: XCTestCase {
    private enum FixtureError: Error { case unexpectedPrepareResult }
    private func loadTiny(indexTopK: Int = 4, captureTargets: Bool = false) throws -> DeepseekV4Model {
        let directory = try XCTUnwrap(
            Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        configuration.indexBlock = 0
        configuration.indexKeep = 0
        configuration.indexTopk = indexTopK
        // Capture target states without instantiating drafter modules: this
        // fixture supplies only target weights and never executes a drafter.
        if captureTargets { configuration.dsparkTargetLayerIDs = Array(0 ..< configuration.hiddenLayers) }
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

    /// Diagnostic only: scalar qualification does not prove that ordinary B4
    /// decode and packed verification choose identical small-matrix arithmetic.
    func testGPUArtifactFourRowDecodeMatchesPackedAndScalar() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["DSV4_PACKED_MULTIROW_DIAGNOSTIC"] == "1" else {
            throw XCTSkip("Set DSV4_PACKED_MULTIROW_DIAGNOSTIC=1 for the four-row diagnostic")
        }
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        guard let requestsPath = environment["DSV4_PACKED_REQUESTS_JSON"] else {
            throw XCTSkip("Set DSV4_PACKED_REQUESTS_JSON to dsv4-dspark-requests.json")
        }
        struct Requests: Decodable {
            struct Request: Decodable { let name: String; let ids: [Int] }
            let requests: [Request]
        }
        struct Replay: Decodable {
            struct Row: Decodable {
                let name: String
                let common_prefix_ids: [Int]
                let comparison_ids: [Int]
            }
            let rows: [Row]
        }
        let requests = try JSONDecoder().decode(
            Requests.self, from: Data(contentsOf: URL(fileURLWithPath: requestsPath)))
        let names = (0 ..< 4).map { "prose-\($0)" }
        let prompts = try names.map { name -> [Int] in
            let matches = requests.requests.filter { $0.name == name }
            XCTAssertEqual(matches.count, 1, name)
            let ids = try XCTUnwrap(matches.first).ids
            XCTAssertFalse(ids.isEmpty, name)
            return ids
        }
        let replay: Replay?
        if let path = environment["DSV4_PACKED_MULTIROW_REPLAY_JSON"] {
            replay = try JSONDecoder().decode(
                Replay.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            XCTAssertEqual(replay?.rows.map { $0.name }, names, "Replay row order must be prose-0...3")
            guard replay?.rows.map({ $0.name }) == names else { return }
            for row in replay!.rows {
                XCTAssertEqual(row.comparison_ids.count, 2, row.name)
                guard row.comparison_ids.count == 2 else { return }
            }
        } else {
            replay = nil
        }
        try Device.withDefaultDevice(.gpu) {
            let model = try loadArtifact()
            XCTAssertFalse(model.configuration.dsparkTargetLayerIDs.isEmpty, "Hidden-state checks must be meaningful")
            func token(_ id: Int) -> MLXArray {
                MLXArray([Int32(id)]).reshaped(1, 1)
            }
            func scalar(_ id: Int, cache: [KVCache]) -> LMOutput {
                var emit = LMOutput.State()
                emit[mtpEmitFlagKey] = true
                return model(LMInput.Text(tokens: token(id)), cache: cache, state: emit)
            }
            func compareCache(_ actual: [KVCache], _ expected: [KVCache], label: String) {
                XCTAssertEqual(actual.count, expected.count, label)
                for (layer, pair) in zip(actual, expected).enumerated() {
                    XCTAssertEqual(pair.0.metaState, pair.1.metaState, "\(label) layer=\(layer) metadata")
                    XCTAssertEqual(pair.0.state.count, pair.1.state.count, "\(label) layer=\(layer) parts")
                    for (part, tensors) in zip(pair.0.state, pair.1.state).enumerated() {
                        assertExactWithDiagnostics(
                            tensors.0, tensors.1, label: "\(label) layer=\(layer) part=\(part)")
                    }
                }
            }
            var initial = [[KVCache]]()
            var comparison = [[Int]]()
            for row in 0 ..< 4 {
                let cache = try model.newCache(parameters: nil)
                guard case .logits(let prepared) = try model.prepare(
                    LMInput(tokens: MLXArray(prompts[row].map(Int32.init))),
                    cache: cache, state: nil, prefill: .init())
                else { throw FixtureError.unexpectedPrepareResult }
                var next = Int(argMax(prepared.logits[0, -1], axis: -1).item(Int32.self))
                for id in replay?.rows[row].common_prefix_ids ?? [] {
                    let result = scalar(id, cache: cache)
                    next = Int(argMax(result.logits[0, -1], axis: -1).item(Int32.self))
                }
                eval(cache)
                initial.append(cache)
                if let replay {
                    comparison.append(replay.rows[row].comparison_ids)
                } else {
                    let probe = scalar(next, cache: cache.map { $0.copy() })
                    comparison.append([next, Int(argMax(probe.logits[0, -1], axis: -1).item(Int32.self))])
                }
            }
            print("packed-four-row prompts=\(prompts.map { $0.count }) commonPrefixes=\(replay?.rows.map { $0.common_prefix_ids.count } ?? [0, 0, 0, 0]) inputs=\(comparison)")
            for width in [1, 2] {
                let ordinary = initial.map { $0.map { $0.copy() } }
                let serial = initial.map { $0.map { $0.copy() } }
                let packedInput = comparison.map { Array($0.prefix(width)) }
                let verified = model.verifyBatchOutput(
                    packedInput, rowCaches: initial, emitMTPState: true)
                XCTAssertEqual(verified.rowOffsets, (0 ..< 4).map { $0 * width })
                let packedHidden = try XCTUnwrap(verified.output.state?[mtpLastHiddenStatesKey])
                for column in 0 ..< width {
                    let batch = model.decodeBatchOutput(
                        MLXArray(comparison.map { Int32($0[column]) }).reshaped(4, 1),
                        rowCaches: ordinary, emitMTPState: true)
                    let batchHidden = try XCTUnwrap(batch.state?[mtpLastHiddenStatesKey])
                    for row in 0 ..< 4 {
                        let expected = scalar(comparison[row][column], cache: serial[row])
                        let hidden = try XCTUnwrap(expected.state?[mtpLastHiddenStatesKey])
                        let position = row * width + column
                        let label = "width=\(width) row=\(row) column=\(column)"
                        let ordinaryLogits = batch.logits[row ..< (row + 1)]
                        let packedLogits = verified.output.logits[0..., position ..< (position + 1)]
                        let ordinaryHidden = batchHidden[row ..< (row + 1)]
                        let selectedHidden = packedHidden[0..., position ..< (position + 1)]
                        for (path, logits, state) in [
                            ("ordinary-vs-scalar", ordinaryLogits, ordinaryHidden),
                            ("packed-vs-scalar", packedLogits, selectedHidden),
                        ] {
                            assertExactWithDiagnostics(logits, expected.logits, label: "\(label) \(path) logits")
                            assertExactWithDiagnostics(state, hidden, label: "\(label) \(path) hidden")
                        }
                        assertExactWithDiagnostics(packedLogits, ordinaryLogits, label: "\(label) packed-vs-ordinary logits")
                        assertExactWithDiagnostics(selectedHidden, ordinaryHidden, label: "\(label) packed-vs-ordinary hidden")
                        compareCache(ordinary[row], serial[row], label: "\(label) ordinary-vs-scalar")
                        compareCache(verified.cachePrefixes[row][column], serial[row], label: "\(label) packed-vs-scalar")
                        compareCache(verified.cachePrefixes[row][column], ordinary[row], label: "\(label) packed-vs-ordinary")
                    }
                }
            }
        }
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

    func testGPUFourRowDecodePackedAndScalarStayExactAcrossPoolBoundaries() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        try Device.withDefaultDevice(.gpu) {
            MLXRandom.seed(2151)
            let quantized = try loadTiny(captureTargets: true)
            // Dense F32 makes small-matrix reduction changes visible even when
            // a quantized tiny fixture happens to retain the same greedy IDs.
            let dense = DeepseekV4Model(quantized.configuration)
            for (fixture, model) in [("quantized", quantized), ("dense-f32", dense)] {
                model.train(false)
                let starts = [3, 127, 7, 255]
                let inputs = [[17, 29], [23, 41], [31, 53], [47, 61]]
                let original = try starts.map { try prefill(model, count: $0) }
                let untouched = original.map { $0.map { $0.copy() } }
                func scalar(_ id: Int, cache: [KVCache]) -> LMOutput {
                    var emit = LMOutput.State()
                    emit[mtpEmitFlagKey] = true
                    return model(
                        LMInput.Text(tokens: MLXArray([Int32(id)]).reshaped(1, 1)),
                        cache: cache, state: emit)
                }
                func compare(_ actual: LMOutput, row: Int, expected: LMOutput, label: String) throws {
                    assertExactWithDiagnostics(
                        actual.logits[row ..< (row + 1)], expected.logits, label: "\(label) logits")
                    let hidden = try XCTUnwrap(actual.state?[mtpLastHiddenStatesKey])
                    let expectedHidden = try XCTUnwrap(expected.state?[mtpLastHiddenStatesKey])
                    assertExactWithDiagnostics(
                        hidden[row ..< (row + 1)], expectedHidden, label: "\(label) hidden")
                }
                for width in [1, 2] {
                    let ordinary = original.map { $0.map { $0.copy() } }
                    let serial = original.map { $0.map { $0.copy() } }
                    let verified = model.verifyBatchOutput(
                        inputs.map { Array($0.prefix(width)) }, rowCaches: original, emitMTPState: true)
                    XCTAssertEqual(verified.rowOffsets, (0 ..< 4).map { $0 * width })
                    var serialPrefixes = Array(repeating: [[KVCache]](), count: 4)
                    for column in 0 ..< width {
                        let batch = model.decodeBatchOutput(
                            MLXArray(inputs.map { Int32($0[column]) }).reshaped(4, 1),
                            rowCaches: ordinary, emitMTPState: true)
                        let packedHidden = try XCTUnwrap(verified.output.state?[mtpLastHiddenStatesKey])
                        for row in 0 ..< 4 {
                            let expected = scalar(inputs[row][column], cache: serial[row])
                            let label = "\(fixture) width=\(width) row=\(row) column=\(column)"
                            try compare(batch, row: row, expected: expected, label: "\(label) ordinary")
                            let position = row * width + column
                            let expectedHidden = try XCTUnwrap(expected.state?[mtpLastHiddenStatesKey])
                            assertExactWithDiagnostics(
                                verified.output.logits[0..., position ..< (position + 1)],
                                expected.logits, label: "\(label) packed logits")
                            assertExactWithDiagnostics(
                                packedHidden[0..., position ..< (position + 1)],
                                expectedHidden, label: "\(label) packed hidden")
                            assertCachesEqual(ordinary[row], serial[row], label: "\(label) ordinary")
                            assertCachesEqual(
                                verified.cachePrefixes[row][column], serial[row], label: "\(label) packed")
                            serialPrefixes[row].append(serial[row].map { $0.copy() })
                        }
                    }
                    for keep in 0 ... width {
                        let retained = (0 ..< 4).map { row in
                            (keep == 0 ? original[row] : verified.cachePrefixes[row][keep - 1])
                                .map { $0.copy() }
                        }
                        let expectedCaches = (0 ..< 4).map { row in
                            (keep == 0 ? untouched[row] : serialPrefixes[row][keep - 1])
                                .map { $0.copy() }
                        }
                        let next = [71, 73, 79, 83]
                        let continued = model.decodeBatchOutput(
                            MLXArray(next.map(Int32.init)).reshaped(4, 1),
                            rowCaches: retained, emitMTPState: true)
                        for row in 0 ..< 4 {
                            let expected = scalar(next[row], cache: expectedCaches[row])
                            let label = "\(fixture) width=\(width) keep=\(keep) row=\(row) continuation"
                            try compare(continued, row: row, expected: expected, label: label)
                            assertCachesEqual(retained[row], expectedCaches[row], label: label)
                            assertCachesEqual(original[row], untouched[row], label: "\(label) unmodified input")
                        }
                    }
                }
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
