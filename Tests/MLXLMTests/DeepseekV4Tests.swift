import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// DeepSeek-V4 against a port of DeepSeek's reference (`inference/model.py`
/// + `kernel.py`). The fixture (`Resources/deepseek_v4_tiny`, weights from
/// its `make_fixture.py`) is a 4-layer model with every V4 path: a
/// hash-routed layer, local, ratio-4 (indexer) and ratio-128 attention, an
/// 8-token window, and the artifact's MXFP4/MXFP8 per-layer quantization.
/// Fixture parity runs on the CPU. The oracle (`make_oracle.py`) prefills the
/// whole prompt in one call with float32 activations, so the parity test
/// loads the fixture with its bf16 tensors widened to float32 as well.
final class DeepseekV4Tests: XCTestCase {
    func testGPURouterScoresMatchFrozenReferenceBits() throws {
        guard Device.defaultDevice().deviceType == .gpu,
            GPU.deviceInfo().architecture == "applegpu_g15d"
        else { throw XCTSkip("Frozen M3 Ultra router arithmetic fixture") }
        // Actual layer-0 router logits and scores from the immutable QAT-off
        // MLX 0.32.4 oracle. Native-log/logAddExp variants fail these values.
        let logits: [UInt32] = [
            3225218394, 3222189349, 3211923427, 3222413284, 3230276969, 3217268808,
            3217943974, 3223332806, 3221304816, 3217019275, 3210747181, 3224934482,
            3217346718, 3220346189, 3210065375, 3209252010, 3217649196, 3223024702,
            3219915628, 3217681364, 3220021372, 3231168264, 3219503136, 3208394419,
        ]
        let expected: [UInt32] = [
            1046941800, 1050911680, 1058187427, 1050642621, 1038867744, 1055054120,
            1054518922, 1049600937, 1052036038, 1055255987, 1058477594, 1047449381,
            1054991542, 1052742984, 1058648462, 1058854834, 1054750618, 1049938888,
            1053046782, 1054725187, 1052971591, 1035922857, 1053343714, 1059075373,
        ]
        let result = deepseekV4RouterSoftplusScores(MLXArray(logits.map { Float(bitPattern: $0) }))
        XCTAssertEqual(result.asArray(Float.self).map(\.bitPattern), expected)
    }


    func testHyperConnectionExpansionRoundsProductsBeforeReducingStreams() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let residual = MLXArray([Float(0.75), -0.75, 0, 0])
                    .reshaped(1, 1, 4, 1).asType(.bfloat16)
                var coefficients = [Float](repeating: 0, count: 16)
                coefficients[0] = 1 + Float(1.1920928955078125e-7)
                coefficients[4] = 1
                let comb = MLXArray(coefficients).reshaped(1, 1, 4, 4)
                let output = deepseekV4HCExpand(
                    MLXArray.zeros([1, 1, 1], dtype: .bfloat16), residual,
                    MLXArray.zeros([1, 1, 4], dtype: .float32), comb)
                // roundF32(0.75 * (1 + 2^-23)) - 0.75 = 2^-23.
                // An exact/FMA dot product instead gives 3 * 2^-25.
                XCTAssertEqual(output.dtype, .bfloat16)
                XCTAssertEqual(
                    output.asType(.float32).asArray(Float.self),
                    [1.1920928955078125e-7, 0, 0, 0])
            }
        }
    }

    func testQueryHeadNormRoundsInverseBeforeMultiplication() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let input = MLXArray([Float(1), 2, 3, 6]).reshaped(1, 1, 1, 4)
                    .asType(.bfloat16)
                let actual = deepseekV4QueryHeadNorm(input, eps: 1e-6)
                XCTAssertEqual(actual.dtype, .bfloat16)
                // Inverse RMS rounds from 0.28284270 to bf16 0.283203125.
                XCTAssertEqual(
                    actual.asType(.float32).asArray(Float.self),
                    [0.283203125, 0.56640625, 0.8515625, 1.703125])
                if device.deviceType == .gpu {
                    XCTAssertEqual(
                        MLXFast.rmsNorm(input, weight: MLXArray.mlxNone, eps: 1e-6)
                            .asType(.float32).asArray(Float.self),
                        [0.283203125, 0.56640625, 0.84765625, 1.6953125])
                }
            }
        }
    }

    /// The third component differs by one bf16 step if the normalized
    /// activation is rounded before multiplying the learned weight.
    func testLearnedRMSNormRoundsAfterWeight() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let input = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 4).asType(.bfloat16)
                let weight = MLXArray(Array(repeating: Float(1.0234375), count: 4))
                    .asType(.bfloat16)
                let norm = DeepseekV4RMSNorm(dimensions: 4, eps: 0)
                norm.update(parameters: ModuleParameters.unflattened(["weight": weight]))
                XCTAssertEqual(norm.parameters().flattened().map { $0.0 }, ["weight"])
                XCTAssertEqual(norm(input).dtype, .bfloat16)
                XCTAssertEqual(
                    norm(input).asType(.float32).asArray(Float.self),
                    [0.373046875, 0.74609375, 1.125, 1.4921875])
            }
        }
    }

    func testHyperConnectionPreservesDeepseekRMSNormInPrefillAndFusedDecode() throws {
        try Device.withDefaultDevice(.gpu) {
            let width = 1024
            let weight = MLXArray(Array(repeating: Float(1.0234375), count: width))
                .asType(.bfloat16)
            let norm = DeepseekV4RMSNorm(dimensions: width, eps: 0)
            norm.update(parameters: ModuleParameters.unflattened(["weight": weight]))
            norm.train(false)
            let connection = GLM5NextHyperConnection(
                hcMult: 4, sinkhornIterations: 1, hcEps: 0, normEps: 0,
                hiddenSize: width, normWeightInFloat32: true, normalizeMixAfterProjection: true)
            connection.train(false)
            let reference: [Float] = [0.373046875, 0.74609375, 1.125, 1.4921875]
            // Zero mixing weights give each of the four streams weight 0.5,
            // so the collapsed activation is exactly twice [1, 2, 3, 4].
            for length in [1, 9] {
                let input = MLXArray((0 ..< (length * 4 * width)).map { Float($0 % 4 + 1) })
                    .reshaped(1, length, 4, width).asType(.bfloat16)
                let expected = (0 ..< (length * width)).map { reference[$0 % 4] }
                for fusedDecode in [false, true] {
                    connection.fusedDecode = fusedDecode
                    let (actual, _, _) = connection.normalizedBranchInput(input, norm: norm)
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self), expected)
                }
                if length == 1 {
                    let mixes = try XCTUnwrap(
                        glm5NextHCMix(input, fn: connection.fn, eps: connection.normEps))
                    let fused = try XCTUnwrap(
                        glm5NextHCNorm(
                            input, mixes: mixes, scale: connection.scale, base: connection.base,
                            normWeight: weight, sinkhornIterations: 1, hcEps: 0, normEps: 0,
                            normWeightInFloat32: true))
                    XCTAssertEqual(fused.0.asType(.float32).asArray(Float.self), expected)
                    // Omitting the option retains the GLM kernel's original arithmetic.
                    let glm = try XCTUnwrap(
                        glm5NextHCNorm(
                            input, mixes: mixes, scale: connection.scale, base: connection.base,
                            normWeight: weight, sinkhornIterations: 1, hcEps: 0, normEps: 0))
                    let glmReference: [Float] = [0.373046875, 0.74609375, 1.1171875, 1.4921875]
                    XCTAssertEqual(
                        glm.0.asType(.float32).asArray(Float.self),
                        (0 ..< width).map { glmReference[$0 % 4] })
                }
            }
        }
    }

    private struct Oracle: Decodable {
        struct Case: Decodable {
            let indexer: String
            let indexBlock: Int
            let indexKeep: Int
            let prefillStep: Int
            let prompt: [Int]
            let greedy: [Int]
            let logprobs: [[Float]]

            enum CodingKeys: String, CodingKey {
                case indexer, prompt, greedy, logprobs
                case indexBlock = "index_block"
                case indexKeep = "index_keep"
                case prefillStep = "prefill_step"
            }
        }
        let decode: Int
        let cases: [Case]
    }

    private var fixture: URL {
        get throws {
            try XCTUnwrap(Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        }
    }

    private func loadTiny(indexBlock: Int, indexKeep: Int, float32: Bool = false) throws
        -> DeepseekV4Model
    {
        let dir = try fixture
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        config.indexBlock = indexBlock
        config.indexKeep = indexKeep
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        if float32 {
            let widened = model.parameters().flattened().map { key, value in
                (key, value.dtype == .bfloat16 ? value.asType(.float32) : value)
            }
            try model.update(parameters: ModuleParameters.unflattened(widened), verify: [.all])
        }
        model.train(false)
        return model
    }

    func testQuantizedMoEPrefillPreservesIndependentTokenOutputs() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            let width = model.layers[0].ffn.gate.weight.dim(1)
            let values: [Float] = (0 ..< (18 * width)).map { index in
                let centered = (index * 17) % 41 - 20
                return Float(centered) / Float(32)
            }
            let input = MLXArray(values).reshaped(2, 9, width).asType(.bfloat16)
            let ids = MLXArray((0 ..< 18).map { Int32($0 % model.vocabularySize) }).reshaped(2, 9)
            // Exercise both hash and score routing with repeated experts across batch rows.
            for layer in model.layers.prefix(2) {
                let projection = try XCTUnwrap(layer.ffn.switchMLP.gateProjection as? QuantizedSwitchLinear)
                XCTAssertEqual(projection.mode, .mxfp4)
                let prefill = layer.ffn(input, inputIDs: ids)
                let flatInput = input.reshaped(18, width)
                let flatIDs = ids.flattened()
                let independent = concatenated((0 ..< 18).map { row in
                    layer.ffn(flatInput[row ..< (row + 1)], inputIDs: flatIDs[row ..< (row + 1)])
                }, axis: 0).reshaped(input.shape)
                XCTAssertEqual(prefill.dtype, .bfloat16)
                XCTAssertEqual(prefill.shape, input.shape)
                let error = (prefill.asType(.float32) - independent.asType(.float32)).asArray(Float.self)
                let expected = independent.asType(.float32).asArray(Float.self)
                let squaredError = error.reduce(Float(0)) { $0 + $1 * $1 }
                let squaredNorm = expected.reduce(Float(0)) { $0 + $1 * $1 }
                XCTAssertLessThan(sqrt(squaredError / max(squaredNorm, 1e-20)), 0.01)
            }
        }
    }

    /// mlx-vlm `generate_step`: prompt chunks of `min(step, remaining - 1)`
    /// while more than one token is left, then the last token, then one
    /// token per step. Returns greedy tokens and per-step log-softmax.
    private func greedy(
        _ model: DeepseekV4Model, prompt: [Int], step: Int, count: Int
    ) throws -> ([Int], [[Float]]) {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            let chunk = MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n)
            eval(model(chunk, cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        var input = MLXArray(rest.map(Int32.init)).reshaped(1, 1)
        var tokens = [Int]()
        var logprobs = [[Float]]()
        for _ in 0 ..< count {
            let logits = model(input, cache: cache)[0, -1]
            let lp = (logits - logSumExp(logits, axis: -1, keepDims: true)).asType(.float32)
            let token = argMax(lp, axis: -1).item(Int.self)
            tokens.append(token)
            logprobs.append(lp.asArray(Float.self))
            input = MLXArray([Int32(token)]).reshaped(1, 1)
        }
        return (tokens, logprobs)
    }

    func testTinyCheckpointMatchesReferenceGreedyDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try fixture
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: dir.appendingPathComponent("oracle.json")))
            XCTAssertEqual(oracle.cases.count, 3)
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
            for item in oracle.cases {
                XCTAssertEqual(item.indexer, "flat")
                let (tokens, logprobs) = try greedy(
                    model, prompt: item.prompt, step: item.prefillStep, count: oracle.decode)
                let label = "\(item.indexer) prompt=\(item.prompt.count) step=\(item.prefillStep)"
                XCTAssertEqual(tokens, item.greedy, label)
                let diff = zip(logprobs, item.logprobs).map { mine, theirs in
                    zip(mine, theirs).map { abs($0 - $1) }.max() ?? 0
                }.max() ?? 0
                print("deepseek_v4 parity \(label): max |dlogprob| = \(diff)")
                // Float32 activations on both sides: equal up to reduction order
                // and the oracle's single-call prefill against this chunk plan.
                XCTAssertLessThan(diff, 1e-3, label)
            }
        }
    }

    /// The serving path (`TokenIterator`, whatever prefill step and chunking
    /// the caller asks for) keeps the final reference chunk intact and
    /// decodes the oracle's tokens.
    func testTokenIteratorUsesReferencePrefillPlan() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try fixture
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: dir.appendingPathComponent("oracle.json")))
            let cases = oracle.cases.filter {
                $0.prefillStep == 2048
            }
            XCTAssertEqual(cases.count, 1)
            for item in cases {
                let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
                for chunking in [PrefillParameters.Chunking.remainder, .balanced, .unchunked] {
                    let parameters = GenerateParameters(
                        maxTokens: oracle.decode, temperature: 0,
                        prefill: PrefillParameters(stepSize: 16, chunking: chunking))
                    var iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(item.prompt.map(Int32.init))),
                        model: model, parameters: parameters)
                    var tokens = [Int]()
                    while let token = iterator.next() { tokens.append(token) }
                    XCTAssertEqual(tokens, item.greedy, "\(item.indexer) \(chunking)")
                }
            }
        }
    }

    func testInferenceWeightPreparationPreservesLoadedModel() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            let attentions = model.layers.map { $0.attn }
            let count = attentions.filter {
                ($0.woA as? QuantizedMultiLinear)?.mode == .mxfp8
            }.count
            XCTAssertGreaterThan(count, 0)
            let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
            let expected = model(tokens, cache: try model.newCache(parameters: nil)).asArray(Float.self)
            model.prepareInferenceWeights()
            XCTAssertEqual(attentions.filter { $0.woA is DeepseekV4CachedMultiLinear }.count, count)
            XCTAssertEqual(model(tokens, cache: try model.newCache(parameters: nil)).asArray(Float.self), expected)
            // The load hook is idempotent and retains registered child modules.
            model.prepareInferenceWeights()
            XCTAssertEqual(attentions.filter { $0.woA is DeepseekV4CachedMultiLinear }.count, count)
            XCTAssertEqual(model(tokens, cache: try model.newCache(parameters: nil)).asArray(Float.self), expected)
        }
    }

    func testPrepareReturnsLastLogitsAndFullContextCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
            let tokens = MLXArray((0 ..< 96).map { Int32($0 % model.vocabularySize) })
            let referenceCache = try model.newCache(parameters: nil)
            let preparedCache = try model.newCache(parameters: nil)
            let expected = model(tokens.reshaped(1, 96), cache: referenceCache)
            let result = try model.prepare(
                LMInput(tokens: tokens), cache: preparedCache, state: nil,
                prefill: PrefillParameters(stepSize: 16))
            guard case .logits(let output) = result else {
                return XCTFail("The final complete reference chunk must return logits")
            }
            XCTAssertEqual(output.logits.shape, [1, 1, model.vocabularySize])
            XCTAssertEqual(output.logits.asArray(Float.self), expected[0..., (-1)..., 0...].asArray(Float.self))
            for (actual, expected) in zip(preparedCache, referenceCache) {
                XCTAssertEqual(actual.offset, expected.offset)
                for (a, b) in zip(actual.state, expected.state) {
                    XCTAssertEqual(a.asType(.float32).asArray(Float.self),
                        b.asType(.float32).asArray(Float.self))
                }
            }
        }
    }

    func testPoolCacheCompletesWindowsAcrossChunksAndDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = DeepseekV4PoolCache(ratio: 4)
            let rows = MLXArray(0 ..< 11).asType(.float32).reshaped(1, 11, 1)
            // A 6-token chunk completes one window and buffers two rows.
            var ready = cache.accumulateWindows(
                kv: rows[0..., 0 ..< 6], gate: rows[0..., 0 ..< 6], offset: 0)
            XCTAssertEqual(ready.kv.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(ready.base, 0)
            XCTAssertEqual(cache.remainder, 2)
            // A 3-token chunk joins the buffered rows: one window from 4.
            ready = cache.accumulateWindows(
                kv: rows[0..., 6 ..< 9], gate: rows[0..., 6 ..< 9], offset: 6)
            XCTAssertEqual(ready.kv.asArray(Float.self), [4, 5, 6, 7])
            XCTAssertEqual(ready.base, 4)
            XCTAssertEqual(cache.remainder, 1)
            _ = cache.appendPooled(MLXArray.zeros([1, 2, 1]))
            // Decode steps complete the next window on its fourth token.
            for t in 9 ..< 11 {
                ready = cache.accumulateWindows(
                    kv: rows[0..., t ..< (t + 1)], gate: rows[0..., t ..< (t + 1)], offset: t)
                XCTAssertEqual(ready.kv.dim(1), 0)
            }
            ready = cache.accumulateWindows(
                kv: MLXArray([Float(11)]).reshaped(1, 1, 1),
                gate: MLXArray([Float(11)]).reshaped(1, 1, 1), offset: 11)
            XCTAssertEqual(ready.kv.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(ready.base, 8)

            // Two pooled entries: visible to query t iff index < (t + 1) / 4.
            let mask = try XCTUnwrap(cache.poolMask(queries: 6, offset: 3))
            XCTAssertEqual(
                mask.asArray(Bool.self),
                [true, false, true, false, true, false, true, false, true, true, true, true])
            XCTAssertNil(cache.poolMask(queries: 1, offset: 12))

            let copy = cache.copy() as! DeepseekV4PoolCache
            XCTAssertEqual(copy.remainder, cache.remainder)
            XCTAssertEqual(copy.pooledCount, 2)
        }
    }

    private func servingGreedy(_ model: DeepseekV4Model, prompt: [Int], count: Int) throws -> [Int] {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
            parameters: GenerateParameters(maxTokens: count, temperature: 0))
        var tokens = [Int]()
        while let token = iterator.next() { tokens.append(token) }
        XCTAssertEqual(tokens.count, count)
        return tokens
    }

    private func prefetchedRow(
        _ model: DeepseekV4Model, prompt: [Int]
    ) throws -> (cache: [KVCache], firstToken: Int32) {
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters(stepSize: 16))
        let output: LMOutput?
        if case .logits(let logits) = result { output = logits } else { output = nil }
        let prepared = try XCTUnwrap(output, "DeepSeek V4 prepare must return full-prompt logits")
        let firstToken = argMax(prepared.logits[0, -1], axis: -1).item(Int32.self)
        for (layer, item) in zip(model.layers, cache) {
            XCTAssertEqual(
                DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local.offset, prompt.count)
        }
        return (cache, firstToken)
    }

    func testContinuousBatchGreedyParityAtOneTwoFourEightRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let lengths = [3, 6, 9, 13, 127, 128, 131, 137]
            let prompts = lengths.enumerated().map { row, count in
                (0 ..< count).map { ($0 * 7 + row * 11 + 2) % 128 }
            }
            for indexBlock in [0] {
                let model = try loadTiny(indexBlock: indexBlock, indexKeep: 2)
                let count = 16
                let serial = try prompts.map {
                    try servingGreedy(model, prompt: $0, count: count)
                }
                for rows in [1, 2, 4, 8] {
                    let prepared = try prompts.prefix(rows).map { try prefetchedRow(model, prompt: $0) }
                    let caches = prepared.map { $0.cache }
                    var inputs = prepared.map { $0.firstToken }
                    var tokens = inputs.map { [Int($0)] }
                    XCTAssertEqual(inputs.map(Int.init), serial.prefix(rows).map { $0[0] }, "first rows=\(rows)")
                    for _ in 1 ..< count {
                        let logits = model.decodeBatch(
                            MLXArray(inputs).reshaped(rows, 1), rowCaches: caches)
                        let next = argMax(logits, axis: -1).asArray(Int32.self)
                        for row in 0 ..< rows { tokens[row].append(Int(next[row])) }
                        inputs = next
                        eval(caches.flatMap { $0 })
                    }
                    XCTAssertEqual(tokens, Array(serial.prefix(rows)), "rows=\(rows) block=\(indexBlock)")
                }
            }
        }
    }

    func testContinuousBatchLongContextGreedyParityAtOneTwoFourEightRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let prompts = [31, 32, 33, 127, 128, 129, 131, 137].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 7 + row * 11 + 2) % 128 }
            }
            for indexBlock in [0] {
                let model = try loadTiny(indexBlock: indexBlock, indexKeep: 2)
                let serial = try prompts.map {
                    try servingGreedy(model, prompt: $0, count: 4)
                }
                for rows in [1, 2, 4, 8] {
                    let prepared = try prompts.prefix(rows).map { try prefetchedRow(model, prompt: $0) }
                    let caches = prepared.map { $0.cache }
                    var inputs = prepared.map { $0.firstToken }
                    XCTAssertEqual(inputs.map(Int.init), serial.prefix(rows).map { $0[0] }, "first long rows=\(rows)")
                    for step in 1 ..< 4 {
                        let next = argMax(
                            model.decodeBatch(MLXArray(inputs).reshaped(rows, 1), rowCaches: caches),
                            axis: -1).asArray(Int32.self)
                        XCTAssertEqual(
                            next.map(Int.init), serial.prefix(rows).map { $0[step] },
                            "long rows=\(rows) block=\(indexBlock) step=\(step)")
                        inputs = next
                        eval(caches.flatMap { $0 })
                    }
                }
            }
        }
    }

    func testContinuousBatchRowsJoinLeaveAndReorderWithoutChangingTokens() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            let prompts = [7, 31, 127, 130].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 3 + row * 17) % 128 }
            }
            let serial = try prompts.map {
                try servingGreedy(model, prompt: $0, count: 16)
            }
            var caches = [Int: [KVCache]]()
            var inputs = Array(repeating: Int32(0), count: prompts.count)
            var positions = Array(repeating: 0, count: prompts.count)
            let schedule = [[0, 1], [1, 0], [1, 0, 2], [2, 1], [3, 2], [0, 3, 2], [2], [3, 0]]
            for active in schedule {
                for row in active where caches[row] == nil {
                    let prepared = try prefetchedRow(model, prompt: prompts[row])
                    caches[row] = prepared.cache
                    inputs[row] = prepared.firstToken
                    XCTAssertEqual(Int(prepared.firstToken), serial[row][0], "first row=\(row)")
                    positions[row] = 1
                }
                let next = argMax(
                    model.decodeBatch(
                        MLXArray(active.map { inputs[$0] }).reshaped(active.count, 1),
                        rowCaches: active.map { caches[$0]! }), axis: -1).asArray(Int32.self)
                for (slot, row) in active.enumerated() {
                    XCTAssertEqual(Int(next[slot]), serial[row][positions[row]], "row=\(row)")
                    inputs[row] = next[slot]
                    positions[row] += 1
                }
                for row in caches.keys {
                    for (layer, item) in zip(model.layers, caches[row]!) {
                        XCTAssertEqual(
                            DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local.offset,
                            prompts[row].count + positions[row] - 1)
                    }
                }
            }
        }
    }

    func testBatchedRoPEKeepsPerRowOffsetsAndInverse() throws {
        Device.withDefaultDevice(.cpu) {
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            let offsets = [0, 3, 127, 128]
            let x = (MLXArray.arange(4 * 2 * 64, dtype: .float32) / 97).reshaped(4, 2, 1, 64)
            for inverse in [false, true] {
                let batched = rope(x, offsets: MLXArray(offsets.map(Int32.init)), inverse: inverse)
                let serial = concatenated(offsets.enumerated().map { row, offset in
                    rope(x[row ..< (row + 1)], offset: offset, inverse: inverse)
                }, axis: 0)
                XCTAssertEqual(batched.asArray(Float.self), serial.asArray(Float.self))
            }
        }
    }

    func testRoPEFloat32PhasesAtPositions257Through320() {
        Device.withDefaultDevice(.gpu) {
            let frequencies = MLXArray((0 ..< 32).map {
                Float(1) / Foundation.pow(Float(10000), Float(2 * $0) / 64)
            })
            let positions = MLXArray((257 ... 320).map(Int32.init)).asType(.float32)
            let angles = positions[0..., .newAxis] * frequencies[.newAxis, 0...]
            let (cosine, sine) = deepseekV4RoPEPhases(angles)
            XCTAssertEqual(cosine.dtype, .float32)
            XCTAssertEqual(sine.dtype, .float32)
            // NumPy float32 arange/outer/cos/sin; little-endian row-major phases.
            let values = cosine.asArray(Float.self) + sine.asArray(Float.self)
            let digest = values.withUnsafeBytes { SHA256.hash(data: Data($0)) }
            XCTAssertEqual(digest.map { String(format: "%02x", $0) }.joined(),
                "1e9c0f920fc12705248915b2a65cf99ea1b7f19564674e161775d87e744c13f3")

            // Layer-0 position 320/head 34/pair 3 cancels near a bf16 boundary.
            let rope = DeepseekV4RoPE(dims: 64, base: 10000, scaling: nil)
            var input = [Float](repeating: 0, count: 68)
            input[0] = 9
            input[10] = 0.51953125
            input[11] = 0.076171875
            let result = rope(MLXArray(input).reshaped(1, 1, 1, 68).asType(.bfloat16), offset: 320)
                .asType(.float32)
            XCTAssertEqual(result[0, 0, 0, 0].item(Float.self), 9)
            XCTAssertEqual(result[0, 0, 0, 11].item(Float.self), 9.715557098388672e-6)
        }
    }

    func testRoPEMatchesCanonicalFloat32Positions() {
        // NumPy float32 reference: plain base 10000, positions 192/2049,
        // pairs 0/1/19/31 of the final 64 channels, input pair [0.75, 1.25].
        let forward: [[Float]] = [
            [-0.25749433, -1.4348159, 1.2815117, 0.6947863,
                -0.38775349, 1.4052215, 0.71775317, 1.2687908],
            [-0.2051065, 1.4432364, -0.35503149, -1.4138433,
                -1.413795, -0.35522372, 0.38486242, 1.4060159],
        ]
        let inverse: [[Float]] = [
            [-1.1448402, -0.90240836, 0.009982467, 1.4577038,
                1.4223735, 0.3191452, 0.7817552, 1.2303897],
            [1.3699646, 0.49819374, -1.080435, -0.9786011,
                0.35188252, -1.4146302, 1.0594906, 1.0012391],
        ]
        let channels = [0, 1, 2, 3, 38, 39, 62, 63]
        let offsets = [192, 2049]
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let rope = DeepseekV4RoPE(dims: 64, base: 10000, scaling: nil)
                let values = (0 ..< (2 * 68)).map { index -> Float in
                    let channel = index % 68
                    return channel < 4 ? 9 : (channel % 2 == 0 ? 0.75 : 1.25)
                }
                let x = MLXArray(values).reshaped(2, 1, 1, 68)
                for (isInverse, expected) in [(false, forward), (true, inverse)] {
                    let result = rope(
                        x, offsets: MLXArray(offsets.map(Int32.init)), inverse: isInverse)
                    for row in 0 ..< 2 {
                        XCTAssertEqual(
                            result[row, 0, 0, ..<4].asArray(Float.self), [9, 9, 9, 9])
                        for (column, channel) in channels.enumerated() {
                            XCTAssertEqual(
                                result[row, 0, 0, channel + 4].item(Float.self),
                                expected[row][column], accuracy: 2e-6)
                        }
                        XCTAssertEqual(
                            result[row ..< (row + 1)].asArray(Float.self),
                            rope(x[row ..< (row + 1)], offset: offsets[row], inverse: isInverse)
                                .asArray(Float.self))
                    }
                }
            }
        }
    }

    func testRoPECompressorScaleKeepsRoundedOffsetsAndSequencePositions() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil, freqScale: 4)
                let x = MLXArray((0 ..< 32).map { Float($0 % 2 == 0 ? 0.75 : 1.25) })
                    .reshaped(1, 2, 16).asType(.bfloat16)
                // Offset 769 rounds down to pool position 192: physical positions 768/772.
                let forward = rope(x, offset: 769).asType(.float32)
                let inverse = rope(x, offsets: MLXArray([Int32(769)]), inverse: true)
                    .asType(.float32)
                XCTAssertEqual(
                    forward[0, 0, ..<4].asArray(Float.self),
                    [-1.1484375, 0.89453125, -1.4609375, 0.03955078125])
                XCTAssertEqual(
                    forward[0, 1, ..<4].asArray(Float.self),
                    [1.4296875, 0.287109375, -0.94140625, -1.109375])
                XCTAssertEqual(
                    inverse[0, 0, ..<4].asArray(Float.self),
                    [1.328125, -0.59765625, 0.72265625, -1.265625])
                XCTAssertEqual(
                    inverse[0, 1, ..<4].asArray(Float.self),
                    [-0.41796875, 1.3984375, -0.53515625, -1.3515625])
            }
        }
    }

    func testBatchedCompressorMatchesIndependentRaggedWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 4, indexKeep: 2)
            for layer in model.layers where layer.attn.ratio > 0 {
                let compressor = try XCTUnwrap(layer.attn.compressor)
                let ratio = layer.attn.ratio
                let starts = [0, ratio - 2, ratio - 1, ratio * 2 - 1]
                let batchedCaches = starts.map { _ in DeepseekV4PoolCache(ratio: ratio) }
                let serialCaches = starts.map { _ in DeepseekV4PoolCache(ratio: ratio) }
                for row in starts.indices where starts[row] > 0 {
                    let prefix = sin(MLXArray.arange(starts[row] * 64, dtype: .float32) / 17)
                        .reshaped(1, starts[row], 64)
                    _ = compressor(prefix, cache: batchedCaches[row], offset: 0)
                    _ = compressor(prefix, cache: serialCaches[row], offset: 0)
                }
                for step in 0 ..< 4 {
                    let x = cos(MLXArray.arange(starts.count * 64, dtype: .float32) / Float(step + 3))
                        .reshaped(starts.count, 1, 64)
                    let actual = compressor.decodeBatch(
                        x, caches: batchedCaches, offsets: starts.map { $0 + step })
                    for row in starts.indices {
                        let expected = compressor(
                            x[row ..< (row + 1)], cache: serialCaches[row], offset: starts[row] + step)
                        XCTAssertEqual(batchedCaches[row].remainder, serialCaches[row].remainder)
                        XCTAssertEqual(actual[row].shape, expected.shape)
                        if expected.size > 0 {
                            XCTAssertLessThan(
                                abs(actual[row] - expected).max().item(Float.self), 1e-5,
                                "ratio=\(ratio) row=\(row) step=\(step)")
                        }
                    }
                }
            }
        }
    }

    /// The pooled entries of a token stream do not depend on how it is split
    /// into calls: one call, uneven chunks, one token at a time, and a
    /// `copy()` / `state` snapshot resumed mid-stream all pool the same
    /// entries (`inference/model.py:309-310, 336-338, 353-360` carry the
    /// previous ratio-4 window across calls).
    func testCompressorPoolsAreChunkingInvariant() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            for layer in model.layers where layer.attn.ratio > 0 {
                let ratio = layer.attn.ratio
                var compressors = [try XCTUnwrap(layer.attn.compressor)]
                if let indexer = layer.attn.indexer { compressors.append(indexer.compressor) }
                let total = ratio == 4 ? 4 * 13 + 3 : 2 * 128 + 5
                let x = sin(MLXArray.arange(total * 64, dtype: .float32) / 7).reshaped(1, total, 64)
                    .asType(.bfloat16)
                for compressor in compressors {
                    func pool(_ chunks: [Int], snapshotAfter: Int? = nil) -> DeepseekV4PoolCache {
                        var cache = DeepseekV4PoolCache(ratio: ratio)
                        var offset = 0
                        for (i, n) in chunks.enumerated() {
                            _ = compressor(x[0..., offset ..< (offset + n)], cache: cache, offset: offset)
                            eval(cache.innerState())
                            offset += n
                            if i == snapshotAfter {
                                let restored = DeepseekV4PoolCache(ratio: ratio)
                                restored.state = (cache.copy() as! DeepseekV4PoolCache).state
                                restored.metaState = cache.metaState
                                cache = restored
                            }
                        }
                        XCTAssertEqual(offset, total)
                        return cache
                    }
                    let whole = pool([total])
                    let expected = try XCTUnwrap(whole.pooled)
                    XCTAssertEqual(expected.dim(1), total / ratio)
                    let uneven = ratio == 4 ? [3, 6, 1, 17, 2, 26] : [100, 1, 60, 100]
                    for (label, cache) in [
                        ("uneven", pool(uneven)),
                        ("one at a time", pool(Array(repeating: 1, count: total))),
                        ("snapshot", pool(uneven, snapshotAfter: 2)),
                    ] {
                        let actual = try XCTUnwrap(cache.pooled, label)
                        XCTAssertEqual(actual.shape, expected.shape, label)
                        let diff = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                            .item(Float.self)
                        XCTAssertLessThanOrEqual(diff, 1e-5, "ratio=\(ratio) \(label)")
                        XCTAssertEqual(cache.remainder, whole.remainder, label)
                    }
                }
            }
        }
    }

    /// The default indexer selects exactly the brute-force top `index_topk`
    /// of `sum_h relu(q_h . k) * scale * w_h / sqrt(H)` over every pooled
    /// entry, serially and in a ragged batch, at pool sizes of 2048 and more
    /// (where mlx-vlm's HISA filter used to replace it).
    func testIndexerSelectsExactTopKOverLargePools() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 2, "hidden_size": 64,
                 "q_lora_rank": 32, "qk_rope_head_dim": 16, "index_n_heads": 4,
                 "index_head_dim": 32, "index_topk": 512, "compress_ratios": [4, 4]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            try config.validateModelConfiguration()
            XCTAssertEqual(config.indexBlock, 0)
            MLXRandom.seed(2111)
            let indexer = DeepseekV4Indexer(config, ratio: 4)
            // The artifact stores this LinearF32 weight in bf16. Keeping the
            // random initializer's float32 weight would hide the premature
            // bf16 projection that this regression must detect.
            indexer.weightsProj.update(parameters: ModuleParameters.unflattened([
                "weight": indexer.weightsProj.weight.asType(.bfloat16),
            ]))
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            func bruteForce(x: MLXArray, qResidual: MLXArray, pool: MLXArray, offset: Int) -> Set<Int32> {
                let q = rope(
                    indexer.wqB(qResidual).reshaped(1, 1, 4, 32).transposed(0, 2, 1, 3), offset: offset)
                    .asType(.float32)
                // Independent reference LinearF32: widening after a bf16
                // projection has already lost the routing coefficients.
                let raw = matmul(x.asType(.float32), indexer.weightsProj.weight.asType(.float32).T)
                let w = raw.reshaped(4, 1, 1) * (indexer.scale * pow(Float(4), -0.5))
                let dots = maximum(matmul(q[0], pool[0].asType(.float32).T), 0)  // [H, 1, P]
                let scores = (dots * w).sum(axis: 0).reshaped(-1)
                let k = min(512, pool.dim(1))
                return Set(argSort(-scores)[..<k].asArray(Int32.self))
            }
            var pools = [MLXArray]()
            var inputs = [(MLXArray, MLXArray, Int)]()
            for count in [2048, 3001, 4096] {
                let pool = MLXRandom.normal([1, count, 32]).asType(.bfloat16)
                let x = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
                let qResidual = MLXRandom.normal([1, 1, 32]).asType(.bfloat16)
                let offset = 4 * count + 1
                let cache = DeepseekV4PoolCache(ratio: 4)
                _ = cache.accumulateWindows(
                    kv: MLXArray.zeros([1, 2, 64], dtype: .bfloat16),
                    gate: MLXArray.zeros([1, 2, 64], dtype: .bfloat16), offset: offset - 2)
                _ = cache.appendPooled(pool)
                let expected = bruteForce(x: x, qResidual: qResidual, pool: pool, offset: offset)
                XCTAssertEqual(expected.count, 512)
                let serial = try XCTUnwrap(
                    indexer(x, qResidual: qResidual, rope: rope, cache: cache, offset: offset))
                XCTAssertEqual(Set(serial.reshaped(-1).asArray(Int32.self)), expected, "serial P=\(count)")
                pools.append(pool)
                inputs.append((x, qResidual, offset))
            }
            let batched = indexer.decodeBatch(
                concatenated(inputs.map { $0.0 }, axis: 0),
                qResidual: concatenated(inputs.map { $0.1 }, axis: 0), rope: rope, pools: pools,
                offsets: MLXArray(inputs.map { Int32($0.2) }))
            for row in pools.indices {
                let expected = bruteForce(
                    x: inputs[row].0, qResidual: inputs[row].1, pool: pools[row], offset: inputs[row].2)
                XCTAssertEqual(
                    Set(batched[row].reshaped(-1).asArray(Int32.self)), expected, "batched row=\(row)")
            }
        }
    }

    /// A prompt chunk's exact selection is the same whether its float32
    /// scores fit one workspace or are tiled over queries.
    func testIndexerPrefillTilesKeepTheExactSelection() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 2, "hidden_size": 64,
                 "q_lora_rank": 32, "qk_rope_head_dim": 16, "index_n_heads": 4,
                 "index_head_dim": 32, "index_topk": 8, "compress_ratios": [4, 4]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            MLXRandom.seed(2151)
            let indexer = DeepseekV4Indexer(config, ratio: 4)
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            let (L, offset) = (40, 4 * 300)
            let pool = MLXRandom.normal([1, 300, 32]).asType(.bfloat16)
            let x = MLXRandom.normal([1, L, 64]).asType(.bfloat16)
            let qResidual = MLXRandom.normal([1, L, 32]).asType(.bfloat16)
            func run(workspace: Int) throws -> MLXArray {
                indexer.scoreWorkspaceBytes = workspace
                let cache = DeepseekV4PoolCache(ratio: 4)
                _ = cache.accumulateWindows(
                    kv: MLXArray.zeros([1, 2, 64], dtype: .bfloat16),
                    gate: MLXArray.zeros([1, 2, 64], dtype: .bfloat16), offset: offset - 2)
                _ = cache.appendPooled(pool)
                return try XCTUnwrap(
                    indexer(x, qResidual: qResidual, rope: rope, cache: cache, offset: offset))
            }
            let whole = try run(workspace: 512 << 20)
            let tiled = try run(workspace: 4 * 4 * 310 * 4 * 7)  // 7 queries per tile
            XCTAssertEqual(whole.shape, [1, L, 8])
            for t in 0 ..< L {
                XCTAssertEqual(
                    Set(whole[0, t].asArray(Int32.self)), Set(tiled[0, t].asArray(Int32.self)),
                    "query \(t)")
            }
        }
    }

    /// DeepSeek's kernel rounds the unnormalized weights against the running
    /// max of each 64-key block, then rescales the float32 accumulator when a
    /// later block raises the max. A query whose first block holds a value of
    /// 128 at score 0 and whose second block holds -47 at score 1 keeps
    /// 128 * exp(-1) - 47; rounding exp(-1) to bf16 (47/128) against the final
    /// max would cancel it to 0.
    func testAttentionRoundsWeightsPerRunningMaxBlock() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 1, "hidden_size": 64,
                 "num_attention_heads": 2, "o_groups": 2, "head_dim": 64, "qk_rope_head_dim": 16,
                 "q_lora_rank": 32, "o_lora_rank": 8, "sliding_window": 128, "compress_ratios": [0]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            let attention = DeepseekV4Attention(config, layerIdx: 0)
            // scale 1/8: q = 8 e0 gives scores k[0].
            var keys = [[Float]](repeating: [Float](repeating: 0, count: 64), count: 128)
            keys[0][1] = 128
            keys[64][0] = 1
            keys[64][1] = -47
            let k = MLXArray(keys.flatMap { $0 }).reshaped(1, 1, 128, 64)
            var query = [Float](repeating: 0, count: 64)
            query[0] = 8
            let q = broadcast(MLXArray(query).reshaped(1, 1, 1, 64), to: [1, 2, 1, 64])
            let out = attention.referenceAttention(
                q, keys: k, valid: MLXArray.ones([1, 1, 128], dtype: .bool))
            // Block 0: max 0, weights 1, accumulator 128. Block 1 raises the max to 1:
            // the accumulator rescales by e^-1 and adds -47; 128 + 1 weights and
            // the sink (e^-1) form the normalizer.
            let e = Float(Foundation.exp(-1.0))
            let expected = (128 * e - 47) / (1 + 128 * e)
            XCTAssertEqual(out[0, 0, 0, 1].item(Float.self), expected, accuracy: 1e-6)
            XCTAssertGreaterThan(out[0, 0, 0, 1].item(Float.self), 1e-3)
        }
    }

    /// The float32 linear (compressor projections, head) on the GPU: the
    /// bfloat16-weight kernel equals a float32 matmul over the widened weight
    /// up to summation order, and a row's result does not depend on how many
    /// rows share the call.
    func testFloat32LinearKernelMatchesWidenedWeight() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        MLXRandom.seed(272)
        for (K, N) in [(64, 129), (4096, 1000)] {
            let weight = MLXRandom.normal([N, K]).asType(.bfloat16)
            let layer = Linear(weight: weight)
            let x = MLXRandom.normal([8, 1, K]).asType(.bfloat16)
            let expected = matmul(x.asType(.float32), weight.asType(.float32).T)
            let full = deepseekV4LinearF32(layer, x)
            XCTAssertEqual(full.dtype, .float32)
            XCTAssertEqual(full.shape, [8, 1, N])
            let scale = abs(expected).max().item(Float.self)
            XCTAssertLessThan(
                abs(full - expected).max().item(Float.self) / scale, 1e-5, "K=\(K) N=\(N)")
            for rows in 1 ... 7 {
                let part = deepseekV4LinearF32(layer, x[..<rows])
                XCTAssertEqual(
                    part.asArray(Float.self), full[..<rows].asArray(Float.self), "rows=\(rows)")
            }
        }
    }

    func testContinuousBatchUsesKeysOnlyWindowsAndIndependentPoolSnapshots() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            XCTAssertTrue(model.batchesExactly)
            XCTAssertFalse(try loadTiny(indexBlock: 4, indexKeep: 2).batchesExactly)
            let prompts = [1, 3, 31, 32, 33, 127, 128, 129].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 5 + row * 7) % 128 }
            }
            let serial = try prompts.map {
                try servingGreedy(model, prompt: $0, count: 4)
            }
            let prepared = try prompts.map { try prefetchedRow(model, prompt: $0) }
            let caches = prepared.map { $0.cache }
            var inputs = prepared.map { $0.firstToken }
            XCTAssertEqual(inputs.map(Int.init), serial.map { $0[0] })
            for step in 1 ..< 4 {
                let next = argMax(
                    model.decodeBatch(MLXArray(inputs).reshaped(8, 1), rowCaches: caches), axis: -1)
                    .asArray(Int32.self)
                XCTAssertEqual(next.map(Int.init), serial.map { $0[step] })
                for row in caches {
                    for (layer, item) in zip(model.layers, row) {
                        let local = DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local
                        XCTAssertEqual(local.state.count, 2)
                        XCTAssertEqual(local.state[1].dim(-1), 0)
                    }
                }
                inputs = next
            }

            let cache = DeepseekV4PoolCache(ratio: 4)
            _ = cache.accumulateWindows(
                kv: MLXArray.zeros([1, 1, 1]), gate: MLXArray.zeros([1, 1, 1]), offset: 0)
            _ = cache.appendPooled(MLXArray([Float(2), 3]).reshaped(1, 2, 1))
            let snapshot = cache.copy() as! DeepseekV4PoolCache
            _ = cache.appendPooled(MLXArray([Float(4)]).reshaped(1, 1, 1))
            XCTAssertEqual(snapshot.pooled!.asArray(Float.self), [2, 3])
            XCTAssertEqual(cache.pooled!.asArray(Float.self), [2, 3, 4])
            XCTAssertEqual(cache.state.last!.dim(1), 3)
            let restored = DeepseekV4PoolCache(ratio: 4)
            restored.state = cache.state
            restored.metaState = cache.metaState
            _ = restored.appendPooled(MLXArray([Float(5)]).reshaped(1, 1, 1))
            XCTAssertEqual(restored.pooled!.asArray(Float.self), [2, 3, 4, 5])
            XCTAssertEqual(cache.pooled!.asArray(Float.self), [2, 3, 4])
            _ = restored.appendPooled(MLXArray(6 ..< 25).asType(.float32).reshaped(1, 19, 1))
            XCTAssertEqual(restored.pooledCount, 23)
            XCTAssertEqual(restored.pooled!.asArray(Float.self), (2 ..< 25).map(Float.init))
            XCTAssertEqual(restored.state.last!.dim(1), 23)
            XCTAssertEqual(snapshot.pooled!.asArray(Float.self), [2, 3])
        }
    }

    func testPoolCacheIncompleteMultiTokenContinuationPreservesBufferedPrefix() throws {
        try Device.withDefaultDevice(.cpu) {
            for ratio in [4, 128] {
                let cache = DeepseekV4PoolCache(ratio: ratio)
                let values = MLXArray(0 ..< ratio).asType(.float32).reshaped(1, ratio, 1)
                let gates = values + 1000
                _ = cache.accumulateWindows(
                    kv: values[0..., ..<1], gate: gates[0..., ..<1], offset: 0)
                var ready = cache.accumulateWindows(
                    kv: values[0..., 1 ..< 3], gate: gates[0..., 1 ..< 3], offset: 1)
                XCTAssertEqual(ready.kv.dim(1), 0)
                XCTAssertEqual(cache.remainder, 3)
                ready = cache.accumulateWindows(
                    kv: values[0..., 3...], gate: gates[0..., 3...], offset: 3)
                XCTAssertEqual(ready.kv.asArray(Float.self), (0 ..< ratio).map(Float.init))
                XCTAssertEqual(ready.gate.asArray(Float.self), (0 ..< ratio).map { Float($0 + 1000) })
                XCTAssertEqual(ready.base, 0)
                XCTAssertEqual(cache.remainder, 0)
            }
        }
    }

    func testArtifactConfigurationTakesOneRatioPerLayer() throws {
        let ratios = [0, 0] + (0 ..< 20).flatMap { _ in [4, 128] } + [4, 0, 0, 0]
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 43,
             "compress_ratios": \(ratios), "num_hash_layers": 3,
             "rope_scaling": {"type": "yarn", "factor": 16,
               "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.compressRatios.count, 43)
        XCTAssertEqual(config.compressRatios.filter { $0 == 4 }.count, 21)
        XCTAssertEqual(config.compressRatios.filter { $0 == 128 }.count, 20)
        XCTAssertEqual(config.compressRatios.last, 4)
        // The artifact sets neither HISA key: exact flat top-k, as DeepSeek's reference.
        XCTAssertEqual(config.indexBlock, 0)
        XCTAssertEqual(config.indexKeep, 0)

        var bad = config
        bad.compressRatios[5] = 2
        XCTAssertThrowsError(try bad.validateModelConfiguration())
    }

    func testSanitizeMapsArtifactLayout() throws {
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 1, "hidden_size": 64,
             "n_routed_experts": 2, "o_groups": 2, "o_lora_rank": 4, "compress_ratios": [0]}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        let weights: [String: MLXArray] = [
            "layers.0.ffn.experts.0.w1.weight": MLXArray.zeros([3, 5]),
            "layers.0.ffn.experts.1.w1.weight": MLXArray.ones([3, 5]),
            "layers.0.ffn.experts.0.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.ffn.experts.1.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.attn.wo_a.weight": MLXArray.zeros([8, 6]),
            "layers.0.hc_attn_fn": MLXArray.zeros([2]),
            "layers.0.hc_ffn_scale": MLXArray.zeros([3]),
            "layers.0.ffn.gate.tid2eid": MLXArray([Int64(1), 0]),
            "layers.0.ffn.gate.weight": MLXArray.zeros([2, 64], dtype: .bfloat16),
            "layers.0.ffn.shared_experts.w3.weight": MLXArray.zeros([2]),
            "layers.1.attn.wkv.weight": MLXArray.zeros([2]),
            "mtp.0.block.attn.wkv.weight": MLXArray.zeros([2]),
            "hc_head_fn": MLXArray.zeros([2]),
            "embed.weight": MLXArray.zeros([2]),
        ]
        let out = DeepseekV4Weights.sanitize(weights, configuration: config)
        XCTAssertEqual(
            Set(out.keys),
            [
                "layers.0.ffn.switch_mlp.gate_proj.weight", "layers.0.ffn.switch_mlp.down_proj.scales",
                "layers.0.attn.wo_a.weight", "layers.0.attn_hc.fn", "layers.0.ffn_hc.scale",
                "layers.0.ffn.gate.tid2eid", "layers.0.ffn.gate.weight",
                "layers.0.ffn.shared_experts.w3.weight",
                "hc_head.fn", "embed.weight",
            ])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]!.shape, [2, 3, 5])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]![1].sum().item(Float.self), 15)
        XCTAssertEqual(out["layers.0.attn.wo_a.weight"]!.shape, [2, 4, 6])
        XCTAssertEqual(out["layers.0.ffn.gate.tid2eid"]!.dtype, .int32)
        XCTAssertEqual(out["layers.0.ffn.gate.weight"]!.dtype, .float32)
    }
    /// The real artifact's tensor names, shapes and dtypes (from the
    /// safetensors headers) against the module tree after sanitize and the
    /// config's per-layer quantization. Placeholders are lazy zeros, so no
    /// weight is read or materialized. Set `DSV4_ARTIFACT_DIR` to the
    /// V4-Flash-0731 MXFP4 directory.
    func testArtifactLayoutMatchesModuleTree() throws {
        guard let path = ProcessInfo.processInfo.environment["DSV4_ARTIFACT_DIR"] else {
            throw XCTSkip("DSV4_ARTIFACT_DIR is not set")
        }
        try Device.withDefaultDevice(.cpu) {
            let dir = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let dtypes: [String: DType] = [
                "BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32, "I64": .int64,
                "I32": .int32,
            ]
            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(atPath: path)
                .filter { $0.hasSuffix(".safetensors") }
            XCTAssertFalse(files.isEmpty)
            for file in files {
                let handle = try FileHandle(forReadingFrom: dir.appendingPathComponent(file))
                defer { try? handle.close() }
                let size = try XCTUnwrap(handle.read(upToCount: 8)).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt64.self)
                }
                let header = try XCTUnwrap(handle.read(upToCount: Int(size)))
                let entries = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: header) as? [String: Any])
                for (name, value) in entries where name != "__metadata__" {
                    let entry = try XCTUnwrap(value as? [String: Any])
                    let shape = try XCTUnwrap(entry["shape"] as? [Int])
                    let dtype = try XCTUnwrap(dtypes[try XCTUnwrap(entry["dtype"] as? String)])
                    weights[name] = MLXArray.zeros(shape, dtype: dtype)
                }
            }
            let model = DeepseekV4Model(config)
            let sanitized = model.sanitize(weights: weights)
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? base.perLayerQuantization?.quantization(layer: path)?.asTuple : nil
            }
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            let modes = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    key, module -> (String, QuantizationMode)? in
                    switch module {
                    case let q as QuantizedLinear: (key, q.mode)
                    case let q as QuantizedSwitchLinear: (key, q.mode)
                    case let q as QuantizedMultiLinear: (key, q.mode)
                    default: nil
                    }
                })
            XCTAssertEqual(modes["layers.2.ffn.switch_mlp.gate_proj"], .mxfp4)
            XCTAssertEqual(modes["layers.2.ffn.shared_experts.w1"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.wo_a"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.indexer.wq_b"], .mxfp8)
            XCTAssertNil(modes["layers.2.attn.compressor.wkv"])
            XCTAssertNil(modes["head"])
        }
    }
}
