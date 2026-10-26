import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class PrismGDNDecodePreworkTests: XCTestCase {
    private func assertIdentical(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let actualBits = actual.asType(.float32).asArray(Float.self).map(\.bitPattern)
        let expectedBits = expected.asType(.float32).asArray(Float.self).map(\.bitPattern)
        let mismatches = zip(actualBits, expectedBits).enumerated().filter { $0.element.0 != $0.element.1 }
        let first = mismatches.first.map {
            "first at \($0.offset): actual=\($0.element.0), expected=\($0.element.1)"
        } ?? "none"
        XCTAssertEqual(actualBits.count, expectedBits.count, file: file, line: line)
        XCTAssertEqual(mismatches.count, 0, "\(mismatches.count) bit mismatches; \(first)", file: file, line: line)
    }

    private func configuration() throws -> Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self,
            from: Data("""
                {"hidden_size":32,"linear_num_value_heads":32,"linear_num_key_heads":16,
                 "linear_key_head_dim":128,"linear_value_head_dim":128,
                 "linear_conv_kernel_dim":4}
                """.utf8))
    }

    private func model(_ dtype: DType) throws -> Qwen35Language.GatedDeltaNet {
        let result = Qwen35Language.GatedDeltaNet(try configuration())
        result.update(parameters: result.parameters().mapValues { $0.asType(dtype) })
        result.conv1d.update(parameters: result.conv1d.parameters().mapValues { $0.asType(.float32) })
        result.train(false)
        return result
    }

    private func cache(_ model: Qwen35Language.GatedDeltaNet, _ dtype: DType) -> MambaCache {
        let result = MambaCache()
        result[0] = MLXRandom.normal([1, 3, model.convDim]).asType(dtype)
        result[1] = MLXRandom.normal([1, model.numVHeads, 128, 128])
        return result
    }

    func testMetalConvMatchesOriginalAndReadsCurrentWeights() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU parity test")
        for dtype in [DType.float16, .float32] {
            // C8192 is the 4B artifact width: 2 * 16 * 128 + 32 * 128.
            for channels in [384, 8192] {
                for seed in [11, 29, 71] {
                    withRandomState(MLXRandom.RandomState(seed: UInt64(seed))) {
                        let conv = Conv1d(
                            inputChannels: channels, outputChannels: channels, kernelSize: 4,
                            groups: channels, bias: false)
                        for scale: Float in [0, 0.0001, 1, 100] {
                            let qkv = (MLXRandom.normal([1, 1, channels]) * scale).asType(dtype)
                            let state = (MLXRandom.normal([1, 3, channels]) * scale).asType(dtype)
                            // Replacing weights after each trace must change its explicit input.
                            conv.update(parameters: ModuleParameters.unflattened([
                                ("weight", MLXRandom.normal([channels, 4, 1]))
                            ]))
                            let got = prismConv4Decode(qkv: qkv, state: state, weight: conv.weight)
                            assertIdentical(got.conv, conv(concatenated([state, qkv], axis: 1)))
                            assertIdentical(got.activated, silu(conv(concatenated([state, qkv], axis: 1))))
                            assertIdentical(
                                got.state, contiguous(concatenated([state[0..., 1..., 0...], qkv], axis: 1)))
                        }
                        let signs = broadcast(
                            MLXArray([Float(1), -1, 1, -1]).reshaped(1, 4, 1),
                            to: [channels, 4, 1])
                        conv.update(parameters: ModuleParameters.unflattened([("weight", signs)]))
                        let state = MLXArray.ones([1, 3, channels], dtype: dtype)
                        let qkv = MLXArray.ones([1, 1, channels], dtype: dtype)
                        assertIdentical(
                            prismConv4Decode(qkv: qkv, state: state, weight: conv.weight).activated,
                            silu(conv(concatenated([state, qkv], axis: 1))))
                    }
                }
            }
        }
    }

    func testSignedZeroAndUnderflowMatchConvolution() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU signed-zero parity test")
        let channels = 384
        for dtype in [DType.float16, .float32] {
            let tiny: Float = dtype == .float16 ? 0.000000059604645 : 1e-20
            var stateValues = [Float](repeating: 0, count: 3 * channels)
            var qkvValues = [Float](repeating: 0, count: channels)
            var weights = [Float](repeating: 1, count: 4 * channels)
            for channel in 0..<channels {
                for tap in 0..<3 {
                    stateValues[tap * channels + channel] =
                        channel % 4 == 1 || (channel % 4 == 2 && tap % 2 == 0) ? -Float.zero : 0
                }
                qkvValues[channel] = channel % 4 == 3 ? -tiny : -Float.zero
                if channel % 4 == 3 { weights[channel * 4 + 3] = tiny }
            }
            let state = MLXArray(stateValues).reshaped(1, 3, channels).asType(dtype)
            let qkv = MLXArray(qkvValues).reshaped(1, 1, channels).asType(dtype)
            let conv = Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: 4,
                groups: channels, bias: false)
            conv.update(parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(weights).reshaped(channels, 4, 1))
            ]))
            let got = prismConv4Decode(qkv: qkv, state: state, weight: conv.weight)
            let rawExpected = conv(concatenated([state, qkv], axis: 1))
            assertIdentical(got.conv, rawExpected)
            let expected = silu(rawExpected)
            assertIdentical(got.activated, expected)
            assertIdentical(got.state, contiguous(concatenated([state[0..., 1..., 0...], qkv], axis: 1)))
        }
    }

    func testOnlyPrismConstructorEnablesConvDecode() throws {
        let data = Data("""
            {"model_type":"prism_hadamard_qwen35","schema_version":2,
             "base_model_type":"qwen3_5","tensor_namespace":"mlx-vlm-qwen3_5",
             "gdn_activation_layout":"grouped",
             "quantization":{"bits":2,"group_size":128,"mode":"affine"},
             "modules":[{"path":"model.embed_tokens","block":512,
                         "embedding":true,"dtype":"float16"}],
             "text_config":{"model_type":"qwen3_5_text","hidden_size":512,
               "num_hidden_layers":2,"intermediate_size":512,"num_attention_heads":1,
               "num_key_value_heads":1,"head_dim":128,"linear_num_value_heads":1,
               "linear_num_key_heads":1,"linear_key_head_dim":128,
               "linear_value_head_dim":128,"linear_conv_kernel_dim":4,
               "vocab_size":4,"full_attention_interval":2,"tie_word_embeddings":true},
             "vision_config":{"model_type":"qwen3_5","depth":1,"hidden_size":8,
               "intermediate_size":16,"out_hidden_size":512,"num_heads":1,
               "patch_size":16,"spatial_merge_size":1,"temporal_patch_size":1,
               "num_position_embeddings":8}}
            """.utf8)
        let config = try JSONDecoder().decode(PrismHadamardQwen35Configuration.self, from: data)
        let prism = try PrismHadamardQwen35(config)
        let ordinary = Qwen35(config.base)
        let prismLayers = prism.loraLayers.compactMap { $0 as? Qwen35Language.DecoderLayer }
        let ordinaryLayers = ordinary.loraLayers.compactMap { $0 as? Qwen35Language.DecoderLayer }
        XCTAssertEqual(prismLayers.compactMap(\.linearAttn).count, 1)
        XCTAssertTrue(prismLayers.compactMap(\.linearAttn).allSatisfy(\.prismConvDecodeEnabled))
        XCTAssertTrue(ordinaryLayers.compactMap(\.linearAttn).allSatisfy { !$0.prismConvDecodeEnabled })
    }

    func testMixedSourceTypesAndStridedInputsMatchConvolution() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU mixed-type parity test")
        let channels = 8192
        for (qkvType, stateType) in [(DType.float16, DType.float32), (.float32, .float16)] {
            let qkv = MLXRandom.normal([1, 1, channels * 2]).asType(qkvType)[0..., 0..., 0..<channels]
            let state = MLXRandom.normal([1, 3, channels * 2]).asType(stateType)[0..., 0..., 0..<channels]
            let conv = Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: 4,
                groups: channels, bias: false)
            let got = prismConv4Decode(qkv: qkv, state: state, weight: conv.weight)
            let expected = conv(concatenated([state, qkv], axis: 1))
            assertIdentical(got.conv, expected)
            assertIdentical(got.activated, silu(expected))
            assertIdentical(got.state, contiguous(concatenated([state[0..., 1..., 0...], qkv], axis: 1)))
        }
    }

    #if DEBUG
        func testRealArtifactConvDecodeIsAdmittedForAllGDNLayerIDs() throws {
            try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU artifact admission test")
            guard let path = ProcessInfo.processInfo.environment["MLX_PRISM_PARITY_FIXTURE"],
                !path.isEmpty
            else { throw XCTSkip("MLX_PRISM_PARITY_FIXTURE is not set") }
            struct Case: Decodable {
                let inputIDs: [Int32]
                enum CodingKeys: String, CodingKey { case inputIDs = "input_ids" }
            }
            struct Fixture: Decodable {
                let modelDirectory: String
                let cases: [Case]
                enum CodingKeys: String, CodingKey {
                    case modelDirectory = "model_directory"
                    case cases
                }
            }
            let decoder = JSONDecoder()
            let fixture = try decoder.decode(
                Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            let item = try XCTUnwrap(fixture.cases.first)
            XCTAssertFalse(item.inputIDs.isEmpty)
            let directory = URL(fileURLWithPath: fixture.modelDirectory, isDirectory: true)
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let config = try decoder.decode(PrismHadamardQwen35Configuration.self, from: data)
            XCTAssertEqual(config.base.textConfiguration.hiddenLayers, 32)
            XCTAssertEqual(config.base.textConfiguration.fullAttentionInterval, 4)
            let base = try decoder.decode(BaseConfiguration.self, from: data)
            let model = try PrismHadamardQwen35(config)
            try loadWeights(
                modelDirectory: directory, model: model,
                quantization: base.perLayerQuantization?.quantization,
                perLayerQuantization: base.perLayerQuantization)
            model.train(false)
            let cache = try model.newCache(parameters: nil)
            let prepared = try model.prepare(
                LMInput(tokens: MLXArray(item.inputIDs).expandedDimensions(axis: 0)),
                cache: cache, state: nil, prefill: PrefillParameters(stepSize: 512))
            let output: LMOutput
            switch prepared {
            case .logits(let result): output = result
            case .tokens(let remaining): output = model(remaining, cache: cache, state: nil)
            }
            let expectedIDs = Set((0..<32).filter { ($0 + 1) % 4 != 0 })
            XCTAssertEqual(expectedIDs.count, 24)
            var observations = [Int: PrismConvDecodeObservation]()
            var installedIDs = Set<Int>()
            for (id, module) in model.loraLayers.enumerated() {
                guard let layer = module as? Qwen35Language.DecoderLayer,
                    let gdn = layer.linearAttn
                else { continue }
                installedIDs.insert(id)
                gdn.prismConvDecodeObserver = { observations[id] = $0 }
            }
            XCTAssertEqual(installedIDs, expectedIDs)
            let token = argMax(output.logits[0..., -1, 0...], axis: -1).item(Int.self)
            let decoded = model(
                LMInput.Text(tokens: MLXArray([Int32(token)]).reshaped(1, 1)),
                cache: cache, state: output.state)
            eval(decoded.logits)
            XCTAssertEqual(Set(observations.keys), expectedIDs)
            for id in expectedIDs.sorted() {
                let observed = try XCTUnwrap(observations[id], "Missing GDN layer \(id)")
                XCTAssertTrue(observed.admitted, "GDN layer \(id)")
                XCTAssertEqual(observed.weightType, .float32, "GDN layer \(id)")
                XCTAssertTrue([DType.float16, .float32].contains(observed.qkvType))
                XCTAssertTrue([DType.float16, .float32].contains(observed.stateType))
                let state = try XCTUnwrap(cache[id] as? MambaCache)
                let promoted: DType = observed.qkvType == .float32 || observed.stateType == .float32
                    ? .float32 : .float16
                XCTAssertEqual(state[0]?.dtype, promoted, "GDN layer \(id) next-state promotion")
            }
            print("Prism conv decode: \(observations.values.filter(\.admitted).count)/24 GDN layers admitted")
        }
    #endif

    func testMultiStepOutputAndBothStatesMatchUnchangedPath() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU parity test")
        for dtype in [DType.float16, .float32] {
            let reference = try model(dtype)
            let candidate = try model(dtype)
            try candidate.update(parameters: reference.parameters(), verify: [.all])
            candidate.prismConvDecodeEnabled = true
            let expectedCache = cache(reference, dtype)
            let actualCache = MambaCache()
            actualCache[0] = expectedCache[0]
            actualCache[1] = expectedCache[1]
            for step in 0..<6 {
                if step == 2 {
                    let replacement = Conv1d(
                        inputChannels: reference.convDim, outputChannels: reference.convDim,
                        kernelSize: 4, groups: reference.convDim, bias: false)
                    replacement.update(parameters: replacement.parameters().mapValues { $0.asType(.float32) })
                    try reference.updateModule(key: "conv1d", replacement)
                    try candidate.updateModule(key: "conv1d", replacement)
                }
                _ = try candidate.prepareFusedInputProjection()
                let input = MLXRandom.normal([1, 1, reference.hiddenSize]).asType(dtype)
                let projected = candidate.projectInputs(input, batch: 1, sequence: 1).qkv
                XCTAssertTrue(candidate.canUsePrismConvDecode(
                    inputs: input, qkv: projected, convState: actualCache[0]!,
                    mask: nil, cache: actualCache, checkpointAfter: nil))
                assertIdentical(candidate(input, cache: actualCache), reference(input, cache: expectedCache))
                assertIdentical(actualCache[0]!, expectedCache[0]!)
                assertIdentical(actualCache[1]!, expectedCache[1]!)
            }
        }
    }

    func testEligibilityRejectsUnsupportedContextsWithoutCacheMutation() throws {
        let gdn = try model(.float16)
        let state = cache(gdn, .float16)
        let input = MLXArray.zeros([1, 1, gdn.hiddenSize], dtype: .float16)
        let qkv = MLXArray.zeros([1, 1, gdn.convDim], dtype: .float16)
        let before = state.state
        func eligible(_ mask: MLXArray? = nil, checkpoint: Int? = nil) -> Bool {
            gdn.canUsePrismConvDecode(
                inputs: input, qkv: qkv, convState: state[0]!,
                mask: mask, cache: state, checkpointAfter: checkpoint)
        }
        XCTAssertFalse(eligible()) // Ordinary Qwen never opts in.
        gdn.prismConvDecodeEnabled = true
        XCTAssertFalse(eligible(MLXArray.ones([1, 1], dtype: .bool)))
        XCTAssertFalse(eligible(checkpoint: 1))
        gdn.train(true)
        XCTAssertFalse(eligible())
        gdn.train(false)
        state.prepare(lengths: [1])
        XCTAssertFalse(eligible())
        state.finalize()
        for (a, b) in zip(state.state, before) { assertIdentical(a, b) }
        for shape in [[2, 1, gdn.hiddenSize], [1, 2, gdn.hiddenSize]] {
            XCTAssertFalse(gdn.canUsePrismConvDecode(
                inputs: MLXArray.zeros(shape, dtype: .float16), qkv: qkv,
                convState: state[0]!, mask: nil, cache: state, checkpointAfter: nil))
        }
        XCTAssertFalse(gdn.canUsePrismConvDecode(
            inputs: input.asType(.bfloat16), qkv: qkv, convState: state[0]!,
            mask: nil, cache: state, checkpointAfter: nil))
        XCTAssertFalse(gdn.canUsePrismConvDecode(
            inputs: input, qkv: qkv.asType(.bfloat16), convState: state[0]!,
            mask: nil, cache: state, checkpointAfter: nil))
        XCTAssertFalse(gdn.canUsePrismConvDecode(
            inputs: input, qkv: qkv, convState: state[0]!.asType(.bfloat16),
            mask: nil, cache: state, checkpointAfter: nil))
        state.saveSpeculativeCheckpoint(convState: state[0]!, recurrentState: state[1]!, advancedBy: 1)
        XCTAssertFalse(eligible())
        _ = state.restoreSpeculativeCheckpoint()
        state.saveSpeculativePrefixes([1: state.state], width: 2)
        XCTAssertFalse(eligible())
        _ = state.restoreSpeculativeCheckpoint(rewinding: 1)
        let convState = state[0]!
        state[0] = nil
        XCTAssertFalse(gdn.canUsePrismConvDecode(
            inputs: input, qkv: qkv, convState: convState,
            mask: nil, cache: state, checkpointAfter: nil))
        state[0] = convState
        state[1] = MLXArray.zeros([1, gdn.numVHeads, 128, 128], dtype: .float16)
        XCTAssertFalse(eligible())
        state[1] = MLXArray.zeros([1, gdn.numVHeads, 64, 128])
        XCTAssertFalse(eligible())
        state[1] = nil
        XCTAssertFalse(eligible())
    }

    func testFallbackOutputsAndCacheMatchOriginal() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "GPU fallback parity test")
        for (batch, length, dtype, training, masked, checkpoint, padded) in [
            (2, 1, DType.float16, false, false, false, false),
            (1, 2, .float16, false, false, false, false),
            (1, 1, .bfloat16, false, false, false, false),
            (1, 1, .float16, true, false, false, false),
            (1, 1, .float16, false, true, false, false),
            (1, 2, .float16, false, false, true, false),
            (1, 1, .float16, false, false, false, true),
        ] {
            let original = try model(dtype)
            original.train(training)
            let candidate = try model(dtype)
            candidate.train(training)
            try candidate.update(parameters: original.parameters(), verify: [.all])
            candidate.prismConvDecodeEnabled = true
            let expected = MambaCache(leftPadding: padded ? [0] : nil)
            expected[0] = MLXRandom.normal([batch, 3, original.convDim]).asType(dtype)
            expected[1] = MLXRandom.normal([batch, original.numVHeads, 128, 128])
            let actual = MambaCache(leftPadding: padded ? [0] : nil)
            actual[0] = expected[0]
            actual[1] = expected[1]
            let input = MLXRandom.normal([batch, length, original.hiddenSize]).asType(dtype)
            let mask = masked ? MLXArray.zeros([batch, length], dtype: .bool) : nil
            assertIdentical(
                candidate(input, mask: mask, cache: actual, checkpointAfter: checkpoint ? 1 : nil),
                original(input, mask: mask, cache: expected, checkpointAfter: checkpoint ? 1 : nil))
            assertIdentical(actual[0]!, expected[0]!)
            assertIdentical(actual[1]!, expected[1]!)
            XCTAssertEqual(actual.metaState, expected.metaState)
            XCTAssertEqual(actual.hasSpeculativeCheckpoint, expected.hasSpeculativeCheckpoint)
        }
    }

    func testCPUAndConvContractFallback() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let gdn = try self.model(.float16)
            gdn.prismConvDecodeEnabled = true
            let state = self.cache(gdn, .float16)
            XCTAssertFalse(gdn.canUsePrismConvDecode(
                inputs: MLXArray.zeros([1, 1, gdn.hiddenSize], dtype: .float16),
                qkv: MLXArray.zeros([1, 1, gdn.convDim], dtype: .float16),
                convState: state[0]!, mask: nil, cache: state, checkpointAfter: nil))
        }
        let gdn = try model(.float16)
        gdn.prismConvDecodeEnabled = true
        let state = cache(gdn, .float16)
        for (bias, stride, padding, dilation, groups) in [
            (true, 1, 0, 1, gdn.convDim), (false, 2, 0, 1, gdn.convDim),
            (false, 1, 1, 1, gdn.convDim), (false, 1, 0, 2, gdn.convDim),
            (false, 1, 0, 1, 1),
        ] {
            let conv = Conv1d(
                inputChannels: gdn.convDim, outputChannels: gdn.convDim, kernelSize: 4,
                stride: stride, padding: padding, dilation: dilation, groups: groups, bias: bias)
            conv.update(parameters: conv.parameters().mapValues { $0.asType(.float32) })
            try gdn.updateModule(key: "conv1d", conv)
            XCTAssertFalse(gdn.canUsePrismConvDecode(
                inputs: MLXArray.zeros([1, 1, gdn.hiddenSize], dtype: .float16),
                qkv: MLXArray.zeros([1, 1, gdn.convDim], dtype: .float16),
                convState: state[0]!, mask: nil, cache: state, checkpointAfter: nil))
        }
    }
}
