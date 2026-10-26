import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class PrismHadamardQwen35Tests: XCTestCase {
    /// Opt in with a local fixture captured from the pinned vendor runtime.
    func testRealArtifactGreedyParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_PRISM_PARITY_FIXTURE"],
            !path.isEmpty
        else { throw XCTSkip("MLX_PRISM_PARITY_FIXTURE is not set") }
        struct Probability: Decodable {
            let tokenID: Int
            let probability: Float
            enum CodingKeys: String, CodingKey {
                case tokenID = "token_id"
                case probability
            }
        }
        struct Case: Decodable {
            let name: String
            let inputIDs: [Int32]
            let expectedOutputIDs: [Int]
            let firstTokenProbabilities: [Probability]
            enum CodingKeys: String, CodingKey {
                case name
                case inputIDs = "input_ids"
                case expectedOutputIDs = "expected_output_ids"
                case firstTokenProbabilities = "first_token_probabilities"
            }
        }
        struct Fixture: Decodable {
            let modelDirectory: String
            let cases: [Case]
            enum CodingKeys: String, CodingKey {
                case modelDirectory = "model_directory"
                case cases
            }
        }
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        guard !fixture.cases.isEmpty,
            Set(fixture.cases.map(\.name)).count == fixture.cases.count
        else {
            XCTFail("Parity fixture must contain cases with unique names")
            return
        }
        let directory = URL(fileURLWithPath: fixture.modelDirectory, isDirectory: true)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        let config = try decoder.decode(PrismHadamardQwen35Configuration.self, from: configData)
        let vocabularySize = config.base.textConfiguration.vocabularySize
        for item in fixture.cases {
            guard !item.name.isEmpty, !item.inputIDs.isEmpty,
                (1 ... 128).contains(item.expectedOutputIDs.count),
                item.inputIDs.allSatisfy({ $0 >= 0 && Int($0) < vocabularySize }),
                item.expectedOutputIDs.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
                item.firstTokenProbabilities.count == 16,
                Set(item.firstTokenProbabilities.map(\.tokenID)).count == 16,
                item.firstTokenProbabilities.allSatisfy({
                    $0.tokenID >= 0 && $0.tokenID < vocabularySize
                        && $0.probability.isFinite && (0 ... 1).contains($0.probability)
                })
            else {
                XCTFail(
                    "\(item.name): invalid token IDs, output length, or 16-token probability oracle"
                )
                return
            }
        }
        let baseConfig = try decoder.decode(BaseConfiguration.self, from: configData)
        let model = try PrismHadamardQwen35(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: baseConfig.perLayerQuantization?.quantization,
            perLayerQuantization: baseConfig.perLayerQuantization)
        // ModelContext switches serving models to inference mode after loading.
        model.train(false)
        for item in fixture.cases {
            let cache = try model.newCache(parameters: nil)
            let prepared = try model.prepare(
                LMInput(tokens: MLXArray(item.inputIDs).expandedDimensions(axis: 0)),
                cache: cache, state: nil, prefill: PrefillParameters(stepSize: 2048))
            var output: LMOutput
            switch prepared {
            case .logits(let result): output = result
            case .tokens(let remaining): output = model(remaining, cache: cache, state: nil)
            }
            let probabilities = MLX.softmax(
                output.logits[0, -1, 0...].asType(.float32), axis: -1, precise: true)
            let tokenIDs = MLXArray(item.firstTokenProbabilities.map { Int32($0.tokenID) })
            let actualProbabilities = probabilities[tokenIDs].asArray(Float.self)
            for (actual, reference) in zip(actualProbabilities, item.firstTokenProbabilities) {
                let context =
                    "\(item.name): first-token probability for ID \(reference.tokenID)"
                XCTAssertTrue(actual.isFinite, context)
                XCTAssertLessThanOrEqual(abs(actual - reference.probability), 1e-3, context)
            }
            var generated = [Int]()
            generated.reserveCapacity(item.expectedOutputIDs.count)
            for index in item.expectedOutputIDs.indices {
                let token = argMax(output.logits[0..., -1, 0...], axis: -1).item(Int.self)
                generated.append(token)
                if index + 1 < item.expectedOutputIDs.count {
                    output = model(
                        LMInput.Text(tokens: MLXArray([Int32(token)]).reshaped(1, 1)),
                        cache: cache, state: output.state)
                }
            }
            let mismatch = zip(generated, item.expectedOutputIDs).enumerated()
                .first { $0.element.0 != $0.element.1 }
            let detail =
                mismatch.map {
                    "first mismatch at output index \($0.offset): native=\($0.element.0), vendor=\($0.element.1)"
                } ?? "all output IDs match"
            XCTAssertEqual(generated, item.expectedOutputIDs, "\(item.name): \(detail)")
        }
    }

    private var batchDeltaKey: LMOutput.Key<MLXArray> { .init("qwen35.ropeDeltas") }
    private var batchPositionsKey: LMOutput.Key<MLXArray> { .init("qwen35.precomputedPositionIds") }

    private func rowState(_ delta: Int32, positions: MLXArray? = nil) -> LMOutput.State {
        var state = LMOutput.State()
        state[batchDeltaKey] = MLXArray([delta])
        state[batchPositionsKey] = positions
        return state
    }

    func testBatchStatePreservesOrderAcrossReorderAndRemoval() throws {
        let model = try PrismHadamardQwen35(decode())
        let states = [
            rowState(2, positions: .zeros([3, 1, 9], dtype: .int32)),
            rowState(-3), rowState(7),
        ]
        for (indices, expected) in [
            ([0, 1, 2], [Int32(2), -3, 7]), ([2, 0], [Int32(7), 2]), ([1], [Int32(-3)]),
        ] {
            let merged = try model.batchDecodeState(
                indices.map { states[$0] }, cacheOffsets: indices.map { [9, 3, 12][$0] })
            XCTAssertEqual(merged.keys, [batchDeltaKey.id])
            XCTAssertEqual(merged[batchDeltaKey]?.asArray(Int32.self), expected)
            try model.validateBatchDecodeState(merged, against: merged)
        }
        // Merging drops only validated prefill positions and leaves original row state intact.
        XCTAssertEqual(states[0].keys, [batchDeltaKey.id, batchPositionsKey.id])
        XCTAssertEqual(states[0][batchPositionsKey]?.shape, [3, 1, 9])
        XCTAssertFalse(
            (Qwen35(try decode().base) as any LanguageModel) is any BatchInvariantDecodeStateModel)
    }

    func testBatchStateRejectsColdMissingMalformedAndUnknownState() throws {
        let model = try PrismHadamardQwen35(decode())
        let valid = rowState(2)
        XCTAssertThrowsError(try model.batchDecodeState([], cacheOffsets: []))
        XCTAssertThrowsError(try model.batchDecodeState([valid], cacheOffsets: []))
        XCTAssertThrowsError(try model.batchDecodeState([valid], cacheOffsets: [0]))
        XCTAssertThrowsError(try model.batchDecodeState([valid], cacheOffsets: [-1]))
        XCTAssertThrowsError(try model.batchDecodeState([nil], cacheOffsets: [2]))
        XCTAssertThrowsError(try model.batchDecodeState([LMOutput.State()], cacheOffsets: [2]))
        var differentDType = valid
        differentDType[batchDeltaKey] = MLXArray([Int64(2)])
        XCTAssertThrowsError(
            try model.batchDecodeState([valid, differentDType], cacheOffsets: [2, 2]))
        for delta in [MLXArray(Int32(2)), MLXArray([Int32(1), 2]), MLXArray([Float(2)])] {
            var invalid = valid
            invalid[batchDeltaKey] = delta
            XCTAssertThrowsError(try model.batchDecodeState([invalid], cacheOffsets: [2]))
        }
        for positions in [
            MLXArray.zeros([3, 1, 0], dtype: .int32),
            MLXArray.zeros([3, 2, 4], dtype: .int32), MLXArray.zeros([3, 1, 4]),
            MLXArray.zeros([3, 4], dtype: .int32),
        ] {
            XCTAssertThrowsError(
                try model.batchDecodeState(
                    [rowState(2, positions: positions)], cacheOffsets: [4]))
        }
        for key in ["unknown", "qwen35.mtpEmitFlag"] {
            var invalid = valid
            invalid[LMOutput.Key<Bool>(key)] = false
            XCTAssertThrowsError(try model.batchDecodeState([invalid], cacheOffsets: [2]))
        }
        var wrongType = valid
        wrongType[LMOutput.Key<String>(batchPositionsKey.id)] = "not an array"
        XCTAssertThrowsError(try model.batchDecodeState([wrongType], cacheOffsets: [2]))
    }

    func testBatchStateRejectsTruncatingDTypesAndEffectivePositionOverflow() throws {
        let model = try PrismHadamardQwen35(decode())
        for array in [MLXArray([Int64.max]), MLXArray([UInt64.max]), MLXArray([Int16(2)])] {
            var invalid = rowState(2)
            invalid[batchDeltaKey] = array
            XCTAssertThrowsError(try model.batchDecodeState([invalid], cacheOffsets: [2]))
        }
        let widePositions = MLXArray([Int64.max]).reshaped(1, 1, 1)
        XCTAssertThrowsError(
            try model.batchDecodeState(
                [rowState(2, positions: broadcast(widePositions, to: [3, 1, 1]))], cacheOffsets: [2]
            ))
        XCTAssertThrowsError(try model.batchDecodeState([rowState(0)], cacheOffsets: [Int.max]))
        XCTAssertThrowsError(try model.batchDecodeState([rowState(Int32.max)], cacheOffsets: [1]))
        XCTAssertThrowsError(try model.batchDecodeState([rowState(-3)], cacheOffsets: [2]))
        // Both boundaries remain representable; there is no wrap or clipping.
        let lower = try model.batchDecodeState([rowState(-2)], cacheOffsets: [2])
        XCTAssertEqual(lower[batchDeltaKey]?.asArray(Int32.self), [-2])
        let upper = try model.batchDecodeState([rowState(1)], cacheOffsets: [Int(Int32.max) - 1])
        XCTAssertEqual(upper[batchDeltaKey]?.asArray(Int32.self), [1])
    }

    func testBatchOutputAcceptsImmutableIdentityAndChecksReplacementValues() throws {
        let model = try PrismHadamardQwen35(decode())
        let merged = try model.batchDecodeState([rowState(2), rowState(-3)], cacheOffsets: [9, 3])
        let expected = try XCTUnwrap(merged[batchDeltaKey])
        let identical = merged
        XCTAssertTrue(identical[batchDeltaKey] === expected)
        try model.validateBatchDecodeState(identical, against: merged)
        var replacement = LMOutput.State()
        replacement[batchDeltaKey] = expected + 0
        XCTAssertFalse(replacement[batchDeltaKey] === expected)
        try model.validateBatchDecodeState(replacement, against: merged)
        replacement[batchDeltaKey] = expected + 1
        XCTAssertThrowsError(try model.validateBatchDecodeState(replacement, against: merged))
        replacement = identical
        replacement[LMOutput.Key<Bool>("unknown")] = false
        XCTAssertThrowsError(try model.validateBatchDecodeState(replacement, against: merged))
    }

    func testBatchOutputMustRetainExactImmutableState() throws {
        let model = try PrismHadamardQwen35(decode())
        let merged = try model.batchDecodeState([rowState(2), rowState(-3)], cacheOffsets: [9, 3])
        XCTAssertThrowsError(try model.validateBatchDecodeState(nil, against: merged))
        XCTAssertThrowsError(try model.validateBatchDecodeState(LMOutput.State(), against: merged))
        for delta in [
            MLXArray([Int32(2), -2]), MLXArray([Int64(2), -3]),
            MLXArray([Int32(2)]), MLXArray([Int32(2), -3]).reshaped(2, 1),
        ] {
            var output = LMOutput.State()
            output[batchDeltaKey] = delta
            XCTAssertThrowsError(try model.validateBatchDecodeState(output, against: merged))
        }
        var unknown = merged
        unknown[LMOutput.Key<Bool>("qwen35.mtpEmitFlag")] = false
        XCTAssertThrowsError(try model.validateBatchDecodeState(unknown, against: merged))
        var positions = merged
        positions[batchPositionsKey] = .zeros([3, 2, 1], dtype: .int32)
        XCTAssertThrowsError(try model.validateBatchDecodeState(positions, against: merged))
    }

    func testPackedWarmPositionsMatchExplicitRowPositionsBeforeCacheMutation() throws {
        let config = try decode().base
        let model = Qwen35(config)
        model.train(false)
        let allOffsets = [9, 3, 12]
        let allDeltas: [Int32] = [2, -1, 7]
        for selected in [[0, 1, 2], [2, 0], [1]] {
            let offsets = selected.map { allOffsets[$0] }
            let deltas = selected.map { allDeltas[$0] }
            let state = try Qwen35BatchDecodeState.merge(
                deltas.map { rowState($0) }, cacheOffsets: offsets)
            let actualCache = PrismPositionProbeCache(offsets: offsets)
            let referenceCache = PrismPositionProbeCache(offsets: offsets)
            let tokens = MLXArray(selected.map { Int32($0 + 1) }).reshaped(selected.count, 1)
            let expectedPositions = MLXArray(zip(offsets, deltas).map { Int32($0.0) + $0.1 })
                .reshaped(1, selected.count, 1)
            let positions = broadcast(expectedPositions, to: [3, selected.count, 1])
            let actual = model.languageModel(
                tokens, cache: [MambaCache(), actualCache], state: state)
            let reference = model.languageModel(
                tokens, cache: [MambaCache(), referenceCache], state: state,
                positionIds: positions)
            XCTAssertEqual(actual.logits.asArray(Float.self), reference.logits.asArray(Float.self))
            XCTAssertEqual(
                actualCache.rowOffsets.asArray(Int32.self), offsets.map { Int32($0 + 1) })
            try Qwen35BatchDecodeState.validate(actual.state, against: state)
        }
    }

    func testRaggedAttentionPreservesPhysicalMaskWidthAcrossContinuation() throws {
        let config = try decode().base.textConfiguration
        let attention = Qwen35Language.Attention(config)
        attention.train(false)
        let outputEmbedding = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        // The two-token gap reproduces min-offset slicing being shorter than packed K/V.
        let batchCache = PrismRaggedAttentionCache(offsets: [5, 3], batched: true)
        let serialCaches = [5, 3].map { PrismRaggedAttentionCache(offsets: [$0], batched: false) }
        for sequence in [1, 2, 1] {
            let offsets = batchCache.offsets
            let values = (0 ..< 2 * sequence * config.hiddenSize).map { Float($0 % 31 - 15) / 31 }
            let input = MLXArray(values).reshaped(2, sequence, config.hiddenSize)
            let positions = stacked(offsets.map { MLXArray($0 ..< ($0 + sequence)).asType(.int32) })
            let positionIDs = broadcast(positions[.newAxis, 0..., 0...], to: [3, 2, sequence])
            guard
                case .array(let mask) = batchCache.makeMask(
                    n: sequence, windowSize: nil, returnArray: true)
            else {
                return XCTFail("Expected a physical ragged mask")
            }
            XCTAssertEqual(mask.dim(-1), (offsets.max() ?? 0) + sequence)
            XCTAssertGreaterThan(mask.dim(-1), batchCache.offset + sequence + 1)
            let actual = attention(input, mask: mask, cache: batchCache, positionIds: positionIDs)
            let expectedRows = serialCaches.enumerated().map { row, cache in
                guard
                    case .array(let rowMask) = cache.makeMask(
                        n: sequence, windowSize: nil, returnArray: true)
                else {
                    preconditionFailure("Expected a serial causal mask")
                }
                return attention(
                    input[row ..< (row + 1)], mask: rowMask, cache: cache,
                    positionIds: positionIDs[0..., row ..< (row + 1), 0...])
            }
            let expected = concatenated(expectedRows, axis: 0)
            XCTAssertTrue(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            let actualLogits = outputEmbedding.asLinear(actual)
            let expectedLogits = outputEmbedding.asLinear(expected)
            XCTAssertTrue(
                allClose(actualLogits, expectedLogits, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            XCTAssertEqual(
                actualLogits.argMax(axis: -1).asArray(Int32.self),
                expectedLogits.argMax(axis: -1).asArray(Int32.self))
            XCTAssertEqual(batchCache.offsets, offsets.map { $0 + sequence })
            XCTAssertEqual(batchCache.presentedLength, (offsets.max() ?? 0) + sequence)
        }
    }

    private func configuration(_ changes: [String: Any] = [:]) throws -> Data {
        var object: [String: Any] = [
            "model_type": "prism_hadamard_qwen35", "schema_version": 2,
            "base_model_type": "qwen3_5", "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": [
                [
                    "path": "model.embed_tokens", "block": 512,
                    "embedding": true, "dtype": "float16",
                ]
            ],
            "text_config": [
                "model_type": "qwen3_5_text", "hidden_size": 512,
                "num_hidden_layers": 2, "intermediate_size": 512,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 128, "linear_num_value_heads": 1,
                "linear_num_key_heads": 1, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "vocab_size": 4, "full_attention_interval": 2,
                "tie_word_embeddings": true,
            ],
            "vision_config": [
                "model_type": "qwen3_5", "depth": 1, "hidden_size": 8,
                "intermediate_size": 16, "out_hidden_size": 512,
                "num_heads": 1, "patch_size": 16, "spatial_merge_size": 1,
                "temporal_patch_size": 1, "num_position_embeddings": 8,
            ],
        ]
        object.merge(changes) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object)
    }

    private func decode(_ changes: [String: Any] = [:]) throws
        -> PrismHadamardQwen35Configuration
    {
        try JSONDecoder().decode(
            PrismHadamardQwen35Configuration.self, from: configuration(changes))
    }

    func testPrismRegistryConstructsFullArtifactModel() async throws {
        let data = try configuration()
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        XCTAssertTrue(model is PrismHadamardQwen35)
    }

    func testMissingManifestIsRejected() throws {
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        object.removeValue(forKey: "modules")
        let data = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(
            try JSONDecoder().decode(PrismHadamardQwen35Configuration.self, from: data))
    }

    func testPackRejectsUnsupportedSchemaAndQuantization() throws {
        for change: [String: Any] in [
            ["schema_version": 1], ["tensor_namespace": "gguf"],
            ["gdn_activation_layout": "interleaved"], ["modules": []],
            ["quantization": ["bits": 4, "group_size": 128, "mode": "affine"]],
            ["quantization": ["bits": 2, "group_size": 64, "mode": "affine"]],
            ["quantization": ["bits": 2, "group_size": 128, "mode": "mxfp4"]],
        ] {
            XCTAssertThrowsError(try decode(change))
        }
    }

    func testManifestRejectsDuplicateUnknownAndMismatchedModules() throws {
        let record: [String: Any] = [
            "path": "model.embed_tokens", "block": 512,
            "embedding": true, "dtype": "float16",
        ]
        XCTAssertThrowsError(try decode(["modules": [record, record]]))
        for (key, value) in [("path", "model.missing"), ("dtype", "bfloat16")] {
            var invalid = record
            invalid[key] = value
            XCTAssertThrowsError(try PrismHadamardQwen35(decode(["modules": [invalid]])))
        }
        var invalid = record
        invalid["embedding"] = false
        XCTAssertThrowsError(try PrismHadamardQwen35(decode(["modules": [invalid]])))
        invalid = record
        invalid["block"] = 1024
        XCTAssertThrowsError(try PrismHadamardQwen35(decode(["modules": [invalid]])))
    }

    func testFusedSignedRotationIsBitwiseReferenceAcrossShapesAndChangedSigns() {
        for dtype in [DType.float16, .float32] {
            for shape in [[1, 512], [2, 3, 1024], [2, 9216]] {
                let count = shape.reduce(1, *)
                let input = MLXArray((0 ..< count).map { Float($0 % 41 - 20) / 32 })
                    .reshaped(shape).asType(dtype)
                let width = shape.last!
                for signPattern in [0, 1] {
                    let signs = MLXArray(
                        (0 ..< width).map {
                            Float(($0 + signPattern) % 3 == 0 ? -1 : 1)
                        })
                    let expected = hadamardTransform(
                        (input.asType(.float32) * signs).reshaped(-1, 512),
                        scale: 1 / sqrt(Float(512))
                    )
                    .reshaped(shape).asType(dtype)
                    let actual = prismHadamardTransform(input, block: 512, signs: signs)
                    XCTAssertEqual(actual.dtype, dtype)
                    XCTAssertEqual(actual.shape, shape)
                    XCTAssertEqual(
                        actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
                        expected.asType(.float32).asArray(Float.self).map(\.bitPattern),
                        "dtype=\(dtype), shape=\(shape), signPattern=\(signPattern)")
                }
            }
        }
    }

    func testSignedRotationUsesFloat32AndInverseOrder() {
        let signs = MLXArray((0 ..< 512).map { Float($0 % 3 == 0 ? -1 : 1) })
        let x = MLXArray((0 ..< 1024).map { Float($0 % 17 - 8) / 8 }).reshaped(2, 512)
        let rotated = prismHadamardTransform(x, block: 512, signs: signs)
        let recovered = prismHadamardTransform(rotated, block: 512, signs: signs, inverse: true)
        XCTAssertLessThan((recovered - x).abs().max().item(Float.self), 1e-5)
        let half = x.asType(.float16)
        XCTAssertEqual(prismHadamardTransform(half, block: 512, signs: signs).dtype, .float16)
    }

    func testPackedLinearAndTiedEmbeddingMatchReferenceAlgebra() throws {
        let width = 512
        let signs = MLXArray((0 ..< width).map { Float($0 % 3 == 0 ? -1 : 1) })
        let dense = MLXArray((0 ..< 4 * width).map { Float($0 % 23 - 11) / 16 })
            .reshaped(4, width).asType(.float16)
        let (weight, scales, biases) = quantized(dense, groupSize: 128, bits: 2)
        let parameters = ModuleParameters.unflattened([
            ("weight", weight), ("scales", scales), ("biases", try XCTUnwrap(biases)),
            ("signs", signs),
        ])
        let linear = try PrismHadamardLinear(width: width, rows: 4, block: 512)
        let embedding = try PrismHadamardEmbedding(width: width, rows: 4, block: 512)
        try linear.update(parameters: parameters, verify: .all)
        try embedding.update(parameters: parameters, verify: .all)
        let x = MLXArray.ones([1, width], dtype: .float16)
        let expected = quantizedMM(
            prismHadamardTransform(x, block: 512, signs: signs), weight,
            scales: scales, biases: biases, groupSize: 128, bits: 2)
        XCTAssertEqual(linear(x).asArray(Float16.self), expected.asArray(Float16.self))
        XCTAssertEqual(embedding.asLinear(x).asArray(Float16.self), expected.asArray(Float16.self))
        let selected = dequantized(
            weight[MLXArray([Int32(2), 0])], scales: scales[MLXArray([Int32(2), 0])],
            biases: biases![MLXArray([Int32(2), 0])], groupSize: 128, bits: 2)
        let expectedRows = prismHadamardTransform(
            selected.asType(.float16), block: 512, signs: signs, inverse: true)
        XCTAssertEqual(
            embedding(MLXArray([Int32(2), 0])).asArray(Float16.self),
            expectedRows.asArray(Float16.self))
    }

    func testSignsFailClosedAndPackedPlaceholdersHaveCorrectShape() throws {
        let linear = try PrismHadamardLinear(width: 512, rows: 4, block: 512)
        XCTAssertEqual(linear.weight.shape, [4, 32])
        XCTAssertEqual(linear.weight.dtype, .uint32)
        XCTAssertEqual(linear.scales.shape, [4, 4])
        XCTAssertThrowsError(try linear.update(parameters: .unflattened([]), verify: .all))
        for invalid in [
            MLXArray.zeros([512]), MLXArray.ones([1, 512]),
            MLXArray.ones([512], dtype: .float16),
        ] {
            XCTAssertThrowsError(
                try linear.update(
                    parameters: .unflattened([("signs", invalid)]), verify: []))
        }
        let unrotated = try PrismHadamardLinear(width: 128, rows: 4, block: 0)
        XCTAssertNil(unrotated.signs)
        XCTAssertThrowsError(try PrismHadamardLinear(width: 128, rows: 4, block: 512))
    }

    func testCustomGDNProjectionsCannotBeBypassedByFourProjectionFusion() throws {
        let config = try decode()
        let layer = Qwen35Language.GatedDeltaNet(config.base.textConfiguration)
        let qkv = try PrismHadamardLinear(width: 512, rows: 384, block: 512)
        let z = try PrismHadamardLinear(width: 512, rows: 128, block: 512)
        let b = try PrismHadamardLinear(width: 512, rows: 1, block: 512)
        let a = try PrismHadamardLinear(width: 512, rows: 1, block: 512)
        let out = try PrismHadamardLinear(width: 128, rows: 512, block: 0)
        try layer.update(
            modules: .unflattened([
                ("in_proj_qkv", qkv), ("in_proj_z", z), ("in_proj_b", b),
                ("in_proj_a", a), ("out_proj", out),
            ]), verify: [])
        layer.fusedInputProjectionEnabled = true
        XCTAssertFalse(try layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        for sequence in [1, 7] {
            let x = MLXArray.ones([1, sequence, 512], dtype: .float16)
            let projected = layer.projectInputs(x, batch: 1, sequence: sequence)
            XCTAssertEqual(projected.b.asArray(Float16.self), b(x).asArray(Float16.self))
            XCTAssertEqual(projected.a.asArray(Float16.self), a(x).asArray(Float16.self))
            XCTAssertTrue(layer.outProj === out)
            XCTAssertEqual(projected.qkv.asArray(Float16.self), qkv(x).asArray(Float16.self))
            XCTAssertEqual(
                projected.z.asArray(Float16.self), z(x).asArray(Float16.self))
        }
    }

    func testSchemaTwoRetainsVisionAndDoesNotShiftNormsOrConv() throws {
        let model = try PrismHadamardQwen35(decode())
        let weights = [
            "vision_tower.patch_embed.proj.weight": MLXArray.ones([1, 1]),
            "language_model.model.norm.weight": MLXArray.ones([512]),
            "language_model.model.layers.0.linear_attn.conv1d.weight": MLXArray.ones([384, 4, 1]),
        ]
        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        for (key, value) in weights {
            XCTAssertEqual(sanitized[key]?.shape, value.shape)
            XCTAssertEqual(sanitized[key]?.sum().item(Float.self), value.sum().item(Float.self))
        }
        XCTAssertTrue(model.languageModel.model.embedTokens is PrismHadamardEmbedding)
        let modules = Dictionary(
            uniqueKeysWithValues: model.languageModel.leafModules().flattened())
        XCTAssertTrue(modules["model.layers.0.linear_attn.in_proj_a"] is Linear)
        XCTAssertFalse(modules["model.layers.0.linear_attn.in_proj_a"] is Quantized)
    }
}

/// Synthetic history makes a wrong absolute query position observable in attention.
private final class PrismPositionProbeCache: BaseKVCache {
    var rowOffsets: MLXArray
    private let priorKeys: MLXArray
    private let priorValues: MLXArray

    init(offsets: [Int]) {
        rowOffsets = MLXArray(offsets.map(Int32.init))
        let values = offsets.flatMap { offset in
            (0 ..< 3 * 128).map { Float(($0 + offset) % 29 - 14) / 29 }
        }
        priorKeys = MLXArray(values).reshaped(offsets.count, 1, 3, 128)
        priorValues = priorKeys * 0.5
        super.init()
        offset = offsets.max() ?? 0
    }

    override var ropeOffset: RoPEOffset { .batch(rowOffsets) }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = (
            concatenated([priorKeys, keys], axis: 2), concatenated([priorValues, values], axis: 2)
        )
        rowOffsets = rowOffsets + keys.dim(2)
        offset += keys.dim(2)
        return result
    }
}

/// Unequal row histories are stored without padding and padded only for attention presentation.
private final class PrismRaggedAttentionCache: BaseKVCache {
    private var rowKeys: [MLXArray]
    private var rowValues: [MLXArray]
    private let batched: Bool
    private(set) var offsets: [Int]
    private(set) var presentedLength = 0

    init(offsets: [Int], batched: Bool) {
        self.offsets = offsets
        self.batched = batched
        rowKeys = offsets.map { length in
            MLXArray((0 ..< length * 128).map { Float($0 % 23 - 11) / 23 }).reshaped(
                1, 1, length, 128)
        }
        rowValues = rowKeys.map { $0 * 0.5 }
        super.init()
        offset = offsets.min() ?? 0
    }

    override var ropeOffset: RoPEOffset {
        batched ? .batch(MLXArray(offsets.map(Int32.init))) : .scalar(offset)
    }

    override func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        let columns = MLXArray(0 ..< ((offsets.max() ?? 0) + n)).asType(.int32)
        let masks = offsets.map { start in
            let positions = MLXArray(start ..< (start + n)).asType(.int32)
            return columns[.newAxis, 0...] .<= positions[0..., .newAxis]
        }
        return .array(stacked(masks, axis: 0).expandedDimensions(axis: 1))
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        for row in offsets.indices {
            rowKeys[row] = concatenated([rowKeys[row], keys[row ..< (row + 1)]], axis: 2)
            rowValues[row] = concatenated([rowValues[row], values[row ..< (row + 1)]], axis: 2)
            offsets[row] += keys.dim(2)
        }
        offset = offsets.min() ?? 0
        presentedLength = offsets.max() ?? 0
        func present(_ rows: [MLXArray]) -> MLXArray {
            concatenated(
                rows.map { row in
                    let padding = presentedLength - row.dim(2)
                    return concatenated(
                        [row, MLXArray.zeros([1, 1, padding, 128], dtype: row.dtype)], axis: 2)
                }, axis: 0)
        }
        return (present(rowKeys), present(rowValues))
    }
}
