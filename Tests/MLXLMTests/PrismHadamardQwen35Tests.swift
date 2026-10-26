import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

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
