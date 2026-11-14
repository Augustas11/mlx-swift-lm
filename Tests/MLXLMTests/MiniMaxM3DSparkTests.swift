import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Config, weight-name and context-cache checks that run without a checkpoint.
final class MiniMaxM3DSparkTests: XCTestCase {
    /// The DSpark fields of mlx-community/MiniMax-M3.1-Flash-oQ3e-mtp
    /// `dspark/config.json` (revision b1084ba6), shrunk to a tiny draft.
    private func configurationObject() -> [String: Any] {
        let text: [String: Any] = [
            "architectures": ["DSparkMiniMaxDraftModel"],
            "hidden_size": 32, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "rms_norm_eps": 1e-6, "use_gemma_norm": true,
            "rope_theta": 10_000_000, "rotary_dim": 4, "partial_rotary_factor": 0.5,
            "hidden_act": "swigluoai", "use_qk_norm": true, "qk_norm_type": "per_head",
            "dense_intermediate_size": 64, "shared_intermediate_size": 16,
            "moe_layer_freq": [0, 0], "sparse_attention_config": NSNull(),
            "layer_types": ["sliding_attention", "sliding_attention"], "sliding_window": 16,
            "attention_output_gate": false, "tie_word_embeddings": false, "model_type": "",
        ]
        return [
            "architectures": ["DSparkMiniMaxDraftModel"], "model_type": "minimax_m3_vl",
            "text_config": text,
            "dspark_block_size": 7, "dspark_markov_rank": 8, "dspark_markov_head_type": "vanilla",
            "dspark_noise_token_id": 60, "dspark_use_dense_ffn": true,
            "dspark_target_layer_ids": [0, 2, 3], "dspark_hybrid_context_fusion": false,
            "enable_confidence_head": true, "confidence_head_with_markov": true,
            "dflash_config": ["num_target_layers": 4, "sliding_window": 16],
            "quantization_config": ["quant_method": "mxfp8", "weight_block_size": [1, 32]],
        ]
    }

    private func decode(_ object: [String: Any]) throws -> MiniMaxM3DSparkConfiguration {
        try MiniMaxM3DSparkConfiguration(data: JSONSerialization.data(withJSONObject: object))
    }

    func testConfigurationReadsDSparkFields() throws {
        let configuration = try decode(configurationObject())
        XCTAssertEqual(configuration.blockSize, 7)
        XCTAssertEqual(configuration.markovRank, 8)
        XCTAssertEqual(configuration.noiseTokenID, 60)
        XCTAssertEqual(configuration.targetLayerIDs, [0, 2, 3])
        XCTAssertEqual(configuration.targetLayerCount, 4)
        XCTAssertEqual(configuration.slidingWindow, 16)
        XCTAssertTrue(configuration.mxfp8)
        XCTAssertEqual(configuration.hiddenSize, 32)
        XCTAssertEqual(configuration.layerCount, 2)
    }

    func testConfigurationRefusesTargetAndUnsupportedHeads() throws {
        var target = configurationObject()
        target["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
        var text = target["text_config"] as! [String: Any]
        text["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
        target["text_config"] = text
        XCTAssertThrowsError(try decode(target))

        var markov = configurationObject()
        markov["dspark_markov_head_type"] = "gated"
        XCTAssertThrowsError(try decode(markov))

        var noConfidence = configurationObject()
        noConfidence["enable_confidence_head"] = false
        XCTAssertThrowsError(try decode(noConfidence))

        var outOfRange = configurationObject()
        outOfRange["dspark_target_layer_ids"] = [0, 4]
        XCTAssertThrowsError(try decode(outOfRange))

        var noise = configurationObject()
        noise["dspark_noise_token_id"] = 64
        XCTAssertThrowsError(try decode(noise))
    }

    func testModelBuildsMXFP8DraftLayersAndSanitizesCheckpointNames() throws {
        let model = MiniMaxM3DSparkDraftModel(try decode(configurationObject()), maximumDepth: 6)
        XCTAssertEqual(model.maximumBlockSize, 7)
        XCTAssertFalse(model.requiresSharedTargetKV)
        XCTAssertTrue(model.requiresGreedySampling)
        let attention = model.layers[0].selfAttn
        let q = try XCTUnwrap(attention.qProj as? QuantizedLinear)
        XCTAssertEqual(q.mode, .mxfp8)
        XCTAssertEqual(q.groupSize, 32)
        XCTAssertEqual(q.weight.shape, [32, 8])
        XCTAssertNil(model.fc as? QuantizedLinear)

        let weights: [String: MLXArray] = [
            "language_model.model.dspark.fc.weight": MLXArray.zeros([32, 96]),
            "language_model.model.dspark.layers.1.decoder_layer.self_attn.q_proj.weight":
                MLXArray.zeros([32, 32], dtype: .uint8),
            "language_model.model.dspark.layers.1.decoder_layer.self_attn.q_proj.weight_scale_inv":
                MLXArray.zeros([32, 1], dtype: .uint8),
            "language_model.model.dspark.markov_head.markov_w1.weight": MLXArray.zeros([64, 8]),
        ]
        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "fc.weight", "layers.1.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.scales",
                "markov_head.markov_w1.weight",
            ])
        XCTAssertEqual(sanitized["layers.1.self_attn.q_proj.weight"]?.dtype, .uint32)
        XCTAssertEqual(sanitized["layers.1.self_attn.q_proj.weight"]?.shape, [32, 8])
    }

    func testContextCacheKeepsWindowAndCountsCommittedPositions() {
        let cache = MiniMaxM3DSparkContextCache(window: 4)
        let first = MLXArray((0 ..< 6).map(Float.init), [1, 1, 3, 2])
        cache.append(keys: first, values: first, committed: 3)
        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(cache.count, 3)
        let second = MLXArray((6 ..< 12).map(Float.init), [1, 1, 3, 2])
        cache.append(keys: second, values: second, committed: 5)
        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(cache.keys!.asArray(Float.self), [4, 5, 6, 7, 8, 9, 10, 11])
        let copied = cache.copy() as! MiniMaxM3DSparkContextCache
        copied.append(keys: second, values: second, committed: 3)
        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(copied.offset, 11)
    }
}
