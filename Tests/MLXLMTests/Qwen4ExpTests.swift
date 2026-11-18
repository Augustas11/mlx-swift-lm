import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Qwen3.8-Flash-Next (`qwen4_exp`) port checks that need no checkpoint.
///
/// The expected n-gram hash constants and rows below come from mlx-lm's
/// `qwen4_exp.py` helpers (`_nth_prime_after`, `_ngram_multipliers`,
/// `NGramEmbedding._ngram_tokens` and its hash), evaluated in Python.
final class Qwen4ExpTests: XCTestCase {

    // MARK: - Fixtures

    /// The heretic-2 oQ4e build's `config.json` without the vision tower and
    /// with three of its per-layer quantization entries.
    private func realConfigData() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "config", withExtension: "json", subdirectory: "qwen4_exp"))
        return try Data(contentsOf: url)
    }

    /// Four layers (three gated delta, one QSA attention), a PLE on layer 2,
    /// an indexer budget of 8 tokens in blocks of 2, and a 4-shard n-gram
    /// table of 420 rows (heads of 101, 103, 107 and 109 rows).
    private func tinyTextJSON(
        layerTypes: String = #"["linear_attention", "linear_attention", "linear_attention", "full_attention"]"#,
        pleLayerIds: String = "[2]",
        extra: String = ""
    ) -> String {
        """
        {
            "model_type": "qwen4_exp_text", "vocab_size": 64, "hidden_size": 32,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "rms_norm_eps": 1e-6, "eos_token_id": 1,
            "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.5,
                "type": "default"},
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 32, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "hc_count": 2, "hc_lowrank": 8,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16,
            "indexer_budget": 8, "indexer_compress_ratio": 2,
            "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 100,
            "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 4,
            "ple_embed_dim": 8, "ple_layer_ids": \(pleLayerIds),
            "layer_types": \(layerTypes),
            "output_gate_type": "sigmoid", "hidden_act": "silu",
            "tie_word_embeddings": false\(extra)
        }
        """
    }

    private func tinyConfiguration(nested: Bool = false) throws -> Qwen4ExpConfiguration {
        let text = tinyTextJSON()
        let json = nested ? #"{"model_type": "qwen4_exp", "text_config": \#(text)}"# : text
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        return config
    }

    private func randomize(_ model: Module, seed: UInt64 = 11) {
        let parameters = model.parameters().flattened()
        let updated: [(String, MLXArray)] = withRandomState(MLXRandom.RandomState(seed: seed)) {
            parameters.map { key, value in
                let noise = MLXRandom.normal(value.shape).asType(value.dtype)
                if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
                    return (key, 1 + 0.1 * noise)
                }
                return (key, 0.2 * noise)
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    // MARK: - Configuration (no Metal)

    func testRealArtifactConfigurationDecodesAndValidates() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: try realConfigData())
        try config.validateModelConfiguration()

        XCTAssertEqual(config.modelType, "qwen4_exp_text")
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.hiddenLayers, 48)
        XCTAssertEqual(config.layerTypes.count, 48)
        for (i, type) in config.layerTypes.enumerated() {
            XCTAssertEqual(type, (i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        XCTAssertEqual(config.ropeTheta, 10_000_000)
        XCTAssertEqual(config.rotaryDimensions, 64)
        XCTAssertEqual(config.ropeType, "default")
        XCTAssertEqual(config.eosTokenId, 248_044)
        XCTAssertEqual(config.pleLayerIds, [2])
        XCTAssertEqual(config.seed, 1234)
        XCTAssertEqual(config.splitNgramParts, 128)
        XCTAssertEqual(config.ngramVocabSizeDivisor, 128)
        XCTAssertEqual(config.outputGateType, "sigmoid")
        XCTAssertEqual(config.numExperts, 512)
        XCTAssertEqual(config.numExpertsPerTok, 10)
        XCTAssertEqual(config.hcCount, 4)
        XCTAssertEqual(config.hcLowrank, 320)
        XCTAssertEqual(config.indexerBudget, 2048)
        XCTAssertEqual(config.indexerCompressRatio, 4)

        // The base build's text_config has no `seed`: the reference default.
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: try realConfigData()) as? [String: Any])
        var text = try XCTUnwrap(object["text_config"] as? [String: Any])
        text.removeValue(forKey: "seed")
        object["text_config"] = text
        let unseeded = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: try JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(unseeded.seed, 1234)

        // The per-layer quantization entries name module paths of `Qwen4ExpModel`.
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: try realConfigData())
        let perLayer = try XCTUnwrap(base.perLayerQuantization)
        let shard = perLayer.quantization(
            layer: "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0")
        XCTAssertEqual(shard?.bits, 4)
        XCTAssertEqual(shard?.groupSize, 32)
    }

    func testRealArtifactHashConstantsMatchReference() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: try realConfigData())
        let constants = Qwen4ExpNGramHash.constants(config, pleLayerIndex: 0)
        XCTAssertEqual(
            constants.sizes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
                20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
                20_000_153, 20_000_159, 20_000_161, 20_000_171,
            ])
        XCTAssertEqual(
            constants.offsets,
            [
                0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165, 120_000_228,
                140_000_297, 160_000_374, 180_000_455, 200_000_548, 220_000_655, 240_000_802,
                260_000_955, 280_001_114, 300_001_275,
            ])
        XCTAssertEqual(constants.multipliers, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071])
        // 128 shards of 2,500,012 rows, as the checkpoint stores them.
        XCTAssertEqual(constants.paddedRows, 320_001_536)
        XCTAssertEqual(constants.paddedRows / config.splitNgramParts, 2_500_012)
        XCTAssertEqual(constants.paddedRows % config.splitNgramParts, 0)

        // A second PLE layer reseeds and continues the prime sequence.
        XCTAssertEqual(
            Qwen4ExpNGramHash.multipliers(vocabSize: 64, ngramSize: 3, seed: 1234 + 10007),
            [125_324_136_723_312_099, 74_677_362_759_709_025, 33_105_235_034_841_755])
    }

    func testConfigurationRejectsWhatThePortDoesNotImplement() throws {
        func rejects(_ json: String, file: StaticString = #filePath, line: UInt = #line) {
            XCTAssertThrowsError(
                try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
                    .validateModelConfiguration(), file: file, line: line)
        }
        // A PLE layer keeps its state in a linear-attention cache.
        rejects(tinyTextJSON(pleLayerIds: "[4]"))
        rejects(tinyTextJSON(layerTypes: #"["linear_attention", "linear_attention", "sliding_attention", "full_attention"]"#))
        rejects(tinyTextJSON(extra: #", "norm_topk_prob": false"#))
        rejects(tinyTextJSON(extra: #", "attention_bias": true"#))
        rejects(tinyTextJSON().replacingOccurrences(of: #""indexer_kv_heads": 1"#, with: #""indexer_kv_heads": 2"#))
        rejects(tinyTextJSON().replacingOccurrences(of: #""type": "default""#, with: #""type": "yarn""#))
        XCTAssertNoThrow(try tinyConfiguration(nested: true))
    }

    // MARK: - Weights (Metal)

    func testNGramRowsMatchReferenceWithAndWithoutCache() throws {
        let model = Qwen4ExpTextModel(try tinyConfiguration())
        let embedding = try XCTUnwrap(model.model.layers[1].ple?.pleEmbedding)
        XCTAssertNil(model.model.layers[0].ple)
        XCTAssertEqual(embedding.headVocabSizes, [101, 103, 107, 109])
        XCTAssertEqual(embedding.table.shards.count, 4)
        XCTAssertEqual(embedding.table.shardRows(0), 105)

        let ids = MLXArray([5, 63, 1, 7, 7, 12, 1, 1, 40] as [Int32]).reshaped(1, 9)
        let expected: [[Int64]] = [
            [27, 170, 282, 375], [49, 155, 226, 357], [7, 186, 232, 356],
            [61, 124, 270, 346], [66, 134, 218, 388], [21, 117, 292, 381],
            [34, 176, 241, 395], [63, 164, 294, 326], [68, 179, 205, 414],
        ]
        let whole = embedding.rows(ids, cache: nil)
        XCTAssertEqual(whole.shape, [1, 9, 4])
        XCTAssertEqual(whole.asArray(Int64.self), expected.flatMap { $0 })

        // The context slot carries the last two tokens across calls.
        let cache = ArraysCache(size: 4)
        let first = embedding.rows(ids[0..., ..<4], cache: cache)
        let rest = embedding.rows(ids[0..., 4...], cache: cache)
        XCTAssertEqual(
            concatenated([first, rest], axis: 1).asArray(Int64.self), expected.flatMap { $0 })
    }

    func testShardedEmbeddingGathersAcrossShards() throws {
        let table = Qwen4ExpShardedEmbedding(totalRows: 10, dimensions: 2, shardCount: 3)
        XCTAssertEqual((0 ..< 3).map(table.shardRows), [4, 3, 3])
        // Row r holds (r, -r) in whichever shard it lives.
        var start = 0
        var updates = [(String, MLXArray)]()
        for i in 0 ..< 3 {
            let rows = table.shardRows(i)
            let values = (start ..< start + rows).flatMap { [Float($0), -Float($0)] }
            updates.append(("shards.\(i).weight", MLXArray(values).reshaped(rows, 2)))
            start += rows
        }
        try table.update(parameters: ModuleParameters.unflattened(updates), verify: [.all])

        let indices = MLXArray([9, 0, 4, 3, 7, 4, 6] as [Int64]).reshaped(1, 7)
        let out = table(indices)
        XCTAssertEqual(out.shape, [1, 7, 2])
        XCTAssertEqual(
            out.asArray(Float.self), [9, -9, 0, 0, 4, -4, 3, -3, 7, -7, 4, -4, 6, -6])
    }

    func testSanitizeMapsCheckpointLayoutsAndRefusesUnknowns() throws {
        let config = try tinyConfiguration(nested: true)
        let model = Qwen4ExpModel(config)
        let ple = "language_model.model.layers.1.ple.ple_embedding."
        let embedding = try XCTUnwrap(model.languageModel.model.layers[1].ple?.pleEmbedding)

        var weights: [String: MLXArray] = [
            "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight": .zeros([64]),
            "language_model.model.layers.0.linear_attn.norm.weight": .zeros([16]),
            "language_model.model.layers.3.self_attn.q_norm.weight": .zeros([16]),
            "language_model.model.layers.1.ple.conv1d.weight": .zeros([64, 4, 1]),
            "language_model.model.layers.0.linear_attn.conv1d.weight": .zeros([160, 1, 4]),
            ple + "ngram_embedding.shards.0.weight": .zeros([105, 2]),
            ple + "ngram_embedding.weight_scale": .ones([1]),
            ple + "layer_multipliers": MLXArray(embedding.multipliers.map { Int64($0) }),
            ple + "ngram_heads_vocab_sizes": MLXArray(embedding.headVocabSizes.map { Int64($0) }),
            ple + "ngram_heads_offsets": MLXArray(embedding.headOffsets.map { Int64($0) }),
            "mtp.layers.0.mlp.gate.weight": .zeros([4, 32]),
            "vision_tower.blocks.0.weight": .zeros([2]),
        ]
        var out = model.sanitize(weights: weights)
        XCTAssertFalse(out.keys.contains { $0.contains(Qwen4ExpWeights.refusalPrefix) }, "\(out.keys)")
        XCTAssertFalse(out.keys.contains { $0.contains("mtp") || $0.contains("vision") })
        XCTAssertFalse(out.keys.contains { $0.hasSuffix("weight_scale") })
        XCTAssertFalse(out.keys.contains { $0.contains("multipliers") || $0.contains("heads_") })
        // mlx-vlm layout (`shards.{i}` names): the `1 + w` norms are folded.
        let lm = "language_model.model.layers."
        XCTAssertEqual(out[lm + "0.attn_hyper_connection.hc_norm.weight"]?.min().item(Float.self), 1)
        XCTAssertEqual(out[lm + "3.self_attn.q_norm.weight"]?.min().item(Float.self), 1)
        XCTAssertEqual(out[lm + "0.linear_attn.norm.weight"]?.max().item(Float.self), 0)
        // torch (C, 1, K) conv weights become (C, K, 1); MLX ones stay.
        XCTAssertEqual(out[lm + "0.linear_attn.conv1d.weight"]?.shape, [160, 4, 1])
        XCTAssertEqual(out[lm + "1.ple.conv1d.weight"]?.shape, [64, 4, 1])
        XCTAssertNotNil(out[lm + "1.ple.ple_embedding.ngram_embedding.shards.0.weight"])

        // HF layout: `shard_{i}` names and stacked experts, norms folded.
        let hf: [String: MLXArray] = [
            "model.language_model.layers.0.mlp.experts.gate_up_proj": .zeros([4, 32, 32]),
            "model.language_model.layers.0.mlp.experts.down_proj": .zeros([4, 32, 16]),
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight":
                .zeros([105, 2]),
            "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight": .zeros([64]),
        ]
        out = model.sanitize(weights: hf)
        XCTAssertEqual(out[lm + "0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(out[lm + "0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(out[lm + "0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 32, 16])
        XCTAssertNotNil(out[lm + "1.ple.ple_embedding.ngram_embedding.shards.3.weight"])
        XCTAssertEqual(out[lm + "0.mlp_hyper_connection.hc_norm.weight"]?.min().item(Float.self), 1)

        // Wrong hash constants, an FP8 table and per-expert tensors refuse the load.
        weights = [
            ple + "layer_multipliers": MLXArray([1, 2, 3] as [Int64]),
            ple + "ngram_embedding.weight_scale": .ones([1]),
            ple + "ngram_embedding.shard_0.weight": .zeros([105, 2], dtype: .uint8),
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": .zeros([16, 32]),
        ]
        weights.removeValue(forKey: ple + "ngram_embedding.shard_0.weight")
        out = model.sanitize(weights: weights)
        let refusals = out.keys.filter { $0.contains(Qwen4ExpWeights.refusalPrefix) }.sorted()
        XCTAssertEqual(refusals.count, 3, "\(refusals)")
        XCTAssertTrue(refusals.contains { $0.contains("ngram_hash_constants_do_not_match_config") })
        XCTAssertTrue(refusals.contains { $0.contains("fp8_ngram_table_not_supported") })
        XCTAssertTrue(refusals.contains { $0.contains("unsupported_expert_layout") })
        XCTAssertThrowsError(
            try model.update(parameters: ModuleParameters.unflattened(out), verify: [.noUnusedKeys]))
    }

    func testPerLayerQuantizationMapsCheckpointNames() throws {
        let model = Qwen4ExpTextModel(try tinyConfiguration())
        let option = BaseConfiguration.QuantizationOption.quantize(
            .init(groupSize: 32, bits: 4))
        let map = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_2": option,
                "language_model.lm_head": .quantize(.init(groupSize: 64, bits: 8)),
                "mtp.fc_hidden": .quantize(.init(groupSize: 64, bits: 8)),
            ])
        let mapped = model.sanitize(perLayerQuantization: map)
        XCTAssertEqual(
            mapped.quantization(layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.2")?
                .groupSize, 32)
        XCTAssertEqual(mapped.quantization(layer: "lm_head")?.bits, 8)
    }

    /// The oQ4e build quantizes a hyper-connection's mix down and block inject
    /// at different widths (layer 16: down 4-bit, unlisted in the config; inject
    /// 5-bit, listed). mlx-lm merges the two and cannot load it; this port keeps
    /// them apart. Through `loadWeights`, each part gets its own config entry
    /// and computes its rows of the dequantized matmul; a missing entry fails
    /// the load instead of reading a part at the wrong width.
    func testMixedPrecisionBlockInjectLoadsApartFromTheMixDown() throws {
        // CPU: the quantized matmuls of this check need no Metal.
        try Device.withDefaultDevice(.cpu) { try checkMixedPrecisionBlockInject() }
    }

    private func checkMixedPrecisionBlockInject() throws {
        let config = try tinyConfiguration(nested: true)
        let source = Qwen4ExpModel(config)
        randomize(source)
        // Layer 0: quantize(model:) updates modules by path, and a sparse layer list
        // (layer 1 alone) is not an array to it.
        let hc = "language_model.model.layers.0.attn_hyper_connection."
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let (down, inject) = withRandomState(MLXRandom.RandomState(seed: 21)) {
            (MLXRandom.normal([8, 64]), MLXRandom.normal([2, 64]))
        }
        let qd = quantized(down, groupSize: 64, bits: 4)
        let qi = quantized(inject, groupSize: 32, bits: 8)
        weights[hc + "input_mix_weight_down.weight"] = qd.wq
        weights[hc + "input_mix_weight_down.scales"] = qd.scales
        weights[hc + "input_mix_weight_down.biases"] = qd.biases
        weights[hc + "block_inject_weight.weight"] = qi.wq
        weights[hc + "block_inject_weight.scales"] = qi.scales
        weights[hc + "block_inject_weight.biases"] = qi.biases

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-mixed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        // As the checkpoint's config: the inject listed, the mix down at the default.
        let listed = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                String(hc.dropLast()) + ".block_inject_weight": .quantize(
                    .init(groupSize: 32, bits: 8))
            ])
        let model = Qwen4ExpModel(config)
        try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: listed)

        let residual = model.languageModel.model.layers[0].attnHyperConnection
        let downLayer = try XCTUnwrap(residual.inputMixWeightDown as? QuantizedLinear)
        let injectLayer = try XCTUnwrap(residual.blockInjectWeight as? QuantizedLinear)
        XCTAssertEqual([downLayer.bits, downLayer.groupSize], [4, 64])
        XCTAssertEqual([injectLayer.bits, injectLayer.groupSize], [8, 32])

        let hyper = withRandomState(MLXRandom.RandomState(seed: 22)) {
            MLXRandom.normal([1, 3, 64])
        }
        let normed = residual.hcNorm(hyper)
        let downRows = dequantized(
            qd.wq, scales: qd.scales, biases: qd.biases, groupSize: 64, bits: 4)
        let injectRows = dequantized(
            qi.wq, scales: qi.scales, biases: qi.biases, groupSize: 32, bits: 8)
        var w = silu(matmul(normed, downRows.T) / 2)
        w = sigmoid(residual.inputMixWeightUp(w)).reshaped(1, 3, 2, 32)
        let wantMixed = (w * normed.reshaped(1, 3, 2, 32)).mean(axis: -2)
        let wantInject = 2 * sigmoid(matmul(normed, injectRows.T) / 2)
        let (mixed, injectScale) = residual.combine(hyper)
        XCTAssertTrue(
            allClose(injectScale, wantInject, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "inject max diff \(abs(injectScale - wantInject).max().item(Float.self))")
        XCTAssertTrue(
            allClose(mixed, wantMixed, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "mixed max diff \(abs(mixed - wantMixed).max().item(Float.self))")

        // Without the inject's entry it would be read at the default 4 bits: refused.
        let unlisted = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:])
        XCTAssertThrowsError(
            try loadWeights(
                modelDirectory: directory, model: Qwen4ExpModel(config),
                perLayerQuantization: unlisted))
    }

    /// On CPU: the Metal float32 GEMM (a prefill of many tokens) and GEMV
    /// (one token) differ by about 1e-3 relative on this hardware (a plain
    /// `[20, 64] x [64, 128]` matmul: GEMM row 0 vs GEMV 1.8e-2 at |x| 27;
    /// GEMV vs CPU 6e-6), which put a constant 2-5e-3 on every decoded
    /// position from the first one, cached state or not. On CPU decode
    /// matches the prefill within 3e-6, so the cached state is exact.
    func testForwardShapesAndIncrementalDecodeMatchesPrefill() throws {
        try Device.withDefaultDevice(.cpu) { try checkIncrementalDecodeMatchesPrefill() }
    }

    private func checkIncrementalDecodeMatchesPrefill() throws {
        let model = Qwen4ExpTextModel(try tinyConfiguration())
        randomize(model)
        let tokens = withRandomState(MLXRandom.RandomState(seed: 5)) {
            MLXRandom.randInt(0 ..< 64, [1, 20]).asType(.int32)
        }

        let full = model(tokens, cache: nil)
        XCTAssertEqual(full.shape, [1, 20, 64])

        // 20 tokens is past the indexer budget (8), so the sparse mask is active.
        let cache = try model.newCache(parameters: nil)
        XCTAssertTrue(cache[3] is Qwen4ExpAttentionCache)
        XCTAssertTrue(cache[1] is ArraysCache && !(cache[1] is MambaCache))
        XCTAssertTrue(cache[0] is MambaCache)
        let prefill = model(tokens[0..., ..<12], cache: cache)
        var steps = [prefill]
        for t in 12 ..< 20 {
            steps.append(model(tokens[0..., t ..< (t + 1)], cache: cache))
        }
        let incremental = concatenated(steps, axis: 1)
        XCTAssertEqual(cache[3].offset, 20)
        XCTAssertTrue(
            allClose(incremental, full, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "max diff \(abs(incremental - full).max().item(Float.self))")
    }

    func testIndexerMaskKeepsBudgetAndOwnBlock() throws {
        let config = try tinyConfiguration()
        let indexer = Qwen4ExpQSAIndexer(config)
        randomize(indexer)
        let x = withRandomState(MLXRandom.RandomState(seed: 9)) {
            MLXRandom.normal([1, 20, 32])
        }
        XCTAssertNil(indexer(x[0..., ..<8], cache: nil), "within the budget the causal mask is used")
        let mask = try XCTUnwrap(indexer(x, cache: nil))
        XCTAssertEqual(mask.shape, [1, 1, 20, 20])
        let rows = mask.asArray(Bool.self)
        for q in 0 ..< 20 {
            let row = Array(rows[(q * 20) ..< (q * 20 + 20)])
            // Causal; at most 4 picked blocks of 2 plus the own partial block.
            XCTAssertFalse(row[(q + 1)...].contains(true), "query \(q) sees the future")
            XCTAssertLessThanOrEqual(row.filter { $0 }.count, 8 + 1)
            if q < 9 {
                // Every complete block fits the budget: the mask is causal.
                XCTAssertEqual(row, (0 ..< 20).map { $0 <= q }, "query \(q)")
            }
        }
    }
}
