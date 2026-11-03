import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class MiniMaxM3Tests: XCTestCase {

    // MARK: - Fixtures

    /// Four layers (one dense MLP, three MoE), every layer block-sparse with
    /// blocks of 4 tokens and 2 selected blocks, so any context past 8 tokens
    /// drops blocks. Dims are multiples of 32 for group-32 quantization.
    private func textConfigJSON(topkBlocks: Int = 2, sparse: [Int] = [1, 1, 1, 1]) -> String {
        """
        {
            "model_type": "minimax_m3", "vocab_size": 64, "hidden_size": 64,
            "intermediate_size": 32, "dense_intermediate_size": 64,
            "shared_intermediate_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "rotary_dim": 16, "rope_theta": 10000000, "rms_norm_eps": 1e-6,
            "use_gemma_norm": true, "use_qk_norm": true, "qk_norm_type": "per_head",
            "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "scoring_func": "sigmoid", "use_routing_bias": true,
            "routed_scaling_factor": 2.0, "moe_layer_freq": [0, 1, 1, 1],
            "pack_shared_expert": false, "tie_word_embeddings": false,
            "sparse_attention_config": {
                "use_sparse_attention": true, "sparse_index_dim": 32,
                "sparse_num_index_heads": 2, "sparse_topk_blocks": \(topkBlocks),
                "sparse_block_size": 4, "sparse_score_type": "max",
                "sparse_init_block": 0, "sparse_local_block": 1,
                "sparse_attention_freq": \(sparse), "sparse_disable_index_value": \(sparse)
            }
        }
        """
    }

    private func configuration(topkBlocks: Int = 2, sparse: [Int] = [1, 1, 1, 1], nested: Bool = false)
        throws -> MiniMaxM3Configuration
    {
        let text = textConfigJSON(topkBlocks: topkBlocks, sparse: sparse)
        let json = nested ? #"{"model_type": "minimax_m3_vl", "text_config": \#(text)}"# : text
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        return config
    }

    /// Seeded noise everywhere, so routing and block selection have no ties.
    private func randomize(_ model: Module, seed: UInt64 = 7) {
        let parameters = model.parameters().flattened()
        let updated: [(String, MLXArray)] = withRandomState(MLXRandom.RandomState(seed: seed)) {
            parameters.map { key, value in
                let noise = MLXRandom.normal(value.shape).asType(value.dtype)
                if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
                    return (key, 0.1 * noise)
                }
                return (key, 0.3 * noise)
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    private func tokens(_ count: Int, seed: UInt64 = 3) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            MLXRandom.randInt(0 ..< 64, [1, count]).asType(.int32)
        }
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: - Configuration

    /// The published M3.1-Flash text config: dense first three layers, every
    /// layer sparse, partial RoPE of 64 dims, 4 index heads over 16 blocks of 128.
    func testArtifactConfigurationDerivesLayout() throws {
        let freq = (0 ..< 60).map { $0 < 3 ? "0" : "1" }.joined(separator: ",")
        let ones = Array(repeating: "1", count: 60).joined(separator: ",")
        let json = """
            {"model_type": "minimax_m3_vl", "text_config": {
                "hidden_size": 6144, "intermediate_size": 3072, "num_hidden_layers": 60,
                "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
                "vocab_size": 200064, "rms_norm_eps": 1e-06, "use_gemma_norm": true,
                "attention_output_gate": false, "rope_theta": 10000000, "rotary_dim": 64,
                "partial_rotary_factor": 0.5, "hidden_act": "swigluoai", "use_qk_norm": true,
                "tie_word_embeddings": false, "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072, "num_local_experts": 128,
                "num_experts_per_tok": 4, "n_shared_experts": 1, "scoring_func": "sigmoid",
                "use_routing_bias": true, "moe_layer_freq": [\(freq)], "qk_norm_type": "per_head",
                "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "routed_scaling_factor": 2.0,
                "sparse_attention_config": {"use_sparse_attention": true, "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4, "sparse_topk_blocks": 16,
                    "sparse_block_size": 128, "sparse_score_type": "max", "sparse_init_block": 0,
                    "sparse_local_block": 1, "sparse_attention_freq": [\(ones)],
                    "sparse_disable_index_value": [\(ones)]},
                "pack_shared_expert": false, "routed_expert_scales": true}}
            """
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.moeLayers.prefix(4), [false, false, false, true])
        XCTAssertEqual(config.moeLayers.filter { $0 }.count, 57)
        XCTAssertTrue(config.sparseLayers.allSatisfy { $0 })
        XCTAssertEqual(config.rotaryDim, 64)
        XCTAssertEqual(config.ropeTheta, 10_000_000)
        XCTAssertEqual(config.swigluBeta, 1.0)
        XCTAssertEqual(
            [config.indexHeads, config.indexHeadDim, config.blockSize, config.topkBlocks,
             config.initBlocks, config.localBlocks],
            [4, 128, 128, 16, 0, 1])
        XCTAssertFalse(config.packSharedExpert)
    }

    /// Layouts this port does not implement are configuration errors, never a
    /// silently different model.
    func testConfigurationRejectsUnsupportedLayouts() throws {
        func decode(_ json: String) throws -> MiniMaxM3Configuration {
            try JSONDecoder().decode(MiniMaxM3Configuration.self, from: Data(json.utf8))
        }
        let base = textConfigJSON()
        for (old, new) in [
            (#""pack_shared_expert": false"#, #""pack_shared_expert": true"#),
            (#""hidden_act": "swigluoai""#, #""hidden_act": "silu""#),
            (#""sparse_score_type": "max""#, #""sparse_score_type": "lse""#),
            (#""qk_norm_type": "per_head""#, #""qk_norm_type": "per_layer""#),
        ] {
            let config = try decode(base.replacingOccurrences(of: old, with: new))
            XCTAssertThrowsError(try config.validateModelConfiguration(), new)
        }
        // pack_shared_expert unset with equal expert sizes means packed (mlx-vlm).
        let unset = try decode(
            base.replacingOccurrences(of: #""pack_shared_expert": false,"#, with: "")
                .replacingOccurrences(of: #""shared_intermediate_size": 64"#,
                    with: #""shared_intermediate_size": 32"#))
        XCTAssertThrowsError(try unset.validateModelConfiguration())
        let nvfp4 = try decode(
            base.replacingOccurrences(of: #""pack_shared_expert": false"#,
                with: #""pack_shared_expert": false, "nvfp4_experts": true"#))
        XCTAssertThrowsError(try nvfp4.validateModelConfiguration())
    }

    // MARK: - Forward and cache

    func testForwardShapesWithAndWithoutCache() throws {
        for nested in [false, true] {
            let config = try configuration(nested: nested)
            let model: any LanguageModel =
                nested ? MiniMaxM3Model(config) : MiniMaxM3TextModel(config)
            randomize(model)
            let input = tokens(13)

            let uncached = model(input, cache: nil)
            XCTAssertEqual(uncached.shape, [1, 13, 64])

            let cache = try model.newCache(parameters: nil)
            XCTAssertEqual(cache.count, 4)
            XCTAssertTrue(cache.allSatisfy { $0 is MiniMaxM3Cache })
            let prefill = model(input, cache: cache)
            let step = model(MLXArray([Int32(5)]).reshaped(1, 1), cache: cache)
            eval(prefill, step)
            XCTAssertEqual(prefill.shape, [1, 13, 64])
            XCTAssertEqual(step.shape, [1, 1, 64])
            XCTAssertEqual(cache[0].offset, 14)
            XCTAssertEqual(cache[0].state.map(\.shape), [[1, 2, 14, 32], [1, 2, 14, 32], [1, 1, 14, 32]])
            XCTAssertLessThan(maxAbsDifference(uncached, prefill), 1e-4)
        }
    }

    /// Dense layers (sparse flag off) use a plain KV cache.
    func testDenseLayersUseSimpleCache() throws {
        let model = MiniMaxM3TextModel(try configuration(sparse: [0, 1, 0, 1]))
        let cache = try model.newCache(parameters: nil)
        XCTAssertTrue(cache[0] is KVCacheSimple)
        XCTAssertTrue(cache[1] is MiniMaxM3Cache)
        XCTAssertNil(model.model.layers[0].selfAttn.indexQProj)
        XCTAssertNotNil(model.model.layers[1].selfAttn.indexQProj)
    }

    func testIncrementalDecodeMatchesPrefill() throws {
        let model = MiniMaxM3TextModel(try configuration())
        randomize(model)
        let input = tokens(40)

        let prefill = model(input, cache: try model.newCache(parameters: nil))
        eval(prefill)

        let cache = try model.newCache(parameters: nil)
        var maxDifference: Float = 0
        for t in 0 ..< 40 {
            let step = model(input[0..., t ..< (t + 1)], cache: cache)
            eval(step)
            maxDifference = max(maxDifference, maxAbsDifference(step[0, 0], prefill[0, t]))
        }
        XCTAssertLessThan(maxDifference, 1e-3, "decode diverged from prefill")
    }

    /// Chunk boundaries that split key blocks must not change the logits.
    func testChunkedPrefillMatchesSinglePrefill() throws {
        let model = MiniMaxM3TextModel(try configuration())
        randomize(model)
        let input = tokens(30)
        let next = MLXArray([Int32(11)]).reshaped(1, 1)

        let single = try model.newCache(parameters: nil)
        let full = model(input, cache: single)
        let fullNext = model(next, cache: single)
        eval(full, fullNext)

        let chunked = try model.newCache(parameters: nil)
        var pieces = [MLXArray]()
        for (start, end) in [(0, 7), (7, 18), (18, 30)] {
            pieces.append(model(input[0..., start ..< end], cache: chunked))
        }
        let chunkedNext = model(next, cache: chunked)
        eval(pieces, chunkedNext)

        XCTAssertLessThan(maxAbsDifference(concatenated(pieces, axis: 1), full), 1e-3)
        XCTAssertLessThan(maxAbsDifference(chunkedNext, fullNext), 1e-3)
    }

    /// Past `topk_blocks * block_size` tokens the indexer drops blocks: the
    /// same weights with every block selectable give different logits, and
    /// identical ones up to that length.
    func testBlockSelectionIsActivePastTopk() throws {
        let sparse = MiniMaxM3TextModel(try configuration(topkBlocks: 2))
        randomize(sparse)
        let dense = MiniMaxM3TextModel(try configuration(topkBlocks: 16))
        try dense.update(parameters: sparse.parameters(), verify: [.all])
        let input = tokens(40)

        let sparseLogits = sparse(input, cache: try sparse.newCache(parameters: nil))
        let denseLogits = dense(input, cache: try dense.newCache(parameters: nil))
        eval(sparseLogits, denseLogits)
        XCTAssertLessThan(
            maxAbsDifference(sparseLogits[0..., ..<8], denseLogits[0..., ..<8]), 1e-4)
        XCTAssertGreaterThan(maxAbsDifference(sparseLogits, denseLogits), 1e-2)
    }

    /// Selection keeps the query's own (local) block, never a future block,
    /// and leaves unused slots at -1 while fewer blocks exist.
    func testSelectBlocksKeepsLocalAndCausalBlocks() throws {
        let model = MiniMaxM3TextModel(try configuration())
        randomize(model)
        let attention = model.model.layers[1].selfAttn
        let (L, total) = (12, 20)
        let queries = withRandomState(MLXRandom.RandomState(seed: 4)) {
            MLXRandom.normal([1, 2, L, 32])
        }
        let keys = withRandomState(MLXRandom.RandomState(seed: 5)) {
            MLXRandom.normal([1, 1, total, 32])
        }
        let blocks = attention.selectBlocks(
            queries: queries, keys: keys, qStart: total - L, sorted: true)
        XCTAssertEqual(blocks.shape, [1, 2, L, 2])
        let values = blocks.asArray(Int32.self)
        for head in 0 ..< 2 {
            for row in 0 ..< L {
                let position = total - L + row
                let selected = (0 ..< 2).map { values[(head * L + row) * 2 + $0] }
                XCTAssertTrue(selected.contains(Int32(position / 4)), "local block kept")
                XCTAssertTrue(selected.allSatisfy { $0 <= Int32(position / 4) }, "causal")
                XCTAssertEqual(selected, selected.sorted { ($0 < 0 ? Int32.max : $0) < ($1 < 0 ? Int32.max : $1) })
            }
        }
        // Position 2 sits in block 0: one valid block, the other slot unused.
        let early = attention.selectBlocks(
            queries: queries[0..., 0..., ..<1], keys: keys[0..., 0..., ..<9], qStart: 2,
            sorted: true)
        XCTAssertEqual(early[0, 0, 0].asArray(Int32.self), [0, -1])
    }

    /// The block-sparse kernel equals masked SDPA over the same selection.
    func testSparseKernelMatchesMaskedAttention() throws {
        let model = MiniMaxM3TextModel(try configuration())
        let attention = model.model.layers[1].selfAttn
        let (B, Hq, Hkv, L, total, D) = (1, 4, 2, 6, 24, 32)
        let (q, k, v, iq, ik) = withRandomState(MLXRandom.RandomState(seed: 11)) {
            (
                MLXRandom.normal([B, Hq, L, D]).asType(.bfloat16),
                MLXRandom.normal([B, Hkv, total, D]).asType(.bfloat16),
                MLXRandom.normal([B, Hkv, total, D]).asType(.bfloat16),
                MLXRandom.normal([B, 2, L, D]), MLXRandom.normal([B, 1, total, D])
            )
        }
        let qStart = total - L
        let blocks = attention.selectBlocks(queries: iq, keys: ik, qStart: qStart, sorted: true)
        let positions = MLXArray(Int32(qStart) ..< Int32(total)).reshaped(1, L)
        let kernel = try XCTUnwrap(
            miniMaxM3SparseAttention(
                queries: q, keys: k, values: v, blockIndices: blocks, queryPositions: positions,
                scale: attention.scale, blockSize: 4))
        let mask = attention.blockMask(blocks, total: total, qStart: qStart)
        let reference = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32),
            scale: attention.scale, mask: .array(mask))
        XCTAssertEqual(kernel.shape, [B, Hq, L, D])
        XCTAssertLessThan(maxAbsDifference(kernel, reference), 2e-2)
    }

    /// Trimming the cache and re-feeding the same tokens reproduces the logits;
    /// a copy is independent of the original.
    func testCacheTrimCopyAndStateRoundTrip() throws {
        let model = MiniMaxM3TextModel(try configuration())
        randomize(model)
        let input = tokens(24)
        let cache = try model.newCache(parameters: nil)
        eval(model(input[0..., ..<20], cache: cache))
        let copy = cache.map { $0.copy() }
        let restored = try model.newCache(parameters: nil)
        for (target, source) in zip(restored, cache) {
            var target = target
            target.state = source.state
        }

        let reference = model(input[0..., 20...], cache: cache)
        eval(reference)
        for layer in cache {
            XCTAssertEqual(layer.trim(4), 4)
        }
        let again = model(input[0..., 20...], cache: cache)
        let fromCopy = model(input[0..., 20...], cache: copy)
        let fromState = model(input[0..., 20...], cache: restored)
        eval(again, fromCopy, fromState)
        XCTAssertEqual(maxAbsDifference(again, reference), 0)
        XCTAssertEqual(maxAbsDifference(fromCopy, reference), 0)
        XCTAssertEqual(maxAbsDifference(fromState, reference), 0)
        XCTAssertEqual(copy[0].offset, 24)
    }

    /// Cached `1 + weight` (prepare) is the same float32 scale as the fallback.
    func testPreparedNormsMatchUnprepared() throws {
        let model = MiniMaxM3TextModel(try configuration())
        randomize(model)
        let input = tokens(10)
        let before = model(input, cache: nil)
        try model.prepare()
        let after = model(input, cache: nil)
        eval(before, after)
        XCTAssertEqual(maxAbsDifference(before, after), 0)
    }

    /// The attention ropes q, k and the index heads in one call; partial
    /// RoPE over the concatenated heads is bitwise the per-tensor RoPE.
    func testBatchedRopeMatchesPerTensorRope() throws {
        let rope = RoPE(dimensions: 64, traditional: false, base: 1e7)
        for (L, offset) in [(1, 9000), (37, 0), (300, 4096)] {
            let shapes = [64, 4, 1, 4].map { [1, L, $0, 128] }
            let parts = shapes.map { MLXRandom.normal($0).asType(.bfloat16) }
            let separate = parts.map { rope($0.transposed(0, 2, 1, 3), offset: offset) }
            let all = rope(concatenated(parts, axis: 2).transposed(0, 2, 1, 3), offset: offset)
            var start = 0
            for (part, expected) in zip(parts, separate) {
                let count = part.dim(2)
                XCTAssertTrue(
                    arrayEqual(all[0..., start ..< start + count], expected).item(Bool.self),
                    "L=\(L) offset=\(offset) heads \(start)..<\(start + count)")
                start += count
            }
        }
    }

    // MARK: - Checkpoint loading

    /// An oQ3e-style `minimax_m3_vl` checkpoint: 3-bit affine routed experts
    /// by default, mxfp8 indexer projections and shared experts, 4/5-bit
    /// affine attention, 8-bit embeddings and head, a float32 router, plus
    /// vision tensors the text model drops. Loading it through the
    /// per-layer quantization map reproduces the quantized reference exactly.
    func testMixedQuantizationCheckpointLoads() throws {
        let config = try configuration(nested: true)
        let reference = MiniMaxM3Model(config)
        randomize(reference)

        var overrides = [String: (Int, Int, QuantizationMode)]()
        for (path, module) in reference.leafModules().flattened() {
            guard module is Linear || module is Embedding || module is SwitchLinear else {
                continue
            }
            if path.hasSuffix("block_sparse_moe.gate") { continue }
            if path.contains("index_") || path.contains("shared_experts") {
                overrides[path] = (32, 8, .mxfp8)
            } else if path.hasSuffix("embed_tokens") || path.hasSuffix("lm_head") {
                overrides[path] = (32, 8, .affine)
            } else if path.contains("self_attn.") {
                overrides[path] = (32, path.contains("layers.1.") ? 5 : 4, .affine)
            } else if path.contains(".mlp.") {
                overrides[path] = (32, 6, .affine)
            }
        }
        quantize(model: reference) { path, module in
            if path.hasSuffix("block_sparse_moe.gate") { return nil }
            guard module is Linear || module is Embedding || module is SwitchLinear else {
                return nil
            }
            return overrides[path] ?? (32, 3, .affine)
        }
        eval(reference)

        var arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        arrays["vision_tower.vision_model.embeddings.patch_embedding.weight"] = MLXArray.zeros([4, 4])
        arrays["multi_modal_projector.linear_1.weight"] = MLXArray.zeros([4, 4])
        arrays["patch_merge_mlp.linear_1.scales"] = MLXArray.zeros([4, 4])
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "minimax-m3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: arrays, url: directory.appending(component: "model.safetensors"))

        let entries = overrides.map { path, q in
            #""\#(path)": {"group_size": \#(q.0), "bits": \#(q.1), "mode": "\#(q.2 == .mxfp8 ? "mxfp8" : "affine")"}"#
        }.joined(separator: ",\n")
        let base = try JSONDecoder.json5().decode(
            BaseConfiguration.self,
            from: Data(
                """
                {"model_type": "minimax_m3_vl",
                 "quantization": {"group_size": 32, "bits": 3, "mode": "affine", \(entries)}}
                """.utf8))

        let model = MiniMaxM3Model(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let index = try XCTUnwrap(
            modules["language_model.model.layers.2.self_attn.index_q_proj"] as? QuantizedLinear)
        XCTAssertEqual(index.mode, .mxfp8)
        XCTAssertNil(index.biases)
        let experts = try XCTUnwrap(
            modules["language_model.model.layers.2.block_sparse_moe.switch_mlp.up_proj"]
                as? QuantizedSwitchLinear)
        XCTAssertEqual([experts.bits, experts.groupSize], [3, 32])
        let gate = try XCTUnwrap(modules["language_model.model.layers.2.block_sparse_moe.gate"])
        XCTAssertFalse(gate is QuantizedLinear)
        XCTAssertEqual((gate as? Linear)?.weight.dtype, .float32)

        let input = tokens(20)
        let expected = reference(input, cache: try reference.newCache(parameters: nil))
        let actual = model(input, cache: try model.newCache(parameters: nil))
        eval(expected, actual)
        XCTAssertEqual(maxAbsDifference(actual, expected), 0)
    }

    /// Raw per-expert tensors stack into `switch_mlp`.
    func testSanitizeStacksExpertsAndDropsVision() throws {
        let config = try configuration(nested: true)
        let model = MiniMaxM3Model(config)
        var weights = [String: MLXArray]()
        for e in 0 ..< 4 {
            for name in ["w1", "w2", "w3"] {
                weights["language_model.model.layers.1.block_sparse_moe.experts.\(e).\(name).weight"] =
                    MLXArray.ones([2, 2]) * Float(e)
            }
        }
        weights["vision_tower.x.weight"] = MLXArray.zeros([1])
        weights["model.language_model.model.norm.weight"] = MLXArray.zeros([64])
        let sanitized = model.sanitize(weights: weights)
        let gate = try XCTUnwrap(
            sanitized["language_model.model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(gate.shape, [4, 2, 2])
        XCTAssertEqual(gate[3, 0, 0].item(Float.self), 3)
        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])
        XCTAssertFalse(sanitized.keys.contains { $0.contains("vision") || $0.contains("experts.0") })
    }
}
