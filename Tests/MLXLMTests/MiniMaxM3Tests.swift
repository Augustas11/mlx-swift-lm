import Foundation
import MLX
@testable import MLXLMCommon
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

    /// Token steps use the M=1 vector product. The 40-token prefill uses the
    /// matrix product. MLX does not make those two kernels agree, and a tight
    /// top-k then turns a few thousandths of projection noise into a different
    /// block set. With every block selectable the logit gap stays the kernel
    /// floor measured on this fixture (under 0.05). A real selection or routing
    /// break is far larger.
    func testIncrementalDecodeMatchesPrefill() throws {
        let model = MiniMaxM3TextModel(try configuration(topkBlocks: 16))
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
        XCTAssertLessThan(maxDifference, 0.05, "decode diverged from prefill by \(maxDifference)")
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

    /// The one-launch norm is bitwise MLX's cast, float32 rms_norm, cast
    /// chain, on single-row and looped row lengths, with and without tails.
    func testFusedNormMatchesMLXChain() throws {
        for dtype in [DType.bfloat16, .float16] {
            for shape in [[3, 7, 128], [2, 6144], [5, 100], [1, 4100], [2368, 128], [4, 4096]] {
                let x = (3 * MLXRandom.normal(shape)).asType(dtype)
                let w = 1 + 0.1 * MLXRandom.normal([shape.last!])
                let expected = MLXFast.rmsNorm(x, weight: w, eps: 1e-6).asType(dtype)
                let actual = try XCTUnwrap(miniMaxM3RMSNorm(x, scale: w, eps: 1e-6))
                XCTAssertEqual(actual.dtype, dtype)
                XCTAssertTrue(
                    arrayEqual(actual, expected).item(Bool.self), "\(dtype) \(shape)")
            }
        }
    }

    /// The norm's float32 copy output: the half-precision output is bitwise
    /// the one-output kernel's, the copy bitwise its cast to float32.
    func testFusedNormFloat32CopyMatchesCast() throws {
        for dtype in [DType.bfloat16, .float16] {
            for shape in [[1, 1, 6144], [4, 1, 64], [5, 100], [1, 4100], [3, 7, 128]] {
                let x = (3 * MLXRandom.normal(shape)).asType(dtype)
                let w = 1 + 0.1 * MLXRandom.normal([shape.last!])
                let single = try XCTUnwrap(miniMaxM3RMSNorm(x, scale: w, eps: 1e-6))
                let pair = try XCTUnwrap(miniMaxM3RMSNormWithFloat32Copy(x, scale: w, eps: 1e-6))
                XCTAssertEqual(pair.count, 2)
                XCTAssertEqual(pair[1].dtype, .float32)
                XCTAssertTrue(bitwiseEqual(pair[0], single), "\(dtype) \(shape)")
                XCTAssertTrue(bitwiseEqual(pair[1], single.asType(.float32)), "\(dtype) \(shape)")
            }
        }
    }

    /// The multi-tensor head norm is bitwise each MLX norm chain, then the
    /// concatenation over heads and the head-major transpose.
    func testHeadNormsMatchPerTensorChain() throws {
        for (B, L, heads) in [(1, 1, [64, 4, 1]), (1, 37, [64, 4, 1, 4]), (2, 5, [8, 2, 1, 2])] {
            let parts = heads.map { (2 * MLXRandom.normal([B, L, $0, 128])).asType(.bfloat16) }
            let scales = heads.map { _ in 1 + 0.1 * MLXRandom.normal([128]) }
            let expected = concatenated(
                zip(parts, scales).map {
                    MLXFast.rmsNorm($0, weight: $1, eps: 1e-6).asType(.bfloat16)
                }, axis: 2
            ).transposed(0, 2, 1, 3)
            let actual = try XCTUnwrap(miniMaxM3HeadNorms(parts, scales: scales, eps: 1e-6))
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), "B=\(B) L=\(L) \(heads)")
        }
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

    // MARK: - Continuous batching and decode fusions

    /// The test model quantized as the oQ3e checkpoint is (4-bit affine
    /// projections, embedding and head, 3-bit routed experts, a float32
    /// router), with bfloat16 activations.
    private func quantizedModel(topkBlocks: Int = 2, prepared: Bool = true) throws
        -> MiniMaxM3TextModel
    {
        let model = MiniMaxM3TextModel(try configuration(topkBlocks: topkBlocks))
        randomize(model)
        let cast = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            key.contains("block_sparse_moe.gate.") ? (key, value) : (key, value.asType(.bfloat16))
        }
        model.update(parameters: ModuleParameters.unflattened(cast))
        quantize(model: model) { path, module in
            if path.hasSuffix("block_sparse_moe.gate") { return nil }
            guard module is Linear || module is Embedding || module is SwitchLinear else {
                return nil
            }
            return (32, module is SwitchLinear ? 3 : 4, .affine)
        }
        if prepared {
            try model.prepare()
        }
        eval(model)
        return model
    }

    private func bitwiseEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
    }

    /// Continuous batching: four rows decoded in one forward through
    /// `MiniMaxM3CacheBatch` give each row its serial logits bit for bit.
    /// The rows straddle the selection threshold (`topk x block` = 8 keys)
    /// during the decode, and one decodes on the sparse kernel (512+ keys).
    func testBatchedRowsMatchSerialBitwise() throws {
        let model = try quantizedModel()
        let lengths = [3, 7, 12, 515]
        var serial = [[KVCache]]()
        var rows = [[KVCache]]()
        var serialTokens = [Int32]()
        for (index, length) in lengths.enumerated() {
            let prompt = tokens(length, seed: UInt64(20 + index))
            let serialCache = try model.newCache(parameters: nil)
            let rowCache = try model.newCache(parameters: nil)
            let a = model(prompt, cache: serialCache)
            let b = model(prompt, cache: rowCache)
            eval(a, b)
            XCTAssertTrue(bitwiseEqual(a, b))
            serial.append(serialCache)
            rows.append(rowCache)
            serialTokens.append(Int32(argMax(a[0, -1]).item(UInt32.self)))
        }
        var batchTokens = serialTokens
        for step in 0 ..< 6 {
            let batch: [KVCache] = (0 ..< rows[0].count).map { layer in
                MiniMaxM3CacheBatch(rowCaches: rows.map { $0[layer] as! any MiniMaxM3SparseCache })
            }
            let batched = model(MLXArray(batchTokens).reshaped(lengths.count, 1), cache: batch)
            let lone = lengths.indices.map {
                model(MLXArray([serialTokens[$0]]).reshaped(1, 1), cache: serial[$0])
            }
            eval([batched] + lone)
            for row in lengths.indices {
                XCTAssertTrue(
                    bitwiseEqual(batched[row ..< (row + 1)], lone[row]),
                    "row \(row) step \(step): max diff \(maxAbsDifference(batched[row ..< (row + 1)], lone[row]))")
                XCTAssertEqual(rows[row][0].offset, serial[row][0].offset)
            }
            batchTokens = argMax(batched[0..., -1], axis: -1).asType(.int32).asArray(Int32.self)
            serialTokens = lone.map { Int32(argMax($0[0, -1]).item(UInt32.self)) }
            XCTAssertEqual(batchTokens, serialTokens, "step \(step)")
        }
        XCTAssertEqual(rows.map { $0[0].offset }, lengths.map { $0 + 6 })
    }

    /// Right-padded rows (`prepare(lengths:)`): each row attends and caches
    /// only its valid leading columns.
    func testBatchedRowLengthsLimitColumns() throws {
        let model = try quantizedModel()
        let prompts = [tokens(10, seed: 31), tokens(6, seed: 32)]
        var serial = [[KVCache]]()
        var rows = [[KVCache]]()
        for prompt in prompts {
            let a = try model.newCache(parameters: nil)
            let b = try model.newCache(parameters: nil)
            eval(model(prompt, cache: a), model(prompt, cache: b))
            serial.append(a)
            rows.append(b)
        }
        let next = tokens(3, seed: 33)
        let batch: [KVCache] = (0 ..< rows[0].count).map { layer in
            MiniMaxM3CacheBatch(rowCaches: rows.map { $0[layer] as! any MiniMaxM3SparseCache })
        }
        batch.forEach { $0.prepare(lengths: [3, 2]) }
        let batched = model(concatenated([next, next], axis: 0), cache: batch)
        batch.forEach { $0.finalize() }
        let lone = [
            model(next, cache: serial[0]), model(next[0..., ..<2], cache: serial[1]),
        ]
        eval([batched] + lone)
        XCTAssertEqual(rows.map { $0[0].offset }, [13, 8])
        XCTAssertLessThan(maxAbsDifference(batched[0 ..< 1], lone[0]), 5e-2)
        XCTAssertLessThan(maxAbsDifference(batched[1 ..< 2, ..<2], lone[1]), 5e-2)
    }

    /// The compiled reference route: sigmoid, biased top-k by
    /// `argPartition`, normalized unbiased weights, ascending expert order.
    private func referenceRoute(k: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        compile { arrays in
            let scores = sigmoid(arrays[0])
            let biased = scores + arrays[1]
            let indices = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var weights = takeAlong(scores, indices, axis: -1)
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            let order = argSort(indices, axis: -1)
            return [takeAlong(indices, order, axis: -1), takeAlong(weights, order, axis: -1)]
        }
    }

    /// The one-launch route is bitwise the compiled route, ties included
    /// (equal biased scores go to the lower expert).
    func testRouteKernelMatchesCompiledRoute() throws {
        for (experts, k, rows) in [(4, 2, 7), (128, 4, 1), (128, 4, 37)] {
            for biasType in [DType.float32, .bfloat16] {
                var gates = 3 * MLXRandom.normal([rows, experts])
                var bias = (0.5 * MLXRandom.normal([experts])).asType(biasType)
                // Exact ties: expert 1 duplicates expert 3, gate and bias.
                gates[0..., 1] = gates[0..., 3]
                bias[1] = bias[3]
                gates = gates.reshaped(1, rows, experts)
                let expected = referenceRoute(k: k)([gates, bias])
                let actual = try XCTUnwrap(miniMaxM3Route(gates: gates, bias: bias, k: k))
                eval(expected, actual)
                XCTAssertTrue(bitwiseEqual(actual[0], expected[0]), "\(experts)/\(k) \(biasType) indices")
                XCTAssertTrue(bitwiseEqual(actual[1], expected[1]), "\(experts)/\(k) \(biasType) weights")
            }
        }
        // The block's own compiled route agrees too.
        let block = try XCTUnwrap(try quantizedModel().model.layers[1].blockSparseMoe)
        let gates = 3 * MLXRandom.normal([1, 5, 4])
        let bias = try XCTUnwrap(block.eScoreCorrectionBias)
        let expected = block.route([gates, bias])
        let actual = try XCTUnwrap(miniMaxM3Route(gates: gates, bias: bias, k: 2))
        XCTAssertTrue(bitwiseEqual(actual[0], expected[0]))
        XCTAssertTrue(bitwiseEqual(actual[1], expected[1]))
    }

    /// A float32 router fed the norm's float32 copy gives the MoE output it
    /// gives casting the half-precision input itself, single and multi-row.
    func testFloat32RouterInputMatchesCast() throws {
        let model = try quantizedModel()
        let block = try XCTUnwrap(model.model.layers[1].blockSparseMoe)
        XCTAssertTrue(block.routerReadsFloat32)
        for shape in [[1, 1, 64], [4, 1, 64], [1, 9, 64]] {
            let x = MLXRandom.normal(shape).asType(.bfloat16)
            let h = MLXRandom.normal(shape).asType(.bfloat16)
            let expected = block(x, residual: h)
            let actual = block(x, residual: h, routerInput: x.asType(.float32))
            eval(expected, actual)
            XCTAssertTrue(bitwiseEqual(actual, expected), "\(shape)")
        }
    }

    /// The one-launch SwiGLU over a fused gate/up product is bitwise the
    /// compiled activation on its two halves: decode (routed and shared) and
    /// prefill shapes, values past and exactly at the clamp, and NaN.
    func testSwiGLUKernelMatchesCompiledActivation() throws {
        let activation = makeMiniMaxM3SwiGLU(alpha: 1.702, limit: 7.0, beta: 1.0)
        for dtype in [DType.bfloat16, .float16, .float32] {
            for shape in [[1, 1, 4, 1, 64], [4, 1, 4, 1, 64], [4, 1, 128], [37, 6144], [3, 5, 2]] {
                var gateUp = (6 * MLXRandom.normal(shape)).asType(dtype)
                let last = shape.count - 1
                var flat = gateUp.reshaped(-1, shape[last])
                flat[0, 0] = MLXArray(Float(7)).asType(dtype)
                flat[0, 1] = MLXArray(Float(-7)).asType(dtype)
                flat[0, shape[last] - 1] = MLXArray(Float.nan).asType(dtype)
                gateUp = flat.reshaped(shape)
                let h = shape[last] / 2
                let expected = activation(gateUp[.ellipsis, ..<h], gateUp[.ellipsis, h...])
                let actual = try XCTUnwrap(
                    miniMaxM3SwiGLU(gateUp, alpha: 1.702, limit: 7.0, beta: 1.0))
                eval(expected, actual)
                XCTAssertEqual(actual.shape, expected.shape)
                XCTAssertEqual(actual.dtype, expected.dtype)
                XCTAssertTrue(
                    arrayEqual(actual, expected, equalNAN: true).item(Bool.self), "\(dtype) \(shape)")
            }
        }
        XCTAssertNil(miniMaxM3SwiGLU(MLXArray.zeros([2, 3]), alpha: 1, limit: 7, beta: 1))
    }

    /// The fused gate/up gather is bitwise the two projections, on decode
    /// (unsorted) and prefill (sorted) shapes, and releases them.
    func testFusedGateUpMatchesSeparateProjections() throws {
        let model = try quantizedModel(prepared: false)
        let block = try XCTUnwrap(model.model.layers[1].blockSparseMoe)
        let inputs = [[1, 1, 64], [4, 1, 64], [1, 40, 64]].map {
            MLXRandom.normal($0).asType(.bfloat16)
        }
        let before = inputs.map { block($0) }
        eval(before)
        try block.fuseGateUp()
        XCTAssertNotNil(block.fused.weight)
        XCTAssertEqual(block.switchMLP.gateProjection.weight.size, 0)
        XCTAssertEqual(block.switchMLP.upProjection.weight.size, 0)
        try block.fuseGateUp()
        let after = inputs.map { block($0) }
        eval(after)
        for (index, (a, b)) in zip(before, after).enumerated() {
            XCTAssertTrue(bitwiseEqual(a, b), "input \(index)")
        }
    }

    /// The fused shared-expert and dense-MLP gate/up product (one quantized
    /// matmul, one-launch SwiGLU) is bitwise the two projections and the
    /// compiled activation, on decode, batched-decode and prefill shapes, and
    /// releases them.
    func testFusedSharedAndDenseGateUpMatchSeparateProjections() throws {
        let model = try quantizedModel(prepared: false)
        let mlps = [
            try XCTUnwrap(model.model.layers[0].mlp),
            try XCTUnwrap(model.model.layers[1].blockSparseMoe?.sharedExperts),
        ]
        let inputs = [[1, 1, 64], [4, 1, 64], [1, 40, 64]].map {
            MLXRandom.normal($0).asType(.bfloat16)
        }
        for (index, mlp) in mlps.enumerated() {
            let before = inputs.map { mlp($0) }
            eval(before)
            try mlp.fuseGateUp()
            XCTAssertNotNil(mlp.fused.weight, "mlp \(index)")
            XCTAssertEqual(mlp.gateProj.weight.size, 0)
            XCTAssertEqual(mlp.upProj.weight.size, 0)
            try mlp.fuseGateUp()
            let after = inputs.map { mlp($0) }
            eval(after)
            for (shape, (a, b)) in zip(inputs.map(\.shape), zip(before, after)) {
                XCTAssertTrue(bitwiseEqual(a, b), "mlp \(index) \(shape)")
            }
            // Each batched row is its single-row result.
            let rows = (0 ..< 4).map { mlp(inputs[1][$0 ..< ($0 + 1)]) }
            eval(rows)
            for row in 0 ..< 4 {
                XCTAssertTrue(bitwiseEqual(after[1][row ..< (row + 1)], rows[row]), "mlp \(index) row \(row)")
            }
        }
    }

    /// The MoE output with the shared expert and the residual folded into
    /// the compiled combine is bitwise the eager adds.
    func testCombineFoldsSharedAndResidualBitwise() throws {
        let model = try quantizedModel(prepared: false)
        let block = try XCTUnwrap(model.model.layers[2].blockSparseMoe)
        for shape in [[1, 1, 64], [1, 9, 64]] {
            let x = MLXRandom.normal(shape).asType(.bfloat16)
            let h = MLXRandom.normal(shape).asType(.bfloat16)
            let bias = try XCTUnwrap(block.eScoreCorrectionBias)
            let routed = block.route([block.gate(x).asType(.float32), bias])
            let routedSum = block.combine([block.switchMLP(x, routed[0]), routed[1]])[0]
            let expected = h + (routedSum + block.sharedExperts!(x))
            let actual = block(x, residual: h)
            eval(expected, actual)
            XCTAssertTrue(bitwiseEqual(actual, expected), "\(shape)")
        }
    }

    /// The one-launch combine is bitwise the compiled combine (and the eager
    /// shared and residual adds it folds), with and without the shared
    /// expert and the residual, on single-row, batched and prefill shapes.
    func testCombineKernelMatchesCompiledCombine() throws {
        let block = try XCTUnwrap(try quantizedModel().model.layers[1].blockSparseMoe)
        for (lead, k, d) in [([1, 1], 2, 64), ([4, 1], 2, 64), ([1, 9], 2, 64), ([2, 1], 4, 6144)] {
            let outputs = (4 * MLXRandom.normal(lead + [k, d])).asType(.bfloat16)
            let raw = MLXRandom.uniform(low: 0.01, high: 1, lead + [k])
            let weights = raw / raw.sum(axis: -1, keepDims: true)
            let shared = MLXRandom.normal(lead + [d]).asType(.bfloat16)
            let residual = (8 * MLXRandom.normal(lead + [d])).asType(.bfloat16)
            let routed = block.combine([outputs, weights])[0]
            for (withShared, withResidual) in [(false, false), (true, false), (false, true), (true, true)] {
                var expected = routed
                if withShared { expected = expected + shared }
                if withResidual { expected = residual + expected }
                let actual = try XCTUnwrap(
                    miniMaxM3Combine(
                        outputs: outputs, weights: weights, factor: block.routedScalingFactor,
                        shared: withShared ? shared : nil, residual: withResidual ? residual : nil))
                // The block's compiled combine agrees as well.
                var inputs = [outputs, weights]
                if withShared { inputs.append(shared) }
                if withResidual, withShared { inputs.append(residual) }
                var blockCombine = block.combine(inputs)[0]
                if withResidual, !withShared { blockCombine = residual + blockCombine }
                eval(expected, actual, blockCombine)
                let label = "\(lead) k\(k) d\(d) shared \(withShared) residual \(withResidual)"
                XCTAssertTrue(bitwiseEqual(actual, expected), label)
                XCTAssertTrue(bitwiseEqual(actual, blockCombine), label)
                // Each batched row is its own single-row combine.
                for row in 0 ..< lead[0] where lead[0] > 1 {
                    let one = try XCTUnwrap(
                        miniMaxM3Combine(
                            outputs: outputs[row ..< (row + 1)], weights: weights[row ..< (row + 1)],
                            factor: block.routedScalingFactor,
                            shared: withShared ? shared[row ..< (row + 1)] : nil,
                            residual: withResidual ? residual[row ..< (row + 1)] : nil))
                    XCTAssertTrue(bitwiseEqual(actual[row ..< (row + 1)], one), "\(label) row \(row)")
                }
            }
        }
    }

    /// Decoding a prepared model (one-launch route, fused gate/up, folded
    /// combine) gives the unprepared model's logits bit for bit.
    func testPreparedQuantizedModelMatchesUnprepared() throws {
        let reference = try quantizedModel(prepared: false)
        let prepared = try quantizedModel(prepared: false)
        try prepared.update(parameters: reference.parameters(), verify: [.all])
        try prepared.prepare()
        let input = tokens(20)
        let a = try reference.newCache(parameters: nil)
        let b = try prepared.newCache(parameters: nil)
        let prefillA = reference(input, cache: a)
        let prefillB = prepared(input, cache: b)
        let stepA = reference(MLXArray([Int32(5)]).reshaped(1, 1), cache: a)
        let stepB = prepared(MLXArray([Int32(5)]).reshaped(1, 1), cache: b)
        eval(prefillA, prefillB, stepA, stepB)
        XCTAssertTrue(bitwiseEqual(prefillA, prefillB))
        XCTAssertTrue(bitwiseEqual(stepA, stepB))
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

    /// Loading fuses each gate/up pair. The separate projections it replaces
    /// must leave memory as each layer fuses: neither the loader's references
    /// nor MLX's buffer cache (whose default bound sits past physical memory)
    /// may keep them. Either one holds a second copy of a 194 GB checkpoint's
    /// gate/up tensors. A loader that still held them releases them only on
    /// return, after the last clear, so they would show here as cache.
    func testLoadingReleasesProjectionsReplacedByFusion() throws {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "minimax-m3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let gateUpBytes = try autoreleasepool { () -> Int in
            let reference = MiniMaxM3Model(try configuration(nested: true))
            randomize(reference)
            quantize(model: reference) { path, module in
                if path.hasSuffix("block_sparse_moe.gate") { return nil }
                guard module is Linear || module is Embedding || module is SwitchLinear else {
                    return nil
                }
                return (32, 3, .affine)
            }
            eval(reference)
            let arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
            try save(arrays: arrays, url: directory.appending(component: "model.safetensors"))
            return arrays.filter {
                $0.key.contains("switch_mlp.gate_proj.") || $0.key.contains("switch_mlp.up_proj.")
            }.values.reduce(0) { $0 + $1.nbytes }
        }
        XCTAssertGreaterThan(gateUpBytes, 0)
        Stream.gpu.synchronize()
        Memory.clearCache()
        let model = MiniMaxM3Model(try configuration(nested: true))
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: .init(groupSize: 32, bits: 3))
        Stream.gpu.synchronize()
        let cached = Memory.cacheMemory
        let up = model.leafModules().flattened().first {
            $0.0.hasSuffix("layers.2.block_sparse_moe.switch_mlp.up_proj")
        }?.1 as? QuantizedSwitchLinear
        XCTAssertEqual(up?.weight.size, 0, "gate/up fused at load")
        XCTAssertLessThan(cached, gateUpBytes / 2)
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
