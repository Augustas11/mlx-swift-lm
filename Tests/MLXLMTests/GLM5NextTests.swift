import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class GLM5NextTests: XCTestCase {

    // MARK: - Fixtures

    /// Six layers: three dense KDA, then DSA (own indexer) + MoE, KDA + MoE,
    /// and DSA with a shared indexer + MoE. Dims are multiples of 32 so every
    /// projection can be quantized at group size 32.
    private func textConfigJSON(indexTopk: Int = 8, nextn: Int = 1, linearHeadDim: Int = 32) -> String {
        """
        {
            "model_type": "glm5_next_text", "vocab_size": 64, "hidden_size": 64,
            "intermediate_size": 96, "moe_intermediate_size": 32,
            "num_hidden_layers": 6, "num_nextn_predict_layers": \(nextn),
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "n_shared_experts": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "routed_scaling_factor": 2.5, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": true, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "kv_lora_rank": 32, "q_lora_rank": 32, "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 32, "v_head_dim": 32, "mla_use_nope": true,
            "rms_norm_eps": 1e-5, "tie_word_embeddings": false, "attention_bias": false,
            "index_topk": \(indexTopk), "index_head_dim": 16, "index_n_heads": 2,
            "index_kpool": 4, "index_kpool_always_select_tail": true,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse"],
            "indexer_types": ["full", "full", "full", "full", "full", "shared"],
            "swiglu_limit": 10.0,
            "linear_attn_config": {"num_heads": 2, "head_dim": \(linearHeadDim),
                "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
            "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "first_k_dense_replace": 3
        }
        """
    }

    private func configuration(
        indexTopk: Int = 8, nextn: Int = 1, nested: Bool = false, linearHeadDim: Int = 32
    ) throws -> GLM5NextConfiguration {
        let text = textConfigJSON(indexTopk: indexTopk, nextn: nextn, linearHeadDim: linearHeadDim)
        let json = nested ? #"{"model_type": "glm5_next", "text_config": \#(text)}"# : text
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        return config
    }

    /// Replace every parameter with seeded noise so routing and top-k selection have no ties.
    private func randomize(_ model: Module, seed: UInt64 = 7) {
        let parameters = model.parameters().flattened()
        let updated: [(String, MLXArray)] = withRandomState(MLXRandom.RandomState(seed: seed)) {
            parameters.map { key, value in
                let noise = MLXRandom.normal(value.shape).asType(value.dtype)
                if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
                    return (key, 1 + 0.1 * noise)
                }
                return (key, 0.15 * noise)
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

    private func setDenseBypass(_ model: GLM5NextTextModel, _ enabled: Bool) {
        for layer in model.model.layers where !layer.isLinear {
            (layer.selfAttn as! GLM5NextAttention).denseBypass = enabled
        }
    }

    // MARK: - Shapes and configuration

    func testForwardShapesWithAndWithoutCache() throws {
        for nested in [false, true] {
            let config = try configuration(nested: nested)
            let model: any LanguageModel =
                nested ? GLM5NextModel(config) : GLM5NextTextModel(config)
            randomize(model)
            let input = tokens(13)

            let uncached = model(input, cache: nil)
            XCTAssertEqual(uncached.shape, [1, 13, 64])

            let cache = try model.newCache(parameters: nil)
            XCTAssertEqual(cache.count, 6)
            XCTAssertTrue(cache[0] is MambaCache)
            XCTAssertTrue(cache[3] is GLM5NextAttentionCache)
            let prefill = model(input, cache: cache)
            let step = model(MLXArray([Int32(5)]).reshaped(1, 1), cache: cache)
            eval(prefill, step)
            XCTAssertEqual(prefill.shape, [1, 13, 64])
            XCTAssertEqual(step.shape, [1, 1, 64])
            XCTAssertEqual(cache[3].offset, 14)
            XCTAssertLessThan(maxAbsDifference(uncached, prefill), 1e-4)
        }
    }

    func testConfigurationDerivesLayerPatternFromLinearAttentionConfig() throws {
        let kda = (0 ..< 45).filter { $0 % 4 != 3 }
        let full = (0 ..< 45).filter { $0 % 4 == 3 }
        let json = """
            {
                "model_type": "glm5_next",
                "text_config": {
                    "model_type": "glm5_next_text", "num_hidden_layers": 45,
                    "hidden_size": 4096, "num_attention_heads": 64, "num_key_value_heads": 64,
                    "index_kpool": 4, "index_topk": 2048, "first_k_dense_replace": 3,
                    "linear_attn_config": {"num_heads": 64, "head_dim": 128,
                        "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                        "kda_layers": \(kda), "full_attn_layers": \(full)}
                }
            }
            """
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.layerTypes.filter { $0 == "linear_attention" }.count, 34)
        XCTAssertEqual(
            config.layerTypes.indices.filter { config.layerTypes[$0] != "linear_attention" },
            full)
        XCTAssertEqual(config.mlpLayerTypes.prefix(4), ["dense", "dense", "dense", "sparse"])
        XCTAssertTrue(config.indexerTypes.allSatisfy { $0 == "full" })
        XCTAssertEqual(config.linearHeadDim, 128)
        XCTAssertEqual(config.linearLowerBound, -5.0)
        XCTAssertEqual(config.indexKpool, 4)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
    }

    // MARK: - Self-consistency

    /// Token-by-token decode must reproduce the full-prefill logits. With
    /// `index_topk` 8 over 40 tokens the indexer's selection is live.
    func testIncrementalDecodeMatchesPrefill() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8))
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

    /// Prefill split into chunks (pool windows straddle chunk boundaries) must
    /// match one-shot prefill, and decode must continue from either.
    func testChunkedPrefillMatchesSinglePrefill() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(model)
        let input = tokens(30)
        let next = MLXArray([Int32(11)]).reshaped(1, 1)

        let single = try model.newCache(parameters: nil)
        let full = model(input, cache: single)
        let fullNext = model(next, cache: single)
        eval(full, fullNext)

        let chunked = try model.newCache(parameters: nil)
        var pieces = [MLXArray]()
        for (start, end) in [(0, 9), (9, 18), (18, 30)] {
            pieces.append(model(input[0..., start ..< end], cache: chunked))
        }
        let chunkedNext = model(next, cache: chunked)
        eval(pieces, chunkedNext)

        XCTAssertLessThan(maxAbsDifference(concatenated(pieces, axis: 1), full), 1e-3)
        XCTAssertLessThan(maxAbsDifference(chunkedNext, fullNext), 1e-3)
    }

    /// While the context fits in `index_topk`, the indexer selects every causal
    /// token, so the dense-MLA bypass must equal the indexer path.
    func testDenseBypassMatchesIndexerPathWithinTopk() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 32))
        randomize(model)
        let input = tokens(20)
        let continuation = tokens(6, seed: 9)

        func run(bypass: Bool) throws -> (MLXArray, [MLXArray]) {
            setDenseBypass(model, bypass)
            let cache = try model.newCache(parameters: nil)
            let prefill = model(input, cache: cache)
            var steps = [MLXArray]()
            for t in 0 ..< 6 {
                steps.append(model(continuation[0..., t ..< (t + 1)], cache: cache))
            }
            eval(prefill, steps)
            return (prefill, steps)
        }

        let (densePrefill, denseSteps) = try run(bypass: true)
        let (sparsePrefill, sparseSteps) = try run(bypass: false)
        setDenseBypass(model, true)

        XCTAssertLessThan(maxAbsDifference(densePrefill, sparsePrefill), 1e-4)
        for (dense, sparse) in zip(denseSteps, sparseSteps) {
            XCTAssertLessThan(maxAbsDifference(dense, sparse), 1e-4)
        }
    }

    /// Past `index_topk` the indexer must actually drop tokens: the same weights
    /// with a larger `index_topk` (all tokens selected) give different logits.
    func testIndexerSelectionIsActivePastTopk() throws {
        let sparse = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(sparse)
        let dense = GLM5NextTextModel(try configuration(indexTopk: 64))
        try dense.update(parameters: sparse.parameters(), verify: [.all])
        let input = tokens(40)

        let sparseLogits = sparse(input, cache: try sparse.newCache(parameters: nil))
        let denseLogits = dense(input, cache: try dense.newCache(parameters: nil))
        eval(sparseLogits, denseLogits)
        // Up to index_topk + kpool - 1 tokens every query sees its whole prefix.
        XCTAssertLessThan(
            maxAbsDifference(sparseLogits[0..., ..<11], denseLogits[0..., ..<11]), 1e-4)
        XCTAssertGreaterThan(maxAbsDifference(sparseLogits, denseLogits), 1e-2)
    }

    /// The per-channel gated delta kernel must match the ops recurrence.
    func testVectorizedGatedDeltaKernelMatchesOps() throws {
        let (B, T, H, Dk, Dv) = (1, 12, 2, 32, 32)
        let (q, k, v, g, beta, state) = withRandomState(MLXRandom.RandomState(seed: 5)) {
            (
                MLXRandom.normal([B, T, H, Dk]) * 0.2,
                MLXRandom.normal([B, T, H, Dk]) * 0.2,
                MLXRandom.normal([B, T, H, Dv]),
                exp(-MLXRandom.uniform(low: 0, high: 1, [B, T, H, Dk])),
                sigmoid(MLXRandom.normal([B, T, H])),
                MLXRandom.normal([B, H, Dv, Dk]) * 0.1
            )
        }
        let (yKernel, sKernel) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: g, beta: beta, state: state, useKernel: true)
        let (yOps, sOps) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: g, beta: beta, state: state, useKernel: false)
        eval(yKernel, sKernel, yOps, sOps)
        XCTAssertLessThan(maxAbsDifference(yKernel, yOps), 1e-4)
        XCTAssertLessThan(maxAbsDifference(sKernel, sOps), 1e-4)
    }

    // MARK: - Fused decode kernels

    /// Cast floating parameters to bfloat16 (the artifact's activation dtype),
    /// keeping the ones the checkpoint stores in float32.
    private func castToBFloat16(_ model: Module) {
        let parameters = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            let keepsFloat32 =
                key.hasSuffix("A_log") || key.hasSuffix("dt_bias") || key.contains("_hc.")
                || key.hasSuffix("mlp.gate.weight") || key.hasSuffix("e_score_correction_bias")
            return (key, keepsFloat32 ? value : value.asType(.bfloat16))
        }
        model.update(parameters: ModuleParameters.unflattened(parameters))
        eval(model)
    }

    private func assertBitwiseEqual(
        _ a: MLXArray?, _ b: MLXArray?, _ message: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let a, let b else {
            return XCTAssertTrue(a == nil && b == nil, message, file: file, line: line)
        }
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, message, file: file, line: line)
        XCTAssertEqual(a.dtype, b.dtype, message, file: file, line: line)
        XCTAssertTrue(
            arrayEqual(a, b).item(Bool.self),
            "\(message): max |diff| \(maxAbsDifference(a, b))", file: file, line: line)
    }

    /// `glm5NextKDADecodeStep` must reproduce the KDA op path bit for bit:
    /// output, conv state and recurrent state, from an empty cache, for single
    /// tokens and for short blocks, over a carried state.
    func testFusedKDADecodeMatchesOpPathBitwise() throws {
        let config = try configuration(linearHeadDim: 128)
        let attention = GLM5NextLinearAttention(config)
        randomize(attention, seed: 11)
        castToBFloat16(attention)
        attention.train(false)
        let input = withRandomState(MLXRandom.RandomState(seed: 13)) {
            MLXRandom.normal([1, 24, config.hiddenSize]).asType(.bfloat16)
        }

        let fusedCache = MambaCache()
        let opCache = MambaCache()
        // Token 0 starts from an empty cache; then 1-token steps, then 3-token blocks.
        var start = 0
        for width in [1, 1, 1, 1, 1, 3, 3, 1, 3, 8] {
            let x = input[0..., start ..< (start + width)]
            attention.fusedDecode = true
            let fused = attention(x, cache: fusedCache)
            attention.fusedDecode = false
            let reference = attention(x, cache: opCache)
            attention.fusedDecode = true
            assertBitwiseEqual(fused, reference, "output at token \(start) width \(width)")
            assertBitwiseEqual(fusedCache[0], opCache[0], "conv state at token \(start)")
            assertBitwiseEqual(fusedCache[1], opCache[1], "recurrent state at token \(start)")
            start += width
        }
        XCTAssertNotNil(
            glm5NextKDADecodeStep(
                q: attention.qProj(input[0..., ..<1]), k: attention.kProj(input[0..., ..<1]),
                v: attention.vProj(input[0..., ..<1]),
                aPre: attention.forgetGate.project(input[0..., ..<1]),
                gatePre: attention.gBProj(attention.gAProj(input[0..., ..<1])),
                bRaw: attention.bProj(input[0..., ..<1]), convState: fusedCache[0]!,
                convWeight: attention.conv1d.weight, aLog: attention.forgetGate.aLog,
                dtBias: attention.forgetGate.dtBias, normWeight: attention.oNorm.weight,
                state: fusedCache[1]!, heads: attention.numHeads, headDim: attention.headDim,
                lowerBound: attention.forgetGate.lowerBound!, normEps: attention.oNorm.eps),
            "the fused kernel must cover the real head_dim 128 decode shape")
        XCTAssertFalse(attention.training, "fused paths only run in evaluation mode")
    }

    /// The HC decode kernels: the mix GEMV reproduces `rmsNorm` + `matmul`
    /// bit for bit; the Sinkhorn + collapse + RMSNorm kernel is the reference
    /// model's (fast exp, FMA collapse), so it stays within an fp32 ulp of the
    /// op path on post/comb and a bf16 ulp on the normalized branch input.
    func testFusedHyperConnectionDecodeKernels() throws {
        var json = textConfigJSON()
        json = json.replacingOccurrences(of: #""hidden_size": 64"#, with: #""hidden_size": 1024"#)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        let connection = GLM5NextHyperConnection(config)
        let norm = RMSNorm(dimensions: 1024, eps: config.rmsNormEps)
        withRandomState(MLXRandom.RandomState(seed: 21)) {
            connection.update(parameters: ModuleParameters.unflattened([
                "fn": 0.05 * MLXRandom.normal(connection.fn.shape),
                "base": 0.5 * MLXRandom.normal(connection.base.shape),
                "scale": 1 + 0.2 * MLXRandom.normal([3]),
            ]))
            norm.update(parameters: ModuleParameters.unflattened([
                "weight": (1 + 0.1 * MLXRandom.normal([1024])).asType(.bfloat16)
            ]))
        }
        connection.train(false)
        norm.train(false)
        for (index, length) in [1, 1, 3, 8].enumerated() {
            let x = withRandomState(MLXRandom.RandomState(seed: UInt64(30 + index))) {
                MLXRandom.normal([1, length, 4, 1024]).asType(.bfloat16)
            }
            let z = MLXFast.rmsNorm(
                x.asType(.float32).reshaped(1, length, 4 * 1024), weight: MLXArray.mlxNone,
                eps: connection.normEps)
            let mixes = try XCTUnwrap(glm5NextHCMix(x, fn: connection.fn, eps: connection.normEps))
            // The reference mixes short blocks one token (one M = 1 GEMV) at a time.
            let tokenwise = concatenated(
                (0 ..< length).map { matmul(z[0..., $0 ..< ($0 + 1)], connection.fn.T) }, axis: 1)
            assertBitwiseEqual(mixes, tokenwise, "mixes at length \(length)")

            connection.fusedDecode = false
            let (normedOps, postOps, combOps) = connection.normalizedBranchInput(x, norm: norm)
            connection.fusedDecode = true
            let fused = try XCTUnwrap(
                glm5NextHCNorm(
                    x, mixes: mixes, scale: connection.scale, base: connection.base,
                    normWeight: norm.weight, sinkhornIterations: connection.sinkhornIterations,
                    hcEps: connection.hcEps, normEps: norm.eps))
            let (normed, post, comb) = connection.normalizedBranchInput(x, norm: norm)
            assertBitwiseEqual(normed, fused.0, "decode path runs the kernel")
            // Past one token the op path mixes with an M = L GEMM (TF32 on NAX
            // GPUs), not the reference's per-token GEMV.
            guard length == 1 else { continue }
            XCTAssertLessThanOrEqual(maxAbsDifference(post, postOps), 1e-6)
            XCTAssertLessThanOrEqual(maxAbsDifference(comb, combOps), 1e-6)
            // One bf16 ulp at the normalized magnitude (|x| < 8).
            XCTAssertLessThanOrEqual(maxAbsDifference(normed, normedOps), 0.0625)
            XCTAssertGreaterThan(
                (normed .== normedOps).asType(.float32).mean().item(Float.self), 0.95)
        }
    }

    /// The deferred one-token hyper-connection chain (`glm5NextHCPre` with
    /// the previous expand folded in, `glm5NextHCPost` for post, comb and the
    /// comb product) reproduces expand + `glm5NextHCMix` + `glm5NextHCNorm`
    /// bit for bit where MLX computes the comb product without NAX; where it
    /// uses NAX (TF32 GEMMs) the model keeps the non-deferred path.
    func testDeferredHyperConnectionDecodeMatchesChainBitwise() throws {
        let json = textConfigJSON().replacingOccurrences(
            of: #""hidden_size": 64"#, with: #""hidden_size": 1024"#)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))

        let model = GLM5NextTextModel(config)
        randomize(model, seed: 61)
        castToBFloat16(model)
        model.train(false)
        let input = tokens(18)
        func decode(deferred: Bool) throws -> [MLXArray] {
            model.model.deferredDecode = deferred
            let cache = try model.newCache(parameters: nil)
            var logits = [model(input[0..., ..<12], cache: cache)[0..., -1]]
            for t in 12 ..< 18 {
                logits.append(model(input[0..., t ..< (t + 1)], cache: cache)[0..., -1])
            }
            eval(logits)
            return logits
        }
        let deferred = try decode(deferred: true)
        let reference = try decode(deferred: false)
        for (step, (a, b)) in zip(deferred, reference).enumerated() {
            assertBitwiseEqual(a, b, "logits at step \(step)")
        }
        let layer = model.model.layers[0]
        XCTAssertTrue(layer.deferredDecodeSupported(dtype: .bfloat16))
        guard glm5NextHCDeferredExact else { return }

        // Kernel level, one half-layer and its successor.
        let connection = layer.attnHC
        let norm = layer.inputLayerNorm
        let (x, y) = withRandomState(MLXRandom.RandomState(seed: 62)) {
            (
                MLXRandom.normal([1, 1, 4, 1024]).asType(.bfloat16),
                MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
            )
        }
        let mixes = try XCTUnwrap(glm5NextHCMix(x, fn: connection.fn, eps: connection.normEps))
        let chain = try XCTUnwrap(
            glm5NextHCNorm(
                x, mixes: mixes, scale: connection.scale, base: connection.base,
                normWeight: norm.weight, sinkhornIterations: connection.sinkhornIterations,
                hcEps: connection.hcEps, normEps: norm.eps))
        let (normalized, h, post, comb, mm) = connection.deferredBranchInput(
            .materialized(x), norm: norm)
        assertBitwiseEqual(normalized, chain.0, "branch input")
        assertBitwiseEqual(h, x, "residual")
        assertBitwiseEqual(post, chain.1, "post")
        assertBitwiseEqual(comb, chain.2, "comb")
        assertBitwiseEqual(
            mm.reshaped(1, 1, 4, 1024),
            matmul(comb.swappedAxes(-1, -2), x.asType(.float32)), "comb product")

        let next = glm5NextHCExpand(y, x, post, comb)
        let (nextNormalized, nextH, _, _, _) = connection.deferredBranchInput(
            .deferred(y: y, residual: x, post: post, comb: comb, mm: mm), norm: norm)
        assertBitwiseEqual(nextH, next, "folded expand")
        let nextMixes = try XCTUnwrap(
            glm5NextHCMix(next, fn: connection.fn, eps: connection.normEps))
        let nextChain = try XCTUnwrap(
            glm5NextHCNorm(
                next, mixes: nextMixes, scale: connection.scale, base: connection.base,
                normWeight: norm.weight, sinkhornIterations: connection.sinkhornIterations,
                hcEps: connection.hcEps, normEps: norm.eps))
        assertBitwiseEqual(nextNormalized, nextChain.0, "branch input after a folded expand")
    }

    /// `glm5NextMoEExperts` (gate/up with the router's selection replayed,
    /// down + routing-weighted combine + shared expert) reproduces the MoE op
    /// path bit for bit on an affine-quantized layer: 4-bit routed experts
    /// and an 8-bit shared expert, as the artifact stores them.
    func testFusedMoEExpertsMatchOpPathBitwise() throws {
        var json = textConfigJSON()
        for (from, to) in [
            (#""hidden_size": 64"#, #""hidden_size": 512"#),
            (#""moe_intermediate_size": 32"#, #""moe_intermediate_size": 512"#),
            (#""n_routed_experts": 4"#, #""n_routed_experts": 64"#),
            (#""num_experts_per_tok": 2"#, #""num_experts_per_tok": 8"#),
        ] {
            json = json.replacingOccurrences(of: from, with: to)
        }
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        let moe = GLM5NextMoE(config)
        randomize(moe, seed: 71)
        castToBFloat16(moe)
        // The router gate stays float32, as sanitize keeps it.
        moe.gate.update(parameters: ModuleParameters.unflattened([
            "weight": moe.gate.weight.asType(.float32)
        ]))
        quantize(model: moe) { path, _ in
            if path.hasPrefix("gate") { return nil }
            return path.hasPrefix("shared_experts") ? (64, 8, .affine) : (64, 4, .affine)
        }
        eval(moe)
        moe.train(false)
        XCTAssertTrue(moe.switchMLP.gateProjection is QuantizedSwitchLinear)
        for seed in 0 ..< 24 {
            let x = withRandomState(MLXRandom.RandomState(seed: UInt64(80 + seed))) {
                (Float(seed % 3 + 1) * MLXRandom.normal([1, 1, 512])).asType(.bfloat16)
            }
            moe.gate.fusedDecode = false
            let reference = moe(x)
            moe.gate.fusedDecode = true
            let fused = moe(x)
            assertBitwiseEqual(fused, reference, "MoE output for input \(seed)")
        }
        let x2 = MLXArray.zeros([1, 512], dtype: .bfloat16)
        let (sig, biased) = glm5NextRouterLogits(
            x2, weight: moe.gate.weight, bias: moe.gate.eScoreCorrectionBias)
        let affine = { (module: Module) -> GLM5NextAffine in
            if let s = module as? QuantizedSwitchLinear {
                let p = s.quantizedParts
                return GLM5NextAffine(
                    weight: p.weight, scales: p.scales, biases: p.biases!, bits: s.bits,
                    groupSize: s.groupSize)
            }
            let l = module as! QuantizedLinear
            return GLM5NextAffine(
                weight: l.weight, scales: l.scales, biases: l.biases!, bits: l.bits,
                groupSize: l.groupSize)
        }
        let shared = try XCTUnwrap(moe.sharedExperts)
        XCTAssertNotNil(
            glm5NextMoEExperts(
                x2, sig: sig, biased: biased, topK: 8, scaling: 2.5, normalize: true, limit: 10,
                gate: affine(moe.switchMLP.gateProjection), up: affine(moe.switchMLP.upProjection),
                down: affine(moe.switchMLP.downProjection), sharedGate: affine(shared.gateProj),
                sharedUp: affine(shared.upProj), sharedDown: affine(shared.downProj)),
            "the fused experts must cover this layer")
    }

    /// `glm5NextMoERouter` reproduces the router op path bit for bit: the
    /// selected experts in argPartition's order (ties to the lower expert),
    /// and the normalized, scaled scores.
    func testFusedMoERouterMatchesOpPathBitwise() throws {
        var json = textConfigJSON()
        for (from, to) in [
            (#""hidden_size": 64"#, #""hidden_size": 512"#),
            (#""n_routed_experts": 4"#, #""n_routed_experts": 64"#),
            (#""num_experts_per_tok": 2"#, #""num_experts_per_tok": 8"#),
        ] {
            json = json.replacingOccurrences(of: from, with: to)
        }
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        let gate = GLM5NextMoEGate(config)
        var weight = withRandomState(MLXRandom.RandomState(seed: 41)) {
            0.05 * MLXRandom.normal([64, 512])
        }
        var bias = withRandomState(MLXRandom.RandomState(seed: 42)) {
            0.01 * MLXRandom.normal([64])
        }
        // Experts 9 and 3 tie exactly with 40 and 17: the lower index must win.
        weight[40] = weight[9]
        weight[17] = weight[3]
        bias[40] = bias[9]
        bias[17] = bias[3]
        gate.update(parameters: ModuleParameters.unflattened([
            "weight": weight, "e_score_correction_bias": bias,
        ]))
        gate.train(false)
        for seed in 0 ..< 32 {
            let x = withRandomState(MLXRandom.RandomState(seed: UInt64(50 + seed))) {
                MLXRandom.normal([1, 1, 512]).asType(.bfloat16)
            }
            gate.fusedDecode = false
            let (indices, scores) = gate(x)
            gate.fusedDecode = true
            let (fusedIndices, fusedScores) = gate(x)
            assertBitwiseEqual(fusedIndices, indices, "experts for input \(seed)")
            assertBitwiseEqual(fusedScores, scores, "scores for input \(seed)")
        }
        XCTAssertNotNil(
            glm5NextMoERouter(
                MLXArray.zeros([1, 512], dtype: .bfloat16), weight: gate.weight,
                bias: gate.eScoreCorrectionBias, topK: 8, scaling: 2.5, normalize: true))
    }

    /// The quantized KDA decode (`glm5NextMultiQMV` for q, k, v, f_a, g_a and
    /// b grouped by bit width, f_b / g_b rows inside the KDA kernel) is bitwise
    /// equal to the op path for the artifact's 8-bit and 5-bit layers, mixed
    /// widths in one layer, and 4-bit.
    func testQuantizedFusedKDADecodeMatchesOpPathBitwise() throws {
        var json = textConfigJSON(linearHeadDim: 128)
        json = json.replacingOccurrences(of: #""hidden_size": 64"#, with: #""hidden_size": 512"#)
        json = json.replacingOccurrences(
            of: #""linear_attn_config": {"num_heads": 2"#, with: #""linear_attn_config": {"num_heads": 8"#)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.linearNumHeads, 8)
        let cases: [(String, (String) -> Int)] = [
            ("8-bit", { _ in 8 }),
            ("5-bit", { _ in 5 }),
            ("4-bit", { _ in 4 }),
            ("mixed", { path in path.hasPrefix("q_proj") || path.hasPrefix("v_proj") ? 5 : 8 }),
        ]
        for (name, bits) in cases {
            let attention = GLM5NextLinearAttention(config)
            randomize(attention, seed: 91)
            castToBFloat16(attention)
            quantize(model: attention) { path, module in
                guard module is Linear else { return nil }
                return (64, bits(path), .affine)
            }
            eval(attention)
            attention.train(false)
            XCTAssertTrue(attention.qProj is QuantizedLinear, name)
            let input = withRandomState(MLXRandom.RandomState(seed: 93)) {
                MLXRandom.normal([1, 12, 512]).asType(.bfloat16)
            }
            let fusedCache = MambaCache()
            let opCache = MambaCache()
            var start = 0
            for width in [1, 1, 1, 6, 1, 1, 1] {
                let x = input[0..., start ..< (start + width)]
                attention.fusedDecode = true
                let fused = attention(x, cache: fusedCache)
                attention.fusedDecode = false
                let reference = attention(x, cache: opCache)
                attention.fusedDecode = true
                assertBitwiseEqual(fused, reference, "\(name) output at token \(start)")
                assertBitwiseEqual(fusedCache[0], opCache[0], "\(name) conv state at \(start)")
                assertBitwiseEqual(fusedCache[1], opCache[1], "\(name) state at \(start)")
                start += width
            }
            let x = input[0..., ..<1].reshaped(1, 512)
            let parts = [
                attention.qProj, attention.kProj, attention.vProj,
                attention.forgetGate.fAProj, attention.gAProj, attention.bProj,
            ].compactMap { glm5NextAffine($0) }
            XCTAssertEqual(parts.count, 6, name)
            if name != "mixed" {
                XCTAssertNotNil(glm5NextMultiQMV(x, parts), "\(name): one multi-qmv dispatch")
            }
            let x3 = x.reshaped(1, 1, 512)
            XCTAssertNotNil(
                glm5NextKDADecodeStep(
                    q: attention.qProj(x3), k: attention.kProj(x3), v: attention.vProj(x3),
                    fa: attention.forgetGate.fAProj(x3), ga: attention.gAProj(x3),
                    fb: try XCTUnwrap(glm5NextAffine(attention.forgetGate.fBProj)),
                    gb: try XCTUnwrap(glm5NextAffine(attention.gBProj)),
                    bRaw: attention.bProj(x3), convState: fusedCache[0]!,
                    convWeight: attention.conv1d.weight, aLog: attention.forgetGate.aLog,
                    dtBias: attention.forgetGate.dtBias, normWeight: attention.oNorm.weight,
                    state: fusedCache[1]!, heads: attention.numHeads, headDim: attention.headDim,
                    lowerBound: attention.forgetGate.lowerBound!, normEps: attention.oNorm.eps),
                "\(name): gate rows run inside the KDA kernel")
        }
    }

    /// `glm5NextDSASelect` reproduces the indexer's op-path selection bit for
    /// bit for one decode token: fp32 scores, argPartition's set with exact
    /// score ties (duplicated pooled keys), chronological order, -1 padding
    /// and the partial-window tail, from "every pool" (<= 512 pools) to a
    /// 64k-token context.
    func testFusedDSASelectMatchesOpPathBitwise() throws {
        var json = textConfigJSON(indexTopk: 2048)
        for (from, to) in [
            (#""index_head_dim": 16"#, #""index_head_dim": 128"#),
            (#""index_n_heads": 2"#, #""index_n_heads": 32"#),
        ] {
            json = json.replacingOccurrences(of: from, with: to)
        }
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        let indexer = GLM5NextIndexer(config)
        randomize(indexer, seed: 101)
        castToBFloat16(indexer)
        indexer.train(false)
        let x = withRandomState(MLXRandom.RandomState(seed: 102)) {
            MLXRandom.normal([1, 1, config.hiddenSize]).asType(.bfloat16)
        }
        let qResid = withRandomState(MLXRandom.RandomState(seed: 103)) {
            MLXRandom.normal([1, 1, config.qLoraRank]).asType(.bfloat16)
        }
        for (pools, remainder) in [(100, 3), (512, 0), (513, 1), (2049, 2), (16389, 3)] {
            var keys = withRandomState(MLXRandom.RandomState(seed: UInt64(104 + pools))) {
                MLXRandom.normal([1, pools, 128]).asType(.bfloat16)
            }
            // Exact score ties across the selection boundary.
            if pools > 600 {
                for (a, b) in [(7, pools - 1), (40, 300), (41, 299), (5, 6)] {
                    keys[0..., b] = keys[0..., a]
                }
            }
            let kvLength = pools * 4 + remainder
            indexer.fusedDecode = false
            let reference = indexer.select(x, qResid: qResid, pooledKeys: keys, kvLength: kvLength)
            indexer.fusedDecode = true
            let fused = indexer.select(x, qResid: qResid, pooledKeys: keys, kvLength: kvLength)
            assertBitwiseEqual(fused, reference, "selection with \(pools) pools")
            if pools == 2049 {
                // Every score equal: the boundary is decided by the tie order alone.
                let same = broadcast(keys[0..., ..<1], to: [1, 1000, 128])
                indexer.fusedDecode = false
                let tiedReference = indexer.select(
                    x, qResid: qResid, pooledKeys: same, kvLength: 4002)
                indexer.fusedDecode = true
                let tied = indexer.select(x, qResid: qResid, pooledKeys: same, kvLength: 4002)
                assertBitwiseEqual(tied, tiedReference, "all-tied selection")
            }
            XCTAssertNotNil(
                glm5NextDSASelect(
                    q: indexer.wqB(qResid), weights: indexer.weightsProj(x), weightScale: 1,
                    pooledKeys: keys, kvLength: kvLength, heads: 32, dim: 128, indexTopk: 2048,
                    kpool: 4, tail: true),
                "the fused selection covers \(pools) pools")
        }
    }

    /// End to end: decode logits with the fused KDA step equal the op path's.
    func testFusedDecodeKeepsModelLogitsBitwise() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8, linearHeadDim: 128))
        randomize(model)
        castToBFloat16(model)
        model.train(false)
        let input = tokens(30)

        func decode(fused: Bool) throws -> [MLXArray] {
            for layer in model.model.layers where layer.isLinear {
                (layer.selfAttn as! GLM5NextLinearAttention).fusedDecode = fused
            }
            // Only the KDA step is compared bitwise here; the HC kernels
            // have their own test.
            for layer in model.model.layers {
                layer.attnHC.fusedDecode = false
                layer.ffnHC.fusedDecode = false
                (layer.mlp as? GLM5NextMoE)?.gate.fusedDecode = false
            }
            let cache = try model.newCache(parameters: nil)
            var logits = [model(input[0..., ..<20], cache: cache)[0..., -1]]
            for t in 20 ..< 30 {
                logits.append(model(input[0..., t ..< (t + 1)], cache: cache)[0..., -1])
            }
            eval(logits)
            return logits
        }
        let fused = try decode(fused: true)
        let reference = try decode(fused: false)
        for (step, (a, b)) in zip(fused, reference).enumerated() {
            assertBitwiseEqual(a, b, "logits at step \(step)")
        }
    }

    // MARK: - Exact trim and batched rows

    /// `GLM5NextAttentionCache.trim` restores the latent, the pooled keys and
    /// the partial pool window of the shorter sequence bit for bit (pools
    /// straddling the trim point rebuild from the kept raw keys), and the next
    /// token's attention output and state equal the untrimmed run's.
    func testLatentCacheTrimRoundTripIsBitEqual() throws {
        let config = try configuration(indexTopk: 8)
        let attention = GLM5NextAttention(config, sharedIndexer: false)
        randomize(attention, seed: 121)
        attention.train(false)
        let x = withRandomState(MLXRandom.RandomState(seed: 122)) {
            MLXRandom.normal([1, 48, config.hiddenSize])
        }
        let maxTrim = 6
        let cache = GLM5NextAttentionCache(kpool: config.indexKpool, maxTrim: maxTrim)
        var snapshots = [0: cache.copy() as! GLM5NextAttentionCache]
        var start = 0
        // Mixed widths so windows complete inside multi-token chunks too.
        for width in [5, 1, 3, 1, 1, 2, 7, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1] {
            _ = attention(x[0..., start ..< (start + width)], cache: cache, previousTopk: [])
            start += width
            eval(cache)
            snapshots[start] = cache.copy() as? GLM5NextAttentionCache
        }
        XCTAssertEqual(start, 42)
        func assertSame(_ a: GLM5NextAttentionCache, _ b: GLM5NextAttentionCache, _ what: String) {
            XCTAssertEqual(a.offset, b.offset, what)
            for (i, name) in ["latent", "pooled", "pending keys", "pending gates"].enumerated() {
                assertBitwiseEqual(a.state[i], b.state[i], "\(what): \(name)")
            }
        }
        // Trim every recorded state back to every earlier recorded state.
        var checked = 0
        for end in snapshots.keys.sorted() where end > 0 {
            for n in 1 ... (maxTrim + 6) where n <= end {
                let trimmed = snapshots[end]!.copy() as! GLM5NextAttentionCache
                let removed = trimmed.trim(n)
                if n <= maxTrim {
                    XCTAssertEqual(removed, n, "trim(\(n)) at \(end) must be exact")
                }
                XCTAssertLessThanOrEqual(removed, n)
                guard let reference = snapshots[end - removed] else { continue }
                assertSame(trimmed, reference, "trim(\(n)) at \(end)")
                // Continue both with the same token: outputs and states agree.
                let next = x[0..., (end - removed) ..< (end - removed + 1)]
                let continued = reference.copy() as! GLM5NextAttentionCache
                let (a, _) = attention(next, cache: trimmed, previousTopk: [])
                let (b, _) = attention(next, cache: continued, previousTopk: [])
                assertBitwiseEqual(a, b, "output after trim(\(n)) at \(end)")
                assertSame(trimmed, continued, "state after trim(\(n)) at \(end) and one token")
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 100)
        // A shared-indexer layer keeps only the latent and trims any amount.
        let shared = GLM5NextAttentionCache(kpool: config.indexKpool, maxTrim: 0)
        _ = shared.appendLatent(MLXArray.zeros([1, 1, 10, 32]))
        XCTAssertEqual(shared.trim(9), 9)
        XCTAssertEqual(shared.offset, 1)
    }

    /// Stack per-row caches into the batched form a continuous-batching
    /// forward uses: recurrent states concatenated along the batch axis, sparse
    /// layers as a `GLM5NextLatentCacheBatch` over the row caches.
    private func batchCaches(_ rows: [[KVCache]]) -> [KVCache] {
        (0 ..< rows[0].count).map { layer -> KVCache in
            if rows[0][layer] is MambaCache {
                let batch = MambaCache()
                batch[0] = concatenated(rows.map { ($0[layer] as! MambaCache)[0]! }, axis: 0)
                batch[1] = concatenated(rows.map { ($0[layer] as! MambaCache)[1]! }, axis: 0)
                return batch
            }
            return GLM5NextLatentCacheBatch(
                rowCaches: rows.map { $0[layer] as! any GLM5NextLatentCache })
        }
    }

    private func randomAffine(_ shape: [Int], bits: Int, groupSize: Int = 64, seed: UInt64)
        -> GLM5NextAffine
    {
        let w = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (0.05 * MLXRandom.normal(shape)).asType(.bfloat16)
        }
        let (q, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        return GLM5NextAffine(
            weight: q, scales: scales, biases: biases!, bits: bits, groupSize: groupSize)
    }

    private func assertRowsEqual(
        _ batched: MLXArray, _ rows: [MLXArray], _ message: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        assertBitwiseEqual(batched, concatenated(rows, axis: 0), message, file: file, line: line)
    }

    /// Every fused decode kernel run on B = 2, 4 and 8 rows equals the B = 1
    /// calls stacked, bit for bit: KDA (precomputed gates, 4/8-bit gate rows,
    /// 5-bit gate rows; one and two tokens per row), multi-qmv, the router, the
    /// MoE experts (one and two tokens per row), the HC mix and norm, and the
    /// deferred HC pre (both forms) and post.
    func testBatchedDecodeKernelsMatchStackedRowsBitwise() throws {
        let (heads, dk) = (2, 128)
        let qkv = heads * dk
        let fb8 = randomAffine([qkv, dk], bits: 8, seed: 201)
        let gb8 = randomAffine([qkv, dk], bits: 8, seed: 202)
        let fb5 = randomAffine([qkv, dk], bits: 5, seed: 203)
        let gb5 = randomAffine([qkv, dk], bits: 5, seed: 204)
        let convWeight = withRandomState(MLXRandom.RandomState(seed: 205)) {
            (0.3 * MLXRandom.normal([3 * qkv, 4, 1])).asType(.bfloat16)
        }
        let aLog = withRandomState(MLXRandom.RandomState(seed: 206)) { MLXRandom.normal([heads]) * 0.5 }
        let dtBias = withRandomState(MLXRandom.RandomState(seed: 207)) { MLXRandom.normal([qkv]) * 0.5 }
        let normWeight = withRandomState(MLXRandom.RandomState(seed: 208)) {
            (1 + 0.1 * MLXRandom.normal([dk])).asType(.bfloat16)
        }
        let projections = (0 ..< 6).map {
            randomAffine([[512, 512, 512, 128, 128, 64][$0], 1024], bits: 8, seed: 210 + UInt64($0))
        }
        let (experts, topK, hidden, inter) = (64, 8, 512, 512)
        let gate = randomAffine([experts, inter, hidden], bits: 4, seed: 220)
        let up = randomAffine([experts, inter, hidden], bits: 4, seed: 221)
        let down = randomAffine([experts, hidden, inter], bits: 4, seed: 222)
        let sharedGate = randomAffine([inter, hidden], bits: 8, seed: 223)
        let sharedUp = randomAffine([inter, hidden], bits: 8, seed: 224)
        let sharedDown = randomAffine([hidden, inter], bits: 8, seed: 225)
        let (routerWeight, routerBias) = withRandomState(MLXRandom.RandomState(seed: 226)) {
            (0.05 * MLXRandom.normal([experts, hidden]), 0.01 * MLXRandom.normal([experts]))
        }
        let D = 1024
        let (fn, scale, base, hcNormWeight) = withRandomState(MLXRandom.RandomState(seed: 227)) {
            (
                0.05 * MLXRandom.normal([24, 4 * D]), 1 + 0.2 * MLXRandom.normal([3]),
                0.5 * MLXRandom.normal([24]), (1 + 0.1 * MLXRandom.normal([D])).asType(.bfloat16)
            )
        }

        for B in [2, 4, 8] {
            func rnd(_ shape: [Int], _ seed: Int, _ scale: Float = 1, _ dtype: DType = .bfloat16)
                -> MLXArray
            {
                withRandomState(MLXRandom.RandomState(seed: UInt64(1000 * B + seed))) {
                    (scale * MLXRandom.normal(shape)).asType(dtype)
                }
            }
            func row(_ a: MLXArray, _ r: Int) -> MLXArray { a[r ..< (r + 1)] }

            // KDA: precomputed gates (1 and 2 tokens per row), gate rows.
            for T in [1, 2] {
                let (q, k, v) = (rnd([B, T, qkv], 1), rnd([B, T, qkv], 2), rnd([B, T, qkv], 3))
                let (aPre, gatePre, bRaw) = (
                    rnd([B, T, qkv], 4), rnd([B, T, qkv], 5), rnd([B, T, heads], 6)
                )
                let (fa, ga) = (rnd([B, T, dk], 7), rnd([B, T, dk], 8))
                let convState = rnd([B, 3, 3 * qkv], 9)
                let state = rnd([B, heads, dk, dk], 10, 0.1, .float32)
                func pre(_ r: Int?) -> (y: MLXArray, convState: MLXArray, state: MLXArray)? {
                    let s = { (a: MLXArray) in r.map { row(a, $0) } ?? a }
                    return glm5NextKDADecodeStep(
                        q: s(q), k: s(k), v: s(v), aPre: s(aPre), gatePre: s(gatePre),
                        bRaw: s(bRaw), convState: s(convState), convWeight: convWeight,
                        aLog: aLog, dtBias: dtBias, normWeight: normWeight, state: s(state),
                        heads: heads, headDim: dk, lowerBound: -5, normEps: 1e-5)
                }
                func rows(
                    _ fb: GLM5NextAffine, _ gb: GLM5NextAffine, _ r: Int?
                ) -> (y: MLXArray, convState: MLXArray, state: MLXArray)? {
                    let s = { (a: MLXArray) in r.map { row(a, $0) } ?? a }
                    return glm5NextKDADecodeStep(
                        q: s(q), k: s(k), v: s(v), fa: s(fa), ga: s(ga), fb: fb, gb: gb,
                        bRaw: s(bRaw), convState: s(convState), convWeight: convWeight,
                        aLog: aLog, dtBias: dtBias, normWeight: normWeight, state: s(state),
                        heads: heads, headDim: dk, lowerBound: -5, normEps: 1e-5)
                }
                var variants: [(String, (Int?) -> (y: MLXArray, convState: MLXArray, state: MLXArray)?)] = [
                    ("precomputed", pre), ("8-bit gate rows", { rows(fb8, gb8, $0) }),
                ]
                if T == 1 { variants.append(("5-bit gate rows", { rows(fb5, gb5, $0) })) }
                for (name, run) in variants {
                    let batched = try XCTUnwrap(run(nil), "KDA \(name) B=\(B) T=\(T)")
                    let singles = (0 ..< B).map { run($0)! }
                    assertRowsEqual(batched.y, singles.map(\.y), "KDA \(name) y B=\(B) T=\(T)")
                    assertRowsEqual(
                        batched.convState, singles.map(\.convState), "KDA \(name) conv B=\(B) T=\(T)")
                    assertRowsEqual(
                        batched.state, singles.map(\.state), "KDA \(name) state B=\(B) T=\(T)")
                }
            }

            // Multi-qmv over one vector per row.
            let xq = rnd([B, 1024], 20)
            let multi = try XCTUnwrap(glm5NextMultiQMV(xq, projections), "multi-qmv B=\(B)")
            let multiRows = (0 ..< B).map { glm5NextMultiQMV(row(xq, $0), projections)! }
            for i in projections.indices {
                assertRowsEqual(multi[i], multiRows.map { $0[i] }, "multi-qmv part \(i) B=\(B)")
            }

            // Router and experts: one and two tokens per row (flattened).
            for T in [1, 2] {
                let x = rnd([B * T, hidden], 30 + T, Float(T))
                let router = try XCTUnwrap(
                    glm5NextMoERouter(
                        x, weight: routerWeight, bias: routerBias, topK: topK, scaling: 2.5,
                        normalize: true))
                let routerRows = (0 ..< (B * T)).map {
                    glm5NextMoERouter(
                        row(x, $0), weight: routerWeight, bias: routerBias, topK: topK,
                        scaling: 2.5, normalize: true)!
                }
                assertRowsEqual(router.0, routerRows.map(\.0), "router experts B=\(B) T=\(T)")
                assertRowsEqual(router.1, routerRows.map(\.1), "router scores B=\(B) T=\(T)")
                func experts(_ x: MLXArray) -> MLXArray? {
                    let (sig, biased) = glm5NextRouterLogits(x, weight: routerWeight, bias: routerBias)
                    return glm5NextMoEExperts(
                        x, sig: sig, biased: biased, topK: topK, scaling: 2.5, normalize: true,
                        limit: 10, gate: gate, up: up, down: down, sharedGate: sharedGate,
                        sharedUp: sharedUp, sharedDown: sharedDown)
                }
                let moe = try XCTUnwrap(experts(x), "experts B=\(B) T=\(T)")
                assertRowsEqual(moe, (0 ..< (B * T)).map { experts(row(x, $0))! }, "experts B=\(B) T=\(T)")
            }

            // Hyper-connections: mix + norm, deferred pre (both forms) and post.
            let hx = rnd([B, 1, 4, D], 40)
            let mixes = try XCTUnwrap(glm5NextHCMix(hx, fn: fn, eps: 1e-5))
            assertRowsEqual(
                mixes, (0 ..< B).map { glm5NextHCMix(row(hx, $0), fn: fn, eps: 1e-5)! }, "hc mix B=\(B)")
            func norm(_ x: MLXArray, _ m: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
                glm5NextHCNorm(
                    x, mixes: m, scale: scale, base: base, normWeight: hcNormWeight,
                    sinkhornIterations: 20, hcEps: 1e-6, normEps: 1e-5)!
            }
            let normed = norm(hx, mixes)
            let normRows = (0 ..< B).map { norm(row(hx, $0), row(mixes, $0)) }
            assertRowsEqual(normed.0, normRows.map(\.0), "hc norm B=\(B)")
            assertRowsEqual(normed.2, normRows.map(\.2), "hc comb B=\(B)")

            func pre(_ x: MLXArray?, _ d: (y: MLXArray, mm: MLXArray, post: MLXArray)?)
                -> (MLXArray, MLXArray, MLXArray)
            {
                glm5NextHCPre(
                    x: x, deferred: d, fn: fn, scale: scale, base: base,
                    normWeight: hcNormWeight, normEps: 1e-5, hcEps: 1e-6, branchNormEps: 1e-5)
            }
            let p = pre(hx, nil)
            let pRows = (0 ..< B).map { pre(row(hx, $0), nil) }
            assertRowsEqual(p.0, pRows.map(\.0), "hc pre normalized B=\(B)")
            assertRowsEqual(p.1, pRows.map(\.1), "hc pre mixes B=\(B)")
            func post(_ m: MLXArray, _ h: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
                glm5NextHCPost(
                    mixes: m, scale: scale, base: base, residual: h, sinkhornIterations: 20,
                    hcEps: 1e-6)
            }
            let q = post(p.1, p.2)
            let qRows = (0 ..< B).map { post(row(p.1, $0), row(p.2, $0)) }
            assertRowsEqual(q.0, qRows.map(\.0), "hc post post B=\(B)")
            assertRowsEqual(q.1, qRows.map(\.1), "hc post comb B=\(B)")
            assertRowsEqual(q.2, qRows.map(\.2), "hc post mm B=\(B)")
            let y = rnd([B, 1, D], 41)
            let d = pre(nil, (y, q.2, q.0))
            let dRows = (0 ..< B).map { pre(nil, (row(y, $0), row(q.2, $0), row(q.0, $0))) }
            assertRowsEqual(d.0, dRows.map(\.0), "hc deferred pre normalized B=\(B)")
            assertRowsEqual(d.1, dRows.map(\.1), "hc deferred pre mixes B=\(B)")
            assertRowsEqual(d.2, dRows.map(\.2), "hc deferred pre h B=\(B)")
        }
    }

    /// A model whose every decode kernel engages: hidden 1024 (deferred HC),
    /// head_dim-128 KDA with 8 heads, 32 indexer heads, 128 experts of 512,
    /// affine-quantized as the artifact (4-bit experts, 8-bit elsewhere).
    private func fusedShapesModel(seed: UInt64) throws -> GLM5NextTextModel {
        var json = textConfigJSON(indexTopk: 8, linearHeadDim: 128)
        for (from, to) in [
            (#""hidden_size": 64"#, #""hidden_size": 1024"#),
            (#""intermediate_size": 96"#, #""intermediate_size": 512"#),
            (#""moe_intermediate_size": 32"#, #""moe_intermediate_size": 512"#),
            (#""n_routed_experts": 4"#, #""n_routed_experts": 128"#),
            (#""num_experts_per_tok": 2"#, #""num_experts_per_tok": 8"#),
            (#""kv_lora_rank": 32"#, #""kv_lora_rank": 64"#),
            (#""q_lora_rank": 32"#, #""q_lora_rank": 64"#),
            (#""qk_nope_head_dim": 32"#, #""qk_nope_head_dim": 64"#),
            (#""v_head_dim": 32"#, #""v_head_dim": 64"#),
            (#""index_n_heads": 2"#, #""index_n_heads": 32"#),
            (#""linear_attn_config": {"num_heads": 2"#, #""linear_attn_config": {"num_heads": 8"#),
        ] {
            json = json.replacingOccurrences(of: from, with: to)
        }
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        let model = GLM5NextTextModel(config)
        randomize(model, seed: seed)
        castToBFloat16(model)
        quantize(model: model) { path, module in
            guard module is Linear || module is SwitchLinear || module is MultiLinear else {
                return nil
            }
            return path.contains("switch_mlp") ? (64, 4, .affine) : (64, 8, .affine)
        }
        eval(model)
        model.train(false)
        return model
    }

    /// Every fused decode kernel covers `model`'s layers for `rows` one-token
    /// rows, so a batched test exercises them rather than the op path.
    private func assertFusedDecodeCovers(_ model: GLM5NextTextModel, rows: Int) throws {
        let layers = model.model.layers
        let x = MLXArray.zeros([rows, model.configuration.hiddenSize], dtype: .bfloat16)
        XCTAssertTrue(layers.allSatisfy { $0.deferredDecodeSupported(dtype: .bfloat16) })
        XCTAssertNotNil(
            glm5NextHCMix(
                MLXArray.zeros([rows, 1, 4, x.dim(1)], dtype: .bfloat16), fn: layers[0].attnHC.fn,
                eps: 1e-5), "HC mix kernel")
        for layer in layers {
            if let kda = layer.selfAttn as? GLM5NextLinearAttention {
                let parts = [
                    kda.qProj, kda.kProj, kda.vProj, kda.forgetGate.fAProj, kda.gAProj, kda.bProj,
                ].compactMap { glm5NextAffine($0) }
                XCTAssertEqual(parts.count, 6)
                XCTAssertNotNil(glm5NextMultiQMV(x, parts), "KDA projections in one multi-qmv")
                XCTAssertNotNil(glm5NextAffine(kda.forgetGate.fBProj))
            }
            if let attention = layer.selfAttn as? GLM5NextAttention, let indexer = attention.indexer {
                XCTAssertNotNil(
                    glm5NextDSASelect(
                        q: MLXArray.zeros([1, indexer.numHeads * indexer.headDim]),
                        weights: MLXArray.zeros([1, indexer.numHeads]), weightScale: 1,
                        pooledKeys: MLXArray.zeros([1, 40, indexer.headDim]), kvLength: 160,
                        heads: indexer.numHeads, dim: indexer.headDim,
                        indexTopk: indexer.indexTopk, kpool: indexer.kpool,
                        tail: indexer.alwaysSelectTail), "DSA selection kernel")
            }
            if let moe = layer.mlp as? GLM5NextMoE {
                let gate = moe.gate
                XCTAssertNotNil(
                    glm5NextMoERouter(
                        x, weight: gate.weight, bias: gate.eScoreCorrectionBias, topK: gate.topK,
                        scaling: gate.scaling, normalize: gate.normalize), "router kernels")
                let shared = try XCTUnwrap(moe.sharedExperts)
                let (sig, biased) = glm5NextRouterLogits(
                    x, weight: gate.weight, bias: gate.eScoreCorrectionBias)
                XCTAssertNotNil(
                    glm5NextMoEExperts(
                        x, sig: sig, biased: biased, topK: gate.topK, scaling: gate.scaling,
                        normalize: gate.normalize, limit: try XCTUnwrap(moe.limit),
                        gate: try XCTUnwrap(glm5NextAffine(moe.switchMLP.gateProjection)),
                        up: try XCTUnwrap(glm5NextAffine(moe.switchMLP.upProjection)),
                        down: try XCTUnwrap(glm5NextAffine(moe.switchMLP.downProjection)),
                        sharedGate: try XCTUnwrap(glm5NextAffine(shared.gateProj)),
                        sharedUp: try XCTUnwrap(glm5NextAffine(shared.upProj)),
                        sharedDown: try XCTUnwrap(glm5NextAffine(shared.downProj))),
                    "MoE expert kernels")
            }
        }
    }

    /// Decode widths per step for `B` rows: two-token (MTP verify) steps only
    /// while the forward stays within 8 flattened tokens, the batched
    /// exactness contract (past the core's qmv batch limit, 12 on M3 Ultra and
    /// 13 on M5 for the real projections, quantized matmuls take qmm).
    private func batchedWidths(_ widths: [Int], rows: Int) -> [Int] {
        widths.map { rows * $0 <= 8 ? $0 : 1 }
    }

    /// Continuous-batching exactness (SPEC-038 FR-CB10): B = 2, 4 and 8 rows
    /// of different context lengths (dense and sparse DSA rows, crossing
    /// `index_topk` and pool windows while decoding) decoded together, one
    /// token per row and then two-token (MTP verify) rows up to 8 flattened
    /// tokens, give each row's serial logits bit for bit, with and without the
    /// deferred HC chain.
    func testBatchedDecodeMatchesSerialRowsBitwise() throws {
        let model = try fusedShapesModel(seed: 131)
        let lengths = [3, 9, 5, 14, 7, 11, 6, 12]
        let prompts = lengths.enumerated().map { tokens($1, seed: UInt64(140 + $0)) }
        let steps = (0 ..< 8).map { tokens(8, seed: UInt64(160 + $0)) }
        let widths = [1, 1, 1, 1, 2, 1, 2]
        try assertFusedDecodeCovers(model, rows: 8)
        for deferred in [true, false] {
            model.model.deferredDecode = deferred
            for B in [2, 4, 8] {
                let widths = batchedWidths(widths, rows: B)
                func prefilled(_ r: Int) throws -> [KVCache] {
                    let cache = try model.newCache(parameters: nil)
                    eval(model(prompts[r], cache: cache))
                    return cache
                }
                var serial = [[MLXArray]]()
                for r in 0 ..< B {
                    let cache = try prefilled(r)
                    var logits = [MLXArray]()
                    var at = 0
                    for w in widths {
                        logits.append(model(steps[r][0..., at ..< (at + w)], cache: cache))
                        eval(logits.last!)
                        at += w
                    }
                    serial.append(logits)
                }
                let caches = batchCaches(try (0 ..< B).map(prefilled))
                var at = 0
                for (s, w) in widths.enumerated() {
                    let input = concatenated(
                        (0 ..< B).map { steps[$0][0..., at ..< (at + w)] }, axis: 0)
                    let logits = model(input, cache: caches)
                    eval(logits)
                    for r in 0 ..< B {
                        assertBitwiseEqual(
                            logits[r ..< (r + 1)], serial[r][s],
                            "deferred=\(deferred) B=\(B) row \(r) step \(s) width \(w)")
                    }
                    at += w
                }
            }
        }
        model.model.deferredDecode = true
    }

    /// The same exactness on the real 4-layer GLM-5.3-Flash cut (set
    /// `GLM5NEXT_CUT_DIR`; run with `MLX_ENABLE_TF32=0` on NAX GPUs): rows of
    /// 2040+ tokens cross `index_topk` = 2048 (dense to sparse DSA) while
    /// decoding together, up to 8 flattened tokens per forward.
    func testBatchedDecodeMatchesSerialRowsOnRealCut() throws {
        guard let path = ProcessInfo.processInfo.environment["GLM5NEXT_CUT_DIR"] else {
            throw XCTSkip("GLM5NEXT_CUT_DIR is not set")
        }
        let dir = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = GLM5NextModel(try JSONDecoder().decode(GLM5NextConfiguration.self, from: data))
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        let prompts = (0 ..< 8).map { r in
            withRandomState(MLXRandom.RandomState(seed: UInt64(170 + r))) {
                MLXRandom.randInt(0 ..< 150_000, [1, 2040 + 3 * r]).asType(.int32)
            }
        }
        let steps = (0 ..< 8).map { r in
            withRandomState(MLXRandom.RandomState(seed: UInt64(180 + r))) {
                MLXRandom.randInt(0 ..< 150_000, [1, 12]).asType(.int32)
            }
        }
        let widths = [1, 1, 1, 1, 1, 1, 2, 1, 1, 2]
        let text = model.languageModel
        for B in [2, 4, 8] {
            let widths = batchedWidths(widths, rows: B)
            func prefilled(_ r: Int) throws -> [KVCache] {
                let cache = try model.newCache(parameters: nil)
                eval(model(prompts[r], cache: cache))
                return cache
            }
            var serial = [[MLXArray]]()
            for r in 0 ..< B {
                let cache = try prefilled(r)
                var logits = [MLXArray]()
                var at = 0
                for w in widths {
                    logits.append(model(steps[r][0..., at ..< (at + w)], cache: cache))
                    eval(logits.last!)
                    at += w
                }
                serial.append(logits)
            }
            let caches = batchCaches(try (0 ..< B).map(prefilled))
            var at = 0
            for (s, w) in widths.enumerated() {
                let input = concatenated((0 ..< B).map { steps[$0][0..., at ..< (at + w)] }, axis: 0)
                let logits = model(input, cache: caches)
                eval(logits)
                for r in 0 ..< B {
                    assertBitwiseEqual(
                        logits[r ..< (r + 1)], serial[r][s], "real cut B=\(B) row \(r) step \(s)")
                }
                at += w
            }
        }
        XCTAssertEqual(text.model.layers.count, 4)
    }

    // MARK: - MTP

    private func emitState(checkpoint: Int? = nil) -> LMOutput.State {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpCacheCheckpointIndexKey] = checkpoint
        return state
    }

    /// Bitwise equality of two cache lists: recurrent states, and the
    /// latent, pooled keys and partial pool window of every sparse layer.
    private func assertCachesEqual(_ a: [KVCache], _ b: [KVCache], _ what: String) {
        for (index, (x, y)) in zip(a, b).enumerated() {
            if let x = x as? MambaCache, let y = y as? MambaCache {
                assertBitwiseEqual(x[0], y[0], "\(what): layer \(index) conv state")
                assertBitwiseEqual(x[1], y[1], "\(what): layer \(index) recurrent state")
            } else {
                XCTAssertEqual(x.offset, y.offset, "\(what): layer \(index) offset")
                for i in 0 ..< 4 {
                    assertBitwiseEqual(x.state[i], y.state[i], "\(what): layer \(index) state \(i)")
                }
            }
        }
    }

    /// An MTP verification forward (two columns, checkpoint after the first)
    /// gives each column the logits of a serial one-token decode bit for bit,
    /// emits the pre-norm hidden state and the last sparse layer's latent,
    /// and a one-token rewind (sparse trim + KDA checkpoint restore) leaves
    /// every cache equal to decoding only the first column.
    func testMTPVerifyColumnsMatchDecodeAndRewindIsExact() throws {
        let model = try fusedShapesModel(seed: 191)
        let prompt = tokens(13, seed: 192)
        let pair = tokens(2, seed: 193)
        let next = tokens(1, seed: 194)
        for deferred in [true, false] {
            model.model.deferredDecode = deferred
            let verified = try model.newCache(parameters: nil)
            let serial = try model.newCache(parameters: nil)
            eval(model(prompt, cache: verified), model(prompt, cache: serial))

            let output = model(LMInput.Text(tokens: pair), cache: verified, state: emitState(checkpoint: 1))
            let first = model(pair[0..., ..<1], cache: serial)
            let afterFirst = serial.map { $0.copy() }
            let second = model(pair[0..., 1...], cache: serial)
            assertBitwiseEqual(output.logits[0..., ..<1], first, "deferred=\(deferred) column 0")
            assertBitwiseEqual(output.logits[0..., 1...], second, "deferred=\(deferred) column 1")

            let hidden = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
            XCTAssertEqual(hidden.shape, [1, 2, model.configuration.hiddenSize])
            let shared = try XCTUnwrap(output.state?[mtpSharedKVStatesKey]?["full_attention"])
            XCTAssertEqual(shared.0.dim(-2), 15)
            XCTAssertEqual(output.state?[mtpSharedKVSourceIndicesKey]?["full_attention"], 5)

            // Both columns kept: the caches equal decoding both tokens.
            let accepted = verified.map { $0.copy() }
            assertCachesEqual(accepted, serial, "deferred=\(deferred) both columns kept")
            XCTAssertEqual(rewindSpeculativePromptCache(verified, numTokens: 1), 1)
            assertCachesEqual(verified, afterFirst, "deferred=\(deferred) after rewind")
            assertBitwiseEqual(
                model(next, cache: verified), model(next, cache: afterFirst),
                "deferred=\(deferred) decode after rewind")
        }
        model.model.deferredDecode = true
        XCTAssertEqual(model.maximumNativeTargetCacheRewind, 1)
    }

    private func copyState(_ state: MTPDrafterState) -> MTPDrafterState {
        MTPDrafterState(
            cache: state.cache.map { $0.copy() }, nextPosition: state.nextPosition,
            seedToken: state.seedToken, seedHidden: state.seedHidden,
            proposalAppended: state.proposalAppended)
    }

    private func assertDrafterStatesEqual(
        _ a: MTPDrafterState, _ b: MTPDrafterState, _ what: String
    ) {
        XCTAssertEqual(a.nextPosition, b.nextPosition, "\(what): next position")
        XCTAssertEqual(a.proposalAppended, b.proposalAppended, "\(what): appended")
        assertBitwiseEqual(a.seedToken, b.seedToken, "\(what): seed token")
        assertBitwiseEqual(a.seedHidden, b.seedHidden, "\(what): seed hidden")
        assertCachesEqual(a.cache, b.cache, what)
    }

    /// The drafter: a tentative chained proposal rolled back by the commit
    /// (exact trim) leaves the state of a commit that never proposed, and a
    /// packed advance of rows with zero and one accepted proposals equals
    /// each row's serial commit bit for bit (6 flattened tokens).
    func testMTPDrafterRollbackAndPackedAdvanceMatchSerialCommit() throws {
        let model = try fusedShapesModel(seed: 201)
        let drafter = GLM5NextMTPDraftModel(model.configuration)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let D = model.configuration.hiddenSize

        var prepared = [MTPDrafterState]()
        for (r, length) in [6, 11, 13].enumerated() {
            let prompt = tokens(length, seed: UInt64(210 + r))
            let cache = try model.newCache(parameters: nil)
            let out = model(LMInput.Text(tokens: prompt), cache: cache, state: emitState())
            let bonus = argMax(out.logits[0..., -1, 0...], axis: -1)
            var state = drafter.makeState(parameters: nil)
            drafter.prepareDrafterState(
                target: model, promptTokens: prompt,
                targetHidden: try XCTUnwrap(out.state?[mtpLastHiddenStatesKey]),
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
            eval(state.cache, state.seedToken!, state.seedHidden!)
            XCTAssertEqual(state.nextPosition, length)
            prepared.append(state)
        }
        let targetHidden = (0 ..< 3).map { r in
            withRandomState(MLXRandom.RandomState(seed: UInt64(220 + r))) {
                MLXRandom.normal([1, 2, D]).asType(.bfloat16)
            }
        }

        // Rollback: a chained proposal appended, then rejected.
        var proposing = copyState(prepared[0])
        proposing.seedToken = nil
        proposing.seedHidden = nil
        let draft = drafter.draftBlock(
            target: model, lastToken: MLXArray([Int32(5)]),
            lastHidden: targetHidden[0][0..., ..<1], sharedKV: [:], positionDeltas: nil,
            queryOffset: 0, blockSize: 2, state: &proposing, sampler: sampler)
        XCTAssertEqual(proposing.proposalAppended, 1)
        var neverProposed = copyState(prepared[0])
        neverProposed.seedToken = nil
        neverProposed.seedHidden = nil
        drafter.commitDrafterState(
            target: model, targetHidden: targetHidden[0], draftTokens: draft, acceptedCount: 0,
            finalToken: MLXArray([Int32(9)]), positionDeltas: nil, state: &proposing,
            sampler: sampler)
        drafter.commitDrafterState(
            target: model, targetHidden: targetHidden[0], draftTokens: draft, acceptedCount: 0,
            finalToken: MLXArray([Int32(9)]), positionDeltas: nil, state: &neverProposed,
            sampler: sampler)
        assertDrafterStatesEqual(proposing, neverProposed, "rolled-back proposal")

        // Packed advance vs serial commits.
        let accepted: [[Int]] = [[], [17], []]
        let finals = [9, 23, 41]
        var serial = [MTPDrafterState]()
        for r in 0 ..< 3 {
            var state = copyState(prepared[r])
            let proposal = state.seedToken!
            state.seedToken = nil
            state.seedHidden = nil
            let draftTokens = accepted[r].isEmpty
                ? proposal : MLXArray(accepted[r].map(Int32.init)).reshaped(1, -1)
            drafter.commitDrafterState(
                target: model, targetHidden: targetHidden[r], draftTokens: draftTokens,
                acceptedCount: accepted[r].count, finalToken: MLXArray([Int32(finals[r])]),
                positionDeltas: nil, state: &state, sampler: sampler)
            serial.append(state)
        }
        let packed = try drafter.advanceAndProposePacked(
            target: model,
            rows: (0 ..< 3).map { r in
                var state = prepared[r]
                state.seedToken = nil
                state.seedHidden = nil
                return MTPPackedDrafterAdvanceRow(
                    targetHidden: targetHidden[r], acceptedTokens: accepted[r],
                    finalToken: finals[r], positionDeltas: nil, state: state)
            },
            sampler: sampler)
        for r in 0 ..< 3 {
            assertDrafterStatesEqual(packed.states[r], serial[r], "packed row \(r)")
            assertBitwiseEqual(
                packed.proposals[r ..< (r + 1)], serial[r].seedToken, "packed proposal \(r)")
        }
        // The packed advance never touches its input states.
        XCTAssertEqual(prepared[1].cache[0].offset, 11)
    }

    /// Proposes the plain-greedy continuation, except a wrong token every
    /// third round, so the iterator exercises the target's accept path
    /// (both verify columns kept) and its in-place rewind.
    private final class OracleDrafter: Module, MTPDrafterModel {
        let expected: [Int]
        let promptLength: Int
        let maximumBlockSize: Int? = 2
        private var round = 0

        init(expected: [Int], promptLength: Int) {
            self.expected = expected
            self.promptLength = promptLength
            super.init()
        }

        func draftBlock(
            target _: any LanguageModel, lastToken _: MLXArray, lastHidden _: MLXArray,
            sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
            queryOffset: Int, blockSize _: Int, sampler _: any LogitSampler
        ) -> MLXArray {
            round += 1
            let next = queryOffset - promptLength + 1
            let token = next < expected.count ? expected[next] : 0
            return MLXArray([Int32(round % 3 == 0 ? (token + 1) % 64 : token)]).reshaped(1, 1)
        }
    }

    /// The MTP iterator over the GLM target with proposals that are accepted
    /// two rounds in three: output equals plain greedy, and the accepted
    /// count shows both paths ran.
    func testMTPIteratorAcceptAndRewindOnGLMTargetMatchPlainGreedy() throws {
        let model = try fusedShapesModel(seed: 251)
        let prompt = tokens(13, seed: 252).reshaped(-1)
        let parameters = greedyParameters(maxTokens: 30)
        var plain = try TokenIterator(prompt: prompt, model: model, parameters: parameters)
        var expected = [Int]()
        while let token = plain.next() { expected.append(token) }
        let drafter = OracleDrafter(expected: expected, promptLength: 13)
        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
            parameters: parameters, blockSize: 2)
        var produced = [Int]()
        while let token = speculative.next() { produced.append(token) }
        XCTAssertNil(speculative.passthroughReason)
        XCTAssertEqual(produced, expected)
        XCTAssertGreaterThan(speculative.acceptedCount, 4)
        XCTAssertLessThan(speculative.acceptedCount, speculative.proposedCount)
    }

    private func greedyParameters(maxTokens: Int) -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        // The MTP iterator prefills the prompt in one forward; match it.
        parameters.prefill.stepSize = Int.max
        parameters.prefill.chunking = .unchunked
        return parameters
    }

    /// Greedy generation through `MTPSpeculativeTokenIterator` with the GLM
    /// drafter equals plain greedy generation token for token.
    private func assertMTPGreedyMatchesPlain(
        _ model: any LanguageModel, configuration: GLM5NextConfiguration, prompt: MLXArray,
        count: Int, label: String
    ) throws -> (proposed: Int, accepted: Int) {
        let parameters = greedyParameters(maxTokens: count)
        var plain = try TokenIterator(prompt: prompt, model: model, parameters: parameters)
        var expected = [Int]()
        while let token = plain.next() { expected.append(token) }

        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: model,
            drafter: GLM5NextMTPDraftModel(configuration), parameters: parameters, blockSize: 2)
        var produced = [Int]()
        while let token = speculative.next() { produced.append(token) }
        XCTAssertNil(speculative.passthroughReason, "\(label): stayed speculative")
        XCTAssertEqual(produced, expected, "\(label): MTP greedy equals plain greedy")
        XCTAssertGreaterThan(speculative.proposedCount, 0, label)
        return (speculative.proposedCount, speculative.acceptedCount)
    }

    func testMTPSpeculativeGreedyMatchesPlainGreedy() throws {
        let model = try fusedShapesModel(seed: 231)
        for deferred in [true, false] {
            model.model.deferredDecode = deferred
            _ = try assertMTPGreedyMatchesPlain(
                model, configuration: model.configuration,
                prompt: tokens(13, seed: 232).reshaped(-1), count: 24,
                label: "deferred=\(deferred)")
        }
        model.model.deferredDecode = true
    }

    /// On the real 4-layer cut (`GLM5NEXT_CUT_DIR`): MTP greedy equals plain
    /// greedy, and the acceptance rate is reported (the cut's trunk is not the
    /// one the head was trained on, so it is a lower bound).
    func testMTPSpeculativeGreedyMatchesPlainGreedyOnRealCut() throws {
        guard let path = ProcessInfo.processInfo.environment["GLM5NEXT_CUT_DIR"] else {
            throw XCTSkip("GLM5NEXT_CUT_DIR is not set")
        }
        let dir = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let configuration = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(configuration)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        XCTAssertTrue(model.languageModel.hasMTPWeights)
        let prompt = withRandomState(MLXRandom.RandomState(seed: 240)) {
            MLXRandom.randInt(0 ..< 150_000, [300]).asType(.int32)
        }
        let (proposed, accepted) = try assertMTPGreedyMatchesPlain(
            model, configuration: configuration, prompt: prompt, count: 64, label: "real cut")
        print("glm5next_mtp_real_cut proposed=\(proposed) accepted=\(accepted)")

        // A verify pass past index_topk (sparse DSA) equals serial decode.
        let long = withRandomState(MLXRandom.RandomState(seed: 241)) {
            MLXRandom.randInt(0 ..< 150_000, [1, 2100]).asType(.int32)
        }
        let pair = MLXArray([Int32(1000), Int32(2000)]).reshaped(1, 2)
        let verified = try model.newCache(parameters: nil)
        let serial = try model.newCache(parameters: nil)
        eval(model(long, cache: verified), model(long, cache: serial))
        let output = model(LMInput.Text(tokens: pair), cache: verified, state: emitState(checkpoint: 1))
        let first = model(pair[0..., ..<1], cache: serial)
        let afterFirst = serial.map { $0.copy() }
        let second = model(pair[0..., 1...], cache: serial)
        assertBitwiseEqual(output.logits[0..., ..<1], first, "real cut verify column 0")
        assertBitwiseEqual(output.logits[0..., 1...], second, "real cut verify column 1")
        XCTAssertEqual(rewindSpeculativePromptCache(verified, numTokens: 1), 1)
        assertCachesEqual(verified, afterFirst, "real cut after rewind")
    }

    // MARK: - Weight names

    /// Tensor names of GLM-5.3-Flash-oQ4e-mtp (`model.safetensors.index.json`,
    /// `.scales`/`.biases` omitted) for a dense KDA layer (0), a DSA + MoE layer
    /// (3), a KDA + MoE layer (4) and the MTP layer.
    private static let artifactLayerKeys: [Int: [String]] = [
        0: [
            "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "ffn_hc.base", "ffn_hc.fn",
            "ffn_hc.scale", "input_layernorm.weight", "mlp.down_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "post_attention_layernorm.weight",
            "self_attn.b_proj.weight", "self_attn.conv1d.weight", "self_attn.forget_gate.A_log",
            "self_attn.forget_gate.dt_bias", "self_attn.forget_gate.f_a_proj.weight",
            "self_attn.forget_gate.f_b_proj.weight", "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight", "self_attn.k_proj.weight", "self_attn.o_norm.weight",
            "self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
        ],
        3: [
            "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "ffn_hc.base", "ffn_hc.fn",
            "ffn_hc.scale", "input_layernorm.weight", "mlp.gate.e_score_correction_bias",
            "mlp.gate.weight", "mlp.shared_experts.down_proj.weight",
            "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
            "mlp.switch_mlp.down_proj.weight", "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.up_proj.weight", "post_attention_layernorm.weight",
            "self_attn.embed_q.weight", "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate", "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.k_norm.weight", "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.wk.weight", "self_attn.indexer.wq_b.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
            "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
            "self_attn.unembed_out.weight",
        ],
        4: [
            "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "ffn_hc.base", "ffn_hc.fn",
            "ffn_hc.scale", "input_layernorm.weight", "mlp.gate.e_score_correction_bias",
            "mlp.gate.weight", "mlp.shared_experts.down_proj.weight",
            "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
            "mlp.switch_mlp.down_proj.weight", "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.up_proj.weight", "post_attention_layernorm.weight",
            "self_attn.b_proj.weight", "self_attn.conv1d.weight", "self_attn.forget_gate.A_log",
            "self_attn.forget_gate.dt_bias", "self_attn.forget_gate.f_a_proj.weight",
            "self_attn.forget_gate.f_b_proj.weight", "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight", "self_attn.k_proj.weight", "self_attn.o_norm.weight",
            "self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
        ],
    ]

    private static let artifactMTPKeys = [
        "block.input_layernorm.weight", "block.mlp.gate.e_score_correction_bias",
        "block.mlp.gate.weight", "block.mlp.shared_experts.down_proj.weight",
        "block.mlp.shared_experts.gate_proj.weight", "block.mlp.shared_experts.up_proj.weight",
        "block.mlp.switch_mlp.down_proj.weight", "block.mlp.switch_mlp.gate_proj.weight",
        "block.mlp.switch_mlp.up_proj.weight", "block.post_attention_layernorm.weight",
        "block.self_attn.embed_q.weight", "block.self_attn.indexer.index_kpool_compress_ape",
        "block.self_attn.indexer.index_kpool_compress_gate", "block.self_attn.indexer.k_norm.bias",
        "block.self_attn.indexer.k_norm.weight", "block.self_attn.indexer.weights_proj.weight",
        "block.self_attn.indexer.wk.weight", "block.self_attn.indexer.wq_b.weight",
        "block.self_attn.kv_a_layernorm.weight", "block.self_attn.kv_a_proj_with_mqa.weight",
        "block.self_attn.o_proj.weight", "block.self_attn.q_a_layernorm.weight",
        "block.self_attn.q_a_proj.weight", "block.self_attn.q_b_proj.weight",
        "block.self_attn.unembed_out.weight", "eh_proj.weight", "enorm.weight", "hnorm.weight",
        "norm.weight",
    ]

    func testModuleNamesMatchArtifactTensorNames() throws {
        let model = GLM5NextModel(try configuration(nested: true))
        let keys = Set(model.parameters().flattened().map(\.0))

        for (layer, expected) in Self.artifactLayerKeys {
            let prefix = "language_model.model.layers.\(layer)."
            let actual = Set(
                keys.filter { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) })
            XCTAssertEqual(actual, Set(expected), "layer \(layer)")
        }
        let mtpPrefix = "language_model.mtp.0."
        let mtp = Set(
            keys.filter { $0.hasPrefix(mtpPrefix) }.map { String($0.dropFirst(mtpPrefix.count)) })
        XCTAssertEqual(mtp, Set(Self.artifactMTPKeys))
        for global in [
            "language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
        ] {
            XCTAssertTrue(keys.contains(global), global)
        }
    }

    /// The artifact's own key layout (plus vision tensors) loads unchanged.
    func testSanitizeKeepsArtifactLayoutAndDropsVision() throws {
        let source = GLM5NextModel(try configuration(nested: true))
        randomize(source)
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["vision_model.blocks.0.attn.qkv.weight"] = MLXArray.zeros([4, 4])
        weights["vision_model.merger.proj.weight"] = MLXArray.zeros([4, 4])

        let target = GLM5NextModel(try configuration(nested: true))
        let sanitized = target.sanitize(weights: weights)
        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("vision") })
        try target.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        XCTAssertTrue(target.languageModel.hasMTPWeights)

        let input = tokens(10)
        XCTAssertEqual(maxAbsDifference(source(input, cache: nil), target(input, cache: nil)), 0)
    }

    /// Hugging Face layout: `model.language_model.*`, `hc_attn_*`, top-level
    /// forget-gate tensors, per-projection convs in torch layout, per-expert
    /// weights, `kv_b_proj`, and the nextn layer at `layers.<L>`.
    func testSanitizeMapsHuggingFaceLayout() throws {
        let config = try configuration(nested: true)
        let source = GLM5NextModel(config)
        randomize(source)
        let reference = Dictionary(uniqueKeysWithValues: source.parameters().flattened())

        var hf = [String: MLXArray]()
        var kvB = [String: (wk: MLXArray?, wv: MLXArray?)]()
        for (key, value) in reference {
            var key = String(key.dropFirst("language_model.".count))
            if key.hasPrefix("mtp.0.") {
                let suffix = String(key.dropFirst("mtp.0.".count))
                let mapped =
                    suffix.hasPrefix("block.")
                    ? String(suffix.dropFirst("block.".count))
                    : suffix == "norm.weight" ? "shared_head.norm.weight" : suffix
                key = "model.layers.\(config.hiddenLayers).\(mapped)"
            }
            if key.hasPrefix("model.") {
                key = "model.language_model." + key.dropFirst("model.".count)
            }
            key = key.replacingOccurrences(of: ".attn_hc.", with: ".hc_attn_")
                .replacingOccurrences(of: ".ffn_hc.", with: ".hc_ffn_")
                .replacingOccurrences(of: ".forget_gate.", with: ".")

            if key.hasSuffix(".self_attn.conv1d.weight") {
                let prefix = String(key.dropLast("conv1d.weight".count))
                let torchLayout = value.movedAxis(source: 1, destination: 2)
                let parts = split(torchLayout, parts: 3, axis: 0)
                for (name, part) in zip(["q", "k", "v"], parts) {
                    hf[prefix + "\(name)_conv1d.weight"] = part
                }
                continue
            }
            if let range = key.range(of: ".mlp.switch_mlp.") {
                for expert in 0 ..< value.dim(0) {
                    let name =
                        key[..<range.lowerBound] + ".mlp.experts.\(expert)."
                        + key[range.upperBound...]
                    hf[String(name)] = value[expert]
                }
                continue
            }
            if key.hasSuffix(".self_attn.embed_q.weight") {
                kvB[String(key.dropLast("embed_q.weight".count)), default: (nil, nil)].wk = value
                continue
            }
            if key.hasSuffix(".self_attn.unembed_out.weight") {
                kvB[String(key.dropLast("unembed_out.weight".count)), default: (nil, nil)].wv =
                    value
                continue
            }
            hf[key] = value
        }
        for (prefix, pair) in kvB {
            let wk = pair.wk!.swappedAxes(-1, -2)
            let combined = concatenated([wk, pair.wv!], axis: 1)
            hf[prefix + "kv_b_proj.weight"] = combined.reshaped(-1, combined.dim(-1))
        }
        hf["model.language_model.layers.\(config.hiddenLayers).shared_head.head.weight"] =
            MLXArray.zeros([4, 4])
        hf["model.visual.patch_embed.proj.weight"] = MLXArray.zeros([4, 4])

        let target = GLM5NextModel(config)
        let sanitized = target.sanitize(weights: hf)
        XCTAssertEqual(Set(sanitized.keys), Set(reference.keys))
        for (key, value) in reference {
            XCTAssertLessThan(maxAbsDifference(sanitized[key]!, value), 1e-6, key)
        }
        try target.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    func testSanitizeWithoutMTPWeightsDropsMTPModules() throws {
        let source = GLM5NextTextModel(try configuration())
        randomize(source)
        let weights = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().filter {
                !$0.0.hasPrefix("mtp.")
            })

        let target = GLM5NextTextModel(try configuration())
        let sanitized = target.sanitize(weights: weights)
        try target.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        XCTAssertFalse(target.hasMTPWeights)

        let input = tokens(9)
        XCTAssertEqual(maxAbsDifference(source(input, cache: nil), target(input, cache: nil)), 0)
    }

    /// Mixed-precision checkpoint through `loadWeights`: per-layer bit widths
    /// keyed by `language_model.*` paths (as in the artifact's `quantization`).
    func testLoadWeightsWithPerLayerQuantization() throws {
        let config = try configuration(nested: true)
        let source = GLM5NextModel(config)
        randomize(source)

        func bits(_ path: String) -> Int {
            if path.hasSuffix("embed_q") || path.hasSuffix("b_proj") { return 5 }
            if path.contains("indexer") || path.hasSuffix("lm_head") { return 8 }
            return 4
        }
        quantize(
            model: source,
            filter: {
                (path: String, module: Module) -> (
                    groupSize: Int, bits: Int, mode: QuantizationMode
                )?
                in
                guard module is Quantizable, !path.hasSuffix("eh_proj") else { return nil }
                return (32, bits(path), .affine)
            })
        eval(source)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm5next-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["vision_model.post_layernorm.weight"] = MLXArray.ones([8])
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, _) in source.leafModules().flattened() where bits(path) != 4 {
            perLayer[path] = .quantize(.init(groupSize: 32, bits: bits(path)))
        }
        let target = GLM5NextModel(config)
        try loadWeights(
            modelDirectory: directory, model: target,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4), perLayerQuantization: perLayer))

        let embedQ = target.languageModel.model.layers[3].selfAttn as! GLM5NextAttention
        XCTAssertEqual((embedQ.embedQ as? QuantizedMultiLinear)?.bits, 5)

        let input = tokens(12)
        XCTAssertEqual(maxAbsDifference(source(input, cache: nil), target(input, cache: nil)), 0)
    }
}
