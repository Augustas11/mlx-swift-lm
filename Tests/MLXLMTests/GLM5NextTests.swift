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
