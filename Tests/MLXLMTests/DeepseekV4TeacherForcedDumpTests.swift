import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Lab tool for a Studio window, skipped unless its environment is set: a
/// teacher-forced pass of a fixed token sequence through the full V4-Flash
/// model, dumping the per-layer state that `scripts/lab/dsv4/
/// dsv4_tf_reference.py dump` writes for DeepSeek's reference, so the two
/// can be diffed layer by layer (`... compare`).
///
/// - `DSV4_ARTIFACT_DIR`: the V4-Flash-0731 MXFP4 directory.
/// - `DSV4_DUMP_TOKENS`: JSON `{"tokens": [...], "positions": [...]}`.
/// - `DSV4_DUMP_OUT`: the safetensors file to write.
///
/// The tokens prefill in the serving plan's bounded reference chunks. At each listed
/// position the dump holds, per layer, the attention and expert inputs, the
/// hidden state after the layer (`h.L`, all hyper-connection streams), the
/// chosen experts (sorted) and, on ratio-4 layers, the exact top-k over the
/// entries visible at that position, recomputed from the captured attention
/// input and the final index pool (sorted, `-1` padded; for an explicit HISA
/// config this is not the selection the forward used); per
/// compressed layer the final pooled entries; and the float32 logits.
final class DeepseekV4TeacherForcedDumpTests: XCTestCase {
    func testLayer2IndexerDiagnosticDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let input = env["DSV4_INDEXER_INPUT"],
            let output = env["DSV4_INDEXER_OUTPUT"]
        else { throw XCTSkip("Indexer diagnostic environment is not set") }
        let dir = URL(fileURLWithPath: artifact)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: input))
        let x = reference["input"]!
        let attn = model.layers[2].attn
        let ix = try XCTUnwrap(attn.indexer)
        let qr = attn.qNorm(deepseekV4Project(attn.wqA, x))
        let q = attn.rope(deepseekV4Project(ix.wqB, qr)
            .reshaped(1, x.dim(1), ix.heads, ix.headDim).transposed(0, 2, 1, 3), offset: 0)
        let raw = deepseekV4LinearF32(ix.weightsProj, x)
        let weights = raw * (ix.scale * pow(Float(ix.heads), -0.5))
        let cache = DeepseekV4PoolCache(ratio: 4)
        let selected = try XCTUnwrap(ix(x, qResidual: qr, rope: attn.rope, cache: cache, offset: 0))
        let pool = try XCTUnwrap(cache.pooled)
        let visible = MLXArray.arange(1, x.dim(1) + 1, dtype: .int32).floorDivide(Int32(4))
            .reshaped(1, x.dim(1), 1)
        let top = MLX.where(selected .< visible, selected, MLXArray(Int32(-1)))
        let dot = matmul(q[0..., 0..., 4096 ..< 5120].asType(.float32),
                         expandedDimensions(pool, axis: 1).swappedAxes(-1, -2).asType(.float32))
        let scores = (maximum(dot, 0)
            * expandedDimensions(weights[0..., 4096 ..< 5120].swappedAxes(-1, -2), axis: -1))
            .sum(axis: 1)
        let marks = MLXArray.arange(0, 1024, step: 64, dtype: .int32)
        let values = ["q_residual": qr, "q": q, "raw_weights": raw.asType(.float32),
                      "weights": weights, "top": top, "pool": pool.asType(.float32),
                      "index_logits": take(dot, marks, axis: 2),
                      "index_scores": take(scores, marks, axis: 1)]
        eval(Array(values.values))
        try save(arrays: values, url: URL(fileURLWithPath: output))
    }

    func testRouterElementaryDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let input = env["DSV4_ROUTER_STAGE_INPUT"], let output = env["DSV4_ROUTER_STAGE_OUTPUT"]
        else { throw XCTSkip("Router diagnostic environment is not set") }
        let x = try MLX.loadArrays(url: URL(fileURLWithPath: input))["router_logits"]!
        let e = exp(-abs(x))
        let l = log1p(e)
        let sp = l + maximum(x, 0)
        let score = sqrt(sp)
        eval(e, l, sp, score)
        try save(arrays: ["exp": e, "log1p": l, "softplus": sp, "score": score],
                 url: URL(fileURLWithPath: output))
    }

    func testLayer0MoESameInputDiagnosticDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let input = env["DSV4_MOE_INPUT"],
            let tokenPath = env["DSV4_DUMP_TOKENS"], let output = env["DSV4_MOE_OUT"]
        else { throw XCTSkip("MoE diagnostic environment is not set") }
        let dir = URL(fileURLWithPath: artifact)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: input))
        struct Request: Decodable { let tokens: [Int] }
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
        let at = reference["positions"]!.asType(.int32)
        let x = reference["ffn_in.0"]!.asType(.bfloat16)
        let ids = MLXArray(request.tokens.map(Int32.init))[at]
        let ffn = model.layers[0].ffn
        if env["DSV4_MOE_OUTPUT_ONLY"] == "1" {
            let result = ffn(x, inputIDs: ids).asType(.float32)
            eval(result)
            let (indices, weights) = ffn.gate(x, inputIDs: ids)
            let shared = ffn.sharedExperts(x).asType(.float32)
            let logits = matmul(x.asType(.float32), ffn.gate.weight.asType(.float32).T)
            let scores = deepseekV4RouterSoftplusScores(logits)
            let selected = takeAlong(scores, indices, axis: -1)
            let denominator = selected.sum(axis: -1, keepDims: true)
            eval(shared, indices, weights, logits, scores, selected, denominator)
            try save(arrays: ["positions": at, "output": result, "shared": shared,
                              "indices": indices, "weights": weights, "router_logits": logits,
                              "router_scores": scores, "selected": selected, "denominator": denominator],
                     url: URL(fileURLWithPath: output))
            return
        }
        let (indices, weights) = ffn.gate(x, inputIDs: ids)
        let k = indices.dim(-1)
        let flat = indices.flattened()
        let order = argSort(flat)
        let inverse = argSort(order)
        let inputs = expandedDimensions(x, axes: [-2, -3]).flattened(start: 0, end: -3)[order.floorDivide(k)]
        let experts = flat[order]
        let routeWeights = weights.flattened()[order].reshaped(-1, 1, 1)
        let gate = ffn.switchMLP.gateProjection(inputs, experts, sortedIndices: true)
        let up = ffn.switchMLP.upProjection(inputs, experts, sortedIndices: true)
        let activation = makeLimitedSwiGLU(limit: config.swigluLimit > 0 ? config.swigluLimit : nil)
        let hidden = (activation(gate.asType(.float32), up.asType(.float32)) * routeWeights).asType(x.dtype)
        let down = ffn.switchMLP.downProjection(hidden, experts, sortedIndices: true)
        func restore(_ v: MLXArray) -> MLXArray {
            scatterUnsort(x: v, invOrder: inverse, shape: indices.shape).squeezed(axis: -2).asType(.float32)
        }
        let routed = restore(down)
        let sortedSlots = argSort(indices, axis: -1)
        let ascending = takeAlong(routed, expandedDimensions(sortedSlots, axis: -1), axis: -2)
        var serialSum = MLXArray.zeros([x.dim(0), x.dim(1)], dtype: .float32)
        for i in 0 ..< k { serialSum = serialSum + ascending[0..., i, 0...] }
        let out = ["positions": at, "input": x.asType(.float32), "indices": indices,
                   "weights": weights, "gate": restore(gate), "up": restore(up),
                   "hidden": restore(hidden), "routed": routed,
                   "sum": routed.sum(axis: -2), "ascending_sum": serialSum,
                   "shared": ffn.sharedExperts(x).asType(.float32),
                   "output": ffn(x, inputIDs: ids).asType(.float32)]
        eval(Array(out.values))
        try save(arrays: out, url: URL(fileURLWithPath: output))
    }

    func testLayer0HCPreDiagnosticDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let tokenPath = env["DSV4_DUMP_TOKENS"],
            let outPath = env["DSV4_HC_DIAGNOSTIC_OUT"]
        else {
            throw XCTSkip(
                "DSV4_ARTIFACT_DIR, DSV4_DUMP_TOKENS and DSV4_HC_DIAGNOSTIC_OUT are not set")
        }

        struct Request: Decodable { let tokens: [Int] }
        let request = try JSONDecoder().decode(
            Request.self, from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
        XCTAssertEqual(request.tokens.count, 8192)

        let dir = URL(fileURLWithPath: artifact)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let ids = MLXArray(request.tokens.map(Int32.init)).reshaped(1, request.tokens.count)
        let hFull = repeated(
            expandedDimensions(model.embed(ids), axis: 2), count: config.hcMult, axis: 2)
        let layer = model.layers[0]
        let connection = layer.attnHC
        func capture(_ h: MLXArray) -> (
            mixes: MLXArray, current: (pre: MLXArray, post: MLXArray, comb: MLXArray),
            oracleComb: MLXArray, attnInput: MLXArray, post: MLXArray, comb: MLXArray
        ) {
            let flat = h.asType(.float32).reshaped(
                h.dim(0), h.dim(1), h.dim(2) * h.dim(3))
            let inverse = rsqrt(
                square(flat).mean(axis: -1, keepDims: true) + connection.normEps)
            let mixes = matmul(flat, connection.fn.T) * inverse
            let current = replaySinkhorn(mixes, connection: connection, precise: true)
            let oracleComb = replaySinkhorn(
                mixes, connection: connection, precise: false).comb
            let (attnInput, post, comb) = connection.normalizedBranchInput(
                h, norm: layer.attnNorm)
            return (mixes, current, oracleComb, attnInput, post, comb)
        }
        let hChunk = hFull[0..., ..<DeepseekV4Model.prefillStepSize]
        let chunk = capture(hChunk)
        let full = capture(hFull)
        let positions = [64] + Array(257 ... 320)
        let at = MLXArray(positions.map(Int32.init))
        let out = [
            "positions": at,
            "hc_residual.0": take(hFull[0], at, axis: 0).asType(.float32),
            "hc_mixes_chunk.0": take(chunk.mixes[0], at, axis: 0),
            "hc_pre_uncompiled_chunk.0": take(chunk.current.pre[0], at, axis: 0),
            "hc_post_uncompiled_chunk.0": take(chunk.current.post[0], at, axis: 0),
            "hc_comb_uncompiled_precise_chunk.0": take(chunk.current.comb[0], at, axis: 0),
            "hc_comb_uncompiled_oracle_chunk.0": take(chunk.oracleComb[0], at, axis: 0),
            "hc_attn_input_chunk.0": take(chunk.attnInput[0], at, axis: 0).asType(.float32),
            "hc_post_chunk.0": take(chunk.post[0], at, axis: 0),
            "hc_comb_chunk.0": take(chunk.comb[0], at, axis: 0),
            "hc_mixes_full.0": take(full.mixes[0], at, axis: 0),
            "hc_pre_uncompiled_full.0": take(full.current.pre[0], at, axis: 0),
            "hc_post_uncompiled_full.0": take(full.current.post[0], at, axis: 0),
            "hc_comb_uncompiled_precise_full.0": take(full.current.comb[0], at, axis: 0),
            "hc_comb_uncompiled_oracle_full.0": take(full.oracleComb[0], at, axis: 0),
            "hc_attn_input_full.0": take(full.attnInput[0], at, axis: 0).asType(.float32),
            "hc_post_full.0": take(full.post[0], at, axis: 0),
            "hc_comb_full.0": take(full.comb[0], at, axis: 0),
        ]
        eval(Array(out.values))
        try save(arrays: out, url: URL(fileURLWithPath: outPath))
    }

    func testFullModelTeacherForcedDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let tokenPath = env["DSV4_DUMP_TOKENS"],
            let outPath = env["DSV4_DUMP_OUT"]
        else { throw XCTSkip("DSV4_ARTIFACT_DIR, DSV4_DUMP_TOKENS and DSV4_DUMP_OUT are not set") }

        struct Request: Decodable {
            let tokens: [Int]
            let positions: [Int]
        }
        let request = try JSONDecoder().decode(
            Request.self, from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
        let tokens = request.tokens
        let positions = request.positions
        XCTAssertFalse(positions.isEmpty)
        XCTAssertTrue(positions.allSatisfy { (0 ..< tokens.count).contains($0) })
        XCTAssertEqual(positions, positions.sorted())

        let dir = URL(fileURLWithPath: artifact)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let caches = try model.newCache(parameters: nil)
        var captured = [String: [MLXArray]]()
        var ropeDiagnostic = [String: MLXArray]()
        func keep(_ key: String, _ value: MLXArray) { captured[key, default: []].append(value) }

        // Supplemental geometry replay; the default remains the serving chunk size.
        let step = env["DSV4_DUMP_PREFILL_STEP"].flatMap(Int.init)
            ?? min(DeepseekV4Model.prefillStepSize, tokens.count)
        guard step > 0, step <= tokens.count else {
            throw NSError(domain: "DSV4DumpGeometry", code: 1)
        }
        for start in stride(from: 0, to: tokens.count, by: step) {
            let end = min(start + step, tokens.count)
            let rows = positions.filter { (start ..< end).contains($0) }.map { Int32($0 - start) }
            let ids = MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, end - start)
            let layerCaches = zip(caches, model.layers).map {
                DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
            }
            var mask: MLXArray?
            if end - start > 1,
                case .array(let array) = layerCaches[0].local.makeMask(
                    n: end - start, windowSize: config.slidingWindow, returnArray: true)
            {
                mask = array
            }
            let at = MLXArray(rows)
            var h = repeated(
                expandedDimensions(model.embed(ids), axis: 2), count: config.hcMult, axis: 2)
            for (l, (layer, cache)) in zip(model.layers, layerCaches).enumerated() {
                if l == 0, start == 0, env["DSV4_ROPE_DIAGNOSTIC_OUT"] != nil {
                    let dense = MLXArray(([64] + Array(257 ... 320)).map(Int32.init))
                    ropeDiagnostic["hc_residual.0"] = take(h[0], dense, axis: 0).asType(.float32)
                }
                let (attnInput, attnPost, attnComb) = layer.attnHC.normalizedBranchInput(
                    h, norm: layer.attnNorm)
                if l == 0, !rows.isEmpty {
                    let attn = layer.attn
                    let offset = cache.local.offset
                    let length = attnInput.dim(1)
                    // Preserve the forward's full-chunk projection geometry:
                    // gathering positions first would select a different GEMM/QMV path.
                    let qResidual = attn.qNorm(deepseekV4Project(attn.wqA, attnInput))
                    let qHead = deepseekV4QueryHeadNorm(
                        deepseekV4Project(attn.wqB, qResidual)
                            .reshaped(1, length, attn.heads, attn.headDim),
                        eps: attn.normEps)
                    let qRotated = attn.rope(qHead.transposed(0, 2, 1, 3), offset: offset)
                        .transposed(0, 2, 1, 3)
                    let kvRotated = attn.rope(
                        attn.kvNorm(deepseekV4Project(attn.wkv, attnInput))
                            .reshaped(1, 1, length, attn.headDim),
                        offset: offset)
                    if start == 0, env["DSV4_ROPE_DIAGNOSTIC_OUT"] != nil {
                        let dense = MLXArray(([64] + Array(257 ... 320)).map(Int32.init))
                        ropeDiagnostic["positions"] = dense
                        ropeDiagnostic["q_head_norm.0"] = take(qHead[0], dense, axis: 0).asType(.float32)
                        ropeDiagnostic["q_post_rope.0"] = take(qRotated[0], dense, axis: 0).asType(.float32)
                        ropeDiagnostic["kv_post_rope.0"] = take(kvRotated[0, 0], dense, axis: 0).asType(.float32)
                    }
                    keep("q_residual.0", take(qResidual[0], at, axis: 0).asType(.float32))
                    keep("q_head_norm.0", take(qHead[0], at, axis: 0).asType(.float32))
                    keep("q_post_rope.0", take(qRotated[0], at, axis: 0).asType(.float32))
                    keep("kv_post_rope.0", take(kvRotated[0, 0], at, axis: 0).asType(.float32))
                }
                let attentionOutput = layer.attn(attnInput, mask: mask, cache: cache)
                if l == 0, start == 0, env["DSV4_ROPE_DIAGNOSTIC_OUT"] != nil {
                    let dense = MLXArray(([64] + Array(257 ... 320)).map(Int32.init))
                    let branch = expandedDimensions(attnPost, axis: -1)
                        * expandedDimensions(attentionOutput.asType(.float32), axis: 2)
                    let products = expandedDimensions(attnComb, axis: -1)
                        * expandedDimensions(h.asType(.float32), axis: -2)
                    ropeDiagnostic["hc_post.0"] = take(attnPost[0], dense, axis: 0)
                    ropeDiagnostic["hc_comb.0"] = take(attnComb[0], dense, axis: 0)
                    ropeDiagnostic["hc_branch.0"] = take(branch[0], dense, axis: 0)
                    ropeDiagnostic["hc_residual_sum.0"] = take(products.sum(axis: 2)[0], dense, axis: 0)
                }
                h = deepseekV4HCExpand(attentionOutput, h, attnPost, attnComb)
                if l == 0, start == 0, env["DSV4_ROPE_DIAGNOSTIC_OUT"] != nil {
                    let dense = MLXArray(([64] + Array(257 ... 320)).map(Int32.init))
                    ropeDiagnostic["attn_branch_out.0"] = take(attentionOutput[0], dense, axis: 0).asType(.float32)
                    ropeDiagnostic["h_after_attn.0"] = take(h[0], dense, axis: 0).asType(.float32)
                }
                if l == 0, !rows.isEmpty {
                    keep("attn_branch_out.0", take(attentionOutput[0], at, axis: 0).asType(.float32))
                    keep("h_after_attn.0", take(h[0], at, axis: 0).asType(.float32))
                }
                let (ffnInput, ffnPost, ffnComb) = layer.ffnHC.normalizedBranchInput(
                    h, norm: layer.ffnNorm)
                let experts = layer.ffn.gate(ffnInput, inputIDs: ids).0
                h = deepseekV4HCExpand(layer.ffn(ffnInput, inputIDs: ids), h, ffnPost, ffnComb)
                if !rows.isEmpty {
                    keep("attn_in.\(l)", take(attnInput[0], at, axis: 0))
                    keep("ffn_in.\(l)", take(ffnInput[0], at, axis: 0))
                    keep("experts.\(l)", sorted(take(experts[0], at, axis: 0).asType(.int32), axis: -1))
                    keep("h.\(l)", take(h[0], at, axis: 0).asType(.float32))
                }
            }
            eval(captured.values.flatMap { $0 } + caches.flatMap { $0.innerState() })
        }

        var out = captured.mapValues { concatenated($0, axis: 0) }
        out["positions"] = MLXArray(positions.map(Int32.init))
        let last = out["h.\(model.layers.count - 1)"]!
        out["logits"] = deepseekV4LinearF32(
            model.head, model.norm(model.hcHead(expandedDimensions(last.asType(.bfloat16), axis: 0))))[0]

        for (l, (layer, cache)) in zip(model.layers, caches).enumerated() where layer.attn.ratio > 0 {
            let layerCache = DeepseekV4LayerCache(cache, ratio: layer.attn.ratio)
            if let pooled = layerCache.pool?.pooled { out["pool.\(l)"] = pooled[0].asType(.float32) }
            guard let indexer = layer.attn.indexer, let indexPool = layerCache.indexPool?.pooled
            else { continue }
            out["index_pool.\(l)"] = indexPool[0].asType(.float32)
            let inputs = out["attn_in.\(l)"]!
            var selections = [MLXArray]()
            for (i, p) in positions.enumerated() {
                let visible = (p + 1) / layer.attn.ratio
                let k = min(indexer.topK, visible)
                guard k > 0 else {
                    selections.append(MLXArray.full([indexer.topK], values: MLXArray(Int32(-1))))
                    continue
                }
                let x = inputs[i ..< (i + 1)].reshaped(1, 1, -1)
                let qResidual = layer.attn.qNorm(layer.attn.wqA(x))
                let q = layer.attn.rope(
                    indexer.wqB(qResidual).reshaped(1, 1, indexer.heads, indexer.headDim)
                        .transposed(0, 2, 1, 3), offset: p
                ).asType(.float32)
                let w = deepseekV4LinearF32(indexer.weightsProj, x).reshaped(indexer.heads, 1, 1)
                    * (indexer.scale * pow(Float(indexer.heads), -0.5))
                let keys = indexPool[0, ..<visible].asType(.float32)
                let scores = (maximum(matmul(q[0], keys.T), 0) * w).sum(axis: 0)
                    .reshaped(-1)
                let top = sorted(argSort(-scores)[..<k].asType(.int32))
                selections.append(
                    k == indexer.topK
                        ? top
                        : concatenated([top, MLXArray.full([indexer.topK - k], values: MLXArray(Int32(-1)))]))
            }
            out["topk_exact.\(l)"] = stacked(selections)
        }
        out = out.mapValues { $0.dtype == .bfloat16 ? $0.asType(.float32) : $0 }
        eval(Array(out.values))
        if let diagnosticPath = env["DSV4_ROPE_DIAGNOSTIC_OUT"] {
            eval(Array(ropeDiagnostic.values))
            try save(arrays: ropeDiagnostic, url: URL(fileURLWithPath: diagnosticPath))
        }
        try save(arrays: out, url: URL(fileURLWithPath: outPath))
    }

    private func replaySinkhorn(
        _ mixes: MLXArray, connection: GLM5NextHyperConnection, precise: Bool
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        let hc = connection.hcMult
        let scale = connection.scale.asType(.float32)
        let base = connection.base.asType(.float32)
        let pre = sigmoid(mixes[.ellipsis, ..<hc] * scale[0] + base[..<hc])
            + connection.hcEps
        let post = 2 * sigmoid(
            mixes[.ellipsis, hc ..< (2 * hc)] * scale[1] + base[hc ..< (2 * hc)])
        let leading = Array(mixes.shape.dropLast())
        var comb = mixes[.ellipsis, (2 * hc)...].reshaped(leading + [hc, hc]) * scale[2]
            + base[(2 * hc)...].reshaped(hc, hc)
        comb = softmax(comb, axis: -1, precise: precise) + connection.hcEps
        comb = comb / (comb.sum(axis: -2, keepDims: true) + connection.hcEps)
        for _ in 0 ..< max(connection.sinkhornIterations - 1, 0) {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + connection.hcEps)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + connection.hcEps)
        }
        return (pre, post, comb)
    }
}
