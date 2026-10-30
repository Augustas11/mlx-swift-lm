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
    /// Head-only diagnostic: named extracted tensors, never a full model load.
    func testFinalHeadSameInputDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let configPath = env["DSV4_HEAD_CONFIG"], let weightPath = env["DSV4_HEAD_WEIGHTS"],
            let referencePath = env["DSV4_HEAD_REFERENCE"], let pythonPath = env["DSV4_HEAD_PYTHON_STAGES"],
            let outputPath = env["DSV4_HEAD_OUT"]
        else { throw XCTSkip("Head same-input diagnostic environment is not set") }
        try Device.withDefaultDevice(.gpu) {
            let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            guard !FileManager.default.fileExists(atPath: output.path) else {
                XCTFail("Head diagnostic output exists; refusing overwrite")
                return
            }
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self,
                from: Data(contentsOf: URL(fileURLWithPath: configPath)))
            let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightPath))
            XCTAssertEqual(Set(weights.keys), Set([
                "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"]))
            let reference = try MLX.loadArrays(url: URL(fileURLWithPath: referencePath))
            let python = try MLX.loadArrays(url: URL(fileURLWithPath: pythonPath))
            let hidden = try XCTUnwrap(reference["h.42"]).asType(.bfloat16).reshaped(1, 158, 4, 4096)
            let frozenLogits = try XCTUnwrap(reference["logits"])
            XCTAssertEqual(config.hiddenSize, 4096)
            XCTAssertEqual(config.hcMult, 4)
            XCTAssertEqual(frozenLogits.shape, [158, 129280])
            let hc = DeepseekV4HyperHead(config)
            try hc.update(parameters: ModuleParameters.unflattened([
                "fn": try XCTUnwrap(weights["hc_head_fn"]),
                "base": try XCTUnwrap(weights["hc_head_base"]),
                "scale": try XCTUnwrap(weights["hc_head_scale"])]), verify: [.all])
            let norm = DeepseekV4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            norm.update(parameters: ModuleParameters.unflattened([
                "weight": try XCTUnwrap(weights["norm.weight"])]))
            let head = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            head.update(parameters: ModuleParameters.unflattened([
                "weight": try XCTUnwrap(weights["head.weight"])]))
            hc.train(false)
            norm.train(false)
            head.train(false)
            var saved = [String: MLXArray]()
            var metrics = [String: Any]()
            func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
                let a = actual.asType(.float32)
                let b = expected.asType(.float32)
                eval(a, b)
                let count = (a .!= b).asType(.int32).sum().item(Int.self)
                let maximum = abs(a - b).max().item(Float.self)
                metrics[name] = ["shape": a.shape, "difference_count": count, "max_abs": maximum]
                print("HEAD \(name) count=\(count) maxabs=\(maximum)")
            }
            func topFive(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
                let a = argSort(-actual, axis: -1)[0..., ..<5]
                let b = argSort(-expected, axis: -1)[0..., ..<5]
                let matches = (sorted(a, axis: -1) .== sorted(b, axis: -1))
                    .asType(.int32).sum(axis: -1) .== Int32(5)
                eval(a, b, matches)
                metrics[name] = ["compared": actual.dim(0),
                    "matching": matches.asType(.int32).sum().item(Int.self),
                    "actual_ids": a.asArray(Int.self), "reference_ids": b.asArray(Int.self)]
            }
            let collapsed = hc(hidden)
            let normalized = norm(collapsed)
            let logits158 = deepseekV4LinearF32(head, normalized)[0]
            saved["swift.collapsed_bf16"] = collapsed[0]
            saved["swift.final_norm_bf16"] = normalized[0]
            saved["swift.head_f32.M158"] = logits158
            for (name, actual, key) in [
                ("collapse.vs_python_old", collapsed[0], "all158.old.collapsed_bf16"),
                ("norm.vs_python_old", normalized[0], "all158.old.final_norm_bf16"),
                ("logits158.vs_python_old", logits158, "all158.old.head_f32"),
                ("collapse.vs_python_reference", collapsed[0], "all158.reference.collapsed_bf16"),
                ("norm.vs_python_reference", normalized[0], "all158.reference.final_norm_bf16")
            ] {
                compare(name, actual, try XCTUnwrap(python[key]))
            }
            compare("logits158.vs_frozen", logits158, frozenLogits)
            topFive("top5.M158.vs_frozen", logits158, frozenLogits)

            // Hold the actual normalized row fixed; only projection geometry changes.
            let last = normalized[0..., 157..<158]
            let custom1 = deepseekV4LinearF32(head, last)[0]
            let generic1 = matmul(last.asType(.float32), head.weight.asType(.float32).T)[0]
            var rowIndices = (0..<8192).map { Int32($0 % 158) }
            rowIndices[8191] = 157
            let indices = MLXArray(rowIndices)
            let denseInput = take(normalized[0], indices, axis: 0).reshaped(1, 8192, 4096)
            XCTAssertTrue(denseInput[0, 8191].asType(.float32).asArray(Float.self)
                == last[0, 0].asType(.float32).asArray(Float.self))
            let dense8192 = deepseekV4LinearF32(head, denseInput)
            eval(dense8192) // Materialize M8192 before selecting the only saved row.
            let denseLast = dense8192[0, 8191..<8192]
            saved["swift.head_f32.customM1"] = custom1
            saved["swift.head_f32.genericM1"] = generic1
            saved["swift.head_f32.denseM8192.last"] = denseLast
            for (name, actual) in [("customM1", custom1), ("genericM1", generic1), ("denseM8192.last", denseLast)] {
                compare(name + ".vs_frozen", actual, frozenLogits[157..<158])
                compare(name + ".vs_python_old_M158", actual, try XCTUnwrap(python["all158.old.head_f32"])[157..<158])
                topFive("top5." + name + ".vs_frozen", actual, frozenLogits[157..<158])
            }
            compare("customM1.vs_genericM1", custom1, generic1)
            compare("denseM8192.vs_genericM1", denseLast, generic1)
            eval(Array(saved.values))
            try save(arrays: saved, url: output.appendingPathComponent("swift-head-stages.safetensors"))
            let data = try JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: output.appendingPathComponent("metrics.json"), options: [.withoutOverwriting])
        }
    }

    /// Frozen small-query attention input, shared by prefill-tail and decode diagnostics.
    func testLayer3TailSameInputDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let inputPath = env["DSV4_TAIL_INPUT"], let configPath = env["DSV4_HEAD_CONFIG"],
            let outputPath = env["DSV4_TAIL_OUT"]
        else { throw XCTSkip("Layer3 tail diagnostic environment is not set") }
        try Device.withDefaultDevice(.gpu) {
            let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            guard !FileManager.default.fileExists(atPath: output.path) else {
                XCTFail("Tail output exists; refusing overwrite")
                return
            }
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self,
                from: Data(contentsOf: URL(fileURLWithPath: configPath)))
            let values = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
            let q = try XCTUnwrap(values["q"])
            let keys = try XCTUnwrap(values["keys"])
            let valid = try XCTUnwrap(values["valid"])
            let expected = try XCTUnwrap(values["reference_attention_tail"])
            let sink = try XCTUnwrap(values["sink"])
            XCTAssertEqual(q.shape, [1, config.attentionHeads, q.dim(2), config.headDim])
            XCTAssertTrue((1...8).contains(q.dim(2)))
            XCTAssertEqual(keys.shape, [1, q.dim(2), keys.dim(2), config.headDim])
            XCTAssertEqual(valid.shape, [1, q.dim(2), keys.dim(2)])
            let attention = DeepseekV4Attention(config, layerIdx: 3)
            attention.update(parameters: ModuleParameters.unflattened(["attn_sink": sink]))
            let fused = try XCTUnwrap(deepseekV4SparseAttention(
                q, keys: keys, valid: valid, sink: sink, scale: attention.scale))
            let graph = attention.graphAttention(q, keys: keys, valid: valid)
            let fusedBF16 = fused.asType(q.dtype)
            let graphBF16 = graph.asType(q.dtype)
            if env["DSV4_REQUIRE_EXACT_ATTENTION"] == "1" {
                eval(fusedBF16, expected)
                XCTAssertTrue(fusedBF16.asType(.float32).asArray(Float.self)
                    == expected.asType(.float32).asArray(Float.self), "Canonical attention BF16 differs from frozen reference")
                let expectedF32 = try XCTUnwrap(values["reference_attention_f32"])
                eval(fused, expectedF32)
                XCTAssertTrue(fused.asArray(Float.self) == expectedF32.asArray(Float.self),
                    "Canonical attention F32 differs from frozen reference")
            }
            var metrics = [String: Any]()
            var comparisons = [
                ("fused_vs_graph_f32", fused, graph),
                ("fused_vs_graph_bf16", fusedBF16, graphBF16),
                ("fused_vs_reference_bf16", fusedBF16, expected),
                ("graph_vs_reference_bf16", graphBF16, expected)
            ]
            if let expectedF32 = values["reference_attention_f32"] {
                comparisons.append(("fused_vs_reference_f32", fused, expectedF32))
                comparisons.append(("graph_vs_reference_f32", graph, expectedF32))
            }
            var diagnostics = [String: MLXArray]()
            var precise: MLXArray?
            if env["DSV4_SPARSE_PRECISE_EXP"] == "1", let kernelPath = env["DSV4_SPARSE_KERNEL_SOURCE"] {
                let sourceFile = try String(contentsOfFile: kernelPath, encoding: .utf8)
                let source = try XCTUnwrap(sourceFile.components(separatedBy: "source: \"\"\"").dropFirst().first)
                    .components(separatedBy: "\"\"\"")[0]
                    .replacingOccurrences(of: "exp(", with: "metal::precise::exp(")
                let kernel = MLXFast.metalKernel(name: "dsv4_diagnostic_precise_sparse_attention",
                    inputNames: ["q", "keys", "valid", "sink", "scale"], outputNames: ["out"], source: source)
                let (B, H, L, D, K) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3), keys.dim(2))
                let result = kernel([q, keys, valid, sink.asType(.float32), MLXArray(attention.scale)],
                    template: [("H", H), ("L", L), ("D", D), ("K", K)],
                    grid: (H * L * 256, B, 1), threadGroup: (256, 1, 1),
                    outputShapes: [q.shape], outputDTypes: [.float32])[0]
                precise = result
                comparisons.append(("precise_vs_reference_bf16", result.asType(q.dtype), expected))
                comparisons.append(("precise_vs_fused_f32", result, fused))
                if let reference = values["reference_attention_f32"] {
                    comparisons.append(("precise_vs_reference_f32", result, reference))
                }
            }
            if env["DSV4_SPARSE_INSTRUMENT"] == "1", let kernelPath = env["DSV4_SPARSE_KERNEL_SOURCE"] {
                let file = try String(contentsOfFile: kernelPath, encoding: .utf8)
                let original = try XCTUnwrap(file.components(separatedBy: "source: \"\"\"" ).dropFirst().first)
                    .components(separatedBy: "\"\"\"" )[0]
                let (B, H, L, D, K) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3), keys.dim(2))
                let blocks = (K + 63) / 64
                let scoreShape = [B, H, L, K]
                let stateShape = [B, H, L, blocks]
                var instrumented = original.replacingOccurrences(
                    of: "if (lane == 0) { scores[j] = present ? dot : -INFINITY; }",
                    with: """
                        if (lane == 0) {
                            scores[j] = present ? dot : -INFINITY;
                            if (k < K) { debug_scores[((size_t(b) * H + h) * L + l) * K + k] = scores[j]; }
                        }
                        """)
                instrumented = instrumented.replacingOccurrences(
                    of: "state[0] = peak;",
                    with: """
                        const size_t debug_row = ((size_t(b) * H + h) * L + l);
                        debug_running[debug_row * ((K + 63) / 64) + start / 64] = peak;
                        debug_norm[debug_row * ((K + 63) / 64) + start / 64] = norm;
                        for (uint z = 0; z < 64 && start + z < K; ++z) {
                            debug_weights[debug_row * K + start + z] = weights[z];
                        }
                        state[0] = peak;
                        """)
                XCTAssertNotEqual(instrumented, original)
                let instrument = MLXFast.metalKernel(name: "dsv4_diagnostic_instrument_sparse",
                    inputNames: ["q", "keys", "valid", "sink", "scale"],
                    outputNames: ["out", "debug_scores", "debug_weights", "debug_running", "debug_norm"],
                    source: instrumented)
                let outputs = instrument([q, keys, valid, sink.asType(.float32), MLXArray(attention.scale)],
                    template: [("H", H), ("L", L), ("D", D), ("K", K)],
                    grid: (H * L * 256, B, 1), threadGroup: (256, 1, 1),
                    outputShapes: [q.shape, scoreShape, scoreShape, stateShape, stateShape],
                    outputDTypes: Array(repeating: .float32, count: 5))
                eval(outputs)
                XCTAssertTrue(outputs[0].asArray(Float.self) == fused.asArray(Float.self),
                    "Instrumentation perturbed fused output; stage conclusions are invalid")
                comparisons.append(("instrumented_vs_fused_f32", outputs[0], fused))
                let expectedScores = try XCTUnwrap(values["scores"])
                let expectedWeights = try XCTUnwrap(values["weights"]).asType(.bfloat16).asType(.float32)
                    .reshaped(scoreShape)
                comparisons.append(("scores_vs_reference_f32", outputs[1], expectedScores))
                comparisons.append(("weights_vs_reference_bf16", outputs[2], expectedWeights))
                comparisons.append(("running_vs_reference_f32", outputs[3], try XCTUnwrap(values["running"])))
                diagnostics["instrumented_f32"] = outputs[0]
                diagnostics["scores"] = outputs[1]
                diagnostics["weights"] = outputs[2]
                diagnostics["running"] = outputs[3]
                diagnostics["online_norm"] = outputs[4]
                let overrideSource = original.replacingOccurrences(of: "dot *= scale;",
                    with: "dot = k < K ? reference_scores[((size_t(b) * H + h) * L + l) * K + k] : -INFINITY;")
                let overrideKernel = MLXFast.metalKernel(name: "dsv4_diagnostic_reference_scores_sparse",
                    inputNames: ["q", "keys", "valid", "sink", "scale", "reference_scores"],
                    outputNames: ["out"], source: overrideSource)
                let override = overrideKernel([q, keys, valid, sink.asType(.float32), MLXArray(attention.scale), expectedScores],
                    template: [("H", H), ("L", L), ("D", D), ("K", K)],
                    grid: (H * L * 256, B, 1), threadGroup: (256, 1, 1),
                    outputShapes: [q.shape], outputDTypes: [.float32])[0]
                comparisons.append(("reference_scores_vs_reference_bf16", override.asType(q.dtype), expected))
                comparisons.append(("reference_scores_vs_reference_f32", override,
                                    try XCTUnwrap(values["reference_attention_f32"])))
                diagnostics["reference_scores_output_f32"] = override
            }
            for (name, a, b) in comparisons {
                eval(a, b)
                let count = (a .!= b).asType(.int32).sum().item(Int.self)
                let maximum = MLX.where(a .== b, MLXArray(Float(0)),
                    abs(a.asType(.float32) - b.asType(.float32))).max().item(Float.self)
                metrics[name] = ["shape": a.shape, "difference_count": count, "max_abs": maximum]
                print("TAIL \(name) count=\(count) maxabs=\(maximum)")
            }
            var saved = ["fused_f32": fused, "graph_f32": graph,
                         "fused_bf16": fusedBF16, "graph_bf16": graphBF16,
                         "reference_bf16": expected]
            saved["precise_f32"] = precise
            saved.merge(diagnostics) { _, new in new }
            try save(arrays: saved, url: output.appendingPathComponent("tail-stages.safetensors"))
            try JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("metrics.json"), options: [.withoutOverwriting])
        }
    }

    /// Real prefill followed by three scalar decode calls, with persistent caches.
    func testActualDecodeLayerDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let oraclePath = env["DSV4_DECODE_ORACLE"],
            let outputPath = env["DSV4_DECODE_OUT"]
        else { throw XCTSkip("Actual decode diagnostic environment is not set") }
        try Device.withDefaultDevice(.gpu) {
            let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            guard !FileManager.default.fileExists(atPath: output.path) else {
                XCTFail("Decode diagnostic output exists; refusing overwrite")
                return
            }
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            struct Oracle: Decodable { let prompt: [Int]; let greedy: [Int] }
            let oracle = try JSONDecoder().decode(Oracle.self,
                from: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
            XCTAssertEqual(oracle.prompt.count, 1024)
            let steps = try XCTUnwrap(Int(env["DSV4_DECODE_STEPS"] ?? "4"))
            guard (1...16).contains(steps), oracle.greedy.count >= steps - 1 else {
                XCTFail("Decode diagnostic requires 1...16 assessed steps and sufficient oracle tokens")
                return
            }
            let dir = URL(fileURLWithPath: artifact)
            let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let model = DeepseekV4Model(config)
            try loadWeights(modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
            model.train(false)
            let caches = try model.newCache(parameters: nil)
            var metadata = [[String: Any]]()
            for step in 0..<steps {
                let tokens = step == 0 ? oracle.prompt : [oracle.greedy[step - 1]]
                let ids = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
                let positions = step == 0
                    ? Array(Set(Array(stride(from: 0, to: 1024, by: 64)) + Array(896..<1024))).sorted()
                    : [0]
                let at = MLXArray(positions.map(Int32.init))
                let layerCaches = zip(caches, model.layers).map {
                    DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
                }
                var mask: MLXArray?
                if tokens.count > 1,
                    case .array(let array) = layerCaches[0].local.makeMask(
                        n: tokens.count, windowSize: config.slidingWindow, returnArray: true) {
                    mask = array
                }
                var h = repeated(expandedDimensions(model.embed(ids), axis: 2), count: config.hcMult, axis: 2)
                var captured = [String: MLXArray]()
                func keep(_ name: String, _ tensor: MLXArray) {
                    captured[name] = take(tensor[0], at, axis: 0).asType(.float32)
                }
                for (index, item) in zip(model.layers, layerCaches).enumerated() {
                    let (layer, cache) = item
                    let residual = h
                    let (attnInput, post, comb) = layer.attnHC.normalizedBranchInput(h, norm: layer.attnNorm)
                    var reconstructed: MLXArray?
                    var leaf = [String: MLXArray]()
                    if step == 1, index == 2, env["DSV4_ACTUAL_QK_OUT"] != nil {
                        let attention = layer.attn
                        let copy = DeepseekV4LayerCache(caches[index].copy(), ratio: attention.ratio)
                        let offset = copy.local.offset
                        let qA = deepseekV4Project(attention.wqA, attnInput)
                        let qr = attention.qNorm(qA)
                        let qB = deepseekV4Project(attention.wqB, qr)
                            .reshaped(1, 1, attention.heads, attention.headDim)
                        let qNormalized = deepseekV4QueryHeadNorm(qB, eps: attention.normEps)
                        let q = attention.rope(qNormalized.transposed(0, 2, 1, 3), offset: offset)
                        let kvProjection = deepseekV4Project(attention.wkv, attnInput)
                        let kvNormalized = attention.kvNorm(kvProjection).reshaped(1, 1, 1, attention.headDim)
                        let kv = attention.rope(kvNormalized, offset: offset)
                        let updated = copy.local.update(keys: kv,
                            values: MLXArray.zeros([1, 1, 1, 0], dtype: kv.dtype)).0
                        let local = DeepseekV4Attention.temporal(updated, cache: copy.local)
                        let (window, windowValid) = attention.windowList(local[0, 0],
                            first: offset + 1 - local.dim(2), positions: [offset])
                        let pool = try XCTUnwrap(attention.compressor)(attnInput,
                            cache: try XCTUnwrap(copy.pool), offset: offset)
                        let selected = try XCTUnwrap(try XCTUnwrap(attention.indexer)(attnInput,
                            qResidual: qr, rope: attention.rope,
                            cache: try XCTUnwrap(copy.indexPool), offset: offset))
                        let selectedKeys = take(pool[0], maximum(selected[0], Int32(0)).flattened(), axis: 0)
                            .reshaped(1, selected.dim(2), attention.headDim)
                        let keys = expandedDimensions(concatenated([window, selectedKeys], axis: 1), axis: 0)
                        let selectedValid = selected[0] .< Int32((offset + 1) / attention.ratio)
                        let valid = expandedDimensions(concatenated([windowValid, selectedValid], axis: 1), axis: 0)
                        let raw = attention.referenceAttention(q, keys: keys, valid: valid).asType(q.dtype)
                        let inverse = attention.rope(raw, offset: offset, inverse: true)
                        let grouped = inverse.reshaped(1, attention.groups, attention.heads / attention.groups,
                            1, attention.headDim).transposed(0, 1, 3, 2, 4)
                            .reshaped(1, attention.groups, 1, attention.heads / attention.groups * attention.headDim)
                        let woA = deepseekV4MultiLinear(attention.woA, grouped)
                        reconstructed = deepseekV4Project(attention.woB, woA.transposed(0, 2, 1, 3).reshaped(1, 1, -1))
                        leaf = ["attn_input": attnInput, "q_a": qA, "q_residual": qr, "q_b": qB,
                                "q_normalized": qNormalized, "q": q, "kv_projection": kvProjection,
                                "kv_normalized": kvNormalized, "kv": kv, "window": window,
                                "selected_indices": selected, "keys": keys, "valid": valid,
                                "raw_attention": raw, "inverse": inverse, "wo_a": woA,
                                "reconstructed_branch": reconstructed!, "pool": pool,
                                "wo_b_input": woA.transposed(0, 2, 1, 3).reshaped(1, 1, -1)]
                    }
                    let branch = layer.attn(attnInput, mask: mask, cache: cache)
                    if let reconstructed {
                        eval(branch, reconstructed)
                        XCTAssertTrue(branch.asType(.float32).asArray(Float.self)
                            == reconstructed.asType(.float32).asArray(Float.self),
                            "Reconstructed layer2 attention differs from actual serving call")
                        leaf["actual_branch"] = branch
                        let frozen = try MLX.loadArrays(url: URL(fileURLWithPath: try XCTUnwrap(env["DSV4_ACTUAL_QK_REFERENCE"])))
                        var leafMetrics = [String: Any]()
                        for name in ["q", "keys", "valid"] {
                            let actual = try XCTUnwrap(leaf[name]).asType(.float32)
                            let expected = try XCTUnwrap(frozen[name]).asType(.float32)
                            eval(actual, expected)
                            leafMetrics[name] = ["actual_shape": actual.shape, "reference_shape": expected.shape,
                                "difference_count": (actual .!= expected).asType(.int32).sum().item(Int.self),
                                "max_abs": abs(actual - expected).max().item(Float.self)]
                        }
                        let leafPath = URL(fileURLWithPath: try XCTUnwrap(env["DSV4_ACTUAL_QK_OUT"]), isDirectory: true)
                        try FileManager.default.createDirectory(at: leafPath, withIntermediateDirectories: false)
                        let widened = leaf.mapValues { $0.asType(.float32) }
                        eval(Array(widened.values))
                        try save(arrays: widened, url: leafPath.appendingPathComponent("actual-qk.safetensors"))
                        try JSONSerialization.data(withJSONObject: leafMetrics, options: [.prettyPrinted, .sortedKeys])
                            .write(to: leafPath.appendingPathComponent("metrics.json"), options: [.withoutOverwriting])
                    }
                    h = deepseekV4HCExpand(branch, residual, post, comb)
                    keep("attn_in.\(index)", attnInput)
                    keep("attn_post.\(index)", post)
                    keep("attn_comb.\(index)", comb)
                    keep("attn_out.\(index)", branch)
                    keep("after_attn.\(index)", h)
                    let (ffnInput, ffnPost, ffnComb) = layer.ffnHC.normalizedBranchInput(h, norm: layer.ffnNorm)
                    let experts = layer.ffn.gate(ffnInput, inputIDs: ids).0
                    let ffnBranch = layer.ffn(ffnInput, inputIDs: ids)
                    h = deepseekV4HCExpand(ffnBranch, h, ffnPost, ffnComb)
                    keep("ffn_in.\(index)", ffnInput)
                    keep("ffn_post.\(index)", ffnPost)
                    keep("ffn_comb.\(index)", ffnComb)
                    keep("ffn_out.\(index)", ffnBranch)
                    keep("h.\(index)", h)
                    captured["experts.\(index)"] = sorted(take(experts[0], at, axis: 0).asType(.int32), axis: -1)
                    if let pool = cache.pool?.pooled { captured["pool.\(index)"] = pool[0].asType(.float32) }
                    if let pool = cache.indexPool?.pooled { captured["index_pool.\(index)"] = pool[0].asType(.float32) }
                }
                let collapsed = model.hcHead(h)
                let normalized = model.norm(collapsed)
                let allLogits = deepseekV4LinearF32(model.head, normalized)
                eval(allLogits) // Preserve actual M1024/M1 head dispatch before row selection.
                captured["logits"] = allLogits[0, -1, 0...]
                captured["head_collapsed"] = collapsed[0, -1, 0...].asType(.float32)
                captured["head_normalized"] = normalized[0, -1, 0...].asType(.float32)
                captured["positions"] = MLXArray(positions.map { Int32(step == 0 ? $0 : 1024 + step - 1) })
                eval(Array(captured.values) + caches.flatMap { $0.innerState() })
                try save(arrays: captured, url: output.appendingPathComponent("step-\(step).safetensors"))
                metadata.append(["assessed_step": step, "fed_tokens": tokens,
                                 "cache_meta_state": caches.map { $0.metaState }])
                print("DECODE TRACE step=\(step) done")
            }
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("metadata.json"), options: [.withoutOverwriting])
        }
    }

    func testHyperConnectionSameInputDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let inputPath = env["DSV4_HC_INPUT"], let configPath = env["DSV4_HC_CONFIG"],
            let outputPath = env["DSV4_HC_OUT"]
        else { throw XCTSkip("HC same-input diagnostic environment is not set") }
        try Device.withDefaultDevice(.gpu) {
            let data = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self,
                from: Data(contentsOf: URL(fileURLWithPath: configPath)))
            let h = try XCTUnwrap(data["h"])
            let branch = try XCTUnwrap(data["branch"])
            XCTAssertEqual(h.shape, [1, 1, 4, 4096])
            let hc = GLM5NextHyperConnection(hcMult: config.hcMult,
                sinkhornIterations: config.hcSinkhornIters, hcEps: config.hcEps,
                normEps: config.rmsNormEps, hiddenSize: config.hiddenSize,
                normWeightInFloat32: true, normalizeMixAfterProjection: true)
            try hc.update(parameters: ModuleParameters.unflattened([
                "fn": try XCTUnwrap(data["fn"]), "base": try XCTUnwrap(data["base"]),
                "scale": try XCTUnwrap(data["scale"])]), verify: [.all])
            let norm = DeepseekV4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            norm.update(parameters: ModuleParameters.unflattened([
                "weight": try XCTUnwrap(data["norm_weight"])]))
            hc.train(false)
            norm.train(false)
            let fusedMix = try XCTUnwrap(glm5NextHCMix(h, fn: hc.fn, eps: hc.normEps,
                normalizeMixAfterProjection: true))
            let flat = h.asType(.float32).reshaped(1, 1, -1)
            let inverse = rsqrt(square(flat).mean(axis: -1, keepDims: true) + hc.normEps)
            let projection = matmul(flat, hc.fn.T)
            let mix = projection * inverse
            let (productionNormalized, productionPost, productionComb) = hc.normalizedBranchInput(h, norm: norm)
            let (collapsed, graphPost, graphComb) = hc(h)
            let graphNormalized = norm(collapsed)
            let graphExpanded = deepseekV4HCExpand(branch, h, graphPost, graphComb)
            let productionExpanded = deepseekV4HCExpand(branch, h, productionPost, productionComb)
            let (replayPre, replayPost, replayComb) = replaySinkhorn(mix, connection: hc, precise: true)
            var saved = [String: MLXArray]()
            var metrics = [String: Any]()
            for (name, a, key) in [
                ("graph_inverse", inverse, "inverse"), ("graph_projection", projection, "projection"),
                ("graph_mix", mix, "mix"), ("fused_mix", fusedMix, "mix"),
                ("graph_collapsed", collapsed, "collapsed"),
                ("graph_normalized", graphNormalized, "normalized"),
                ("graph_post", graphPost, "post"), ("graph_comb", graphComb, "comb"),
                ("graph_expanded", graphExpanded, "expanded"),
                ("production_normalized", productionNormalized, "normalized"),
                ("production_post", productionPost, "post"), ("production_comb", productionComb, "comb"),
                ("production_expanded", productionExpanded, "expanded"),
                ("uncompiled_pre", replayPre, "pre"), ("uncompiled_post", replayPost, "post"),
                ("uncompiled_comb", replayComb, "comb")
            ] {
                let a = a.asType(.float32), b = try XCTUnwrap(data[key]).asType(.float32)
                eval(a, b)
                let delta = a - b
                metrics[name] = ["count": (a .!= b).asType(.int32).sum().item(Int.self),
                    "max_abs": abs(delta).max().item(Float.self),
                    "rel_l2": (sqrt(square(delta).sum()) / sqrt(square(b).sum())).item(Float.self)]
                saved[name] = a
                if name.hasPrefix("production_") {
                    XCTAssertTrue(a.asArray(Float.self) == b.asArray(Float.self),
                        "Canonical production HC differs from frozen reference: \(name)")
                }
            }
            // Compare each ordinary decode row with the identical batched/packed input.
            hc.fusedDecode = false
            let varied = concatenated([h, -h, h * Float(2), h * Float(0.5),
                h * Float(0.25), -h * Float(0.5), h * Float(0.125), -h * Float(2)], axis: 0)
            for (geometry, inputs) in [("rows", varied), ("positions", varied.transposed(1, 0, 2, 3))] {
                let axis = geometry == "rows" ? 0 : 1
                let widths = geometry == "rows" ? [1, 2, 4, 8] : Array(1...8)
                for width in widths {
                    let batch = axis == 0 ? inputs[..<width] : inputs[0..., ..<width]
                    let (normalized, post, comb) = hc.normalizedBranchInput(batch, norm: norm)
                    let expanded = deepseekV4HCExpand(
                        broadcast(branch, to: Array(batch.shape.prefix(2)) + [config.hiddenSize]),
                        batch, post, comb)
                    var singles = [[MLXArray]](repeating: [], count: 4)
                    for index in 0..<width {
                        let row = axis == 0 ? batch[index..<index+1] : batch[0..., index..<index+1]
                        let (n, p, c) = hc.normalizedBranchInput(row, norm: norm)
                        let e = deepseekV4HCExpand(branch, row, p, c)
                        for (part, tensor) in [n, p, c, e].enumerated() { singles[part].append(tensor) }
                    }
                    for (part, tensor) in [normalized, post, comb, expanded].enumerated() {
                        let scalar = concatenated(singles[part], axis: axis).asType(.float32)
                        let actual = tensor.asType(.float32)
                        eval(actual, scalar)
                        let names = ["normalized", "post", "comb", "expanded"]
                        XCTAssertTrue(actual.asArray(Float.self) == scalar.asArray(Float.self),
                            "Canonical HC row/position differs: \(geometry) \(width) \(names[part])")
                        metrics["production_\(geometry)_\(width)_\(names[part])"] = [
                            "count": (actual .!= scalar).asType(.int32).sum().item(Int.self),
                            "max_abs": abs(actual - scalar).max().item(Float.self)]
                    }
                }
            }
            let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            guard !FileManager.default.fileExists(atPath: output.path) else {
                XCTFail("HC diagnostic output exists; refusing overwrite")
                return
            }
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            try save(arrays: saved, url: output.appendingPathComponent("hc-stages.safetensors"))
            try JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("metrics.json"), options: [.withoutOverwriting])
        }
    }

    func testGroupedOutputProjectionSameInputDiagnostic() throws {
        let env = ProcessInfo.processInfo.environment
        guard let stagePath = env["DSV4_PROJECTION_STAGES"], let weightPath = env["DSV4_PROJECTION_WEIGHTS"],
            let actualPath = env["DSV4_PROJECTION_ACTUAL"], let outputPath = env["DSV4_PROJECTION_OUT"]
        else { throw XCTSkip("Grouped projection diagnostic environment is not set") }
        try Device.withDefaultDevice(.gpu) {
            let stages = try MLX.loadArrays(url: URL(fileURLWithPath: stagePath))
            let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightPath))
            let actual = try MLX.loadArrays(url: URL(fileURLWithPath: actualPath))
            let x = try XCTUnwrap(stages["input_grouped"])
            let dense = try XCTUnwrap(stages["dense_weight"])
            let weight = try XCTUnwrap(weights["layers.2.attn.wo_a.weight"]).reshaped(8, 1024, -1)
            let scales = try XCTUnwrap(weights["layers.2.attn.wo_a.scales"]).reshaped(8, 1024, -1)
            XCTAssertEqual(x.shape, [1, 8, 1, 4096])
            XCTAssertEqual(x.dtype, .bfloat16)
            let quantized = QuantizedMultiLinear(weight: weight, scales: scales, biases: nil,
                groupSize: 32, bits: 8, mode: .mxfp8)
            let qBF16 = quantized(x)
            let qF32 = quantized(x.asType(.float32))
            let dBF16 = matmul(x, dense.swappedAxes(-1, -2))
            let dF32 = matmul(x.asType(.float32), dense.asType(.float32).swappedAxes(-1, -2))
            let swiftDense = dequantized(weight, scales: scales, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8)
                .asType(.bfloat16)
            eval(swiftDense, dense)
            XCTAssertTrue(swiftDense.asType(.float32).asArray(Float.self)
                == dense.asType(.float32).asArray(Float.self), "Swift dequantized weight differs from reference")
            // Both CB rows and packed target positions occupy the outer dimension.
            let varied = concatenated([x, -x, x * Float(2), x * Float(0.5),
                x * Float(0.25), -x * Float(0.5), x * Float(0.125), -x * Float(2)], axis: 0)
            let full = deepseekV4MultiLinear(quantized, varied)
            var rowChecks = 0
            for rows in [1, 2, 4, 8] {
                let prefix = deepseekV4MultiLinear(quantized, varied[..<rows])
                eval(prefix, full)
                XCTAssertTrue(prefix.asType(.float32).asArray(Float.self)
                    == full[..<rows].asType(.float32).asArray(Float.self), "Dense grouped prefix rows=\(rows)")
                rowChecks += 1
            }
            for row in 0..<8 {
                let scalar = deepseekV4MultiLinear(quantized, varied[row..<row+1])
                eval(scalar, full)
                XCTAssertTrue(scalar.asType(.float32).asArray(Float.self)
                    == full[row..<row+1].asType(.float32).asArray(Float.self), "Dense grouped row/packed position=\(row)")
                rowChecks += 1
            }
            MLXRandom.seed(2151)
            let fixture = QuantizedMultiLinear(weight: MLXRandom.normal([8, 1024, 4096]).asType(.bfloat16),
                groupSize: 32, bits: 8, mode: .mxfp8)
            let fixtureX = MLXRandom.normal([1, 8, 1, 4096]).asType(.bfloat16)
            let oldFixture = fixture(fixtureX)
            let fixtureWeight = dequantized(fixture.weight, scales: fixture.scales, biases: nil,
                groupSize: 32, bits: 8, mode: .mxfp8).asType(.bfloat16)
            let denseFixture = matmul(fixtureX, fixtureWeight.swappedAxes(-1, -2))
            eval(oldFixture, denseFixture)
            var metrics: [String: Any] = ["row_and_packed_exact_checks": rowChecks,
                "swift_dequantized_vs_reference_count": 0,
                "synthetic_seed": 2151,
                "synthetic_old_quantized_vs_dense_bf16_count":
                    (oldFixture .!= denseFixture).asType(.int32).sum().item(Int.self)]
            for (name, a, b) in [
                ("production_helper_vs_reference_bf16", deepseekV4MultiLinear(quantized, x),
                 try XCTUnwrap(stages["reference_wo_a"])),
                ("quantized_vs_actual_bf16", qBF16, try XCTUnwrap(actual["wo_a"])),
                ("dense_vs_reference_bf16", dBF16, try XCTUnwrap(stages["reference_wo_a"])),
                ("dense_vs_reference_f32", dF32, try XCTUnwrap(stages["reference_wo_a_f32"])),
                ("quantized_vs_dense_bf16", qBF16, dBF16),
                ("quantized_vs_dense_f32", qF32, dF32)
            ] {
                let a = a.asType(.float32), b = b.asType(.float32)
                eval(a, b)
                let delta = a - b
                metrics[name] = ["count": (a .!= b).asType(.int32).sum().item(Int.self),
                    "max_abs": abs(delta).max().item(Float.self),
                    "rel_l2": (sqrt((delta * delta).sum()) / sqrt((b * b).sum())).item(Float.self)]
            }
            XCTAssertTrue(qBF16.asType(.float32).asArray(Float.self)
                == actual["wo_a"]!.asArray(Float.self), "Projection probe differs from actual branch")
            XCTAssertTrue(dBF16.asType(.float32).asArray(Float.self)
                == stages["reference_wo_a"]!.asArray(Float.self), "Dense projection differs from frozen reference")
            XCTAssertTrue(deepseekV4MultiLinear(quantized, x).asType(.float32).asArray(Float.self)
                == stages["reference_wo_a"]!.asArray(Float.self), "Production projection differs from frozen reference")
            let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            try save(arrays: ["quantized_bf16": qBF16, "quantized_f32": qF32,
                              "dense_bf16": dBF16, "dense_f32": dF32],
                     url: output.appendingPathComponent("projection-stages.safetensors"))
            try JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("metrics.json"), options: [.withoutOverwriting])
        }
    }

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
