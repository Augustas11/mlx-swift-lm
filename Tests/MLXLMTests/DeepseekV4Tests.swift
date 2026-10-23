import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// DeepSeek-V4 against the mlx-vlm 0.7.10 reference. The fixture
/// (`Resources/deepseek_v4_tiny`, made by its `make_fixture.py`) is a 4-layer
/// model with every V4 path: a hash-routed layer, local, ratio-4 (indexer,
/// HISA and flat selection) and ratio-128 attention, an 8-token window, and
/// the artifact's MXFP4/MXFP8 per-layer quantization. Everything runs on the
/// CPU; the oracle is mlx-vlm's `generate_step` on the CPU, so the expected
/// tokens and log-probabilities are those of the same op graph.
final class DeepseekV4Tests: XCTestCase {

    private struct Oracle: Decodable {
        struct Case: Decodable {
            let indexer: String
            let indexBlock: Int
            let indexKeep: Int
            let prefillStep: Int
            let prompt: [Int]
            let greedy: [Int]
            let logprobs: [[Float]]

            enum CodingKeys: String, CodingKey {
                case indexer, prompt, greedy, logprobs
                case indexBlock = "index_block"
                case indexKeep = "index_keep"
                case prefillStep = "prefill_step"
            }
        }
        let decode: Int
        let cases: [Case]
    }

    private var fixture: URL {
        get throws {
            try XCTUnwrap(Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        }
    }

    private func loadTiny(indexBlock: Int, indexKeep: Int) throws -> DeepseekV4Model {
        let dir = try fixture
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        config.indexBlock = indexBlock
        config.indexKeep = indexKeep
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        return model
    }

    /// mlx-vlm `generate_step`: prompt chunks of `min(step, remaining - 1)`
    /// while more than one token is left, then the last token, then one
    /// token per step. Returns greedy tokens and per-step log-softmax.
    private func greedy(
        _ model: DeepseekV4Model, prompt: [Int], step: Int, count: Int
    ) throws -> ([Int], [[Float]]) {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            let chunk = MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n)
            eval(model(chunk, cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        var input = MLXArray(rest.map(Int32.init)).reshaped(1, 1)
        var tokens = [Int]()
        var logprobs = [[Float]]()
        for _ in 0 ..< count {
            let logits = model(input, cache: cache)[0, -1]
            let lp = (logits - logSumExp(logits, axis: -1, keepDims: true)).asType(.float32)
            let token = argMax(lp, axis: -1).item(Int.self)
            tokens.append(token)
            logprobs.append(lp.asArray(Float.self))
            input = MLXArray([Int32(token)]).reshaped(1, 1)
        }
        return (tokens, logprobs)
    }

    func testTinyCheckpointMatchesReferenceGreedyDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try fixture
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: dir.appendingPathComponent("oracle.json")))
            XCTAssertEqual(oracle.cases.count, 6)
            var models = [String: DeepseekV4Model]()
            for item in oracle.cases {
                let model =
                    try models[item.indexer]
                    ?? loadTiny(indexBlock: item.indexBlock, indexKeep: item.indexKeep)
                models[item.indexer] = model
                let (tokens, logprobs) = try greedy(
                    model, prompt: item.prompt, step: item.prefillStep, count: oracle.decode)
                let label = "\(item.indexer) prompt=\(item.prompt.count) step=\(item.prefillStep)"
                XCTAssertEqual(tokens, item.greedy, label)
                let diff = zip(logprobs, item.logprobs).map { mine, theirs in
                    zip(mine, theirs).map { abs($0 - $1) }.max() ?? 0
                }.max() ?? 0
                print("deepseek_v4 parity \(label): max |dlogprob| = \(diff)")
                // Same MLX core and the same CPU op graph: equal up to reduction order.
                XCTAssertLessThan(diff, 1e-3, label)
            }
        }
    }

    /// The serving path (`TokenIterator`, whatever prefill step and chunking
    /// the caller asks for) prefills with the reference plan, so it decodes
    /// the step-2048 oracle cases exactly.
    func testTokenIteratorUsesReferencePrefillPlan() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try fixture
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: dir.appendingPathComponent("oracle.json")))
            let cases = oracle.cases.filter {
                $0.prefillStep == DeepseekV4Model.prefillStepSize
            }
            XCTAssertEqual(cases.count, 2)
            for item in cases {
                let model = try loadTiny(indexBlock: item.indexBlock, indexKeep: item.indexKeep)
                for chunking in [PrefillParameters.Chunking.remainder, .balanced, .unchunked] {
                    let parameters = GenerateParameters(
                        maxTokens: oracle.decode, temperature: 0,
                        prefill: PrefillParameters(stepSize: 16, chunking: chunking))
                    var iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(item.prompt.map(Int32.init))),
                        model: model, parameters: parameters)
                    var tokens = [Int]()
                    while let token = iterator.next() { tokens.append(token) }
                    XCTAssertEqual(tokens, item.greedy, "\(item.indexer) \(chunking)")
                }
            }
        }
    }

    func testPoolCacheCompletesWindowsAcrossChunksAndDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = DeepseekV4PoolCache(ratio: 4)
            let rows = MLXArray(0 ..< 11).asType(.float32).reshaped(1, 11, 1)
            // A 6-token chunk completes one window and buffers two rows.
            var ready = cache.accumulateWindows(
                kv: rows[0..., 0 ..< 6], gate: rows[0..., 0 ..< 6], offset: 0)
            XCTAssertEqual(ready.kv.asArray(Float.self), [0, 1, 2, 3])
            XCTAssertEqual(ready.base, 0)
            XCTAssertEqual(cache.remainder, 2)
            // A 3-token chunk joins the buffered rows: one window from 4.
            ready = cache.accumulateWindows(
                kv: rows[0..., 6 ..< 9], gate: rows[0..., 6 ..< 9], offset: 6)
            XCTAssertEqual(ready.kv.asArray(Float.self), [4, 5, 6, 7])
            XCTAssertEqual(ready.base, 4)
            XCTAssertEqual(cache.remainder, 1)
            _ = cache.appendPooled(MLXArray.zeros([1, 2, 1]))
            // Decode steps complete the next window on its fourth token.
            for t in 9 ..< 11 {
                ready = cache.accumulateWindows(
                    kv: rows[0..., t ..< (t + 1)], gate: rows[0..., t ..< (t + 1)], offset: t)
                XCTAssertEqual(ready.kv.dim(1), 0)
            }
            ready = cache.accumulateWindows(
                kv: MLXArray([Float(11)]).reshaped(1, 1, 1),
                gate: MLXArray([Float(11)]).reshaped(1, 1, 1), offset: 11)
            XCTAssertEqual(ready.kv.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(ready.base, 8)

            // Two pooled entries: visible to query t iff index < (t + 1) / 4.
            let mask = try XCTUnwrap(cache.poolMask(queries: 6, offset: 3))
            XCTAssertEqual(
                mask.asArray(Bool.self),
                [true, false, true, false, true, false, true, false, true, true, true, true])
            XCTAssertNil(cache.poolMask(queries: 1, offset: 12))

            let copy = cache.copy() as! DeepseekV4PoolCache
            XCTAssertEqual(copy.remainder, cache.remainder)
            XCTAssertEqual(copy.pooledCount, 2)
        }
    }

    func testArtifactConfigurationTakesOneRatioPerLayer() throws {
        let ratios = [0, 0] + (0 ..< 20).flatMap { _ in [4, 128] } + [4, 0, 0, 0]
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 43,
             "compress_ratios": \(ratios), "num_hash_layers": 3,
             "rope_scaling": {"type": "yarn", "factor": 16,
               "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.compressRatios.count, 43)
        XCTAssertEqual(config.compressRatios.filter { $0 == 4 }.count, 21)
        XCTAssertEqual(config.compressRatios.filter { $0 == 128 }.count, 20)
        XCTAssertEqual(config.compressRatios.last, 4)
        // The reference enables HISA by default; the artifact sets neither key.
        XCTAssertEqual(config.indexBlock, 64)
        XCTAssertEqual(config.indexKeep, 16)

        var bad = config
        bad.compressRatios[5] = 2
        XCTAssertThrowsError(try bad.validateModelConfiguration())
    }

    func testSanitizeMapsArtifactLayout() throws {
        let json = """
            {"model_type": "deepseek_v4", "num_hidden_layers": 1, "hidden_size": 64,
             "n_routed_experts": 2, "o_groups": 2, "o_lora_rank": 4, "compress_ratios": [0]}
            """
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
        let weights: [String: MLXArray] = [
            "layers.0.ffn.experts.0.w1.weight": MLXArray.zeros([3, 5]),
            "layers.0.ffn.experts.1.w1.weight": MLXArray.ones([3, 5]),
            "layers.0.ffn.experts.0.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.ffn.experts.1.w2.scales": MLXArray.zeros([3, 2]),
            "layers.0.attn.wo_a.weight": MLXArray.zeros([8, 6]),
            "layers.0.hc_attn_fn": MLXArray.zeros([2]),
            "layers.0.hc_ffn_scale": MLXArray.zeros([3]),
            "layers.0.ffn.gate.tid2eid": MLXArray([Int64(1), 0]),
            "layers.0.ffn.shared_experts.w3.weight": MLXArray.zeros([2]),
            "layers.1.attn.wkv.weight": MLXArray.zeros([2]),
            "mtp.0.block.attn.wkv.weight": MLXArray.zeros([2]),
            "hc_head_fn": MLXArray.zeros([2]),
            "embed.weight": MLXArray.zeros([2]),
        ]
        let out = DeepseekV4Weights.sanitize(weights, configuration: config)
        XCTAssertEqual(
            Set(out.keys),
            [
                "layers.0.ffn.switch_mlp.gate_proj.weight", "layers.0.ffn.switch_mlp.down_proj.scales",
                "layers.0.attn.wo_a.weight", "layers.0.attn_hc.fn", "layers.0.ffn_hc.scale",
                "layers.0.ffn.gate.tid2eid", "layers.0.ffn.shared_experts.w3.weight",
                "hc_head.fn", "embed.weight",
            ])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]!.shape, [2, 3, 5])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]![1].sum().item(Float.self), 15)
        XCTAssertEqual(out["layers.0.attn.wo_a.weight"]!.shape, [2, 4, 6])
        XCTAssertEqual(out["layers.0.ffn.gate.tid2eid"]!.dtype, .int32)
    }
    /// The real artifact's tensor names, shapes and dtypes (from the
    /// safetensors headers) against the module tree after sanitize and the
    /// config's per-layer quantization. Placeholders are lazy zeros, so no
    /// weight is read or materialized. Set `DSV4_ARTIFACT_DIR` to the
    /// V4-Flash-0731 MXFP4 directory.
    func testArtifactLayoutMatchesModuleTree() throws {
        guard let path = ProcessInfo.processInfo.environment["DSV4_ARTIFACT_DIR"] else {
            throw XCTSkip("DSV4_ARTIFACT_DIR is not set")
        }
        try Device.withDefaultDevice(.cpu) {
            let dir = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let dtypes: [String: DType] = [
                "BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32, "I64": .int64,
                "I32": .int32,
            ]
            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(atPath: path)
                .filter { $0.hasSuffix(".safetensors") }
            XCTAssertFalse(files.isEmpty)
            for file in files {
                let handle = try FileHandle(forReadingFrom: dir.appendingPathComponent(file))
                defer { try? handle.close() }
                let size = try XCTUnwrap(handle.read(upToCount: 8)).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt64.self)
                }
                let header = try XCTUnwrap(handle.read(upToCount: Int(size)))
                let entries = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: header) as? [String: Any])
                for (name, value) in entries where name != "__metadata__" {
                    let entry = try XCTUnwrap(value as? [String: Any])
                    let shape = try XCTUnwrap(entry["shape"] as? [Int])
                    let dtype = try XCTUnwrap(dtypes[try XCTUnwrap(entry["dtype"] as? String)])
                    weights[name] = MLXArray.zeros(shape, dtype: dtype)
                }
            }
            let model = DeepseekV4Model(config)
            let sanitized = model.sanitize(weights: weights)
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? base.perLayerQuantization?.quantization(layer: path)?.asTuple : nil
            }
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            let modes = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    key, module -> (String, QuantizationMode)? in
                    switch module {
                    case let q as QuantizedLinear: (key, q.mode)
                    case let q as QuantizedSwitchLinear: (key, q.mode)
                    case let q as QuantizedMultiLinear: (key, q.mode)
                    default: nil
                    }
                })
            XCTAssertEqual(modes["layers.2.ffn.switch_mlp.gate_proj"], .mxfp4)
            XCTAssertEqual(modes["layers.2.ffn.shared_experts.w1"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.wo_a"], .mxfp8)
            XCTAssertEqual(modes["layers.2.attn.indexer.wq_b"], .mxfp8)
            XCTAssertNil(modes["layers.2.attn.compressor.wkv"])
            XCTAssertNil(modes["head"])
        }
    }
}
