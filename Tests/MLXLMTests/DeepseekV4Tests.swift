import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// DeepSeek-V4 against a port of DeepSeek's reference (`inference/model.py`
/// + `kernel.py`). The fixture (`Resources/deepseek_v4_tiny`, weights from
/// its `make_fixture.py`) is a 4-layer model with every V4 path: a
/// hash-routed layer, local, ratio-4 (indexer) and ratio-128 attention, an
/// 8-token window, and the artifact's MXFP4/MXFP8 per-layer quantization.
/// Everything runs on the CPU. The oracle (`make_oracle.py`) prefills the
/// whole prompt in one call with float32 activations, so the parity test
/// loads the fixture with its bf16 tensors widened to float32 as well.
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

    private func loadTiny(indexBlock: Int, indexKeep: Int, float32: Bool = false) throws
        -> DeepseekV4Model
    {
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
        if float32 {
            let widened = model.parameters().flattened().map { key, value in
                (key, value.dtype == .bfloat16 ? value.asType(.float32) : value)
            }
            try model.update(parameters: ModuleParameters.unflattened(widened), verify: [.all])
        }
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
            XCTAssertEqual(oracle.cases.count, 3)
            let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
            for item in oracle.cases {
                XCTAssertEqual(item.indexer, "flat")
                let (tokens, logprobs) = try greedy(
                    model, prompt: item.prompt, step: item.prefillStep, count: oracle.decode)
                let label = "\(item.indexer) prompt=\(item.prompt.count) step=\(item.prefillStep)"
                XCTAssertEqual(tokens, item.greedy, label)
                let diff = zip(logprobs, item.logprobs).map { mine, theirs in
                    zip(mine, theirs).map { abs($0 - $1) }.max() ?? 0
                }.max() ?? 0
                print("deepseek_v4 parity \(label): max |dlogprob| = \(diff)")
                // Float32 activations on both sides: equal up to reduction order
                // and the oracle's single-call prefill against this chunk plan.
                XCTAssertLessThan(diff, 1e-3, label)
            }
        }
    }

    /// The serving path (`TokenIterator`, whatever prefill step and chunking
    /// the caller asks for) prefills with the fixed 2048-token plan and
    /// decodes the oracle's tokens.
    func testTokenIteratorUsesReferencePrefillPlan() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try fixture
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: dir.appendingPathComponent("oracle.json")))
            let cases = oracle.cases.filter {
                $0.prefillStep == DeepseekV4Model.prefillStepSize
            }
            XCTAssertEqual(cases.count, 1)
            for item in cases {
                let model = try loadTiny(indexBlock: 0, indexKeep: 0, float32: true)
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

    private func prefetchedRow(_ model: DeepseekV4Model, prompt: [Int]) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters(stepSize: 16))
        guard case .tokens(let last) = result else {
            XCTFail("DeepSeek V4 prepare must leave the last token for decode")
            return cache
        }
        XCTAssertEqual(last.tokens.asArray(Int32.self), [Int32(prompt.last!)])
        for (layer, item) in zip(model.layers, cache) {
            XCTAssertEqual(
                DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local.offset, prompt.count - 1)
        }
        return cache
    }

    func testContinuousBatchGreedyParityAtOneTwoFourEightRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let lengths = [3, 6, 9, 13, 127, 128, 131, 137]
            let prompts = lengths.enumerated().map { row, count in
                (0 ..< count).map { ($0 * 7 + row * 11 + 2) % 128 }
            }
            for indexBlock in [0] {
                let model = try loadTiny(indexBlock: indexBlock, indexKeep: 2)
                let count = 16
                let serial = try prompts.map {
                    try greedy(model, prompt: $0, step: DeepseekV4Model.prefillStepSize, count: count).0
                }
                for rows in [1, 2, 4, 8] {
                    let caches = try prompts.prefix(rows).map { try prefetchedRow(model, prompt: $0) }
                    var inputs = prompts.prefix(rows).map { Int32($0.last!) }
                    var tokens = Array(repeating: [Int](), count: rows)
                    for _ in 0 ..< count {
                        let logits = model.decodeBatch(
                            MLXArray(inputs).reshaped(rows, 1), rowCaches: caches)
                        let next = argMax(logits, axis: -1).asArray(Int32.self)
                        for row in 0 ..< rows { tokens[row].append(Int(next[row])) }
                        inputs = next
                        eval(caches.flatMap { $0 })
                    }
                    XCTAssertEqual(tokens, Array(serial.prefix(rows)), "rows=\(rows) block=\(indexBlock)")
                }
            }
        }
    }

    func testContinuousBatchLongContextGreedyParityAtOneTwoFourEightRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let prompts = [31, 32, 33, 127, 128, 129, 131, 137].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 7 + row * 11 + 2) % 128 }
            }
            for indexBlock in [0] {
                let model = try loadTiny(indexBlock: indexBlock, indexKeep: 2)
                let serial = try prompts.map {
                    try greedy(model, prompt: $0, step: DeepseekV4Model.prefillStepSize, count: 4).0
                }
                for rows in [1, 2, 4, 8] {
                    let caches = try prompts.prefix(rows).map { try prefetchedRow(model, prompt: $0) }
                    var inputs = prompts.prefix(rows).map { Int32($0.last!) }
                    for step in 0 ..< 4 {
                        let next = argMax(
                            model.decodeBatch(MLXArray(inputs).reshaped(rows, 1), rowCaches: caches),
                            axis: -1).asArray(Int32.self)
                        XCTAssertEqual(
                            next.map(Int.init), serial.prefix(rows).map { $0[step] },
                            "long rows=\(rows) block=\(indexBlock) step=\(step)")
                        inputs = next
                        eval(caches.flatMap { $0 })
                    }
                }
            }
        }
    }

    func testContinuousBatchRowsJoinLeaveAndReorderWithoutChangingTokens() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            let prompts = [7, 31, 127, 130].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 3 + row * 17) % 128 }
            }
            let serial = try prompts.map {
                try greedy(model, prompt: $0, step: DeepseekV4Model.prefillStepSize, count: 16).0
            }
            var caches = [Int: [KVCache]]()
            var inputs = prompts.map { Int32($0.last!) }
            var positions = Array(repeating: 0, count: prompts.count)
            let schedule = [[0, 1], [1, 0], [1, 0, 2], [2, 1], [3, 2], [0, 3, 2], [2], [3, 0]]
            for active in schedule {
                for row in active where caches[row] == nil {
                    caches[row] = try prefetchedRow(model, prompt: prompts[row])
                }
                let next = argMax(
                    model.decodeBatch(
                        MLXArray(active.map { inputs[$0] }).reshaped(active.count, 1),
                        rowCaches: active.map { caches[$0]! }), axis: -1).asArray(Int32.self)
                for (slot, row) in active.enumerated() {
                    XCTAssertEqual(Int(next[slot]), serial[row][positions[row]], "row=\(row)")
                    inputs[row] = next[slot]
                    positions[row] += 1
                }
                for row in caches.keys {
                    for (layer, item) in zip(model.layers, caches[row]!) {
                        XCTAssertEqual(
                            DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local.offset,
                            prompts[row].count - 1 + positions[row])
                    }
                }
            }
        }
    }

    func testBatchedRoPEKeepsPerRowOffsetsAndInverse() throws {
        Device.withDefaultDevice(.cpu) {
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            let offsets = [0, 3, 127, 128]
            let x = (MLXArray.arange(4 * 2 * 64, dtype: .float32) / 97).reshaped(4, 2, 1, 64)
            for inverse in [false, true] {
                let batched = rope(x, offsets: MLXArray(offsets.map(Int32.init)), inverse: inverse)
                let serial = concatenated(offsets.enumerated().map { row, offset in
                    rope(x[row ..< (row + 1)], offset: offset, inverse: inverse)
                }, axis: 0)
                XCTAssertEqual(batched.asArray(Float.self), serial.asArray(Float.self))
            }
        }
    }

    func testBatchedCompressorMatchesIndependentRaggedWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 4, indexKeep: 2)
            for layer in model.layers where layer.attn.ratio > 0 {
                let compressor = try XCTUnwrap(layer.attn.compressor)
                let ratio = layer.attn.ratio
                let starts = [0, ratio - 2, ratio - 1, ratio * 2 - 1]
                let batchedCaches = starts.map { _ in DeepseekV4PoolCache(ratio: ratio) }
                let serialCaches = starts.map { _ in DeepseekV4PoolCache(ratio: ratio) }
                for row in starts.indices where starts[row] > 0 {
                    let prefix = sin(MLXArray.arange(starts[row] * 64, dtype: .float32) / 17)
                        .reshaped(1, starts[row], 64)
                    _ = compressor(prefix, cache: batchedCaches[row], offset: 0)
                    _ = compressor(prefix, cache: serialCaches[row], offset: 0)
                }
                for step in 0 ..< 4 {
                    let x = cos(MLXArray.arange(starts.count * 64, dtype: .float32) / Float(step + 3))
                        .reshaped(starts.count, 1, 64)
                    let actual = compressor.decodeBatch(
                        x, caches: batchedCaches, offsets: starts.map { $0 + step })
                    for row in starts.indices {
                        let expected = compressor(
                            x[row ..< (row + 1)], cache: serialCaches[row], offset: starts[row] + step)
                        XCTAssertEqual(batchedCaches[row].remainder, serialCaches[row].remainder)
                        XCTAssertEqual(actual[row].shape, expected.shape)
                        if expected.size > 0 {
                            XCTAssertLessThan(
                                abs(actual[row] - expected).max().item(Float.self), 1e-5,
                                "ratio=\(ratio) row=\(row) step=\(step)")
                        }
                    }
                }
            }
        }
    }

    /// The pooled entries of a token stream do not depend on how it is split
    /// into calls: one call, uneven chunks, one token at a time, and a
    /// `copy()` / `state` snapshot resumed mid-stream all pool the same
    /// entries (`inference/model.py:309-310, 336-338, 353-360` carry the
    /// previous ratio-4 window across calls).
    func testCompressorPoolsAreChunkingInvariant() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            for layer in model.layers where layer.attn.ratio > 0 {
                let ratio = layer.attn.ratio
                var compressors = [try XCTUnwrap(layer.attn.compressor)]
                if let indexer = layer.attn.indexer { compressors.append(indexer.compressor) }
                let total = ratio == 4 ? 4 * 13 + 3 : 2 * 128 + 5
                let x = sin(MLXArray.arange(total * 64, dtype: .float32) / 7).reshaped(1, total, 64)
                    .asType(.bfloat16)
                for compressor in compressors {
                    func pool(_ chunks: [Int], snapshotAfter: Int? = nil) -> DeepseekV4PoolCache {
                        var cache = DeepseekV4PoolCache(ratio: ratio)
                        var offset = 0
                        for (i, n) in chunks.enumerated() {
                            _ = compressor(x[0..., offset ..< (offset + n)], cache: cache, offset: offset)
                            eval(cache.innerState())
                            offset += n
                            if i == snapshotAfter {
                                let restored = DeepseekV4PoolCache(ratio: ratio)
                                restored.state = (cache.copy() as! DeepseekV4PoolCache).state
                                restored.metaState = cache.metaState
                                cache = restored
                            }
                        }
                        XCTAssertEqual(offset, total)
                        return cache
                    }
                    let whole = pool([total])
                    let expected = try XCTUnwrap(whole.pooled)
                    XCTAssertEqual(expected.dim(1), total / ratio)
                    let uneven = ratio == 4 ? [3, 6, 1, 17, 2, 26] : [100, 1, 60, 100]
                    for (label, cache) in [
                        ("uneven", pool(uneven)),
                        ("one at a time", pool(Array(repeating: 1, count: total))),
                        ("snapshot", pool(uneven, snapshotAfter: 2)),
                    ] {
                        let actual = try XCTUnwrap(cache.pooled, label)
                        XCTAssertEqual(actual.shape, expected.shape, label)
                        let diff = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                            .item(Float.self)
                        XCTAssertLessThanOrEqual(diff, 1e-5, "ratio=\(ratio) \(label)")
                        XCTAssertEqual(cache.remainder, whole.remainder, label)
                    }
                }
            }
        }
    }

    /// The default indexer selects exactly the brute-force top `index_topk`
    /// of `sum_h relu(q_h . k) * scale * w_h / sqrt(H)` over every pooled
    /// entry, serially and in a ragged batch, at pool sizes of 2048 and more
    /// (where mlx-vlm's HISA filter used to replace it).
    func testIndexerSelectsExactTopKOverLargePools() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 2, "hidden_size": 64,
                 "q_lora_rank": 32, "qk_rope_head_dim": 16, "index_n_heads": 4,
                 "index_head_dim": 32, "index_topk": 512, "compress_ratios": [4, 4]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            try config.validateModelConfiguration()
            XCTAssertEqual(config.indexBlock, 0)
            MLXRandom.seed(2111)
            let indexer = DeepseekV4Indexer(config, ratio: 4)
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            func bruteForce(x: MLXArray, qResidual: MLXArray, pool: MLXArray, offset: Int) -> Set<Int32> {
                let q = rope(
                    indexer.wqB(qResidual).reshaped(1, 1, 4, 32).transposed(0, 2, 1, 3), offset: offset)
                    .asType(.float32)
                let w = indexer.weightsProj(x).asType(.float32).reshaped(4, 1, 1) * pow(Float(4), -0.5)
                let dots = maximum(matmul(q[0], pool[0].asType(.float32).T), 0)  // [H, 1, P]
                let scores = (dots * indexer.scale * w).sum(axis: 0).reshaped(-1)
                let k = min(512, pool.dim(1))
                return Set(argSort(-scores)[..<k].asArray(Int32.self))
            }
            var pools = [MLXArray]()
            var inputs = [(MLXArray, MLXArray, Int)]()
            for count in [2048, 3001, 4096] {
                let pool = MLXRandom.normal([1, count, 32]).asType(.bfloat16)
                let x = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
                let qResidual = MLXRandom.normal([1, 1, 32]).asType(.bfloat16)
                let offset = 4 * count + 1
                let cache = DeepseekV4PoolCache(ratio: 4)
                _ = cache.accumulateWindows(
                    kv: MLXArray.zeros([1, 2, 64], dtype: .bfloat16),
                    gate: MLXArray.zeros([1, 2, 64], dtype: .bfloat16), offset: offset - 2)
                _ = cache.appendPooled(pool)
                let expected = bruteForce(x: x, qResidual: qResidual, pool: pool, offset: offset)
                XCTAssertEqual(expected.count, 512)
                let serial = try XCTUnwrap(
                    indexer(x, qResidual: qResidual, rope: rope, cache: cache, offset: offset))
                XCTAssertEqual(Set(serial.reshaped(-1).asArray(Int32.self)), expected, "serial P=\(count)")
                pools.append(pool)
                inputs.append((x, qResidual, offset))
            }
            let batched = indexer.decodeBatch(
                concatenated(inputs.map { $0.0 }, axis: 0),
                qResidual: concatenated(inputs.map { $0.1 }, axis: 0), rope: rope, pools: pools,
                offsets: MLXArray(inputs.map { Int32($0.2) }))
            for row in pools.indices {
                let expected = bruteForce(
                    x: inputs[row].0, qResidual: inputs[row].1, pool: pools[row], offset: inputs[row].2)
                XCTAssertEqual(
                    Set(batched[row].reshaped(-1).asArray(Int32.self)), expected, "batched row=\(row)")
            }
        }
    }

    /// A prompt chunk's exact selection is the same whether its float32
    /// scores fit one workspace or are tiled over queries.
    func testIndexerPrefillTilesKeepTheExactSelection() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 2, "hidden_size": 64,
                 "q_lora_rank": 32, "qk_rope_head_dim": 16, "index_n_heads": 4,
                 "index_head_dim": 32, "index_topk": 8, "compress_ratios": [4, 4]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            MLXRandom.seed(2151)
            let indexer = DeepseekV4Indexer(config, ratio: 4)
            let rope = DeepseekV4RoPE(dims: 16, base: 160000, scaling: nil)
            let (L, offset) = (40, 4 * 300)
            let pool = MLXRandom.normal([1, 300, 32]).asType(.bfloat16)
            let x = MLXRandom.normal([1, L, 64]).asType(.bfloat16)
            let qResidual = MLXRandom.normal([1, L, 32]).asType(.bfloat16)
            func run(workspace: Int) throws -> MLXArray {
                indexer.scoreWorkspaceBytes = workspace
                let cache = DeepseekV4PoolCache(ratio: 4)
                _ = cache.accumulateWindows(
                    kv: MLXArray.zeros([1, 2, 64], dtype: .bfloat16),
                    gate: MLXArray.zeros([1, 2, 64], dtype: .bfloat16), offset: offset - 2)
                _ = cache.appendPooled(pool)
                return try XCTUnwrap(
                    indexer(x, qResidual: qResidual, rope: rope, cache: cache, offset: offset))
            }
            let whole = try run(workspace: 512 << 20)
            let tiled = try run(workspace: 4 * 4 * 310 * 4 * 7)  // 7 queries per tile
            XCTAssertEqual(whole.shape, [1, L, 8])
            for t in 0 ..< L {
                XCTAssertEqual(
                    Set(whole[0, t].asArray(Int32.self)), Set(tiled[0, t].asArray(Int32.self)),
                    "query \(t)")
            }
        }
    }

    /// DeepSeek's kernel rounds the unnormalized weights against the running
    /// max of each 64-key block, then rescales the float32 accumulator when a
    /// later block raises the max. A query whose first block holds a value of
    /// 128 at score 0 and whose second block holds -47 at score 1 keeps
    /// 128 * exp(-1) - 47; rounding exp(-1) to bf16 (47/128) against the final
    /// max would cancel it to 0.
    func testAttentionRoundsWeightsPerRunningMaxBlock() throws {
        try Device.withDefaultDevice(.cpu) {
            let json = """
                {"model_type": "deepseek_v4", "num_hidden_layers": 1, "hidden_size": 64,
                 "num_attention_heads": 2, "o_groups": 2, "head_dim": 64, "qk_rope_head_dim": 16,
                 "q_lora_rank": 32, "o_lora_rank": 8, "sliding_window": 128, "compress_ratios": [0]}
                """
            let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
            let attention = DeepseekV4Attention(config, layerIdx: 0)
            // scale 1/8: q = 8 e0 gives scores k[0].
            var keys = [[Float]](repeating: [Float](repeating: 0, count: 64), count: 128)
            keys[0][1] = 128
            keys[64][0] = 1
            keys[64][1] = -47
            let k = MLXArray(keys.flatMap { $0 }).reshaped(1, 1, 128, 64)
            var query = [Float](repeating: 0, count: 64)
            query[0] = 8
            let q = broadcast(MLXArray(query).reshaped(1, 1, 1, 64), to: [1, 2, 1, 64])
            let out = attention.referenceAttention(
                q, keys: k, valid: MLXArray.ones([1, 1, 128], dtype: .bool))
            // Block 0: max 0, weights 1, accumulator 128. Block 1 raises the max to 1:
            // the accumulator rescales by e^-1 and adds -47; 128 + 1 weights and
            // the sink (e^-1) form the normalizer.
            let e = Float(Foundation.exp(-1.0))
            let expected = (128 * e - 47) / (1 + 128 * e)
            XCTAssertEqual(out[0, 0, 0, 1].item(Float.self), expected, accuracy: 1e-6)
            XCTAssertGreaterThan(out[0, 0, 0, 1].item(Float.self), 1e-3)
        }
    }

    /// The float32 linear (compressor projections, head) on the GPU: the
    /// bfloat16-weight kernel equals a float32 matmul over the widened weight
    /// up to summation order, and a row's result does not depend on how many
    /// rows share the call.
    func testFloat32LinearKernelMatchesWidenedWeight() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        MLXRandom.seed(272)
        for (K, N) in [(64, 129), (4096, 1000)] {
            let weight = MLXRandom.normal([N, K]).asType(.bfloat16)
            let layer = Linear(weight: weight)
            let x = MLXRandom.normal([8, 1, K]).asType(.bfloat16)
            let expected = matmul(x.asType(.float32), weight.asType(.float32).T)
            let full = deepseekV4LinearF32(layer, x)
            XCTAssertEqual(full.dtype, .float32)
            XCTAssertEqual(full.shape, [8, 1, N])
            let scale = abs(expected).max().item(Float.self)
            XCTAssertLessThan(
                abs(full - expected).max().item(Float.self) / scale, 1e-5, "K=\(K) N=\(N)")
            for rows in 1 ... 7 {
                let part = deepseekV4LinearF32(layer, x[..<rows])
                XCTAssertEqual(
                    part.asArray(Float.self), full[..<rows].asArray(Float.self), "rows=\(rows)")
            }
        }
    }

    func testContinuousBatchUsesKeysOnlyWindowsAndIndependentPoolSnapshots() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadTiny(indexBlock: 0, indexKeep: 0)
            XCTAssertTrue(model.batchesExactly)
            XCTAssertFalse(try loadTiny(indexBlock: 4, indexKeep: 2).batchesExactly)
            let prompts = [1, 3, 31, 32, 33, 127, 128, 129].enumerated().map { row, count in
                (0 ..< count).map { ($0 * 5 + row * 7) % 128 }
            }
            let serial = try prompts.map {
                try greedy(model, prompt: $0, step: DeepseekV4Model.prefillStepSize, count: 4).0
            }
            let caches = try prompts.map { try prefetchedRow(model, prompt: $0) }
            var inputs = prompts.map { Int32($0.last!) }
            for step in 0 ..< 4 {
                let next = argMax(
                    model.decodeBatch(MLXArray(inputs).reshaped(8, 1), rowCaches: caches), axis: -1)
                    .asArray(Int32.self)
                XCTAssertEqual(next.map(Int.init), serial.map { $0[step] })
                for row in caches {
                    for (layer, item) in zip(model.layers, row) {
                        let local = DeepseekV4LayerCache(item, ratio: layer.attn.ratio).local
                        XCTAssertEqual(local.state.count, 2)
                        XCTAssertEqual(local.state[1].dim(-1), 0)
                    }
                }
                inputs = next
            }

            let cache = DeepseekV4PoolCache(ratio: 4)
            _ = cache.accumulateWindows(
                kv: MLXArray.zeros([1, 1, 1]), gate: MLXArray.zeros([1, 1, 1]), offset: 0)
            _ = cache.appendPooled(MLXArray([Float(2), 3]).reshaped(1, 2, 1))
            let snapshot = cache.copy() as! DeepseekV4PoolCache
            _ = cache.appendPooled(MLXArray([Float(4)]).reshaped(1, 1, 1))
            XCTAssertEqual(snapshot.pooled!.asArray(Float.self), [2, 3])
            XCTAssertEqual(cache.pooled!.asArray(Float.self), [2, 3, 4])
            XCTAssertEqual(cache.state.last!.dim(1), 3)
            let restored = DeepseekV4PoolCache(ratio: 4)
            restored.state = cache.state
            restored.metaState = cache.metaState
            _ = restored.appendPooled(MLXArray([Float(5)]).reshaped(1, 1, 1))
            XCTAssertEqual(restored.pooled!.asArray(Float.self), [2, 3, 4, 5])
            XCTAssertEqual(cache.pooled!.asArray(Float.self), [2, 3, 4])
            _ = restored.appendPooled(MLXArray(6 ..< 25).asType(.float32).reshaped(1, 19, 1))
            XCTAssertEqual(restored.pooledCount, 23)
            XCTAssertEqual(restored.pooled!.asArray(Float.self), (2 ..< 25).map(Float.init))
            XCTAssertEqual(restored.state.last!.dim(1), 23)
            XCTAssertEqual(snapshot.pooled!.asArray(Float.self), [2, 3])
        }
    }

    func testPoolCacheIncompleteMultiTokenContinuationPreservesBufferedPrefix() throws {
        try Device.withDefaultDevice(.cpu) {
            for ratio in [4, 128] {
                let cache = DeepseekV4PoolCache(ratio: ratio)
                let values = MLXArray(0 ..< ratio).asType(.float32).reshaped(1, ratio, 1)
                let gates = values + 1000
                _ = cache.accumulateWindows(
                    kv: values[0..., ..<1], gate: gates[0..., ..<1], offset: 0)
                var ready = cache.accumulateWindows(
                    kv: values[0..., 1 ..< 3], gate: gates[0..., 1 ..< 3], offset: 1)
                XCTAssertEqual(ready.kv.dim(1), 0)
                XCTAssertEqual(cache.remainder, 3)
                ready = cache.accumulateWindows(
                    kv: values[0..., 3...], gate: gates[0..., 3...], offset: 3)
                XCTAssertEqual(ready.kv.asArray(Float.self), (0 ..< ratio).map(Float.init))
                XCTAssertEqual(ready.gate.asArray(Float.self), (0 ..< ratio).map { Float($0 + 1000) })
                XCTAssertEqual(ready.base, 0)
                XCTAssertEqual(cache.remainder, 0)
            }
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
        // The artifact sets neither HISA key: exact flat top-k, as DeepSeek's reference.
        XCTAssertEqual(config.indexBlock, 0)
        XCTAssertEqual(config.indexKeep, 0)

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
            "layers.0.ffn.gate.weight": MLXArray.zeros([2, 64], dtype: .bfloat16),
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
                "layers.0.ffn.gate.tid2eid", "layers.0.ffn.gate.weight",
                "layers.0.ffn.shared_experts.w3.weight",
                "hc_head.fn", "embed.weight",
            ])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]!.shape, [2, 3, 5])
        XCTAssertEqual(out["layers.0.ffn.switch_mlp.gate_proj.weight"]![1].sum().item(Float.self), 15)
        XCTAssertEqual(out["layers.0.attn.wo_a.weight"]!.shape, [2, 4, 6])
        XCTAssertEqual(out["layers.0.ffn.gate.tid2eid"]!.dtype, .int32)
        XCTAssertEqual(out["layers.0.ffn.gate.weight"]!.dtype, .float32)
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
