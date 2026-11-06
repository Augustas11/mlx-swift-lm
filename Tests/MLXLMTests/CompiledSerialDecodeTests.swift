// Copyright © 2026 Apple Inc.
//
// Compiled generic serial decode (issue #406, fork port of
// ml-explore/mlx-swift-lm#550): `TokenIterator` with
// `GenerateParameters.compiledDecode` must produce the same greedy tokens and
// leave the same `KVCacheSimple` state as eager decode, across prompt lengths,
// capacity growth, cache trim/reuse and failure/retrace paths.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CompiledSerialDecodeTests: XCTestCase {

    private let hiddenLayers = 2

    /// A tiny deterministic Qwen2. `hiddenSize` 64 gives a 32-wide head, which
    /// KV quantization needs.
    private func makeQwen2(hiddenSize: Int = 16) throws -> Qwen2Model {
        MLXRandom.seed(7)
        let configuration = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: Data(
                """
                {
                    "hidden_size": \(hiddenSize), "num_hidden_layers": \(hiddenLayers),
                    "intermediate_size": 32, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
                    "vocab_size": 64, "tie_word_embeddings": true
                }
                """.utf8))
        let model = Qwen2Model(configuration)
        eval(model)
        return model
    }

    private func makeLlama() -> LlamaModel {
        MLXRandom.seed(11)
        let model = LlamaModel(
            LlamaConfiguration(
                hiddenSize: 16, hiddenLayers: hiddenLayers, intermediateSize: 32,
                attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 64, kvHeads: 1))
        eval(model)
        return model
    }

    private func prompt(length: Int, salt: Int = 0) -> [Int32] {
        (0 ..< length).map { Int32(($0 * 29 + salt * 13 + 3) % 64) }
    }

    private struct Run {
        var tokens: [Int]
        var compiledSteps: Int
        var compiledBlocks: Int?
    }

    /// Drive a `TokenIterator` for `count` tokens at temperature 0.
    private func generate(
        _ model: any LanguageModel, prompt: [Int32], cache: [KVCache],
        maxTokens: Int?, count: Int, compiled: Bool,
        configure: (inout GenerateParameters) -> Void = { _ in }
    ) throws -> Run {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.compiledDecode = compiled
        configure(&parameters)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
            parameters: parameters)
        var tokens: [Int] = []
        while tokens.count < count, let token = iterator.next() {
            tokens.append(token)
        }
        return Run(
            tokens: tokens, compiledSteps: iterator.compiledDecodeStepCount,
            compiledBlocks: iterator.compiledDecoder?.compiledStepCount)
    }

    private func assertSameCacheState(
        _ eager: [KVCache], _ compiled: [KVCache], _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(eager.count, compiled.count, label, file: file, line: line)
        for (layer, (lhs, rhs)) in zip(eager, compiled).enumerated() {
            XCTAssertTrue(
                type(of: rhs) == KVCacheSimple.self,
                "\(label): layer \(layer) must stay a KVCacheSimple", file: file, line: line)
            XCTAssertEqual(lhs.offset, rhs.offset, "\(label): layer \(layer) offset",
                file: file, line: line)
            let (a, b) = (lhs.state, rhs.state)
            XCTAssertEqual(a.count, b.count, label, file: file, line: line)
            for (index, (x, y)) in zip(a, b).enumerated() {
                XCTAssertEqual(
                    x.shape, y.shape, "\(label): layer \(layer) state \(index) shape",
                    file: file, line: line)
                XCTAssertTrue(
                    allClose(x, y, rtol: 1e-4, atol: 1e-4).item(Bool.self),
                    "\(label): layer \(layer) state \(index) diverged", file: file, line: line)
            }
        }
    }

    // MARK: - Parity across prompt lengths

    private func assertParityAcrossPromptLengths(
        _ model: any LanguageModel, family: String
    ) throws {
        // 1: no prefill beyond the first forward; 255/300: adoption of a
        // KVCacheSimple buffer at and past one 256-row step; 600: a buffer the
        // default 512-token prefill chunking wrote.
        for length in [1, 2, 7, 33, 255, 300, 600] {
            let tokens = prompt(length: length, salt: length)
            let eagerCache = try model.newCache(parameters: nil)
            let compiledCache = try model.newCache(parameters: nil)
            let eager = try generate(
                model, prompt: tokens, cache: eagerCache, maxTokens: 24, count: 24,
                compiled: false)
            let compiled = try generate(
                model, prompt: tokens, cache: compiledCache, maxTokens: 24, count: 24,
                compiled: true)

            XCTAssertEqual(eager.compiledSteps, 0)
            XCTAssertEqual(
                compiled.compiledSteps, 23,
                "\(family) prompt \(length): every decode step after the first compiles")
            XCTAssertEqual(
                eager.tokens, compiled.tokens, "\(family) prompt \(length): greedy tokens diverged")
            assertSameCacheState(eagerCache, compiledCache, "\(family) prompt \(length)")
        }
    }

    func testQwen2IteratorCompiledMatchesEagerAcrossPromptLengths() throws {
        try assertParityAcrossPromptLengths(makeQwen2(), family: "Qwen2")
    }

    func testLlamaIteratorCompiledMatchesEagerAcrossPromptLengths() throws {
        try assertParityAcrossPromptLengths(makeLlama(), family: "Llama")
    }

    func testCompiledDecodeIsOffByDefault() throws {
        XCTAssertFalse(GenerateParameters().compiledDecode)
        let model = try makeQwen2()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt(length: 5))), model: model,
            parameters: GenerateParameters(maxTokens: 4, temperature: 0))
        while iterator.next() != nil {}
        XCTAssertEqual(iterator.compiledDecodeStepCount, 0)
        XCTAssertNil(iterator.compiledDecoder)
    }

    // MARK: - Capacity growth (retrace)

    /// Unbounded generation crosses a 256-row capacity block: the decoder
    /// re-adopts the layers into a larger cache and traces a new step. The new
    /// trace must continue exactly where the old one stopped.
    func testCapacityGrowthRetracesWithoutDivergence() throws {
        let model = try makeQwen2()
        // After prefill and the first eager step the offset is length + 1; each
        // length leaves fewer than 39 rows in its first 256-row block.
        for length in [230, 250, 480, 500] {
            let tokens = prompt(length: length, salt: 3)
            let eagerCache = try model.newCache(parameters: nil)
            let compiledCache = try model.newCache(parameters: nil)
            let eager = try generate(
                model, prompt: tokens, cache: eagerCache, maxTokens: nil, count: 40,
                compiled: false)
            let compiled = try generate(
                model, prompt: tokens, cache: compiledCache, maxTokens: nil, count: 40,
                compiled: true)

            XCTAssertEqual(eager.tokens, compiled.tokens, "prompt \(length): growth diverged")
            XCTAssertGreaterThanOrEqual(
                compiled.compiledBlocks ?? 0, 2, "prompt \(length): expected a capacity retrace")
            assertSameCacheState(eagerCache, compiledCache, "growth prompt \(length)")
        }
    }

    func testCapacityBlocksStayAboveOffsetAndRespectRemainingTokens() {
        let step = SerialCompiledDecoder.capacityStep
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: 0, remainingTokens: nil), step)
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: step - 1, remainingTokens: nil), step)
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: step, remainingTokens: nil), 2 * step)
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: 10, remainingTokens: 4), 14)
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: 10, remainingTokens: 10_000), step)
        XCTAssertEqual(SerialCompiledDecoder.capacity(after: 10, remainingTokens: 0), step)
    }

    // MARK: - Conversation-cache reuse

    /// Turn 1 decodes compiled into the caller's cache, the cache is trimmed
    /// (as a conversation-cache LCP hit trims), and turn 2 continues on it.
    /// Both turns must match an eager run doing the same, and the cache must
    /// round-trip through the prompt-cache ABI as plain `KVCacheSimple`.
    func testCompiledDecodeSurvivesTrimAndReuse() throws {
        let models: [any LanguageModel] = [try makeQwen2(), makeLlama()]
        for model in models {
            let eagerCache = try model.newCache(parameters: nil)
            let compiledCache = try model.newCache(parameters: nil)

            let first = prompt(length: 40, salt: 1)
            let eager1 = try generate(
                model, prompt: first, cache: eagerCache, maxTokens: 20, count: 20,
                compiled: false)
            let compiled1 = try generate(
                model, prompt: first, cache: compiledCache, maxTokens: 20, count: 20,
                compiled: true)
            XCTAssertEqual(eager1.tokens, compiled1.tokens, "turn 1 diverged")
            assertSameCacheState(eagerCache, compiledCache, "after turn 1")

            for (eager, compiled) in zip(eagerCache, compiledCache) {
                XCTAssertEqual(eager.trim(7), 7)
                XCTAssertEqual(compiled.trim(7), 7)
            }

            let second = prompt(length: 9, salt: 2)
            let eager2 = try generate(
                model, prompt: second, cache: eagerCache, maxTokens: 30, count: 30,
                compiled: false)
            let compiled2 = try generate(
                model, prompt: second, cache: compiledCache, maxTokens: 30, count: 30,
                compiled: true)
            XCTAssertEqual(eager2.tokens, compiled2.tokens, "turn 2 after trim diverged")
            XCTAssertGreaterThan(compiled2.compiledSteps, 0)
            assertSameCacheState(eagerCache, compiledCache, "after turn 2")

            // Disk-cache ABI: the compiled cache serializes as KVCacheSimple
            // ("KVCache") with exactly the written prefix.
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: compiledCache)
            let (restored, _) = try loadPromptCache(url: url)
            assertSameCacheState(eagerCache, restored, "restored from disk")
        }
    }

    /// A cache restored through `state` (as the cold tier and the disk cache
    /// restore one) decodes compiled exactly like eager.
    func testCompiledDecodeOnStateRestoredCache() throws {
        let model = try makeQwen2()
        let warm = try model.newCache(parameters: nil)
        _ = try generate(
            model, prompt: prompt(length: 30), cache: warm, maxTokens: 1, count: 1,
            compiled: false)
        eval(warm)

        func restoredCopy() -> [KVCache] {
            warm.map { layer in
                let simple = KVCacheSimple()
                simple.state = layer.state.map { $0[.ellipsis] }
                return simple
            }
        }
        let eagerCache = restoredCopy()
        let compiledCache = restoredCopy()
        let next = prompt(length: 3, salt: 9)
        let eager = try generate(
            model, prompt: next, cache: eagerCache, maxTokens: 16, count: 16, compiled: false)
        let compiled = try generate(
            model, prompt: next, cache: compiledCache, maxTokens: 16, count: 16, compiled: true)
        XCTAssertEqual(eager.tokens, compiled.tokens)
        XCTAssertEqual(compiled.compiledSteps, 15)
        assertSameCacheState(eagerCache, compiledCache, "state-restored cache")
    }

    /// A layer trimmed while the decoder holds it is detected; the decoder no
    /// longer owns the layers and a fresh one adopts them at the new offset.
    func testDecoderDetectsExternalTrimAndReadopts() throws {
        let model = try makeQwen2()
        let cache = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt(length: 12)).reshaped(1, -1), cache: cache))

        let decoder = try XCTUnwrap(
            SerialCompiledDecoder.make(model: model, cache: cache, remainingTokens: nil))
        eval(decoder.step(MLXArray([Int32(5)]).reshaped(1, 1), remainingTokens: nil))
        XCTAssertTrue(decoder.owns(cache))
        XCTAssertEqual(cache.map(\.offset), [13, 13])

        cache.forEach { $0.trim(2) }
        XCTAssertFalse(decoder.owns(cache), "an external trim must invalidate the decoder")
        let readopted = try XCTUnwrap(
            SerialCompiledDecoder.make(model: model, cache: cache, remainingTokens: nil))
        XCTAssertEqual(readopted.processedTokenCount, 11)
    }

    // MARK: - Eligibility

    func testIneligibleCachesStayEagerAndMatch() throws {
        let model = try makeQwen2(hiddenSize: 64)
        let tokens = prompt(length: 20)

        // A RotatingKVCache (maxKVSize) and a dynamically quantized cache
        // (kvBits) are not compiled; the flag must be inert for them.
        let configurations: [(String, (inout GenerateParameters) -> Void)] = [
            ("maxKVSize", { $0.maxKVSize = 128 }),
            ("kvBits", { $0.kvBits = 8; $0.kvGroupSize = 32 }),
        ]
        for (label, configure) in configurations {
            var parameters = GenerateParameters(maxTokens: 12, temperature: 0)
            configure(&parameters)
            let eager = try generate(
                model, prompt: tokens, cache: try model.newCache(parameters: parameters),
                maxTokens: 12, count: 12, compiled: false, configure: configure)
            let flagged = try generate(
                model, prompt: tokens, cache: try model.newCache(parameters: parameters),
                maxTokens: 12, count: 12, compiled: true, configure: configure)
            XCTAssertEqual(flagged.compiledSteps, 0, "\(label) must not compile")
            XCTAssertEqual(eager.tokens, flagged.tokens, "\(label) diverged")
        }

        // A subclass of KVCacheSimple keeps its own update semantics.
        let chunked: [KVCache] = (0 ..< hiddenLayers).map { _ in ChunkedKVCache(chunkSize: 16) }
        let run = try generate(
            model, prompt: tokens, cache: chunked, maxTokens: 6, count: 6, compiled: true)
        XCTAssertEqual(run.compiledSteps, 0)
    }

    func testDecoderRejectsMismatchedTopology() throws {
        let model = try makeQwen2()
        let full = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt(length: 4)).reshaped(1, -1), cache: full))
        XCTAssertNil(
            SerialCompiledDecoder.make(
                model: model, cache: Array(full.prefix(1)), remainingTokens: nil))
        XCTAssertNil(
            SerialCompiledDecoder.make(
                model: model, cache: try model.newCache(parameters: nil), remainingTokens: nil),
            "an empty cache has nothing to adopt")
    }

    // MARK: - Failure and retrace do not poison the compile cache

    /// A rejected step (bad shape, exhausted capacity) is refused on the host
    /// before dispatch: later steps on the same session still match eager.
    func testSessionFailureDoesNotPoisonLaterSteps() throws {
        let model = try makeQwen2()
        let promptArray = MLXArray(prompt(length: 6)).reshaped(1, -1)
        let referenceCache = try model.newCache(parameters: nil)
        let referencePrefill = model(promptArray, cache: referenceCache)
        let session = try CompiledDecodeSession(model: model, prompt: promptArray, capacity: 16)
        eval(referencePrefill, session.prefillLogits)

        var token = Int(argMax(referencePrefill[0..., -1, 0...]).item(Int32.self))
        for step in 0 ..< 8 {
            if step == 3 {
                XCTAssertThrowsError(try session.step(MLXArray([Int32(1), 2]).reshaped(1, 2)))
                XCTAssertThrowsError(try session.step(MLXArray([Int32(1)])))
            }
            let input = MLXArray([Int32(token)]).reshaped(1, 1)
            let reference = model(input, cache: referenceCache)
            let compiled = try session.step(input)
            eval(reference, compiled)
            XCTAssertTrue(
                allClose(reference, compiled, rtol: 1e-4, atol: 1e-4).item(Bool.self),
                "step \(step) diverged after a rejected step")
            token = Int(argMax(reference[0..., -1, 0...]).item(Int32.self))
        }
        XCTAssertEqual(session.processedTokenCount, 14)
    }

    /// Compiled functions are erased from the compile cache when released, and
    /// a new one can be allocated at the same address. Creating and dropping
    /// many sessions with different capacities and prompts, interleaved with
    /// eager and compiled iterators on the same model, must never replay a
    /// stale trace.
    func testReleasedSessionsNeverLeakTracesIntoNewOnes() throws {
        let model = try makeQwen2()
        for round in 0 ..< 6 {
            let length = 3 + round * 5
            let promptArray = MLXArray(prompt(length: length, salt: round)).reshaped(1, -1)
            let capacity = length + 4 + round
            let referenceCache = try model.newCache(parameters: nil)
            let referencePrefill = model(promptArray, cache: referenceCache)
            eval(referencePrefill)
            do {
                let session = try CompiledDecodeSession(
                    model: model, prompt: promptArray, capacity: capacity)
                var token = Int(argMax(referencePrefill[0..., -1, 0...]).item(Int32.self))
                while session.remainingCapacity > 0 {
                    let input = MLXArray([Int32(token)]).reshaped(1, 1)
                    let reference = model(input, cache: referenceCache)
                    let compiled = try session.step(input)
                    eval(reference, compiled)
                    XCTAssertTrue(
                        allClose(reference, compiled, rtol: 1e-4, atol: 1e-4).item(Bool.self),
                        "round \(round) diverged")
                    token = Int(argMax(reference[0..., -1, 0...]).item(Int32.self))
                }
            }

            let tokens = prompt(length: length + 2, salt: round + 10)
            let eager = try generate(
                model, prompt: tokens, cache: try model.newCache(parameters: nil),
                maxTokens: 10, count: 10, compiled: false)
            let compiled = try generate(
                model, prompt: tokens, cache: try model.newCache(parameters: nil),
                maxTokens: 10, count: 10, compiled: true)
            XCTAssertEqual(eager.tokens, compiled.tokens, "round \(round) iterator diverged")
        }
    }

    /// Opt-in real-checkpoint parity and timing for the iterator path. Set
    /// `MLX_QWEN25_7B_PATH` to a local Qwen2.5-7B-Instruct-4bit directory.
    func testQwen25SevenBIteratorCompiledParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_QWEN25_7B_PATH"] else {
            throw XCTSkip("set MLX_QWEN25_7B_PATH to run the Qwen2.5-7B iterator parity check")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let configurationData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder.json5().decode(
            Qwen2Configuration.self, from: configurationData)
        let baseConfiguration = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: configurationData)
        let model = Qwen2Model(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfiguration.perLayerQuantization)

        for length in [7, 300, 2_000] {
            let tokens = (0 ..< length).map { Int32(($0 * 7_919 + 151) % 150_000) }
            var seconds: [Bool: Double] = [:]
            var runs: [Bool: Run] = [:]
            for compiled in [false, true] {
                let start = Date()
                runs[compiled] = try generate(
                    model, prompt: tokens, cache: try model.newCache(parameters: nil),
                    maxTokens: 128, count: 128, compiled: compiled)
                seconds[compiled] = Date().timeIntervalSince(start)
            }
            XCTAssertEqual(runs[false]?.tokens, runs[true]?.tokens, "prompt \(length) diverged")
            print(
                String(
                    format: "[QWEN2.5-7B iterator] prompt %d: eager %.2f s, compiled %.2f s",
                    length, seconds[false] ?? 0, seconds[true] ?? 0))
        }
    }
}
