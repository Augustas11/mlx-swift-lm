// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// #2373: `TokenIterator.next()` settles `KVCacheSimple` and `RotatingKVCache`
/// through the token alone; other caches keep their `state` in the evaluation.
final class TokenIteratorDecodeSettleTests: XCTestCase {

    private func makeTinyModel() -> LlamaModel {
        MLXRandom.seed(2373)
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        return model
    }

    private let prompt: [Int32] = [1, 2, 3, 4, 5]

    /// The pre-#2373 loop: a whole short prompt as the first forward, then one
    /// greedy token per step with `[token] + state` evaluated asynchronously.
    private func referenceTokens(
        model: any LanguageModel, parameters: GenerateParameters, count: Int
    ) throws -> [Int] {
        let cache = try model.newCache(parameters: parameters)
        var logits = model(MLXArray(prompt)[.newAxis], cache: cache)
        var y = argMax(logits[0..., -1, 0...], axis: -1)
        asyncEval(y)
        var tokens: [Int] = []
        for _ in 0 ..< count {
            let previous = y
            logits = model(previous[.newAxis], cache: cache)
            y = argMax(logits[0..., -1, 0...], axis: -1)
            asyncEval([y] + cache.flatMap { $0.state })
            tokens.append(previous.item(Int.self))
        }
        return tokens
    }

    private func iteratorTokens(
        model: any LanguageModel, parameters: GenerateParameters, count: Int
    ) throws -> (tokens: [Int], cache: [KVCache]) {
        let cache = try model.newCache(parameters: parameters)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
            parameters: parameters)
        var tokens: [Int] = []
        for _ in 0 ..< count {
            guard let token = iterator.next() else { break }
            tokens.append(token)
        }
        return (tokens, cache)
    }

    func testSimpleCacheSettlesThroughTheTokenAndKeepsTokens() throws {
        let model = makeTinyModel()
        let parameters = GenerateParameters(maxTokens: 40, temperature: 0)
        let (tokens, cache) = try iteratorTokens(model: model, parameters: parameters, count: 40)
        XCTAssertEqual(tokens.count, 40)
        XCTAssertTrue(cache.allSatisfy { type(of: $0) == KVCacheSimple.self })

        // Nothing but the token is evaluated for these caches; the rows the
        // decode wrote are there all the same.
        XCTAssertTrue(decodeSettleArrays(cache).isEmpty)
        XCTAssertEqual(cache[0].offset, prompt.count + 40)
        XCTAssertEqual(cache[0].state[0].dim(2), cache[0].offset)

        XCTAssertEqual(
            tokens, try referenceTokens(model: model, parameters: parameters, count: 40))
    }

    func testRotatingCacheSettlesThroughTheTokenAndKeepsTokensAcrossWrap() throws {
        let model = makeTinyModel()
        // 5 prompt rows plus 80 decode rows wrap a 32-row ring twice.
        let parameters = GenerateParameters(maxTokens: 80, maxKVSize: 32, temperature: 0)
        let (tokens, cache) = try iteratorTokens(model: model, parameters: parameters, count: 80)
        XCTAssertEqual(tokens.count, 80)
        XCTAssertTrue(cache.allSatisfy { type(of: $0) == RotatingKVCache.self })

        XCTAssertTrue(decodeSettleArrays(cache).isEmpty)

        XCTAssertEqual(
            tokens, try referenceTokens(model: model, parameters: parameters, count: 80))
    }

    /// LFM3 with conv, sliding-attention and full-attention layers, at the size
    /// `LFM3Tests` uses, with random weights.
    private func makeTinyHybridModel() throws -> LFM3Model {
        MLXRandom.seed(2373)
        let configuration = try JSONDecoder().decode(
            LFM3Configuration.self,
            from: Data(
                """
                {
                    "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                    "intermediate_size": 48, "num_hidden_layers": 6,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                    "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": false,
                    "sliding_window": 4, "use_sandwich_norm": true,
                    "tie_word_embeddings": true,
                    "layer_types": [
                        "conv", "sliding_attention", "full_attention", "conv",
                        "sliding_attention", "full_attention"
                    ],
                    "rope_parameters": {
                        "full_attention": {
                            "rope_type": "default", "rope_theta": 1000000.0,
                            "partial_rotary_factor": 0.5
                        },
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                    }
                }
                """.utf8))
        let model = LFM3Model(configuration)
        eval(model)
        return model
    }

    func testHybridConvAttentionCacheKeepsConvStateAndKeepsTokens() throws {
        let model = try makeTinyHybridModel()
        // 200 decode steps: the 4-row sliding window wraps many times and the
        // full-attention caches grow to 205 rows.
        let parameters = GenerateParameters(maxTokens: 200, temperature: 0)
        let (tokens, cache) = try iteratorTokens(model: model, parameters: parameters, count: 200)
        XCTAssertEqual(tokens.count, 200)
        XCTAssertTrue(cache.contains { $0 is MambaCache })
        XCTAssertTrue(cache.contains { type(of: $0) == KVCacheSimple.self })

        // Plain attention layers add nothing; every other layer adds its state, in order.
        let kept = cache.filter {
            type(of: $0) != KVCacheSimple.self && type(of: $0) != RotatingKVCache.self
        }
        XCTAssertTrue(kept.allSatisfy { $0 is MambaCache })
        let expected = kept.flatMap { $0.state }
        let settled = decodeSettleArrays(cache)
        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(settled.count, expected.count)
        XCTAssertTrue(
            zip(settled, expected).allSatisfy { arrayEqual($0, $1).item(Bool.self) })

        XCTAssertEqual(
            tokens, try referenceTokens(model: model, parameters: parameters, count: 200))
    }

    func testOtherCachesKeepState() throws {
        let quantized = QuantizedKVCache(groupSize: 64, bits: 4)
        let keys = MLXRandom.normal([1, 4, 3, 64])
        _ = quantized.updateQuantized(keys: keys, values: keys)
        XCTAssertEqual(decodeSettleArrays([quantized]).count, quantized.state.count)
        XCTAssertTrue(
            zip(decodeSettleArrays([quantized]), quantized.state).allSatisfy {
                arrayEqual($0, $1).item(Bool.self)
            })
    }
}
