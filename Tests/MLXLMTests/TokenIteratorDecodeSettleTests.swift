// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// #2373: `TokenIterator.next()` settles `KVCacheSimple` and `RotatingKVCache`
/// through their stored buffers, not through `state` slices of the live rows.
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
        model: LlamaModel, parameters: GenerateParameters, count: Int
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
        model: LlamaModel, parameters: GenerateParameters, count: Int
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

    func testSimpleCacheSettlesStoredBuffersAndKeepsTokens() throws {
        let model = makeTinyModel()
        let parameters = GenerateParameters(maxTokens: 40, temperature: 0)
        let (tokens, cache) = try iteratorTokens(model: model, parameters: parameters, count: 40)
        XCTAssertEqual(tokens.count, 40)
        XCTAssertTrue(cache.allSatisfy { type(of: $0) == KVCacheSimple.self })

        // Offset 45 sits inside the first 256-row block: `state` would slice, the
        // settle arrays are the stored buffers themselves.
        let settled = decodeSettleArrays(cache)
        XCTAssertEqual(settled.count, 2 * cache.count)
        for (array, layer) in zip(settled, cache.flatMap { $0.innerState() }) {
            XCTAssertTrue(array === layer)
            XCTAssertEqual(array.dim(2), 256)
        }
        XCTAssertEqual(cache[0].state[0].dim(2), cache[0].offset)

        XCTAssertEqual(
            tokens, try referenceTokens(model: model, parameters: parameters, count: 40))
    }

    func testRotatingCacheSettlesStoredBuffersAndKeepsTokensAcrossWrap() throws {
        let model = makeTinyModel()
        // 5 prompt rows plus 80 decode rows wrap a 32-row ring twice.
        let parameters = GenerateParameters(maxTokens: 80, maxKVSize: 32, temperature: 0)
        let (tokens, cache) = try iteratorTokens(model: model, parameters: parameters, count: 80)
        XCTAssertEqual(tokens.count, 80)
        XCTAssertTrue(cache.allSatisfy { type(of: $0) == RotatingKVCache.self })

        let settled = decodeSettleArrays(cache)
        XCTAssertEqual(settled.count, 2 * cache.count)
        for (array, layer) in zip(settled, cache.flatMap { $0.innerState() }) {
            XCTAssertTrue(array === layer)
        }

        XCTAssertEqual(
            tokens, try referenceTokens(model: model, parameters: parameters, count: 80))
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
