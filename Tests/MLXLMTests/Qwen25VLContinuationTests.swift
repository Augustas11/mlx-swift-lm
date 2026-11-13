// Copyright © 2026 Apple Inc.
//
// Equivalence tests for Qwen2.5-VL and Qwen2-VL windowed prefill and warm
// (cached-prefix) continuation, on tiny random-weight models so they run in CI
// without downloads. The invariant under test mirrors Qwen35ContinuationTests:
// however a prompt reaches the KV cache — one shot, windowed chunks, or split
// across a warm continuation — the next-token logits must match, because M-RoPE
// positions must be anchored at the cache offset (plus the carried rope delta),
// never restarted at zero.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }
    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}

final class Qwen25VLContinuationTests: XCTestCase {

    // MARK: - Tiny models

    private func makeTinyQwen25VL() throws -> Qwen25VL {
        let json = """
            {
                "model_type": "qwen2_5_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "sliding_window": 32768,
                "use_sliding_window": false,
                "max_window_layers": 2,
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vision_token_id": 504,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "window_size": 64,
                    "fullatt_block_indexes": [1],
                    "tokens_per_second": 2,
                    "in_chans": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen25VLConfiguration.self, from: Data(json.utf8))
        return Qwen25VL(config)
    }

    private func makeTinyQwen2VL() throws -> Qwen2VL {
        let json = """
            {
                "model_type": "qwen2_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "image_token_id": 500,
                "video_token_id": 501,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "embed_dim": 32,
                    "hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "mlp_ratio": 2.0,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen2VLConfiguration.self, from: Data(json.utf8))
        return Qwen2VL(config)
    }

    // MARK: - Shared assertions

    /// The Qwen2 family and Qwen2.5-VL share the tiny-model token ids.
    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    // MARK: - Qwen2.5-VL

    func testQwen25VLWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedPrefillMatchesSingleShot() throws {
        try continuation.assertWindowedTextPrefill(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(makeTinyQwen25VL())
    }

    /// Through the `PreparedInputSplitting` conformance, so the config-derived token
    /// ids and merge size are exercised alongside the split itself.
    func testQwen25VLAppendOnlyMediaSplitMatchesFullPrefill() throws {
        let model = try makeTinyQwen25VL()
        try continuation.assertAppendOnlyMediaSplit(
            model,
            split: { (model as PreparedInputSplitting).splitPreparedInput($0, droppingFirst: $1) },
            expectsIsolation: true)
    }

    // MARK: - Qwen2-VL

    func testQwen2VLWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(makeTinyQwen2VL())
    }

    func testQwen2VLWindowedPrefillMatchesSingleShot() throws {
        try continuation.assertWindowedTextPrefill(makeTinyQwen2VL())
    }

    /// Post-vision decode uses fused RoPE at cacheOffset + delta. That must
    /// match the one-token continuation, which still applies M-RoPE position ids.
    func testQwen2VLImageTokenDecodeMatchesPositionIds() throws {
        let model = try makeTinyQwen2VL()
        try withRandomState(MLXRandom.RandomState(seed: 5)) {
            let image = continuation.image()
            let prefix = concatenated(
                [
                    continuation.textTokens(10), continuation.imageRun(),
                    continuation.textTokens(8, seed: 5),
                ], axis: 1)
            let next = continuation.textTokens(1, seed: 9)

            let cacheM = try model.newCache(parameters: nil)
            let prepared = try model.prepare(
                LMInput(text: .init(tokens: prefix), image: image),
                cache: cacheM, state: nil, prefill: PrefillParameters())
            guard case .logits(let prefixOut) = prepared else {
                XCTFail("expected logits")
                return
            }
            let continued = try model.prepare(
                LMInput(text: .init(tokens: next)),
                cache: cacheM, state: prefixOut.state, prefill: PrefillParameters())
            guard case .logits(let positionOut) = continued else {
                XCTFail("expected logits")
                return
            }

            let cacheF = try model.newCache(parameters: nil)
            _ = try model.prepare(
                LMInput(text: .init(tokens: prefix), image: image),
                cache: cacheF, state: nil, prefill: PrefillParameters())
            let fused = try model.tokenOnlyForward(next, cache: cacheF)

            let positionLogits = positionOut.logits[0..., -1, 0...]
            let fusedLogits = fused[0..., -1, 0...]
            eval(positionLogits, fusedLogits)
            let diff = abs(positionLogits - fusedLogits).max().item(Float.self)
            XCTAssertLessThanOrEqual(
                diff, MatmulPrecision.splitTolerance,
                "fused post-vision decode diverged from M-RoPE position ids by \(diff)")
        }
    }

    /// macprovider #2201: a token-only forward (text prompt, no image) trapped in
    /// the `LanguageModel` default. It must run and match the text-input forward
    /// for a prefill and the decode step after it.
    func testQwen2VLTokenOnlyForwardMatchesTextInputForward() throws {
        let model = try makeTinyQwen2VL()
        let prompt = MLXArray((1 ... 9).map(Int32.init)).reshaped(1, 9)
        let next = MLXArray([Int32(42)]).reshaped(1, 1)

        let tokenCache = try model.newCache(parameters: nil)
        let tokenPrefill = model(prompt, cache: tokenCache)
        let tokenDecode = model(next, cache: tokenCache)

        let textCache = try model.newCache(parameters: nil)
        let textPrefill = model(LMInput.Text(tokens: prompt), cache: textCache, state: nil).logits
        let textDecode = model(LMInput.Text(tokens: next), cache: textCache, state: nil).logits

        eval(tokenPrefill, tokenDecode, textPrefill, textDecode)
        XCTAssertEqual(tokenPrefill.shape, [1, 9, 512])
        XCTAssertEqual(tokenCache.first?.offset, 10)
        XCTAssertTrue(allClose(tokenPrefill, textPrefill, atol: 1e-5).item(Bool.self))
        XCTAssertTrue(allClose(tokenDecode, textDecode, atol: 1e-5).item(Bool.self))
    }

    /// macprovider #2201 audit (code and architecture MEDIUMs): after an image
    /// prefill, text positions continue at `offset + rope delta`. A token-only
    /// step must match the state-carrying step, and refuse when the cache's
    /// position state is unknown, instead of silently using cache-offset positions.
    func testQwen2VLTokenOnlyStepAfterImagePrefillMatchesStatePath() throws {
        let model = try makeTinyQwen2VL()
        let tokens = concatenated(
            [
                continuation.textTokens(5), continuation.imageRun(),
                continuation.textTokens(6, seed: 3),
            ],
            axis: 1)
        let image = continuation.image()
        let next = MLXArray([Int32(42), Int32(43)]).reshaped(1, 2)

        func imagePrefill() throws -> ([any KVCache], LMOutput.State?) {
            let cache = try model.newCache(parameters: nil)
            let (_, state) = try continuation.lastLogits(
                model.prepare(
                    LMInput(text: .init(tokens: tokens), image: image),
                    cache: cache, state: nil, prefill: PrefillParameters()))
            return (cache, state)
        }

        let (stateCache, state) = try imagePrefill()
        let viaState = model(LMInput.Text(tokens: next), cache: stateCache, state: state).logits

        let (tokenCache, _) = try imagePrefill()
        let viaTokens = model(next, cache: tokenCache)

        let (offsetCache, _) = try imagePrefill()
        let offsetOnly = model(LMInput.Text(tokens: next), cache: offsetCache, state: nil).logits

        eval(viaState, viaTokens, offsetOnly)
        XCTAssertGreaterThan(
            continuation.maxAbsDiff(viaState, offsetOnly), 1e-3,
            "fixture must make the rope delta matter")
        XCTAssertLessThan(continuation.maxAbsDiff(viaState, viaTokens), 1e-4)

        // A cache this model did not prepare (a copy of a vision cache) has an
        // unknown M-RoPE delta: the token-only path refuses instead of guessing.
        let (visionCache, _) = try imagePrefill()
        let copied = visionCache.map { $0.copy() }
        XCTAssertThrowsError(try model.tokenOnlyPositionState(cache: copied))
        let refused = model(next, cache: copied)
        eval(refused)
        XCTAssertTrue(all(isNaN(refused)).item(Bool.self))
        // A fresh cache is plain text and stays allowed.
        XCTAssertNil(try model.tokenOnlyPositionState(cache: try model.newCache(parameters: nil)))
    }

    /// macprovider #2201 audit (security MEDIUMs): the recorded delta holds only
    /// while the cache still holds the image span. A trim within the trailing
    /// text keeps the token-only step exact; a trim into the image makes the
    /// state unknown, and it stays unknown after the cache grows back past the
    /// span. The throwing forward refuses without stopping the process.
    func testQwen2VLTokenOnlyStepAfterTrimMatchesOrRefuses() throws {
        let model = try makeTinyQwen2VL()
        // 5 text + vision start + 4 image tokens + 6 text: the image span ends at 10.
        let tokens = concatenated(
            [
                continuation.textTokens(5), continuation.imageRun(),
                continuation.textTokens(6, seed: 3),
            ],
            axis: 1)
        let image = continuation.image()
        let next = MLXArray([Int32(42), Int32(43)]).reshaped(1, 2)

        func imagePrefill() throws -> ([any KVCache], LMOutput.State?) {
            let cache = try model.newCache(parameters: nil)
            let (_, state) = try continuation.lastLogits(
                model.prepare(
                    LMInput(text: .init(tokens: tokens), image: image),
                    cache: cache, state: nil, prefill: PrefillParameters()))
            return (cache, state)
        }
        func trim(_ cache: [any KVCache], _ n: Int) {
            for layer in cache { XCTAssertEqual(layer.trim(n), n) }
        }

        // Trim within the trailing text: offset 13 is past the image end.
        let (stateCache, state) = try imagePrefill()
        trim(stateCache, 3)
        let viaState = model(LMInput.Text(tokens: next), cache: stateCache, state: state).logits
        let (tokenCache, _) = try imagePrefill()
        trim(tokenCache, 3)
        let viaTokens = try model.tokenOnlyForward(next, cache: tokenCache)
        let (offsetCache, _) = try imagePrefill()
        trim(offsetCache, 3)
        let offsetOnly = model(LMInput.Text(tokens: next), cache: offsetCache, state: nil).logits
        eval(viaState, viaTokens, offsetOnly)
        XCTAssertGreaterThan(
            continuation.maxAbsDiff(viaState, offsetOnly), 1e-3,
            "fixture must make the rope delta matter")
        XCTAssertLessThan(continuation.maxAbsDiff(viaState, viaTokens), 1e-4)

        // Trim into the image: offset 9 is inside the span.
        let (intoImage, _) = try imagePrefill()
        trim(intoImage, 7)
        XCTAssertThrowsError(try model.tokenOnlyPositionState(cache: intoImage))
        XCTAssertThrowsError(try model.tokenOnlyForward(next, cache: intoImage)) {
            XCTAssertTrue($0 is Qwen2VL.UnknownPositionStateError)
        }
        XCTAssertEqual(intoImage.first?.offset, 9, "a refused forward must not touch the cache")

        // The protocol overload refuses without stopping the process: all-NaN
        // logits of the right shape, cache untouched.
        let refused = model(next, cache: intoImage)
        eval(refused)
        XCTAssertEqual(refused.shape, [1, 2, 512])
        XCTAssertTrue(all(isNaN(refused)).item(Bool.self))
        XCTAssertEqual(intoImage.first?.offset, 9)

        // Regrowing past the span through the state path leaves it unknown.
        let (regrown, _) = try imagePrefill()
        trim(regrown, 7)
        _ = model(LMInput.Text(tokens: next), cache: regrown, state: nil).logits
        XCTAssertEqual(regrown.first?.offset, 11)
        XCTAssertThrowsError(try model.tokenOnlyPositionState(cache: regrown))
    }

    /// macprovider #2201 audit: concurrent sessions on one model each touch
    /// only their own cache's registry entry, and a cache's key is stable.
    func testQwen2VLRopeDeltaRegistryIsPerCacheUnderConcurrency() {
        let registry = Qwen2VLRopeDeltaRegistry()
        let caches: [[any KVCache]] = (0 ..< 64).map { _ in [KVCacheSimple()] }
        let deltas = (0 ..< 64).map { MLXArray(Int32($0 + 1)) }
        XCTAssertTrue((caches[0].first as AnyObject) === (caches[0].first as AnyObject))

        let failures = LockedCounter()
        DispatchQueue.concurrentPerform(iterations: 64 * 20) { step in
            let i = step % 64
            let vision = i.isMultiple(of: 2)
            registry.record(vision ? deltas[i] : nil, visionEnd: 0, for: caches[i])
            switch registry.lookup(caches[i]) {
            case .known(let delta?) where vision && delta === deltas[i]: break
            case .known(nil) where !vision: break
            default: failures.increment()
            }
        }
        XCTAssertEqual(failures.value, 0)
    }

    func testQwen2VLWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(makeTinyQwen2VL())
    }

    /// `Qwen2VL` does not conform, so the split runs through `QwenVL` directly
    /// with this model's ids. Its vision attention stays inside each image
    /// (macprovider #2016), so, as for Qwen2.5-VL, the split must match the
    /// full prefill.
    func testQwen2VLAppendOnlyMediaSplitMatchesFullPrefill() throws {
        let model = try makeTinyQwen2VL()
        try continuation.assertAppendOnlyMediaSplit(
            model,
            split: {
                QwenVL.splitPreparedInput(
                    $0,
                    droppingFirst: $1,
                    imageTokenId: model.config.baseConfiguration.imageTokenId,
                    videoTokenId: model.config.baseConfiguration.videoTokenId,
                    mergeSize: model.config.visionConfiguration.spatialMergeSize)
            },
            expectsIsolation: true)
    }
}
