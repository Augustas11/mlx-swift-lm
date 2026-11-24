import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// A staged speculative round on a real (tiny, random) Inkling 2 model: committing `n` of `L`
/// verified rows must leave the cache that an `n`-row forward leaves, so the next step decodes
/// the same; and the MTP emit prefill must fill the cache as ordinary prefill does under the
/// caller's chunking. Exercises the model's own conv-tail record sites. CPU only.
final class Inkling2SpeculativeRoundTests: XCTestCase {

    private func resource(_ name: String) throws -> Data {
        let directory = try XCTUnwrap(Bundle.module.url(forResource: "inkling2", withExtension: nil))
        return try Data(contentsOf: directory.appendingPathComponent(name))
    }

    /// Mini's topology (24 layers, the looped core, 48 visits) at toy widths, a 6-slot ring so
    /// short prompts wrap it, and 8-bit affine experts that run on the CPU.
    private func tinyModel() throws -> Inkling2Model {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: resource("oq4e-config.json")) as? [String: Any])
        var text = try XCTUnwrap(json["text_config"] as? [String: Any])
        let tiny: [String: Any] = [
            "hidden_size": 64, "vocab_size": 64, "unpadded_vocab_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 32,
            "d_rel": 8, "rel_extent": 6, "sliding_window_size": 6, "log_scaling_n_floor": 4,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 32,
        ]
        text.merge(tiny) { _, new in new }
        var loop = try XCTUnwrap(text["loop_config"] as? [String: Any])
        loop["gate_rank"] = 8
        text["loop_config"] = loop
        json["text_config"] = text
        json["mtp_config"] = nil
        json["quantization_config"] = ["group_size": 32, "bits": 8, "mode": "affine"]
        let data = try JSONSerialization.data(withJSONObject: json)
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        MLXRandom.seed(7)
        let model = Inkling2Model(config, quantization: base.perLayerQuantization)
        eval(model)
        return model
    }

    private func tokens(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init))
    }

    private func close(_ a: MLXArray, _ b: MLXArray, _ note: String) {
        XCTAssertEqual(a.shape, b.shape, note)
        XCTAssertTrue(allClose(a, b, rtol: 1e-4, atol: 1e-4).item(Bool.self), note)
    }

    private func expectTailsMatch(_ a: [KVCache], _ b: [KVCache], _ note: String) {
        for (visit, (x, y)) in zip(a, b).enumerated() {
            let (tx, ty) = (((x as! CacheList)[1]) as! ArraysCache, ((y as! CacheList)[1]) as! ArraysCache)
            XCTAssertTrue(tx is ConvolutionTailCache, "\(note): visit \(visit) tail type")
            for slot in 0 ..< 4 {
                switch (tx[slot], ty[slot]) {
                case (nil, nil): continue
                case let (p?, q?): close(p, q, "\(note): visit \(visit) slot \(slot)")
                default: XCTFail("\(note): visit \(visit) slot \(slot) present on one side")
                }
            }
        }
    }

    func testCommittingAPrefixOfAVerifyRoundMatchesWritingOnlyThatPrefix() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let prompt = tokens([3, 9, 27, 17, 51, 25, 11, 33, 35])  // 9 > the 6-slot ring
            let verify = [5, 15, 45, 7]
            for retaining in [0, 1, 3, 4] {
                let note = "retaining \(retaining)"
                let live = KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)
                eval(model(prompt[.newAxis], cache: live.cache))
                live.commitProcessedTokens(prompt.size)
                let oracle = live.copy()

                let round = try XCTUnwrap(live.beginRound(maximumPositions: verify.count), note)
                let verified = model(tokens(verify)[.newAxis], cache: round.caches)
                eval(verified)
                live.commit(round, retaining: retaining)

                if retaining > 0 {
                    let kept = model(tokens(Array(verify[..<retaining]))[.newAxis], cache: oracle.cache)
                    close(verified[0..., ..<retaining, 0...], kept, "\(note): verified logits")
                    oracle.commitProcessedTokens(retaining)
                }
                expectTailsMatch(live.cache, oracle.cache, note)
                XCTAssertEqual(live.processedTokenCount, oracle.processedTokenCount, note)

                // The next two steps decode identically from both caches.
                for (step, token) in [61, 13].enumerated() {
                    let a = model(tokens([token])[.newAxis], cache: live.cache)
                    let b = model(tokens([token])[.newAxis], cache: oracle.cache)
                    close(a, b, "\(note): step \(step) logits")
                }
            }
        }
    }

    func testRewindingTheLastCommitMatchesWritingOnlyWhatWasKept() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let prompt = tokens([3, 9, 27, 17, 51, 25, 11, 33])
            let live = KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)
            eval(model(prompt[.newAxis], cache: live.cache))
            live.commitProcessedTokens(prompt.size)
            let oracle = live.copy()

            let round = try XCTUnwrap(live.beginRound(maximumPositions: 4))
            eval(model(tokens([5, 15, 45, 7])[.newAxis], cache: round.caches))
            live.commit(round, retaining: 4)
            XCTAssertEqual(live.rewindLastRound(3), 3)

            eval(model(tokens([5])[.newAxis], cache: oracle.cache))
            oracle.commitProcessedTokens(1)
            expectTailsMatch(live.cache, oracle.cache, "rewind")
            close(
                model(tokens([61])[.newAxis], cache: live.cache),
                model(tokens([61])[.newAxis], cache: oracle.cache), "after rewind")
        }
    }

    func testEmitPrefillFillsTheCacheAsOrdinaryPrefillUnderTheCallersChunks() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let prompt = LMInput(tokens: tokens([3, 9, 27, 17, 51, 25, 11, 33, 35, 1, 2]))
            let prefill = PrefillParameters(stepSize: 4)

            let ordinaryCache = try model.newCache(parameters: nil)
            guard case .tokens(let rest) = try model.prepare(
                prompt, cache: ordinaryCache, state: nil, prefill: prefill)
            else { return XCTFail("ordinary prepare returned logits") }
            let ordinaryLogits = model(rest.tokens[.newAxis], cache: ordinaryCache)

            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            state[mtpTargetLayerIdsKey] = [5, 23, 41, 48]
            let emitCache = try model.newCache(parameters: nil)
            guard case .logits(let output) = try model.prepare(
                prompt, cache: emitCache, state: state, prefill: prefill)
            else { return XCTFail("emit prepare returned tokens") }

            close(output.logits, ordinaryLogits, "last-token logits")
            expectTailsMatch(emitCache, ordinaryCache, "emit prefill")
            let taps = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
            XCTAssertEqual(taps.shape, [1, prompt.text.tokens.size, 4 * 64])
            XCTAssertEqual(taps.dtype, .bfloat16)
        }
    }

    func testTheIteratorKeepsTheCallersPrefillForInkling() throws {
        let model = try Device.withDefaultDevice(.cpu) { try tinyModel() }
        XCTAssertTrue((model as Any) is any MTPChunkedPromptHiddenModel)
    }
}
