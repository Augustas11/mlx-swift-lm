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
    /// short prompts wrap it, and 8-bit affine experts that run on the CPU. Every parameter is
    /// random (the module initialises the conv taps and experts to zero), so every tail and the
    /// MoE reach the logits.
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
        // `BaseConfiguration` reads `quantization`; the fixture's per-layer map names nvfp4.
        let affine8: [String: Any] = ["group_size": 32, "bits": 8, "mode": "affine"]
        json["quantization"] = affine8
        json["quantization_config"] = affine8
        let data = try JSONSerialization.data(withJSONObject: json)
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        MLXRandom.seed(7)
        let model = Inkling2Model(config, quantization: base.perLayerQuantization)
        XCTAssertEqual(model.expertFormats.first?.gateUp, Inkling2ExpertFormat(mode: .affine, bits: 8, groupSize: 32))
        let randomized = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.dtype == .uint32 || value.dtype == .uint8 {
                return (key, MLXRandom.randInt(0 ..< 200, value.shape).asType(value.dtype))
            }
            return (key, (MLXRandom.normal(value.shape) * 0.05).asType(value.dtype))
        }
        model.update(parameters: ModuleParameters.unflattened(randomized))
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

    private func identical(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        zip(a, b).allSatisfy { x, y in
            let (sx, sy) = (x.state, y.state)
            return sx.count == sy.count
                && zip(sx, sy).allSatisfy { $0.shape == $1.shape && arrayEqual($0, $1).item(Bool.self) }
        }
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

            func ordinary(_ prefill: PrefillParameters) throws -> ([KVCache], MLXArray) {
                let cache = try model.newCache(parameters: nil)
                guard case .tokens(let rest) = try model.prepare(
                    prompt, cache: cache, state: nil, prefill: prefill)
                else { throw XCTSkip("ordinary prepare returned logits") }
                let logits = model(rest.tokens[.newAxis], cache: cache)
                eval(logits, cache)
                return (cache, logits)
            }
            func emitting(_ prefill: PrefillParameters) throws -> ([KVCache], LMOutput) {
                var state = LMOutput.State()
                state[mtpEmitFlagKey] = true
                state[mtpTargetLayerIdsKey] = [5, 23, 41, 48]
                let cache = try model.newCache(parameters: nil)
                guard case .logits(let output) = try model.prepare(
                    prompt, cache: cache, state: state, prefill: prefill)
                else { throw XCTSkip("emit prepare returned tokens") }
                eval(output.logits, cache)
                return (cache, output)
            }

            // The same chunks run the same ops, so the caches are bit-identical.
            let stepFour = PrefillParameters(stepSize: 4)
            let (ordinaryCache, ordinaryLogits) = try ordinary(stepFour)
            let (emitCache, output) = try emitting(stepFour)
            XCTAssertTrue(identical(emitCache, ordinaryCache), "emit prefill at step 4")
            XCTAssertTrue(arrayEqual(output.logits, ordinaryLogits).item(Bool.self), "last-token logits")
            let taps = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
            XCTAssertEqual(taps.shape, [1, prompt.text.tokens.size, 4 * 64])
            XCTAssertEqual(taps.dtype, .bfloat16)

            // Negative control: the check above sees a chunking change.
            let (unchunkedCache, _) = try emitting(PrefillParameters(chunking: .unchunked))
            XCTAssertFalse(
                identical(unchunkedCache, ordinaryCache),
                "an unchunked prefill left the same cache as step 4: the check cannot see chunking")
        }
    }

    func testTheIteratorKeepsTheCallersPrefillOnlyForChunkAggregatingTargets() throws {
        let inkling = try Device.withDefaultDevice(.cpu) { try tinyModel() }
        let caller = PrefillParameters(stepSize: 512)
        let kept = MTPSpeculativeTokenIterator.promptPrefill(
            caller, target: inkling, needsPromptHidden: true)
        XCTAssertEqual(kept.stepSize, 512)
        guard case .balanced = kept.chunking else { return XCTFail("chunking replaced for Inkling 2") }

        let other = Device.withDefaultDevice(.cpu) { LlamaModel(
            LlamaConfiguration(
                hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 8,
                rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)) }
        let replaced = MTPSpeculativeTokenIterator.promptPrefill(
            caller, target: other, needsPromptHidden: true)
        XCTAssertEqual(replaced.stepSize, Int.max)
        guard case .unchunked = replaced.chunking else { return XCTFail("other targets must run unchunked") }
        XCTAssertEqual(
            MTPSpeculativeTokenIterator.promptPrefill(caller, target: other, needsPromptHidden: false)
                .stepSize, 512)
    }

    // MARK: End to end through the MTP iterator

    /// Proposes the true greedy continuation (from an ordinary run) with one wrong token per
    /// round at a rotating position, so rounds accept every count from 0 to `blockSize - 1`.
    /// With exact verify and rollback the speculative stream equals ordinary decode.
    private final class OracleDrafter: Module, StatefulMTPDrafterModel {
        let expected: [Int]
        let promptLength: Int
        let vocabulary: Int
        private(set) var rounds = 0

        init(expected: [Int], promptLength: Int, vocabulary: Int) {
            self.expected = expected
            self.promptLength = promptLength
            self.vocabulary = vocabulary
            super.init()
        }

        var maximumBlockSize: Int? { 8 }
        var requiresSharedTargetKV: Bool { false }
        var requiresPromptPrefill: Bool { true }
        var requiresGreedySampling: Bool { true }
        var targetLayerIds: [Int]? { [5, 23, 41, 48] }
        func isCompatible(with target: any LanguageModel) -> Bool { true }
        func validateCompatibility(with target: any LanguageModel) throws {}
        func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
            MTPDrafterState(cache: [])
        }
        func prepareDrafterState(
            target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
            firstBonus: MLXArray, positionDeltas: MLXArray?, state: inout MTPDrafterState,
            sampler: any LogitSampler
        ) {}
        func commitDrafterState(
            target: any LanguageModel, targetHidden: MLXArray, draftTokens: MLXArray,
            acceptedCount: Int, finalToken: MLXArray, positionDeltas: MLXArray?,
            state: inout MTPDrafterState, sampler: any LogitSampler
        ) {}
        func draftBlock(
            target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
            sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
            queryOffset: Int, blockSize: Int, sampler: any LogitSampler
        ) -> MLXArray {
            var state = makeState(parameters: nil)
            return draftBlock(
                target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
                positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
                state: &state, sampler: sampler)
        }
        func draftBlock(
            target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
            sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
            queryOffset: Int, blockSize: Int, state: inout MTPDrafterState,
            sampler: any LogitSampler
        ) -> MLXArray {
            // The bonus `lastToken` is output index `queryOffset - promptLength`; draft what
            // follows it.
            let bonus = queryOffset - promptLength
            let wrongAt = rounds % blockSize  // blockSize - 1 means no wrong draft this round
            rounds += 1
            let drafts = (1 ..< blockSize).map { k -> Int32 in
                let index = bonus + k
                let truth = index < expected.count ? expected[index] : 0
                return Int32(k - 1 == wrongAt ? (truth + 1) % vocabulary : truth)
            }
            return MLXArray(drafts, [1, blockSize - 1])
        }
    }

    func testMTPIteratorWithAnOracleDrafterMatchesOrdinaryDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let prompt = [3, 9, 27, 17, 51, 25, 11, 33, 35, 1, 2]
            let input = LMInput(tokens: tokens(prompt))
            var parameters = GenerateParameters(maxTokens: 48, temperature: 0)
            parameters.prefill = PrefillParameters(stepSize: 4)

            var plain = try TokenIterator(input: input, model: model, parameters: parameters)
            var expected = [Int]()
            while let token = plain.next() { expected.append(token) }

            let drafter = OracleDrafter(expected: expected, promptLength: prompt.count, vocabulary: 64)
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: model, drafter: drafter, parameters: parameters,
                blockSize: 8)
            var produced = [Int]()
            while let token = iterator.next() { produced.append(token) }

            XCTAssertNil(iterator.passthroughReason)
            XCTAssertEqual(produced, expected, "speculative stream differs from ordinary decode")
            XCTAssertGreaterThan(iterator.acceptedCount, 0)
            XCTAssertLessThan(iterator.acceptedCount, iterator.proposedCount)
        }
    }
}
