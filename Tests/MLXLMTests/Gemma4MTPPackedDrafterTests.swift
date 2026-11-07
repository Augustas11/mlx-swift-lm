// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// The Gemma 4 assistant drafting for the text-only `gemma4_unified` target (MLXLLM): every
/// emitted stream must be the target's own greedy stream, whatever the drafter proposes.
@Suite(.serialized)
struct Gemma4MTPPackedDrafterTests {
    static let vocabularySize = 12

    /// A vocabulary this small makes a random drafter agree with the target often enough that
    /// every stream below both accepts and rejects proposals.
    static let smallVocabularySize = 3

    static func target(
        slidingWindow: Int, vocabularySize: Int = vocabularySize, seed: UInt64 = 7
    ) throws -> MLXLLM.Gemma4Model {
        let json = """
            {
              "model_type": "gemma4_unified",
              "vocab_size": \(vocabularySize),
              "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 16,
                "num_hidden_layers": 3,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": \(slidingWindow),
                "attention_k_eq_v": true,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
                "rope_parameters": {
                  "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                  "sliding_attention": {"rope_theta": 10000.0}
                },
                "tie_word_embeddings": true
              }
            }
            """
        MLXRandom.seed(seed)
        let model = MLXLLM.Gemma4Model(
            try JSONDecoder().decode(MLXLLM.Gemma4Configuration.self, from: Data(json.utf8)))
        eval(model)
        return model
    }

    static func drafterConfigJSON(slidingWindow: Int, vocabularySize: Int = vocabularySize)
        -> String
    {
        """
        {
          "model_type": "gemma4_unified_assistant",
          "backbone_hidden_size": 16,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "block_size": 4,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": \(vocabularySize),
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 0,
            "sliding_window": \(slidingWindow),
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    }

    static func drafter(
        slidingWindow: Int, vocabularySize: Int = vocabularySize, seed: UInt64 = 11
    ) throws -> Gemma4AssistantDraftModel {
        MLXRandom.seed(seed)
        let model = Gemma4AssistantDraftModel(
            try JSONDecoder().decode(
                Gemma4AssistantConfiguration.self,
                from: Data(
                    drafterConfigJSON(slidingWindow: slidingWindow, vocabularySize: vocabularySize)
                        .utf8)))
        eval(model)
        return model
    }

    static func greedy(_ model: any LanguageModel, prompt: [Int32], count: Int) throws -> [Int] {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: model,
            cache: model.newCache(parameters: nil),
            parameters: GenerateParameters(maxTokens: count, temperature: 0))
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return tokens
    }

    // MARK: - Speculative iterator, depth k

    @Test(arguments: [(1, 5), (2, 5), (3, 5), (3, 13)])
    func iteratorMatchesSerialGreedy(depth: Int, promptLength: Int) throws {
        let window = 8
        let vocabulary = Self.smallVocabularySize
        let target = try Self.target(slidingWindow: window, vocabularySize: vocabulary)
        let drafter = try Self.drafter(slidingWindow: window, vocabularySize: vocabulary)
        let prompt = (0 ..< promptLength).map { Int32(($0 * 5 + 1) % vocabulary) }
        let count = 40

        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            mainModel: target,
            drafter: drafter,
            mainCache: target.newCache(parameters: nil),
            parameters: GenerateParameters(maxTokens: count, temperature: 0),
            blockSize: depth + 1)
        var speculative: [Int] = []
        while let token = iterator.next() { speculative.append(token) }

        #expect(iterator.passthroughReason == nil, "the drafter stood down")
        #expect(iterator.acceptedCount > 0, "no draft was accepted")
        #expect(iterator.proposedCount > iterator.acceptedCount, "no draft was rejected")
        #expect(speculative == (try Self.greedy(target, prompt: prompt, count: count)))
    }

    // MARK: - Packed advance

    /// Drives the target and the packed drafter the way a depth-one native round does: verify
    /// `[final, proposal]` in one forward, keep the accepted prefix, rewind the rejected column,
    /// and ask ``advanceAndProposePacked`` for the next proposal from the verification state.
    @Test
    func packedAdvanceLoopMatchesSerialGreedy() throws {
        let window = 64
        let vocabulary = Self.smallVocabularySize
        let target = try Self.target(slidingWindow: window, vocabularySize: vocabulary)
        let drafter = try Self.drafter(slidingWindow: window, vocabularySize: vocabulary)
        let prompt: [Int32] = [1, 0, 2, 2, 1]
        let count = 40
        let sampler = ArgMaxSampler()
        let cache = try target.newCache(parameters: nil)

        func forward(_ tokens: [Int]) -> (logits: MLXArray, state: LMOutput.State, offset: Int) {
            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            let output = target(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])),
                cache: cache, state: state)
            return (output.logits, output.state!, cache[0].offset)
        }
        func advance(
            _ verified: (logits: MLXArray, state: LMOutput.State, offset: Int),
            accepted: [Int], final: Int
        ) throws -> Int {
            let state = verified.state
            let result = try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: state[mtpLastHiddenStatesKey]!,
                        acceptedTokens: accepted,
                        finalToken: final,
                        positionDeltas: nil,
                        state: drafter.makeState(parameters: nil),
                        sharedKV: state[mtpSharedKVStatesKey],
                        sharedKVOffset: verified.offset)
                ],
                sampler: sampler)
            return result.proposals.item(Int.self)
        }

        let prefill = forward(prompt.map(Int.init))
        var final = argMax(prefill.logits[0, -1], axis: -1).item(Int.self)
        var proposal = try advance(
            prefill, accepted: prompt.dropFirst().map(Int.init), final: final)
        var emitted = [final]
        var accepted = 0
        var rejected = 0
        while emitted.count < count {
            let verify = forward([final, proposal])
            let targets = argMax(verify.logits[0], axis: -1).asArray(Int.self)
            if targets[0] == proposal {
                accepted += 1
                final = targets[1]
                emitted += [proposal, final]
                proposal = try advance(verify, accepted: [proposal], final: final)
            } else {
                rejected += 1
                final = targets[0]
                emitted.append(final)
                #expect(trimPromptCache(cache, numTokens: 1) == 1)
                proposal = try advance(verify, accepted: [], final: final)
            }
        }

        #expect(accepted > 0, "no proposal was accepted, so the accept path is untested")
        #expect(rejected > 0, "no proposal was rejected, so the rewind path is untested")
        #expect(
            Array(emitted.prefix(count)) == (try Self.greedy(target, prompt: prompt, count: count)))
    }

    /// The packed proposal is the stateless depth-one draft at the final token's position.
    @Test
    func packedProposalEqualsDraftBlockOverTheAcceptedPrefix() throws {
        let window = 4
        let target = try Self.target(slidingWindow: 64)
        let drafter = try Self.drafter(slidingWindow: window)
        let cache = try target.newCache(parameters: nil)
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        let tokens = MLXArray([Int32]([1, 2, 3, 4, 5, 6, 7, 8, 9])).reshaped([1, 9])
        let state = target(LMInput.Text(tokens: tokens), cache: cache, state: emit).state!
        let hidden = state[mtpLastHiddenStatesKey]!
        let sharedKV = state[mtpSharedKVStatesKey]!

        // Verification input = the last 3 columns; one proposal accepted, one rejected.
        let row = MTPPackedDrafterAdvanceRow(
            targetHidden: hidden[0..., 6 ..< 9, 0...],
            acceptedTokens: [8],
            finalToken: 5,
            positionDeltas: nil,
            state: drafter.makeState(parameters: nil),
            sharedKV: sharedKV,
            sharedKVOffset: 9)
        let packed = try drafter.advanceAndProposePacked(
            target: target, rows: [row], sampler: ArgMaxSampler())

        let full = sharedKV["full_attention"]!
        let sliding = sharedKV["sliding_attention"]!
        let expected = drafter.draftBlock(
            target: target,
            lastToken: MLXArray([Int32(5)]),
            lastHidden: hidden[0..., 7 ..< 8, 0...],
            sharedKV: [
                "full_attention": (full.0[0..., 0..., ..<8, 0...], full.1[0..., 0..., ..<8, 0...]),
                "sliding_attention": (
                    sliding.0[0..., 0..., 4 ..< 8, 0...], sliding.1[0..., 0..., 4 ..< 8, 0...]
                ),
            ],
            positionDeltas: nil,
            queryOffset: 8,
            blockSize: 2,
            sampler: ArgMaxSampler())
        #expect(packed.proposals.shape == [1, 1])
        #expect(packed.proposals.item(Int.self) == expected.item(Int.self))
        #expect(packed.states[0].seedToken?.item(Int.self) == expected.item(Int.self))
        #expect(packed.states[0].nextPosition == 8)
    }

    @Test
    func packedRowsNeverReadEachOther() throws {
        let target = try Self.target(slidingWindow: 64)
        let drafter = try Self.drafter(slidingWindow: 64)
        func row(_ tokens: [Int32], final: Int) throws -> MTPPackedDrafterAdvanceRow {
            let cache = try target.newCache(parameters: nil)
            var emit = LMOutput.State()
            emit[mtpEmitFlagKey] = true
            let state = target(
                LMInput.Text(tokens: MLXArray(tokens).reshaped([1, tokens.count])),
                cache: cache, state: emit
            ).state!
            return MTPPackedDrafterAdvanceRow(
                targetHidden: state[mtpLastHiddenStatesKey]![0..., (-2)..., 0...],
                acceptedTokens: [],
                finalToken: final,
                positionDeltas: nil,
                state: drafter.makeState(parameters: nil),
                sharedKV: state[mtpSharedKVStatesKey],
                sharedKVOffset: tokens.count)
        }
        let a = try row([1, 2, 3, 4, 5, 6], final: 3)
        let b = try row([9, 8, 7], final: 10)
        let sampler = ArgMaxSampler()
        let both = try drafter.advanceAndProposePacked(
            target: target, rows: [a, b], sampler: sampler)
        let alone = try [a, b].map {
            try drafter.advanceAndProposePacked(target: target, rows: [$0], sampler: sampler)
                .proposals
        }
        #expect(both.proposals.shape == [2, 1])
        #expect(both.proposals.asArray(Int32.self) == alone.flatMap { $0.asArray(Int32.self) })
    }

    @Test
    func malformedRowsFailBeforeAnyForward() throws {
        let target = try Self.target(slidingWindow: 64)
        let drafter = try Self.drafter(slidingWindow: 64)
        let hidden = MLXArray.zeros([1, 2, 16])
        let kv = MLXArray.zeros([1, 1, 4, 8])
        let sharedKV = ["full_attention": (kv, kv), "sliding_attention": (kv, kv)]
        let state = drafter.makeState(parameters: nil)
        let sampler = ArgMaxSampler()

        #expect(throws: MTPPackedDrafterError.incompatibleState(row: 0)) {
            try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: hidden, acceptedTokens: [], finalToken: 1,
                        positionDeltas: nil, state: state)
                ],
                sampler: sampler)
        }
        #expect(
            throws: MTPPackedDrafterError.invalidAcceptedCount(
                row: 0, acceptedCount: 2, inputCount: 2)
        ) {
            try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: hidden, acceptedTokens: [1, 2], finalToken: 1,
                        positionDeltas: nil, state: state, sharedKV: sharedKV, sharedKVOffset: 4)
                ],
                sampler: sampler)
        }
    }

    // MARK: - Embedded assistant

    @Test
    func embeddedAssistantLoadsFromTheTargetDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-mtp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let assistant = try JSONSerialization.jsonObject(
            with: Data(Self.drafterConfigJSON(slidingWindow: 8).utf8))
        let config: [String: Any] = [
            "model_type": "gemma4_unified",
            "text_config": [
                "model_type": "gemma4_unified_text", "mtp_assistant_config": assistant,
            ],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))

        let source = try Self.drafter(slidingWindow: 8, seed: 3)
        let parameters = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().map {
                (Gemma4AssistantDraftModel.embeddedTensorPrefix + $0.0, $0.1)
            })
        let weights = directory.appendingPathComponent(
            Gemma4AssistantDraftModel.embeddedWeightFileName)
        try save(arrays: parameters, url: weights)

        let loaded = try Gemma4AssistantDraftModel.loadEmbedded(targetDirectory: directory)
        let loadedParameters = Dictionary(
            uniqueKeysWithValues: loaded.parameters().flattened().map { ($0.0, $0.1) })
        #expect(loadedParameters.count == parameters.count)
        for (key, value) in source.parameters().flattened() {
            #expect(loadedParameters[key].map { arrayEqual($0, value).item(Bool.self) } == true)
        }

        var stray = parameters
        stray["language_model.model.norm.weight"] = MLXArray.ones([16])
        try save(arrays: stray, url: weights)
        #expect(
            throws: Gemma4EmbeddedAssistantError.unexpectedTensor(
                "language_model.model.norm.weight")
        ) {
            try Gemma4AssistantDraftModel.loadEmbedded(targetDirectory: directory)
        }
    }

    @Test
    func targetLoaderLeavesTheEmbeddedAssistantToTheDrafter() throws {
        let target = try Self.target(slidingWindow: 8)
        let sanitized = target.sanitize(weights: [
            "language_model.mtp.pre_projection.weight": MLXArray.zeros([8, 32]),
            "language_model.model.norm.weight": MLXArray.ones([16]),
        ])
        #expect(Set(sanitized.keys) == ["language_model.model.norm.weight"])
    }
}
