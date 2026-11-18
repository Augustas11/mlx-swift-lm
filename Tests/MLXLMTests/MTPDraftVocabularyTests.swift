// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// MARK: - DraftVocabulary

@Test
func testDraftVocabularyCanonicalizesAndValidates() throws {
    let vocabulary = try DraftVocabulary(ids: [5, 1, 5, 3], vocabularySize: 8, requiredIds: [3])
    #expect(vocabulary.ids == [1, 3, 5])
    #expect(throws: DraftVocabulary.Error.empty) {
        try DraftVocabulary(ids: [Int](), vocabularySize: 8)
    }
    #expect(throws: DraftVocabulary.Error.idOutOfRange(id: 8, vocabularySize: 8)) {
        try DraftVocabulary(ids: [1, 8], vocabularySize: 8)
    }
    #expect(throws: DraftVocabulary.Error.idOutOfRange(id: -1, vocabularySize: 8)) {
        try DraftVocabulary(ids: [-1], vocabularySize: 8)
    }
    #expect(throws: DraftVocabulary.Error.missingRequiredIds([7])) {
        try DraftVocabulary(ids: [1, 2], vocabularySize: 8, requiredIds: [2, 7])
    }
    #expect(throws: DraftVocabulary.Error.invalidVocabularySize(0)) {
        try DraftVocabulary(ids: [0], vocabularySize: 0)
    }
}

@Test
func testDraftVocabularySidecarRoundTrips() throws {
    let vocabulary = try DraftVocabulary(ids: [9, 2, 4], vocabularySize: 10)
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("draft-vocab-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try vocabulary.write(to: url)
    #expect(try DraftVocabulary(contentsOf: url, requiredIds: [9]) == vocabulary)
    #expect(throws: DraftVocabulary.Error.missingRequiredIds([3])) {
        try DraftVocabulary(contentsOf: url, requiredIds: [3])
    }
    let wrong = Data(#"{"format": "other", "version": 1, "vocab_size": 4, "ids": [1]}"#.utf8)
    #expect(throws: DraftVocabulary.Error.unsupportedFormat("other v1")) {
        try DraftVocabulary(jsonData: wrong)
    }
}

@Test
func testDraftVocabularyMostFrequentFromJSONL() throws {
    let jsonl = Data(
        """
        [1, 2, 2, 3]
        {"token_ids": [2, 3, 3, 3, 4]}

        {"tokens": [5, 1]}
        """.utf8)
    let sequences = try DraftVocabulary.sequences(jsonl: jsonl)
    #expect(sequences == [[1, 2, 2, 3], [2, 3, 3, 3, 4], [5, 1]])
    // Counts: 3 -> 4, 2 -> 3, 1 -> 2, 4 -> 1, 5 -> 1.
    let top2 = try DraftVocabulary.mostFrequent(
        in: sequences, topN: 2, vocabularySize: 16, specialIds: [15])
    #expect(top2.ids == [2, 3, 15])
    // Tie between 4 and 5 resolves to the smaller id.
    let top4 = try DraftVocabulary.mostFrequent(
        in: sequences, topN: 4, vocabularySize: 16, specialIds: [])
    #expect(top4.ids == [1, 2, 3, 4])
    #expect(throws: DraftVocabulary.Error.idOutOfRange(id: 5, vocabularySize: 5)) {
        try DraftVocabulary.mostFrequent(
            in: sequences, topN: 2, vocabularySize: 5, specialIds: [])
    }
}

// MARK: - Projection exactness

/// Subset logits vs the full head's columns at the same ids: the largest
/// absolute difference (0 means bit-equal) and whether greedy agrees
/// wherever the full argmax is in the subset.
private func compareSubsetToFull(
    full: MLXArray, subset: MLXArray, projection: DraftVocabularyProjection
) -> (maxDiff: Float, argmaxChecked: Int) {
    let gathered = take(full, projection.idArray, axis: -1)
    let maxDiff = abs(gathered.asType(.float32) - subset.asType(.float32)).max()
        .item(Float.self)
    let fullArgmax = argMax(full, axis: -1).reshaped(-1).asArray(Int.self)
    let subsetArgmax = projection.idArray[argMax(subset, axis: -1)].reshaped(-1)
        .asArray(Int.self)
    let ids = Set(projection.vocabulary.ids.map(Int.init))
    var checked = 0
    for (f, s) in zip(fullArgmax, subsetArgmax) where ids.contains(f) {
        #expect(f == s)
        checked += 1
    }
    return (maxDiff, checked)
}

@Suite(.serialized)
struct DraftVocabularyProjectionTests {
    let vocab = 2048

    /// About a third of the vocabulary, always including 0 and the last id.
    func vocabulary(seed: UInt64) throws -> DraftVocabulary {
        if seed == 0 {
            return try DraftVocabulary(
                ids: [0, 3, 17, 64, 65, 300, 511, 512, 1000, 1500, 2000, 2046, 2047],
                vocabularySize: vocab)
        }
        let mask = MLXRandom.uniform(0 ..< 1, [vocab], key: MLXRandom.key(seed)) .< 0.33
        let ids = (0 ..< vocab).filter { mask[$0].item(Bool.self) }
        return try DraftVocabulary(
            ids: ids + [0, vocab - 1], vocabularySize: vocab, requiredIds: [0, vocab - 1])
    }

    /// Hidden states covering decode (one row, one column), batched decode,
    /// and multi-column (prefill-like) shapes.
    /// Input widths 256 and 512: MLX's quantized matrix-vector fast path
    /// needs width % 512 == 0, so both its general and fast kernels run.
    func hiddenShapes(dims: Int, dtype: DType) -> [MLXArray] {
        [[1, 1, dims], [4, 1, dims], [1, 7, dims], [2, 33, dims]].enumerated().map {
            MLXRandom.normal($0.element, key: MLXRandom.key(UInt64(100 + $0.offset)))
                .asType(dtype)
        }
    }

    func check(
        _ label: String, projection: DraftVocabularyProjection, dims: Int, dtype: DType,
        full: (MLXArray) -> MLXArray
    ) {
        for hidden in hiddenShapes(dims: dims, dtype: dtype) {
            let f = full(hidden)
            let s = projection.logits(hidden)
            eval(f, s)
            #expect(s.shape == Array(f.shape.dropLast()) + [projection.vocabulary.ids.count])
            let (maxDiff, checked) = compareSubsetToFull(full: f, subset: s, projection: projection)
            #expect(maxDiff == 0, "\(label) \(hidden.shape): max |diff| \(maxDiff)")
            // A third of the vocabulary holds some full argmax in every
            // multi-row shape; a 13-id subset need not.
            #expect(
                checked > 0 || hidden.shape == [1, 1, dims] || projection.vocabulary.ids.count < 100
            )
        }
    }

    @Test(arguments: [256, 512])
    func linearFloat32AndBFloat16(dims: Int) throws {
        let vocabulary = try vocabulary(seed: 1)
        for dtype in [DType.float32, .bfloat16] {
            for bias in [false, true] {
                MLXRandom.seed(11)
                let linear = Linear(dims, vocab, bias: bias)
                linear.update(parameters: linear.parameters().mapValues { $0.asType(dtype) })
                eval(linear)
                let projection = try DraftVocabularyProjection(
                    vocabulary: vocabulary, linear: linear)
                check(
                    "Linear \(dtype) bias=\(bias)", projection: projection, dims: dims, dtype: dtype
                ) {
                    linear($0)
                }
            }
        }
    }

    @Test(arguments: [256, 512])
    func quantizedLinear(dims: Int) throws {
        for (bits, groupSize, seed) in [
            (4, 64, 2), (8, 64, 2), (4, 32, 2), (6, 64, 2), (4, 64, 0), (8, 64, 0),
        ] as [(Int, Int, UInt64)] {
            let vocabulary = try vocabulary(seed: seed)
            MLXRandom.seed(12)
            let linear = QuantizedLinear(
                dims, vocab, bias: false, groupSize: groupSize, bits: bits)
            eval(linear)
            let projection = try DraftVocabularyProjection(vocabulary: vocabulary, linear: linear)
            let gathered = try #require(projection.linear as? QuantizedLinear)
            #expect(gathered.bits == bits && gathered.groupSize == groupSize)
            #expect(gathered.weight.dim(0) % DraftVocabularyProjection.rowMultiple == 0)
            #expect(
                gathered.weight.dim(0) - vocabulary.ids.count
                    < DraftVocabularyProjection.rowMultiple)
            check(
                "QuantizedLinear \(bits)b g\(groupSize) K=\(vocabulary.ids.count)",
                projection: projection, dims: dims, dtype: .bfloat16
            ) {
                linear($0)
            }
        }
    }

    @Test(arguments: [256, 512])
    func tiedEmbeddingFallback(dims: Int) throws {
        let vocabulary = try vocabulary(seed: 3)
        MLXRandom.seed(13)
        let embedding = Embedding(embeddingCount: vocab, dimensions: dims)
        eval(embedding)
        let projection = try DraftVocabularyProjection(
            vocabulary: vocabulary, lmHead: nil, embedding: embedding)
        check("Embedding", projection: projection, dims: dims, dtype: .float32) {
            embedding.asLinear($0)
        }

        let quantized = QuantizedEmbedding(embedding, groupSize: 64, bits: 4)
        eval(quantized)
        let qProjection = try DraftVocabularyProjection(
            vocabulary: vocabulary, lmHead: nil, embedding: quantized)
        check("QuantizedEmbedding", projection: qProjection, dims: dims, dtype: .bfloat16) {
            quantized.asLinear($0)
        }
    }

    @Test
    func rejectsMismatchedOrUnsupportedHeads() throws {
        let vocabulary = try DraftVocabulary(ids: [1, 2], vocabularySize: 8)
        #expect(throws: DraftVocabulary.Error.headRowMismatch(vocabularySize: 8, headRows: 9)) {
            try DraftVocabularyProjection(vocabulary: vocabulary, linear: Linear(4, 9))
        }
        final class RotatedEmbedding: Embedding {
            override func asLinear(_ x: MLXArray) -> MLXArray { super.asLinear(-x) }
        }
        #expect(throws: DraftVocabulary.Error.unsupportedHead("RotatedEmbedding")) {
            try DraftVocabularyProjection(
                vocabulary: vocabulary,
                embedding: RotatedEmbedding(embeddingCount: 8, dimensions: 4))
        }
    }

    @Test
    func sampleMapsSubsetPositionsBackToIds() throws {
        let vocabulary = try DraftVocabulary(ids: [3, 5, 7], vocabularySize: 8)
        let projection = try DraftVocabularyProjection(
            vocabulary: vocabulary, linear: Linear(4, 8))
        let logits = MLXArray([Float(0), 2, 1, 9, 1, 0], [2, 3])
        let sampled = projection.sample(
            subsetLogits: logits, sampler: GenerateParameters(temperature: 0).sampler())
        #expect(sampled.asArray(Int.self) == [5, 3])
    }
}

// MARK: - Qwen-style drafter through the MTP iterator

private func qwenDraftVocabConfigJSON(tied: Bool) -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "intermediate_size": 64,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 16,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 16,
      "linear_value_head_dim": 16,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 64,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 256,
      "tie_word_embeddings": \(tied),
      "attention_bias": false,
      "full_attention_interval": 2,
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

@Suite(.serialized)
struct Qwen35DraftVocabularyTests {
    func greedyParameters(maxTokens: Int) -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefill.stepSize = Int.max
        parameters.prefill.chunking = .unchunked
        return parameters
    }

    func models(tied: Bool) throws -> (Qwen35TextModel, Qwen35MTPDraftModel) {
        let cfg = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(qwenDraftVocabConfigJSON(tied: tied).utf8))
        MLXRandom.seed(2428)
        let target = Qwen35TextModel(cfg)
        let drafter = Qwen35MTPDraftModel(cfg)
        eval(target, drafter)
        return (target, drafter)
    }

    func run(
        target: Qwen35TextModel, drafter: Qwen35MTPDraftModel, prompt: MLXArray, count: Int
    ) throws -> (tokens: [Int], proposed: Int, accepted: Int) {
        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: target, drafter: drafter,
            parameters: greedyParameters(maxTokens: count), blockSize: 2)
        var produced = [Int]()
        while let token = speculative.next() { produced.append(token) }
        #expect(speculative.passthroughReason == nil)
        return (produced, speculative.proposedCount, speculative.acceptedCount)
    }

    @Test(arguments: [true, false])
    func subsetDraftingKeepsGreedyOutput(tied: Bool) throws {
        let (target, drafter) = try models(tied: tied)
        let prompt = MLXRandom.randInt(0 ..< 64, [11], key: MLXRandom.key(5)).asType(.int32)
        var plain = try TokenIterator(
            prompt: prompt, model: target, parameters: greedyParameters(maxTokens: 24))
        var expected = [Int]()
        while let token = plain.next() { expected.append(token) }

        // nil: today's path.
        #expect(drafter.draftVocabulary == nil)
        let baseline = try run(target: target, drafter: drafter, prompt: prompt, count: 24)
        #expect(baseline.tokens == expected)

        // The whole vocabulary as a subset drafts exactly what nil drafts.
        try drafter.installDraftVocabulary(
            DraftVocabulary(ids: 0 ..< 64, vocabularySize: 64), target: target)
        let whole = try run(target: target, drafter: drafter, prompt: prompt, count: 24)
        #expect(whole.tokens == expected)
        #expect(whole.proposed == baseline.proposed && whole.accepted == baseline.accepted)

        // A strict subset: output unchanged, only acceptance may move.
        try drafter.installDraftVocabulary(
            DraftVocabulary(ids: stride(from: 0, to: 64, by: 3), vocabularySize: 64),
            target: target)
        let subset = try run(target: target, drafter: drafter, prompt: prompt, count: 24)
        #expect(subset.tokens == expected)
        #expect(subset.proposed > 0)

        // Uninstall restores the full head.
        try drafter.installDraftVocabulary(nil, target: target)
        #expect(drafter.draftVocabulary == nil)
        let restored = try run(target: target, drafter: drafter, prompt: prompt, count: 24)
        #expect(restored == baseline)
    }

    /// Every proposal of the subset drafter lies in the subset, through the
    /// serial commit and the packed advance.
    @Test
    func proposalsStayInSubset() throws {
        let (target, drafter) = try models(tied: false)
        let vocabulary = try DraftVocabulary(ids: [1, 4, 9, 16, 25, 36, 49], vocabularySize: 64)
        try drafter.installDraftVocabulary(vocabulary, target: target)
        let allowed = Set(vocabulary.ids.map(Int.init))
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(3), 7, 11, 13]).reshaped(1, 4)
        let hidden = MLXRandom.normal([1, 4, 32], key: MLXRandom.key(9))

        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(2)]), positionDeltas: nil, state: &state,
            sampler: sampler)
        #expect(allowed.contains(state.seedToken!.item(Int.self)))
        let seed = drafter.draftBlock(
            target: target, lastToken: MLXArray([Int32(2)]), lastHidden: hidden[0..., (-1)...],
            sharedKV: [:], positionDeltas: nil, queryOffset: 4, blockSize: 2, state: &state,
            sampler: sampler)
        #expect(allowed.contains(seed.item(Int.self)))

        let packed = try drafter.advanceAndProposePacked(
            target: target,
            rows: [
                MTPPackedDrafterAdvanceRow(
                    targetHidden: MLXRandom.normal([1, 2, 32]),
                    acceptedTokens: [seed.item(Int.self)], finalToken: 5, positionDeltas: nil,
                    state: state),
                MTPPackedDrafterAdvanceRow(
                    targetHidden: MLXRandom.normal([1, 2, 32]), acceptedTokens: [],
                    finalToken: 6, positionDeltas: nil, state: state),
            ], sampler: sampler)
        for token in packed.proposals.reshaped(-1).asArray(Int.self) {
            #expect(allowed.contains(token))
        }
    }
}
