// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A subset of token ids the MTP drafter scores (issue 2428).
///
/// The drafter computes logits over these ids only, through output-layer rows
/// gathered once (``DraftVocabularyProjection``); the target still verifies
/// every draft over the full vocabulary, so greedy output is unchanged and
/// only acceptance can move. A draft the target would not pick is rejected as
/// usual; a token outside the subset is never drafted but is still emitted
/// by the target's own verification column.
///
/// Sidecar file (JSON, UTF-8), `format` and `version` checked on load:
///
/// ```json
/// {"format": "mtp-draft-vocabulary", "version": 1, "vocab_size": 151936,
///  "ids": [0, 1, 2, 13, 151643, 151645]}
/// ```
///
/// `vocab_size` is the number of output rows of the head the subset is
/// gathered from (the lm_head or tied embedding row count, which may exceed
/// the tokenizer's vocabulary). `ids` are stored sorted and unique.
public struct DraftVocabulary: Sendable, Equatable {
    public static let format = "mtp-draft-vocabulary"
    public static let version = 1

    /// Sorted, unique token ids.
    public let ids: [Int32]
    /// Output rows of the full head.
    public let vocabularySize: Int

    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        case empty
        case invalidVocabularySize(Int)
        case idOutOfRange(id: Int, vocabularySize: Int)
        case missingRequiredIds([Int])
        case unsupportedFormat(String)
        case headRowMismatch(vocabularySize: Int, headRows: Int)
        case unsupportedHead(String)

        public var description: String {
            switch self {
            case .empty: "draft vocabulary is empty"
            case .invalidVocabularySize(let n): "draft vocabulary size \(n) is not positive"
            case .idOutOfRange(let id, let n): "draft vocabulary id \(id) is outside 0..<\(n)"
            case .missingRequiredIds(let ids): "draft vocabulary misses required ids \(ids)"
            case .unsupportedFormat(let s): "unsupported draft vocabulary sidecar: \(s)"
            case .headRowMismatch(let n, let rows):
                "draft vocabulary is for \(n) output rows, the head has \(rows)"
            case .unsupportedHead(let type): "cannot gather draft rows from a \(type) head"
            }
        }
    }

    /// Validates and canonicalizes (sorts, removes duplicates) `ids`.
    /// `requiredIds` (EOS and other special ids) must all be present.
    public init<S: Sequence>(ids: S, vocabularySize: Int, requiredIds: [Int] = []) throws
    where S.Element: BinaryInteger {
        guard vocabularySize > 0 else { throw Error.invalidVocabularySize(vocabularySize) }
        var unique = Set<Int>()
        for id in ids {
            let id = Int(id)
            guard (0 ..< vocabularySize).contains(id) else {
                throw Error.idOutOfRange(id: id, vocabularySize: vocabularySize)
            }
            unique.insert(id)
        }
        guard !unique.isEmpty else { throw Error.empty }
        let missing = Set(requiredIds).subtracting(unique).sorted()
        guard missing.isEmpty else { throw Error.missingRequiredIds(missing) }
        self.ids = unique.sorted().map(Int32.init)
        self.vocabularySize = vocabularySize
    }

    private struct Sidecar: Codable {
        var format: String
        var version: Int
        var vocabSize: Int
        var ids: [Int]

        enum CodingKeys: String, CodingKey {
            case format, version, ids
            case vocabSize = "vocab_size"
        }
    }

    /// Loads and validates a sidecar file (format above).
    public init(contentsOf url: URL, requiredIds: [Int] = []) throws {
        try self.init(jsonData: Data(contentsOf: url), requiredIds: requiredIds)
    }

    public init(jsonData: Data, requiredIds: [Int] = []) throws {
        let sidecar = try JSONDecoder().decode(Sidecar.self, from: jsonData)
        guard sidecar.format == Self.format, sidecar.version == Self.version else {
            throw Error.unsupportedFormat("\(sidecar.format) v\(sidecar.version)")
        }
        try self.init(
            ids: sidecar.ids, vocabularySize: sidecar.vocabSize, requiredIds: requiredIds)
    }

    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(
            Sidecar(
                format: Self.format, version: Self.version, vocabSize: vocabularySize,
                ids: ids.map(Int.init)))
    }

    public func write(to url: URL) throws {
        try jsonData().write(to: url, options: .atomic)
    }

    /// The `topN` most frequent ids across `sequences` (ties broken by the
    /// smaller id), united with `specialIds`. Ids outside the vocabulary are
    /// rejected.
    public static func mostFrequent<S: Sequence>(
        in sequences: S, topN: Int, vocabularySize: Int, specialIds: [Int]
    ) throws -> DraftVocabulary where S.Element: Sequence, S.Element.Element: BinaryInteger {
        var counts = [Int: Int]()
        for sequence in sequences {
            for id in sequence {
                let id = Int(id)
                guard (0 ..< vocabularySize).contains(id) else {
                    throw Error.idOutOfRange(id: id, vocabularySize: vocabularySize)
                }
                counts[id, default: 0] += 1
            }
        }
        let top = counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .prefix(max(topN, 0)).map(\.key)
        return try DraftVocabulary(
            ids: top + specialIds, vocabularySize: vocabularySize, requiredIds: specialIds)
    }

    /// Token-id sequences from JSONL: each non-empty line is either a JSON
    /// array of ids or an object with a `token_ids` (or `tokens`) array.
    public static func sequences(jsonl data: Data) throws -> [[Int]] {
        struct Line: Decodable {
            var tokenIds: [Int]?
            var tokens: [Int]?
            enum CodingKeys: String, CodingKey {
                case tokens
                case tokenIds = "token_ids"
            }
        }
        let decoder = JSONDecoder()
        var result = [[Int]]()
        for line in data.split(separator: UInt8(ascii: "\n")) {
            let line = Data(line)
            guard line.contains(where: { !$0.isWhitespaceByte }) else { continue }
            if let ids = try? decoder.decode([Int].self, from: line) {
                result.append(ids)
                continue
            }
            let object = try decoder.decode(Line.self, from: line)
            guard let ids = object.tokenIds ?? object.tokens else {
                throw Error.unsupportedFormat("JSONL line without token_ids or tokens")
            }
            result.append(ids)
        }
        return result
    }
}

extension UInt8 {
    fileprivate var isWhitespaceByte: Bool {
        self == 0x20 || self == 0x09 || self == 0x0A || self == 0x0D
    }
}

/// The output head restricted to a ``DraftVocabulary``: the head's rows for
/// those ids, gathered once at install (output dimension is the row axis of
/// `Linear.weight` `[out, in]`, of `QuantizedLinear.weight/scales/biases`
/// `[out, packed]`, and of an embedding table `[vocab, dim]`). Group size,
/// bits, mode and global scale are kept, so each subset logit runs the same
/// kernel class over the same row as the full head. The gathered rows are
/// padded to a multiple of ``rowMultiple`` by repeating the last id (the
/// logits are cut back to `K`): matrix-vector kernels pick their fast path
/// by output-row count (MLX `qmv_fast`, the GLM fused QMV need rows % 8),
/// and a full vocabulary normally meets it.
///
/// Drafted tokens are `ids[sample(subsetLogits)]`. Greedy drafting proposes
/// the full argmax whenever that id is in the subset. Sampled drafting draws
/// from the distribution renormalized over the subset, not the full one; the
/// target-side verification is unchanged either way.
public struct DraftVocabularyProjection: @unchecked Sendable {
    public let vocabulary: DraftVocabulary
    /// `vocabulary.ids` as an int32 array `[K]`.
    public let idArray: MLXArray
    let head: Head

    /// Output rows of the gathered head are padded to a multiple of this.
    public static let rowMultiple = 8

    /// `ids` padded with the last id to a multiple of ``rowMultiple``.
    private static func gatherIds(_ vocabulary: DraftVocabulary) -> MLXArray {
        let ids = vocabulary.ids
        let pad = (rowMultiple - ids.count % rowMultiple) % rowMultiple
        return MLXArray(ids + Array(repeating: ids[ids.count - 1], count: pad))
    }

    enum Head {
        case linear(Linear)
        case embedding(Embedding)
    }

    /// Gathers from `lmHead` when present, else from the tied `embedding`
    /// (`embedding.asLinear`), matching the full head the drafter uses.
    public init(vocabulary: DraftVocabulary, lmHead: Linear?, embedding: Embedding) throws {
        if let lmHead {
            try self.init(vocabulary: vocabulary, linear: lmHead)
        } else {
            try self.init(vocabulary: vocabulary, embedding: embedding)
        }
    }

    public init(vocabulary: DraftVocabulary, linear: Linear) throws {
        let ids = Self.gatherIds(vocabulary)
        try Self.checkRows(vocabulary, linear.weight.dim(0))
        let sliced: Linear
        switch linear {
        case let q as QuantizedLinear where type(of: q) == QuantizedLinear.self:
            sliced = QuantizedLinear(
                weight: q.weight[ids], bias: q.bias?[ids], scales: q.scales[ids],
                biases: q.biases?[ids], groupSize: q.groupSize, bits: q.bits, mode: q.mode,
                globalScale: q.globalScale, referencePrefill: q.referencePrefill)
        case let l where type(of: l) == Linear.self:
            sliced = Linear(
                weight: l.weight[ids], bias: l.bias?[ids], referencePrefill: l.referencePrefill)
        default:
            throw DraftVocabulary.Error.unsupportedHead(String(describing: type(of: linear)))
        }
        sliced.train(false)
        eval(sliced)
        self.vocabulary = vocabulary
        self.idArray = MLXArray(vocabulary.ids)
        self.head = .linear(sliced)
    }

    public init(vocabulary: DraftVocabulary, embedding: Embedding) throws {
        let ids = Self.gatherIds(vocabulary)
        try Self.checkRows(vocabulary, embedding.weight.dim(0))
        let sliced: Embedding
        switch embedding {
        case let q as QuantizedEmbedding where type(of: q) == QuantizedEmbedding.self:
            sliced = QuantizedEmbedding(
                weight: q.weight[ids], scales: q.scales[ids], biases: q.biases?[ids],
                groupSize: q.groupSize, bits: q.bits, mode: q.mode, globalScale: q.globalScale)
        case let e where type(of: e) == Embedding.self:
            sliced = Embedding(weight: e.weight[ids])
        default:
            // Subclasses (e.g. a Hadamard-rotated table) override asLinear;
            // a plain row gather would not reproduce their logits.
            throw DraftVocabulary.Error.unsupportedHead(String(describing: type(of: embedding)))
        }
        eval(sliced)
        self.vocabulary = vocabulary
        self.idArray = MLXArray(vocabulary.ids)
        self.head = .embedding(sliced)
    }

    private static func checkRows(_ vocabulary: DraftVocabulary, _ rows: Int) throws {
        guard rows == vocabulary.vocabularySize else {
            throw DraftVocabulary.Error.headRowMismatch(
                vocabularySize: vocabulary.vocabularySize, headRows: rows)
        }
    }

    /// The gathered head as a `Linear`, when gathered from one (its rows
    /// include the padding).
    public var linear: Linear? {
        if case .linear(let linear) = head { return linear }
        return nil
    }

    /// Subset logits `[..., K]`; column `k` is the full head's column
    /// `ids[k]`. `project` applies a linear head (models with their own
    /// projection kernel pass it here).
    public func logits(
        _ hidden: MLXArray, project: (Linear, MLXArray) -> MLXArray = { $0($1) }
    ) -> MLXArray {
        let logits =
            switch head {
            case .linear(let linear): project(linear, hidden)
            case .embedding(let embedding): embedding.asLinear(hidden)
            }
        let count = vocabulary.ids.count
        return logits.dim(-1) == count ? logits : logits[.ellipsis, ..<count]
    }

    /// Samples from subset logits and maps the subset positions back to
    /// token ids, keeping the sampler's output shape and dtype.
    public func sample(subsetLogits: MLXArray, sampler: any LogitSampler) -> MLXArray {
        let positions = sampler.sample(logits: subsetLogits)
        return idArray[positions].asType(positions.dtype)
    }
}

/// A drafter whose draft head can be restricted to a ``DraftVocabulary``.
///
/// Call once after the target's weights are final (after loading and any
/// quantization), before generation: the rows are gathered from the target's
/// head at this call. `nil` restores the full head. Not safe to call while
/// the drafter is generating.
public protocol MTPDraftVocabularyInstallable: AnyObject {
    var draftVocabulary: DraftVocabulary? { get }
    func installDraftVocabulary(_ vocabulary: DraftVocabulary?, target: any LanguageModel) throws
}
