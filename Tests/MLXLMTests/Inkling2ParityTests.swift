import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Inkling 2 against the mlx-vlm oracle (Blaizzy/mlx-vlm@f1618d6d). The fixtures
/// (`Resources/inkling2/tiny-*`, made on the Studio by `tools/inkling2_parity/make_tiny_fixture.py`)
/// hold random float32 weights for an 8-layer loop config (core run three times, 16 visits),
/// 4 experts, an 8-token window and a 6-token global band, all shorter than the 24-token prompt.
///
/// - tiny-source: source checkpoint layout, NVFP4 (per-expert amax) and MXFP8 experts.
/// - tiny-oq: the same weights in the oQ layout; the same oracle.
/// - tiny-oq-affine: oQ layout with affine 8-bit attention, embedding and head, NVFP4 and
///   affine 4-bit experts.
///
/// Tolerance: max |Δlogit| < 1e-3 over every row, and identical greedy tokens. Runs on the CPU,
/// as the oracle does; `INKLING2_PARITY_DEVICE=gpu` runs it on the GPU.
final class Inkling2ParityTests: XCTestCase {

    private struct Oracle: Decodable {
        struct Case: Decodable {
            let prefillStep: Int
            let greedy: [Int]
            let logits: [[Float]]

            enum CodingKeys: String, CodingKey {
                case greedy, logits
                case prefillStep = "prefill_step"
            }
        }
        let prompt: [Int]
        let fullLogits: [[Float]]
        let cases: [Case]

        enum CodingKeys: String, CodingKey {
            case prompt, cases
            case fullLogits = "full_logits"
        }
    }

    private static let tolerance: Float = 1e-3

    private var device: Device {
        ProcessInfo.processInfo.environment["INKLING2_PARITY_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func fixture(_ name: String) throws -> URL {
        let root = try XCTUnwrap(Bundle.module.url(forResource: "inkling2", withExtension: nil))
        let directory = root.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("oracle.json").path)
        else {
            throw XCTSkip(
                "\(name) fixture absent; generate it on the Studio with "
                    + "`python3 -I tools/inkling2_parity/make_tiny_fixture.py --mlx-vlm <mlx-vlm@f1618d6d>`")
        }
        return directory
    }

    private func load(_ directory: URL) throws -> Inkling2Model {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
        try configuration.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = Inkling2Model(configuration, quantization: base.perLayerQuantization)
        // tiny-source carries the source checkpoint's sidecar; the oQ fixtures have none.
        try model.applyQuantizationMetadata(directory: directory)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        return model
    }

    private func oracle(_ directory: URL) throws -> Oracle {
        try JSONDecoder().decode(
            Oracle.self, from: Data(contentsOf: directory.appendingPathComponent("oracle.json")))
    }

    private func maxDifference(_ mine: [[Float]], _ theirs: [[Float]]) -> Float {
        XCTAssertEqual(mine.count, theirs.count)
        return zip(mine, theirs).map { a, b in
            XCTAssertEqual(a.count, b.count)
            return zip(a, b).map { abs($0 - $1) }.max() ?? 0
        }.max() ?? 0
    }

    private func rows(_ array: MLXArray) -> [[Float]] {
        (0 ..< array.dim(0)).map { array[$0].asType(.float32).asArray(Float.self) }
    }

    private func decode(_ model: Inkling2Model, prompt: [Int], step: Int, count: Int) throws -> (
        [Int], [[Float]]
    ) {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            let chunk = MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n)
            eval(model(chunk, cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        var input = MLXArray(rest.map(Int32.init)).reshaped(1, rest.count)
        var tokens = [Int]()
        var logits = [[Float]]()
        for _ in 0 ..< count {
            let row = model(input, cache: cache)[0, -1]
            eval(row)
            eval(cache)
            let token = argMax(row, axis: -1).item(Int.self)
            tokens.append(token)
            logits.append(row.asType(.float32).asArray(Float.self))
            input = MLXArray([Int32(token)]).reshaped(1, 1)
        }
        return (tokens, logits)
    }

    private func checkParity(_ name: String) throws {
        let directory = try fixture(name)
        try Device.withDefaultDevice(device) {
            let model = try load(directory)
            let expected = try oracle(directory)
            XCTAssertGreaterThan(expected.prompt.count, model.configuration.text.slidingWindow)

            let input = MLXArray(expected.prompt.map(Int32.init)).reshaped(1, expected.prompt.count)
            let full = maxDifference(rows(model(input, cache: nil)[0]), expected.fullLogits)
            print("inkling2 parity \(name) uncached: max |dlogit| = \(full)")
            XCTAssertLessThan(full, Self.tolerance, "\(name) uncached")

            for item in expected.cases {
                let (tokens, logits) = try decode(
                    model, prompt: expected.prompt, step: item.prefillStep,
                    count: item.greedy.count)
                XCTAssertEqual(tokens, item.greedy, "\(name) prefill step \(item.prefillStep)")
                let difference = maxDifference(logits, item.logits)
                print("inkling2 parity \(name) step \(item.prefillStep): max |dlogit| = \(difference)")
                XCTAssertLessThan(difference, Self.tolerance, "\(name) prefill step \(item.prefillStep)")
            }
        }
    }

    func testSourceLayoutMatchesOracle() throws {
        try checkParity("tiny-source")
    }

    func testOQLayoutMatchesOracle() throws {
        try checkParity("tiny-oq")
    }

    func testOQAffineLayoutMatchesOracle() throws {
        try checkParity("tiny-oq-affine")
    }
}

/// Real-weight parity against `tools/inkling2_parity/dump_real_oracle.py` (mlx-vlm f1618d6d,
/// 1024 and 8192 tokens, prefill 2048). Needs `INKLING2_REAL_MODEL` (the source checkpoint
/// directory, with `hf_quant_config.json`, or the oQ4e-mtp directory) and `INKLING2_REAL_ORACLE`
/// (the dump's output directory for that checkpoint). The layout is read from the checkpoint's
/// index with the loader's own `Inkling2WeightLayout.kind`; `INKLING2_REAL_LAYOUT` (`source` or
/// `oq`), when set, must agree, and so must the layout the oracle's `oracle.json` records
/// (oracles written before it recorded one are source oracles). `INKLING2_REAL_MODE=exact` (the
/// default) requires every greedy token to match; `report` only records the comparison.
/// `INKLING2_REAL_REPORT` writes the numbers as JSON. Runs on the GPU unless
/// `INKLING2_REAL_DEVICE=cpu`; use the oracle's device.
///
/// The path follows ``Inkling2FastPath``: compiled and fused by default, the plain op-by-op
/// path (mlx-vlm's portable path) with `INKLING2_COMPILED=0`.
///
/// Where the logits differ, per decode step: the largest |Δlogit| and the vocabulary id it sits
/// at, that id's rank and logit in the oracle row, the largest |Δlogit| over the oracle's top
/// ``topK`` ids, the mean |Δlogit| and the KL divergence. The worst step is repeated at the top
/// of each row, so a tail-only difference (large max, small top-k difference and KL) is told
/// apart from one bad step or one bad vocabulary row.
final class Inkling2RealParityTests: XCTestCase {
    private struct Step: Encodable {
        let maxLogitDifference: Float
        let maxDifferenceVocabularyId: Int
        /// The oracle's logit at that id, ours, and the id's rank in the oracle row (0 = top).
        let referenceLogit: Float
        let portLogit: Float
        let referenceRank: Int
        let topKLogitDifference: Float
        let meanAbsoluteLogitDifference: Float
        let kl: Float
    }

    private struct Row: Encodable {
        let layout: String
        /// `compiled` (``Inkling2FastPath``, the default) or `plain` (`INKLING2_COMPILED=0`).
        let path: String
        let length: Int
        let steps: Int
        let firstMismatch: Int?
        let matchingTokens: Int
        let meanKL: Float
        let maxKL: Float
        let maxKLStep: Int
        let maxLogitDifference: Float
        let maxDifferenceStep: Int
        let maxDifferenceVocabularyId: Int
        let maxDifferenceReferenceLogit: Float
        let maxDifferencePortLogit: Float
        let maxDifferenceReferenceRank: Int
        /// Largest |Δlogit| over the oracle's top-k ids, across all steps.
        let maxTopKLogitDifference: Float
        let topK: Int
        let prefillSeconds: Double
        /// Forward, argmax and cache evaluation only; the diagnostics above are not timed.
        let decodeTokensPerSecond: Double
        let perStep: [Step]
    }

    /// The oracle's schedule: chunks of 2048 over all prompt tokens but the last.
    private static let prefillStep = 2048
    private static let topK = 20

    /// `source` or `oq`, from the tensor names in `model.safetensors.index.json`.
    private static func layout(of directory: URL) throws -> String {
        let url = directory.appendingPathComponent("model.safetensors.index.json")
        guard
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
            let map = index["weight_map"] as? [String: Any]
        else {
            throw XCTSkip("\(url.path) has no weight_map")
        }
        switch try Inkling2WeightLayout.kind(of: map.keys.filter { !Inkling2WeightLayout.isDropped($0) }) {
        case .oq: return "oq"
        case .source: return "source"
        }
    }

    /// The layout `oracle.json` records; oracles from before slice 4 are source oracles.
    private static func oracleLayout(_ directory: URL) throws -> String {
        let url = directory.appendingPathComponent("oracle.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return "source" }
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        return meta?["layout"] as? String ?? "source"
    }

    /// Compares one row of our logits with the oracle's row; both float32 `[vocabulary]`.
    private func compare(_ row: MLXArray, _ reference: MLXArray) -> Step {
        let referenceLog = reference - logSumExp(reference, axis: -1, keepDims: true)
        let log = row - logSumExp(row, axis: -1, keepDims: true)
        let kl = (exp(referenceLog) * (referenceLog - log)).sum()
        let difference = abs(row - reference)
        let worst = argMax(difference)
        let top = argPartition(-reference, kth: Self.topK - 1)[..<Self.topK]
        let topDifference = take(difference, top).max()
        let referenceAtWorst = take(reference, worst)
        let rank = (reference .> referenceAtWorst).asType(.int32).sum()
        let mean = difference.mean()
        eval(kl, worst, topDifference, referenceAtWorst, rank, mean)
        let id = worst.item(Int.self)
        return Step(
            maxLogitDifference: difference[id].item(Float.self),
            maxDifferenceVocabularyId: id,
            referenceLogit: referenceAtWorst.item(Float.self),
            portLogit: row[id].item(Float.self),
            referenceRank: rank.item(Int.self),
            topKLogitDifference: topDifference.item(Float.self),
            meanAbsoluteLogitDifference: mean.item(Float.self),
            kl: kl.item(Float.self))
    }

    func testGreedyTokensMatchMLXVLMOracle() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["INKLING2_REAL_MODEL"],
            let oraclePath = environment["INKLING2_REAL_ORACLE"]
        else {
            throw XCTSkip("INKLING2_REAL_MODEL and INKLING2_REAL_ORACLE are not set")
        }
        let exact = (environment["INKLING2_REAL_MODE"] ?? "exact") == "exact"
        let device: Device = environment["INKLING2_REAL_DEVICE"] == "cpu" ? .cpu : .gpu

        try Device.withDefaultDevice(device) {
            let directory = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)
            let oracleDirectory = URL(fileURLWithPath: (oraclePath as NSString).expandingTildeInPath)
            let layout = try Self.layout(of: directory)
            if let wanted = environment["INKLING2_REAL_LAYOUT"] {
                XCTAssertEqual(layout, wanted, "INKLING2_REAL_LAYOUT disagrees with \(modelPath)")
                guard layout == wanted else { return }
            }
            let oracleLayout = try Self.oracleLayout(oracleDirectory)
            XCTAssertEqual(oracleLayout, layout, "the oracle in \(oraclePath) is for the other layout")
            guard oracleLayout == layout else { return }
            print("inkling2 real parity layout=\(layout) model=\(modelPath) oracle=\(oraclePath)")
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let model = Inkling2Model(configuration, quantization: base.perLayerQuantization)
            try model.applyQuantizationMetadata(directory: directory)
            try loadWeights(
                modelDirectory: directory, model: model,
                perLayerQuantization: base.perLayerQuantization)
            model.train(false)

            let oracles = try FileManager.default.contentsOfDirectory(
                at: oracleDirectory, includingPropertiesForKeys: nil
            )
            .filter { $0.lastPathComponent.hasPrefix("oracle-") && $0.pathExtension == "safetensors" }
            .sorted { $0.path.compare($1.path, options: .numeric) == .orderedAscending }
            XCTAssertFalse(oracles.isEmpty, "no oracle files in \(oraclePath)")

            var report = [Row]()
            for url in oracles {
                let arrays = try loadArrays(url: url)
                let prompt = try XCTUnwrap(arrays["prompt"])
                let expected = try XCTUnwrap(arrays["greedy"]).asArray(Int32.self).map(Int.init)
                let logits = try XCTUnwrap(arrays["logits"])

                let cache = try model.newCache(parameters: nil)
                let begin = Date()
                guard
                    case .tokens(let rest) = try model.prepare(
                        LMInput(tokens: prompt), cache: cache,
                        prefill: PrefillParameters(stepSize: Self.prefillStep, chunking: .remainder))
                else {
                    XCTFail("prepare should hand tokens to the iterator")
                    continue
                }
                eval(cache)
                let prefillSeconds = Date().timeIntervalSince(begin)
                XCTAssertEqual(rest.tokens.size, 1, "prepare leaves the last prompt token")

                var input = rest.tokens.reshaped(1, rest.tokens.size)
                var tokens = [Int]()
                var steps = [Step]()
                var decodeSeconds = 0.0
                for step in expected.indices {
                    let stepBegin = Date()
                    let row = model(input, cache: cache)[0, -1].asType(.float32)
                    eval(row)
                    eval(cache)
                    tokens.append(argMax(row, axis: -1).item(Int.self))
                    decodeSeconds += Date().timeIntervalSince(stepBegin)

                    steps.append(compare(row, logits[step]))

                    // Follow the oracle's token so one miss does not corrupt later comparisons.
                    input = MLXArray([Int32(expected[step])]).reshaped(1, 1)
                }

                let worst = steps.indices.max { steps[$0].maxLogitDifference < steps[$1].maxLogitDifference } ?? 0
                let worstKL = steps.indices.max { steps[$0].kl < steps[$1].kl } ?? 0
                let result = Row(
                    layout: layout, path: Inkling2FastPath.enabled ? "compiled" : "plain",
                    length: prompt.size, steps: expected.count,
                    firstMismatch: zip(tokens, expected).enumerated()
                        .first { $0.element.0 != $0.element.1 }?.offset,
                    matchingTokens: zip(tokens, expected).filter { $0 == $1 }.count,
                    meanKL: steps.map(\.kl).reduce(0, +) / Float(max(steps.count, 1)),
                    maxKL: steps.isEmpty ? 0 : steps[worstKL].kl,
                    maxKLStep: worstKL,
                    maxLogitDifference: steps.isEmpty ? 0 : steps[worst].maxLogitDifference,
                    maxDifferenceStep: worst,
                    maxDifferenceVocabularyId: steps.isEmpty ? -1 : steps[worst].maxDifferenceVocabularyId,
                    maxDifferenceReferenceLogit: steps.isEmpty ? 0 : steps[worst].referenceLogit,
                    maxDifferencePortLogit: steps.isEmpty ? 0 : steps[worst].portLogit,
                    maxDifferenceReferenceRank: steps.isEmpty ? -1 : steps[worst].referenceRank,
                    maxTopKLogitDifference: steps.map(\.topKLogitDifference).max() ?? 0,
                    topK: Self.topK,
                    prefillSeconds: prefillSeconds,
                    decodeTokensPerSecond: Double(expected.count) / decodeSeconds,
                    perStep: steps)
                report.append(result)
                let mismatch = result.firstMismatch.map { "\($0)" } ?? "none"
                let stepMaxima = steps.map { ($0.maxLogitDifference * 100).rounded() / 100 }
                let fields: [String] = [
                    "inkling2 real parity layout=\(layout) path=\(result.path) length=\(result.length)",
                    "matching=\(result.matchingTokens)/\(result.steps) firstMismatch=\(mismatch)",
                    "meanKL=\(result.meanKL) maxKL=\(result.maxKL)@step\(result.maxKLStep)",
                    "maxDiff=\(result.maxLogitDifference)@step\(result.maxDifferenceStep)/id\(result.maxDifferenceVocabularyId)",
                    "oracleLogit=\(result.maxDifferenceReferenceLogit) oracleRank=\(result.maxDifferenceReferenceRank)",
                    "portLogit=\(result.maxDifferencePortLogit) top\(Self.topK)Diff=\(result.maxTopKLogitDifference)",
                    "prefill=\(result.prefillSeconds)s decode=\(result.decodeTokensPerSecond)tok/s",
                    "stepMaxDiffs=\(stepMaxima)",
                ]
                print(fields.joined(separator: " "))
                if exact {
                    XCTAssertEqual(tokens, expected, "greedy tokens at prompt length \(prompt.size)")
                }
            }

            if let path = environment["INKLING2_REAL_REPORT"] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: URL(fileURLWithPath: path))
            }
        }
    }
}
