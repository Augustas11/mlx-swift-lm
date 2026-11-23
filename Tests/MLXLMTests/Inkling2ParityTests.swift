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

/// Real-weight parity against `tools/inkling2_parity/dump_real_oracle.py` (mlx-vlm f1618d6d on
/// the source checkpoint, 1024 and 8192 tokens, prefill 2048). Needs `INKLING2_REAL_MODEL` (the
/// source checkpoint directory, with `hf_quant_config.json`) and `INKLING2_REAL_ORACLE` (the
/// dump's output directory). `INKLING2_REAL_MODE=exact` (the default) requires every greedy
/// token to match; `report` only records the comparison. `INKLING2_REAL_REPORT` writes the
/// numbers as JSON. Runs on the GPU unless `INKLING2_REAL_DEVICE=cpu`; use the oracle's device.
final class Inkling2RealParityTests: XCTestCase {
    private struct Row: Encodable {
        let length: Int
        let steps: Int
        let firstMismatch: Int?
        let matchingTokens: Int
        let meanKL: Float
        let maxLogitDifference: Float
        let prefillSeconds: Double
        let decodeTokensPerSecond: Double
    }

    /// The oracle's schedule: chunks of 2048 over all prompt tokens but the last.
    private static let prefillStep = 2048

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

            let oracleDirectory = URL(fileURLWithPath: (oraclePath as NSString).expandingTildeInPath)
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
                var divergences = [Float]()
                var worst: Float = 0
                let decodeBegin = Date()
                for step in expected.indices {
                    let row = model(input, cache: cache)[0, -1].asType(.float32)
                    eval(row)
                    eval(cache)
                    tokens.append(argMax(row, axis: -1).item(Int.self))

                    let reference = logits[step]
                    let referenceLog = reference - logSumExp(reference, axis: -1, keepDims: true)
                    let log = row - logSumExp(row, axis: -1, keepDims: true)
                    divergences.append(
                        (exp(referenceLog) * (referenceLog - log)).sum().item(Float.self))
                    worst = max(worst, abs(row - reference).max().item(Float.self))

                    // Follow the oracle's token so one miss does not corrupt later comparisons.
                    input = MLXArray([Int32(expected[step])]).reshaped(1, 1)
                }
                let decodeSeconds = Date().timeIntervalSince(decodeBegin)

                let result = Row(
                    length: prompt.size, steps: expected.count,
                    firstMismatch: zip(tokens, expected).enumerated()
                        .first { $0.element.0 != $0.element.1 }?.offset,
                    matchingTokens: zip(tokens, expected).filter { $0 == $1 }.count,
                    meanKL: divergences.reduce(0, +) / Float(max(divergences.count, 1)),
                    maxLogitDifference: worst,
                    prefillSeconds: prefillSeconds,
                    decodeTokensPerSecond: Double(expected.count) / decodeSeconds)
                report.append(result)
                print("inkling2 real parity \(result)")
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
