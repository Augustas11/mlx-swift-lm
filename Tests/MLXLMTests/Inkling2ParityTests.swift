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
