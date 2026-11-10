import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Single-stream serving throughput through the production `generate()` loop (the
/// `TokenIterator` path that serving uses), for `tools/ricursive_parity/bench_job.sh`.
/// Needs `RICURSIVE_BENCH_MODEL`, `RICURSIVE_BENCH_PROMPT` (a JSON array of token ids)
/// and `RICURSIVE_BENCH_REPORT`. Timing matches `bench_mlx_lm.py`: prefill tok/s is
/// prompt tokens over the time to the first token, decode tok/s is the remaining
/// tokens over the time from the first to the last token.
final class RicursiveServingBenchTests: XCTestCase {
    /// No end-of-sequence or unknown token, so every run decodes the full count.
    private struct NoStopTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private struct Run: Encodable {
        let prefillTokensPerSecond: Double
        let decodeTokensPerSecond: Double
        let generatedTokens: Int
        let peakMemoryGB: Double

        enum CodingKeys: String, CodingKey {
            case prefillTokensPerSecond = "prefill_tokens_per_second"
            case decodeTokensPerSecond = "decode_tokens_per_second"
            case generatedTokens = "generated_tokens"
            case peakMemoryGB = "peak_memory_gb"
        }
    }

    private struct Report: Encodable {
        let engine = "mlx-swift-lm"
        let promptTokens: Int
        let prefillStep: Int
        let runs: [Run]
        let medianPrefillTokensPerSecond: Double
        let medianDecodeTokensPerSecond: Double

        enum CodingKeys: String, CodingKey {
            case engine, runs
            case promptTokens = "prompt_tokens"
            case prefillStep = "prefill_step"
            case medianPrefillTokensPerSecond = "median_prefill_tokens_per_second"
            case medianDecodeTokensPerSecond = "median_decode_tokens_per_second"
        }
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    }

    func testServingThroughput() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["RICURSIVE_BENCH_MODEL"],
            let promptPath = environment["RICURSIVE_BENCH_PROMPT"],
            let reportPath = environment["RICURSIVE_BENCH_REPORT"]
        else {
            throw XCTSkip("RICURSIVE_BENCH_MODEL, _PROMPT and _REPORT are not set")
        }
        let tokens = Int(environment["RICURSIVE_BENCH_TOKENS"] ?? "") ?? 128
        let runCount = Int(environment["RICURSIVE_BENCH_RUNS"] ?? "") ?? 3

        let directory = URL(fileURLWithPath: modelPath)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(RicursiveConfiguration.self, from: configData)
        try configuration.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let model = RicursiveModel(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let prompt = try JSONDecoder().decode(
            [Int].self, from: Data(contentsOf: URL(fileURLWithPath: promptPath)))
        let parameters = GenerateParameters(maxTokens: tokens, temperature: 0)
        let prefillStep = parameters.prefill.stepSize ?? 2048

        func runOnce(_ count: Int) throws -> Run {
            var parameters = parameters
            parameters.maxTokens = count
            Memory.peakMemory = 0
            let begin = Date()
            var first: Date?
            var last = begin
            var generated = 0
            _ = try generate(
                promptTokens: prompt, parameters: parameters, model: model,
                tokenizer: NoStopTokenizer()
            ) { _ in
                let now = Date()
                if first == nil { first = now }
                last = now
                generated += 1
                return .more
            }
            let firstToken = try XCTUnwrap(first)
            return Run(
                prefillTokensPerSecond: Double(prompt.count)
                    / firstToken.timeIntervalSince(begin),
                decodeTokensPerSecond: Double(generated - 1) / last.timeIntervalSince(firstToken),
                generatedTokens: generated,
                peakMemoryGB: Double(Memory.peakMemory) / 1e9)
        }

        _ = try runOnce(min(tokens, 8))  // warm-up, untimed
        let runs = try (0 ..< runCount).map { _ in try runOnce(tokens) }
        for run in runs {
            XCTAssertEqual(run.generatedTokens, tokens)
        }
        let report = Report(
            promptTokens: prompt.count, prefillStep: prefillStep, runs: runs,
            medianPrefillTokensPerSecond: median(runs.map(\.prefillTokensPerSecond)),
            medianDecodeTokensPerSecond: median(runs.map(\.decodeTokensPerSecond)))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath: reportPath))
        print(
            "ricursive bench \(prompt.count): prefill \(report.medianPrefillTokensPerSecond) tok/s, decode \(report.medianDecodeTokensPerSecond) tok/s"
        )
    }
}
