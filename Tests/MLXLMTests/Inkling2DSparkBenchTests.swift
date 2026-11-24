import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Opt-in goal metric for Inkling 2 Mini with its bundled DSpark draft, on real weights: greedy
/// token parity against ordinary decode and decode tok/s, acceptance and prefill time for both,
/// at the same prefill step. Skips unless `INKLING2_DSPARK_MODEL` is set.
///
/// - `INKLING2_DSPARK_MODEL`: the target directory (with `mtp.safetensors`).
/// - `INKLING2_DSPARK_PROMPTS`: JSON array of token-id arrays (the megakernel-zoo bench's
///   `prompts/natural.json`), else one built-in prompt.
/// - `INKLING2_DSPARK_PROMPT_COUNT` (4), `INKLING2_DSPARK_TOKENS` (512), `INKLING2_DSPARK_BLOCK`
///   (8), `INKLING2_DSPARK_PREFILL_STEP` (512, macprovider's default), `INKLING2_DSPARK_REPORT`
///   (JSON path).
///
/// Prints one `DSPARK_ROW` per prompt and a `DSPARK_SUMMARY`. Fails on any token mismatch.
final class Inkling2DSparkBenchTests: XCTestCase {

    private struct Row: Codable {
        let prompt: Int
        let promptTokens: Int
        let tokens: Int
        let parity: Bool
        let firstMismatch: Int?
        let plainDecodeTokS: Double
        let dsparkDecodeTokS: Double
        let plainPrefillS: Double
        let dsparkPrefillS: Double
        let accepted: Int
        let proposed: Int
        let passthroughReason: String?
    }

    func testDSparkParityAndSpeedAgainstOrdinaryDecode() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["INKLING2_DSPARK_MODEL"] else {
            throw XCTSkip("INKLING2_DSPARK_MODEL is not set")
        }
        let promptCount = Int(environment["INKLING2_DSPARK_PROMPT_COUNT"] ?? "") ?? 4
        let maxTokens = Int(environment["INKLING2_DSPARK_TOKENS"] ?? "") ?? 512
        let blockSize = Int(environment["INKLING2_DSPARK_BLOCK"] ?? "") ?? 8
        let prefillStep = Int(environment["INKLING2_DSPARK_PREFILL_STEP"] ?? "") ?? 512

        var prompts: [[Int]] = [[200005, 1428, 200008, 2500, 6668, 495, 8592, 13, 200007]]
        if let path = environment["INKLING2_DSPARK_PROMPTS"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
            prompts = try JSONDecoder().decode([[Int]].self, from: data)
        }
        prompts = Array(prompts.prefix(promptCount))
        XCTAssertFalse(prompts.isEmpty)

        try Device.withDefaultDevice(.gpu) {
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
            let drafter = try Inkling2DSparkDraftModel.loadEmbedded(
                targetDirectory: directory, target: model)

            var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            parameters.prefill = PrefillParameters(stepSize: prefillStep)

            struct Sample {
                var tokens = [Int]()
                var prefill = 0.0
                var decode = 0.0
                var accepted = 0
                var proposed = 0
                var passthrough: String?
            }
            func plain(_ input: LMInput, _ parameters: GenerateParameters) throws -> Sample {
                var sample = Sample()
                let start = Date.timeIntervalSinceReferenceDate
                var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
                sample.prefill = Date.timeIntervalSinceReferenceDate - start
                let decodeStart = Date.timeIntervalSinceReferenceDate
                while let token = iterator.next() { sample.tokens.append(token) }
                Stream().synchronize()
                sample.decode = Date.timeIntervalSinceReferenceDate - decodeStart
                return sample
            }
            func dspark(_ input: LMInput, _ parameters: GenerateParameters) throws -> Sample {
                var sample = Sample()
                let start = Date.timeIntervalSinceReferenceDate
                var iterator = try MTPSpeculativeTokenIterator(
                    input: input, mainModel: model, drafter: drafter, parameters: parameters,
                    blockSize: blockSize)
                sample.prefill = Date.timeIntervalSinceReferenceDate - start
                let decodeStart = Date.timeIntervalSinceReferenceDate
                while let token = iterator.next() { sample.tokens.append(token) }
                Stream().synchronize()
                sample.decode = Date.timeIntervalSinceReferenceDate - decodeStart
                sample.accepted = iterator.acceptedCount
                sample.proposed = iterator.proposedCount
                sample.passthrough = iterator.passthroughReason
                return sample
            }

            // Compile kernels and realise the weights before timing.
            var warm = parameters
            warm.maxTokens = 16
            let warmInput = LMInput(tokens: MLXArray(prompts[0].map(Int32.init)))
            _ = try plain(warmInput, warm)
            _ = try dspark(warmInput, warm)
            Memory.clearCache()

            var rows = [Row]()
            for (index, prompt) in prompts.enumerated() {
                let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
                // Alternate the order to spread thermal drift.
                let (p, d): (Sample, Sample)
                if index % 2 == 0 {
                    p = try plain(input, parameters)
                    Memory.clearCache()
                    d = try dspark(input, parameters)
                } else {
                    d = try dspark(input, parameters)
                    Memory.clearCache()
                    p = try plain(input, parameters)
                }
                Memory.clearCache()
                let mismatch = zip(p.tokens, d.tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                    ?? (p.tokens.count == d.tokens.count ? nil : min(p.tokens.count, d.tokens.count))
                let row = Row(
                    prompt: index, promptTokens: prompt.count, tokens: p.tokens.count,
                    parity: mismatch == nil, firstMismatch: mismatch,
                    plainDecodeTokS: Double(p.tokens.count) / p.decode,
                    dsparkDecodeTokS: Double(d.tokens.count) / d.decode,
                    plainPrefillS: p.prefill, dsparkPrefillS: d.prefill,
                    accepted: d.accepted, proposed: d.proposed,
                    passthroughReason: d.passthrough)
                rows.append(row)
                print(
                    String(
                        format:
                            "DSPARK_ROW prompt=%d prompt_tokens=%d tokens=%d parity=%@ first_mismatch=%@ plain_tok_s=%.1f dspark_tok_s=%.1f gain=%.3f accepted=%d proposed=%d prefill_s=%.3f/%.3f passthrough=%@",
                        index, prompt.count, p.tokens.count, row.parity ? "yes" : "NO",
                        mismatch.map(String.init) ?? "-", row.plainDecodeTokS, row.dsparkDecodeTokS,
                        row.dsparkDecodeTokS / row.plainDecodeTokS, d.accepted, d.proposed,
                        p.prefill, d.prefill, d.passthrough ?? "-"))
                XCTAssertNil(mismatch, "prompt \(index): DSpark tokens differ from ordinary decode")
                XCTAssertNil(d.passthrough, "prompt \(index): the iterator fell back to passthrough")
            }

            func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
            let plainTokS = median(rows.map(\.plainDecodeTokS))
            let dsparkTokS = median(rows.map(\.dsparkDecodeTokS))
            let accepted = rows.map(\.accepted).reduce(0, +)
            let proposed = rows.map(\.proposed).reduce(0, +)
            print(
                String(
                    format:
                        "DSPARK_SUMMARY prompts=%d tokens=%d block=%d prefill_step=%d parity=%d/%d plain_tok_s=%.1f dspark_tok_s=%.1f gain=%.3f acceptance=%d/%d",
                    rows.count, maxTokens, blockSize, prefillStep, rows.filter(\.parity).count,
                    rows.count, plainTokS, dsparkTokS, dsparkTokS / plainTokS, accepted, proposed))
            if let path = environment["INKLING2_DSPARK_REPORT"] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(rows).write(to: URL(fileURLWithPath: path))
            }
        }
    }
}
