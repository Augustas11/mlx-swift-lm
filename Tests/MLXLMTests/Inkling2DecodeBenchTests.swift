import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Inkling 2 decode speed and its per-phase time split (slice 5 prep). Needs
/// `INKLING2_BENCH_MODEL` (the source or oQ checkpoint directory); skipped otherwise. Run it alone
/// in a Studio window, release build, GPU.
///
/// For each context in `INKLING2_BENCH_CONTEXTS` (default `1024,8192`): a seeded prompt of that
/// many ordinary-text ids, prefill in 2048-token chunks, 4 warm-up steps, then three decodes on
/// the same cache:
///
/// 1. synchronous, as `dump_real_oracle.py` times mlx-vlm (`eval(row, token)` every step):
///    tokens/s, and per token the host time spent building the graph and the time waiting for
///    its evaluation;
/// 2. pipelined, as `TokenIterator` and mlx-vlm's `generate_step` run (the next step's graph is
///    built while the GPU runs the current one): tokens/s;
/// 3. profiled: ``Inkling2PhaseProfiler`` puts an `eval` barrier after every phase; ms per token
///    per phase, and what no phase holds.
///
/// `INKLING2_BENCH_STEPS` (64) and `INKLING2_BENCH_PROFILE_STEPS` (16) set the step counts;
/// `INKLING2_BENCH_REPORT` writes the numbers as JSON. Each result line starts with
/// `inkling2 decode bench`.
final class Inkling2DecodeBenchTests: XCTestCase {
    private struct PhaseTime: Encodable {
        let phase: String
        let millisecondsPerToken: Double
        let callsPerToken: Double
        /// Share of the profiled step.
        let share: Double
    }

    private struct Row: Encodable {
        let context: Int
        let prefillSeconds: Double
        let synchronousTokensPerSecond: Double
        let buildMillisecondsPerToken: Double
        let evalMillisecondsPerToken: Double
        let pipelinedTokensPerSecond: Double
        let profiledMillisecondsPerToken: Double
        let unattributedMillisecondsPerToken: Double
        let phases: [PhaseTime]
    }

    private static let prefillStep = 2048
    private static let warmUp = 4

    private func seconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// Ordinary text ids below the special-token range, from a fixed linear congruential sequence.
    private func prompt(length: Int) -> MLXArray {
        var state: UInt64 = 2467 &+ UInt64(length)
        let ids = (0 ..< length).map { _ -> Int32 in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int32((state >> 33) % 199_998)
        }
        return MLXArray(ids)
    }

    func testDecodeSpeedAndPhaseSplit() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["INKLING2_BENCH_MODEL"] else {
            throw XCTSkip("INKLING2_BENCH_MODEL is not set")
        }
        let contexts = (environment["INKLING2_BENCH_CONTEXTS"] ?? "1024,8192")
            .split(separator: ",").compactMap { Int($0) }
        let steps = Int(environment["INKLING2_BENCH_STEPS"] ?? "") ?? 64
        let profiledSteps = Int(environment["INKLING2_BENCH_PROFILE_STEPS"] ?? "") ?? 16

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

            var report = [Row]()
            for context in contexts {
                let cache = try model.newCache(parameters: nil)
                var start = DispatchTime.now().uptimeNanoseconds
                guard
                    case .tokens(let rest) = try model.prepare(
                        LMInput(tokens: prompt(length: context)), cache: cache,
                        prefill: PrefillParameters(stepSize: Self.prefillStep, chunking: .remainder))
                else {
                    XCTFail("prepare should hand tokens to the iterator")
                    continue
                }
                eval(cache)
                let prefillSeconds = seconds(since: start)

                var input = rest.tokens.reshaped(1, rest.tokens.size)
                func step() -> MLXArray {
                    let token = argMax(model(input, cache: cache)[0, -1], axis: -1)
                    eval(token)
                    input = token.reshaped(1, 1)
                    return token
                }
                for _ in 0 ..< Self.warmUp { _ = step() }

                // 1. Synchronous.
                var build = 0.0
                var wait = 0.0
                for _ in 0 ..< steps {
                    start = DispatchTime.now().uptimeNanoseconds
                    let token = argMax(model(input, cache: cache)[0, -1], axis: -1)
                    build += seconds(since: start)
                    let built = DispatchTime.now().uptimeNanoseconds
                    eval(token)
                    _ = token.item(Int.self)
                    wait += seconds(since: built)
                    input = token.reshaped(1, 1)
                }
                let synchronous = Double(steps) / (build + wait)

                // 2. Pipelined.
                var y = argMax(model(input, cache: cache)[0, -1], axis: -1)
                asyncEval(y)
                start = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< steps {
                    let next = argMax(model(y.reshaped(1, 1), cache: cache)[0, -1], axis: -1)
                    asyncEval(next)
                    _ = y.item(Int.self)
                    y = next
                }
                _ = y.item(Int.self)
                let pipelined = Double(steps) / seconds(since: start)
                input = y.reshaped(1, 1)

                // 3. Profiled.
                let profiler = Inkling2PhaseProfiler()
                Inkling2PhaseProfiler.active = profiler
                start = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< profiledSteps { _ = step() }
                let profiledSeconds = seconds(since: start)
                Inkling2PhaseProfiler.active = nil

                let perToken = 1000 / Double(max(profiledSteps, 1))
                let attributed = profiler.seconds.values.reduce(0, +)
                let phases = Inkling2PhaseProfiler.Phase.allCases.map { phase in
                    PhaseTime(
                        phase: phase.rawValue,
                        millisecondsPerToken: (profiler.seconds[phase] ?? 0) * perToken,
                        callsPerToken: Double(profiler.calls[phase] ?? 0) / Double(max(profiledSteps, 1)),
                        share: (profiler.seconds[phase] ?? 0) / profiledSeconds)
                }
                let row = Row(
                    context: context, prefillSeconds: prefillSeconds,
                    synchronousTokensPerSecond: synchronous,
                    buildMillisecondsPerToken: build * 1000 / Double(steps),
                    evalMillisecondsPerToken: wait * 1000 / Double(steps),
                    pipelinedTokensPerSecond: pipelined,
                    profiledMillisecondsPerToken: profiledSeconds * perToken,
                    unattributedMillisecondsPerToken: (profiledSeconds - attributed) * perToken,
                    phases: phases)
                report.append(row)
                let summary: [String] = [
                    "inkling2 decode bench context=\(context) prefill=\(prefillSeconds)s",
                    "sync=\(synchronous)tok/s build=\(row.buildMillisecondsPerToken)ms",
                    "eval=\(row.evalMillisecondsPerToken)ms pipelined=\(pipelined)tok/s",
                    "profiled=\(row.profiledMillisecondsPerToken)ms/token",
                    "unattributed=\(row.unattributedMillisecondsPerToken)ms",
                ]
                print(summary.joined(separator: " "))
                for phase in phases {
                    let line: [String] = [
                        "inkling2 decode bench context=\(context) phase=\(phase.phase)",
                        "ms/token=\(phase.millisecondsPerToken) calls/token=\(phase.callsPerToken)",
                        "share=\(phase.share)",
                    ]
                    print(line.joined(separator: " "))
                }
            }

            if let path = environment["INKLING2_BENCH_REPORT"] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: URL(fileURLWithPath: path))
            }
        }
    }
}
