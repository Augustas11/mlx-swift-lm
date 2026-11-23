import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Inkling 2 single-row prefill and decode speed on both paths, and the per-phase time split of
/// the plain path. Needs `INKLING2_BENCH_MODEL` (the source or oQ checkpoint directory); skipped
/// otherwise. Run it alone in a Studio window, release build, GPU.
///
/// For each path in `INKLING2_BENCH_PATHS` (default `compiled,plain`; ``Inkling2FastPath``) and
/// each context in `INKLING2_BENCH_CONTEXTS` (default `1024,8192,32768`):
///
/// 1. warm-up: a prefill of `min(context, 4096)` tokens on a fresh cache, which traces every
///    compiled function and kernel for the chunk shapes the timed prefill uses;
/// 2. prefill of a seeded prompt of `context` ordinary-text ids in 2048-token chunks, timed to
///    the evaluated cache: prefill tokens/s;
/// 3. 4 decode warm-up steps, then pipelined decode as `TokenIterator` and mlx-vlm's
///    `generate_step` run (the next step's graph is built while the GPU runs the current one):
///    decode tokens/s, the number rule 15 compares with the reference;
/// 4. synchronous decode (`eval` and `item` every step): tokens/s, host graph-build and
///    eval-wait milliseconds per token;
/// 5. plain path only: ``Inkling2PhaseProfiler`` with an `eval` barrier after every phase.
///
/// No diagnostics and no extra syncs run inside a timed loop. `INKLING2_BENCH_STEPS` (64) and
/// `INKLING2_BENCH_PROFILE_STEPS` (16) set the step counts; `INKLING2_BENCH_REPORT` writes the
/// numbers as JSON. Every result prints as one `BENCH_ROW path=... ctx=... decode_tps=...
/// prefill_tps=...` line; phases print as `BENCH_PHASE`.
final class Inkling2DecodeBenchTests: XCTestCase {
    private struct PhaseTime: Encodable {
        let phase: String
        let millisecondsPerToken: Double
        let callsPerToken: Double
        /// Share of the profiled step.
        let share: Double
    }

    private struct Row: Encodable {
        let path: String
        let context: Int
        let prefillSeconds: Double
        let prefillTokensPerSecond: Double
        let decodeTokensPerSecond: Double
        let synchronousTokensPerSecond: Double
        let buildMillisecondsPerToken: Double
        let evalMillisecondsPerToken: Double
        let profiledMillisecondsPerToken: Double?
        let unattributedMillisecondsPerToken: Double?
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

    /// Prefill `tokens` into a fresh cache; the cache and the token left for decode.
    private func prefill(_ model: Inkling2Model, _ tokens: MLXArray) throws -> ([KVCache], MLXArray)? {
        let cache = try model.newCache(parameters: nil)
        guard
            case .tokens(let rest) = try model.prepare(
                LMInput(tokens: tokens), cache: cache,
                prefill: PrefillParameters(stepSize: Self.prefillStep, chunking: .remainder))
        else { return nil }
        eval(cache)
        return (cache, rest.tokens)
    }

    func testDecodeSpeedAndPhaseSplit() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["INKLING2_BENCH_MODEL"] else {
            throw XCTSkip("INKLING2_BENCH_MODEL is not set")
        }
        let contexts = (environment["INKLING2_BENCH_CONTEXTS"] ?? "1024,8192,32768")
            .split(separator: ",").compactMap { Int($0) }
        let paths = (environment["INKLING2_BENCH_PATHS"] ?? "compiled,plain")
            .split(separator: ",").map(String.init)
        let steps = Int(environment["INKLING2_BENCH_STEPS"] ?? "") ?? 64
        let profiledSteps = Int(environment["INKLING2_BENCH_PROFILE_STEPS"] ?? "") ?? 16
        let wasEnabled = Inkling2FastPath.enabled
        defer { Inkling2FastPath.enabled = wasEnabled }

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
            for path in paths {
                XCTAssertTrue(["compiled", "plain"].contains(path), "unknown path \(path)")
                Inkling2FastPath.enabled = path == "compiled"
                for context in contexts {
                    let tokens = prompt(length: context)

                    // 1. Warm-up prefill.
                    _ = try prefill(model, tokens[..<min(context, 4096)])
                    Memory.clearCache()

                    // 2. Prefill.
                    var start = DispatchTime.now().uptimeNanoseconds
                    guard let prefilled = try prefill(model, tokens) else {
                        XCTFail("prepare should hand tokens to the iterator")
                        continue
                    }
                    let (cache, rest) = prefilled
                    let prefillSeconds = seconds(since: start)

                    var input = rest.reshaped(1, rest.size)
                    func step() {
                        let token = argMax(model(input, cache: cache)[0, -1], axis: -1)
                        eval(token)
                        input = token.reshaped(1, 1)
                    }
                    for _ in 0 ..< Self.warmUp { step() }

                    // 3. Pipelined decode.
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

                    // 4. Synchronous decode.
                    var build = 0.0
                    var wait = 0.0
                    for _ in 0 ..< steps {
                        start = DispatchTime.now().uptimeNanoseconds
                        let token = argMax(model(input, cache: cache)[0, -1], axis: -1)
                        build += seconds(since: start)
                        let built = DispatchTime.now().uptimeNanoseconds
                        _ = token.item(Int.self)
                        wait += seconds(since: built)
                        input = token.reshaped(1, 1)
                    }

                    // 5. Profiled (plain path).
                    var phases = [PhaseTime]()
                    var profiledMilliseconds: Double?
                    var unattributedMilliseconds: Double?
                    if path == "plain", profiledSteps > 0 {
                        let profiler = Inkling2PhaseProfiler()
                        Inkling2PhaseProfiler.active = profiler
                        start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0 ..< profiledSteps { step() }
                        let profiledSeconds = seconds(since: start)
                        Inkling2PhaseProfiler.active = nil
                        let perToken = 1000 / Double(profiledSteps)
                        phases = Inkling2PhaseProfiler.Phase.allCases.map { phase in
                            PhaseTime(
                                phase: phase.rawValue,
                                millisecondsPerToken: (profiler.seconds[phase] ?? 0) * perToken,
                                callsPerToken: Double(profiler.calls[phase] ?? 0) / Double(profiledSteps),
                                share: (profiler.seconds[phase] ?? 0) / profiledSeconds)
                        }
                        profiledMilliseconds = profiledSeconds * perToken
                        unattributedMilliseconds =
                            (profiledSeconds - profiler.seconds.values.reduce(0, +)) * perToken
                    }

                    let row = Row(
                        path: path, context: context, prefillSeconds: prefillSeconds,
                        prefillTokensPerSecond: Double(context) / prefillSeconds,
                        decodeTokensPerSecond: pipelined,
                        synchronousTokensPerSecond: Double(steps) / (build + wait),
                        buildMillisecondsPerToken: build * 1000 / Double(steps),
                        evalMillisecondsPerToken: wait * 1000 / Double(steps),
                        profiledMillisecondsPerToken: profiledMilliseconds,
                        unattributedMillisecondsPerToken: unattributedMilliseconds,
                        phases: phases)
                    report.append(row)
                    let fields: [String] = [
                        "BENCH_ROW path=\(path) ctx=\(context)",
                        "decode_tps=\(row.decodeTokensPerSecond)",
                        "prefill_tps=\(row.prefillTokensPerSecond)",
                        "sync_tps=\(row.synchronousTokensPerSecond)",
                        "build_ms=\(row.buildMillisecondsPerToken) eval_ms=\(row.evalMillisecondsPerToken)",
                        "profiled_ms=\(profiledMilliseconds.map { "\($0)" } ?? "-")",
                        "unattributed_ms=\(unattributedMilliseconds.map { "\($0)" } ?? "-")",
                    ]
                    print(fields.joined(separator: " "))
                    for phase in phases {
                        let line: [String] = [
                            "BENCH_PHASE path=\(path) ctx=\(context) phase=\(phase.phase)",
                            "ms_per_token=\(phase.millisecondsPerToken)",
                            "calls_per_token=\(phase.callsPerToken) share=\(phase.share)",
                        ]
                        print(line.joined(separator: " "))
                    }
                    Memory.clearCache()
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
