import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Lab harness for a real MiniMax M3 checkpoint, loaded through
/// `LLMModelFactory`. Skipped unless `MINIMAX_M3_MODEL_DIR` is set.
///
/// - `MINIMAX_M3_MODE=parity`: for each prompt in `MINIMAX_M3_PROMPTS`
///   (`{"steps": N, "chunk": C, "prompts": [{"name", "ids"}]}`), prefill in
///   chunks of `C` tokens, then decode `N` greedy tokens, writing every step's
///   token and top-10 float32 logits as JSON lines to `MINIMAX_M3_OUT`.
/// - `MINIMAX_M3_MODE=perf`: for each cell in `MINIMAX_M3_PROMPTS`
///   (`{"max_tokens", "chunk", "eos", "prompts": [{"name", "ids", "max_tokens"?}]}`),
///   greedy generation through `TokenIterator`, writing TTFT, prefill and
///   decode tokens/s and peak memory per cell.
final class MiniMaxM3ParityHarness: XCTestCase {

    private struct NoTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { NoTokenizer() }
    }

    /// Token ids come from the reference side; nothing here needs text.
    private struct NoTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private struct Prompt: Decodable {
        let name: String
        let ids: [Int]
        let maxTokens: Int?

        enum CodingKeys: String, CodingKey {
            case name, ids
            case maxTokens = "max_tokens"
        }
    }

    private struct Plan: Decodable {
        let prompts: [Prompt]
        let steps: Int?
        let chunk: Int?
        let maxTokens: Int?
        let eos: [Int]?

        enum CodingKeys: String, CodingKey {
            case prompts, steps, chunk, eos
            case maxTokens = "max_tokens"
        }
    }

    private final class JSONLines {
        let handle: FileHandle

        init(path: String) throws {
            FileManager.default.createFile(atPath: path, contents: nil)
            handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        }

        func write(_ object: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            data.append(0x0a)
            handle.write(data)
            try handle.synchronize()
        }
    }

    func testRealCheckpoint() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MINIMAX_M3_MODEL_DIR"] else {
            throw XCTSkip("MINIMAX_M3_MODEL_DIR is not set")
        }
        let promptsPath = try XCTUnwrap(env["MINIMAX_M3_PROMPTS"])
        let outPath = try XCTUnwrap(env["MINIMAX_M3_OUT"])
        let mode = env["MINIMAX_M3_MODE"] ?? "parity"
        let plan = try JSONDecoder().decode(
            Plan.self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
        let out = try JSONLines(path: outPath)

        let loadStart = Date()
        let context = try await LLMModelFactory.shared.load(
            from: URL(fileURLWithPath: modelDir), using: NoTokenizerLoader())
        let model = context.model
        try out.write([
            "event": "load", "s": Date().timeIntervalSince(loadStart),
            "model": String(describing: type(of: model)),
            "active_gb": Double(Memory.activeMemory) / 1e9,
        ])

        // Wire the weights for the whole run, as mlx-lm / mlx-vlm generation
        // does; unwired, a ~190 GB checkpoint is re-paged on every decode step.
        let wired = WiredMemoryTicket(
            size: GPU.maxRecommendedWorkingSetBytes() ?? 0, policy: MLX.WiredSumPolicy(),
            manager: .shared, kind: .active)
        let chunk = plan.chunk ?? 2048
        try await wired.withWiredLimit {
            if mode == "perf" {
                try perf(model: model, plan: plan, chunk: chunk, out: out)
            } else {
                try parity(model: model, plan: plan, chunk: chunk, out: out)
            }
        }
        try out.write(["event": "done"])
    }

    private func parity(model: any LanguageModel, plan: Plan, chunk: Int, out: JSONLines) throws {
        let steps = plan.steps ?? 128
        for prompt in plan.prompts {
            let cache = try model.newCache(parameters: nil)
            let start = Date()
            var logits = MLXArray(0)
            var at = 0
            while at < prompt.ids.count {
                let end = min(at + chunk, prompt.ids.count)
                let input = MLXArray(prompt.ids[at ..< end].map(Int32.init)).reshaped(1, -1)
                let output = model(input, cache: cache)
                if end == prompt.ids.count {
                    logits = output[0, -1]
                    eval(logits)
                } else {
                    eval(cache.flatMap { $0.innerState() })
                }
                at = end
            }
            let prefillSeconds = Date().timeIntervalSince(start)
            var tokens = [Int]()
            for step in 0 ..< steps {
                let values = logits.asType(.float32)
                let order = argSort(-values)[..<10]
                let top = values.take(order)
                eval(order, top)
                let ids = order.asArray(UInt32.self).map(Int.init)
                let scores = top.asArray(Float.self).map(Double.init)
                let token = ids[0]
                tokens.append(token)
                try out.write([
                    "prompt": prompt.name, "step": step, "token": token,
                    "top_ids": ids, "top_logits": scores,
                ])
                let next = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
                logits = next[0, -1]
                eval(logits)
            }
            try out.write([
                "prompt": prompt.name, "event": "summary", "prompt_tokens": prompt.ids.count,
                "prefill_s": prefillSeconds, "tokens": tokens,
            ])
            Memory.clearCache()
        }
    }

    private func perf(model: any LanguageModel, plan: Plan, chunk: Int, out: JSONLines) throws {
        let eos = Set(plan.eos ?? [])
        for prompt in plan.prompts {
            Memory.clearCache()
            Memory.peakMemory = 0
            let maxTokens = prompt.maxTokens ?? plan.maxTokens ?? 256
            let parameters = GenerateParameters(
                maxTokens: maxTokens, temperature: 0,
                prefill: .init(stepSize: chunk, chunking: .remainder))
            let start = Date()
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt.ids.map(Int32.init))), model: model,
                parameters: parameters)
            var first: Date?
            var last = start
            var generated = 0
            var finish = "length"
            var tail = [Int]()
            while let token = iterator.next() {
                last = Date()
                if first == nil { first = last }
                generated += 1
                tail.append(token)
                if eos.contains(token) {
                    finish = "stop"
                    break
                }
            }
            let ttft = (first ?? last).timeIntervalSince(start)
            let decodeSeconds = last.timeIntervalSince(first ?? last)
            try out.write([
                "cell": prompt.name, "prompt_tokens": prompt.ids.count, "ttft_s": ttft,
                "prefill_iter_s": iterator.promptPrefillTime,
                "prefill_tps": Double(prompt.ids.count) / ttft,
                "decode_tps": generated > 1 ? Double(generated - 1) / decodeSeconds : 0,
                "gen_tokens": generated, "finish": finish,
                "peak_gb": Double(Memory.peakMemory) / 1e9, "tail_ids": Array(tail.suffix(48)),
            ])
        }
    }
}
