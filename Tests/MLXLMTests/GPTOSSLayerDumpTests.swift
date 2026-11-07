import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// The lab-only GPT-OSS layer dump (#2255) is off unless its environment
/// variable names a directory, and writes only the forwards it was given.
final class GPTOSSLayerDumpTests: XCTestCase {

    private func tinyModel() throws -> GPTOSSModel {
        let config = try JSONDecoder().decode(
            GPTOSSConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gpt_oss", "num_hidden_layers": 2, "num_local_experts": 4,
                    "num_experts_per_tok": 2, "vocab_size": 32, "rms_norm_eps": 1e-5,
                    "hidden_size": 16, "intermediate_size": 16, "head_dim": 8,
                    "num_attention_heads": 2, "num_key_value_heads": 1,
                    "sliding_window": 4, "rope_theta": 10000.0,
                    "layer_types": ["sliding_attention", "full_attention"]
                }
                """.utf8))
        return GPTOSSModel(config)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gptoss-layer-dump-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testDumpIsOffByDefault() throws {
        XCTAssertNil(GPTOSSLayerDump.make(environment: [:]))
        XCTAssertNil(GPTOSSLayerDump.make(environment: [GPTOSSLayerDump.directoryKey: ""]))
        XCTAssertNil(
            GPTOSSLayerDump.make(environment: [GPTOSSLayerDump.forwardsKey: "8"]),
            "a forward budget alone does not turn the dump on")

        try XCTSkipIf(
            ProcessInfo.processInfo.environment[GPTOSSLayerDump.directoryKey] != nil,
            "the test host sets the dump variable")
        let model = try tinyModel()
        XCTAssertNil(model.model.layerDump)
        let cache = try model.newCache(parameters: nil)
        eval(model(MLXArray(Array(Int32(1) ... Int32(5))).reshaped(1, 5), cache: cache))
    }

    func testDumpWritesOnlyItsForwardBudget() throws {
        let directory = try temporaryDirectory()
        let dump = try XCTUnwrap(
            GPTOSSLayerDump.make(environment: [
                GPTOSSLayerDump.directoryKey: directory.path,
                GPTOSSLayerDump.forwardsKey: "1",
            ]))
        let model = try tinyModel()
        model.model.layerDump = dump
        let cache = try model.newCache(parameters: nil)

        let hidden = model.model(
            MLXArray(Array(Int32(1) ... Int32(5))).reshaped(1, 5), cache: cache)
        eval(hidden)
        eval(model(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache))

        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        XCTAssertEqual(
            names,
            [
                "manifest.jsonl", "native_f0_embed.npy", "native_f0_final_norm.npy",
                "native_f0_layer00.npy", "native_f0_layer00_attn.npy", "native_f0_layer01.npy",
            ])
        let finalNorm = try loadArray(
            url: directory.appendingPathComponent("native_f0_final_norm.npy"))
        XCTAssertEqual(finalNorm.shape, [1, 5, 16])
        XCTAssertTrue(allClose(finalNorm, hidden.asType(.float32)).item(Bool.self))

        let manifest = try String(
            contentsOf: directory.appendingPathComponent("manifest.jsonl"), encoding: .utf8)
        XCTAssertEqual(
            manifest, #"{"forward":0,"tokens":5,"cache_offset":0,"dtype":"float32"}"# + "\n")
    }
}
