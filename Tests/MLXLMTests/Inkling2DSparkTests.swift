import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Inkling 2 Mini's bundled DSpark draft (`mtp.safetensors`): the module tree against the
/// checkpoint, and the target's draft taps. CPU only; no weight-sized array is evaluated.
///
/// Fixture `Resources/inkling2/oq4e-mtp-tensors.json`: dtype and shape of every tensor in
/// mlx-community/Inkling-2-Mini-oQ4e-mtp @ 3eea0427 `mtp.safetensors`, read from its header.
final class Inkling2DSparkTests: XCTestCase {

    private struct TensorList: Decodable {
        let tensors: [String: Entry]
        struct Entry: Decodable {
            let dtype: String
            let shape: [Int]
            init(from decoder: Decoder) throws {
                var c = try decoder.unkeyedContainer()
                dtype = try c.decode(String.self)
                shape = try c.decode([Int].self)
            }
        }
    }

    private func resource(_ name: String) throws -> Data {
        let directory = try XCTUnwrap(Bundle.module.url(forResource: "inkling2", withExtension: nil))
        return try Data(contentsOf: directory.appendingPathComponent(name))
    }

    private func configuration() throws -> Inkling2Configuration {
        try JSONDecoder().decode(Inkling2Configuration.self, from: resource("oq4e-config.json"))
    }

    func testDraftModuleTreeMatchesTheBundledCheckpoint() throws {
        let draft = try Inkling2DSparkDraftModel(configuration())
        let expected = try JSONDecoder().decode(
            TensorList.self, from: resource("oq4e-mtp-tensors.json")
        ).tensors
        let actual = Dictionary(uniqueKeysWithValues: draft.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
        for (key, entry) in expected {
            XCTAssertEqual(actual[key], entry.shape, key)
        }
    }

    func testDraftReadsTheTargetVisitsItsConfigNames() throws {
        let config = try configuration()
        let dspark = try XCTUnwrap(config.dspark)
        let draft = try Inkling2DSparkDraftModel(config)
        XCTAssertEqual(draft.targetLayerIds, dspark.targetLayerIds)
        // Mini: three visit taps and the final-norm output (id = visit count).
        XCTAssertEqual(dspark.targetLayerIds.last, Inkling2LoopSchedule(config.text).count)
        XCTAssertEqual(draft.maximumBlockSize, dspark.blockSize)
        XCTAssertTrue(draft.requiresGreedySampling)
        XCTAssertTrue(draft.requiresPromptPrefill)
        XCTAssertFalse(draft.requiresSharedTargetKV)
    }

    func testDraftRequiresTheMTPConfig() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: resource("oq4e-config.json")) as? [String: Any])
        json["mtp_config"] = nil
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertThrowsError(try Inkling2DSparkDraftModel(config)) {
            XCTAssertEqual($0 as? Inkling2DSparkError, .missingDraftConfig)
        }
    }
}
