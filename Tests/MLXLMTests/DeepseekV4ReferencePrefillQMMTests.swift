import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class DeepseekV4ReferencePrefillQMMTests: XCTestCase {
    func testUnsupportedDTypeReturnsNil() {
        XCTAssertNil(deepseekV4ReferencePrefillQMM(
            MLXArray.zeros([2, 32], dtype: .float32),
            weight: MLXArray.zeros([4, 4], dtype: .uint32),
            scales: MLXArray.zeros([4, 1], dtype: .uint8)))
    }

    func testGPUReferenceProjectionFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["DSV4_PREFILL_QMM_FIXTURE"] else {
            throw XCTSkip("Reference prefill QMM fixture is not configured")
        }
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let weight = try XCTUnwrap(fixture["weight"])
        let scales = try XCTUnwrap(fixture["scales"])
        for rows in [1, 2, 3, 5, 8, 17, 18, 23, 32, 33, 64, 65, 94, 128, 129, 256] {
            let input = try XCTUnwrap(fixture["x_\(rows)"])
            let expected = try XCTUnwrap(fixture["y_\(rows)"])
            let result = try XCTUnwrap(deepseekV4ReferencePrefillQMM(
                input, weight: weight, scales: scales))
            let actualValues = result.asType(.float32).asArray(Float.self)
            let expectedValues = expected.asType(.float32).asArray(Float.self)
            let differences = zip(actualValues, expectedValues).filter { $0 != $1 }.count
            XCTAssertEqual(actualValues.count, expectedValues.count)
            XCTAssertEqual(differences, 0, "rows=\(rows)")
        }
    }
}
