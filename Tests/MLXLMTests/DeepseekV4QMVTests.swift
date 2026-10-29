import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4QMVTests: XCTestCase {
    private func layer(k: Int = 256, n: Int = 16) -> QuantizedLinear {
        // Exercise both signs, subnormals and finite E4M3 exponent ranges.
        // Avoid NaN encodings so equality also detects arithmetic differences.
        let bytes = (0 ..< (n * k)).map { i -> UInt8 in
            let magnitude = UInt8((i * 37 + i / k * 11) % 127)
            return magnitude | (i % 3 == 0 ? 128 : 0)
        }
        let packed = stride(from: 0, to: bytes.count, by: 4).map { i in
            UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8
                | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24
        }
        let scales = (0 ..< (n * k / 32)).map { UInt8(112 + ($0 * 7) % 16) }
        return QuantizedLinear(
            weight: MLXArray(packed).reshaped(n, k / 4),
            scales: MLXArray(scales).reshaped(n, k / 32), biases: nil,
            groupSize: 32, bits: 8, mode: .mxfp8)
    }

    private func input(_ rows: Int, k: Int, dtype: DType) -> MLXArray {
        MLXArray((0 ..< (rows * k)).map { Float(($0 * 13 + $0 / k * 17) % 109 - 54) / 64 })
            .reshaped(rows, k).asType(dtype)
    }

    private func assertBitsEqual(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
            expected.asType(.float32).asArray(Float.self).map(\.bitPattern),
            file: file, line: line)
    }

    func testGPUExactSerialRowsAndCompanionIndependence() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        for dtype in [DType.bfloat16, .float16] {
            for k in [256, 512, 1024] {
                let projection = layer(k: k)
                for rows in [2, 3, 4, 8] {
                    let x = input(rows, k: k, dtype: dtype)
                    let serial = concatenated(
                        (0 ..< rows).map { projection(x[$0 ..< ($0 + 1)]) }, axis: 0)
                    let result = try XCTUnwrap(deepseekV4QMVBatch(x, projection))
                    assertBitsEqual(result, serial)
                    assertBitsEqual(deepseekV4Project(projection, x), serial)
                    assertBitsEqual(
                        deepseekV4Project(projection, x.reshaped(rows, 1, k)),
                        serial.reshaped(rows, 1, 16))
                    // Row zero stays identical when every other activation changes.
                    let changed = concatenated(
                        [
                            x[0 ..< 1], -x[1 ..< rows],
                        ], axis: 0)
                    let independent = try XCTUnwrap(deepseekV4QMVBatch(changed, projection))
                    assertBitsEqual(independent[0 ..< 1], result[0 ..< 1])
                    // MLXFast must materialize strided rows before the raw-pointer kernel.
                    let strided = input(rows * 2, k: k, dtype: dtype)[.stride(by: 2)]
                    let stridedSerial = concatenated(
                        (0 ..< rows).map { projection(strided[$0 ..< ($0 + 1)]) }, axis: 0)
                    assertBitsEqual(
                        try XCTUnwrap(deepseekV4QMVBatch(strided, projection)), stridedSerial)
                }
            }
        }
    }

    func testGPUZeroE8M0ScaleEncodingMatchesSerial() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let original = layer()
        let projection = QuantizedLinear(
            weight: original.weight, scales: MLXArray.zeros([16, 8], dtype: .uint8),
            biases: nil, groupSize: 32, bits: 8, mode: .mxfp8)
        let x = input(3, k: 256, dtype: .bfloat16)
        let serial = concatenated((0 ..< 3).map { projection(x[$0 ..< ($0 + 1)]) }, axis: 0)
        assertBitsEqual(try XCTUnwrap(deepseekV4QMVBatch(x, projection)), serial)
    }

    func testGPURejectsUnsupportedShapesDTypesAndFormats() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("GPU only") }
        let projection = layer()
        let x = input(3, k: 256, dtype: .bfloat16)
        for rows in [1, 9] {
            XCTAssertNil(deepseekV4QMVBatch(input(rows, k: 256, dtype: .bfloat16), projection))
        }
        XCTAssertNil(deepseekV4QMVBatch(x.asType(.float32), projection))
        assertBitsEqual(
            deepseekV4Project(projection, x.asType(.float32)), projection(x.asType(.float32)))
        XCTAssertNil(deepseekV4QMVBatch(x.reshaped(3, 1, 256), projection))
        XCTAssertNil(deepseekV4QMVBatch(input(3, k: 128, dtype: .bfloat16), layer(k: 128)))
        XCTAssertNil(deepseekV4QMVBatch(x, layer(n: 9)))
        let mxfp4 = QuantizedLinear(256, 16, bias: false, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(deepseekV4QMVBatch(x, mxfp4))
        assertBitsEqual(deepseekV4Project(mxfp4, x), mxfp4(x))
        let affine = QuantizedLinear(256, 16, bias: false, groupSize: 32, bits: 8)
        XCTAssertNil(deepseekV4QMVBatch(x, affine))
        assertBitsEqual(deepseekV4Project(affine, x), affine(x))
        let biased = QuantizedLinear(
            weight: projection.weight, bias: MLXArray.zeros([16], dtype: .bfloat16),
            scales: projection.scales, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8)
        XCTAssertNil(deepseekV4QMVBatch(x, biased))
        assertBitsEqual(deepseekV4Project(biased, x), biased(x))
    }

    func testCPUUsesOriginalLayerFallback() {
        Device.withDefaultDevice(.cpu) {
            let x = input(3, k: 256, dtype: .float32)
            let mxfp8 = layer()
            XCTAssertNil(deepseekV4QMVBatch(x.asType(.bfloat16), mxfp8))
            let affine = QuantizedLinear(256, 16, bias: false, groupSize: 32, bits: 8)
            XCTAssertNil(deepseekV4QMVBatch(x, affine))
            assertBitsEqual(deepseekV4Project(affine, x), affine(x))
            let dense = Linear(256, 16, bias: false)
            assertBitsEqual(deepseekV4Project(dense, x), dense(x))
        }
    }
}
