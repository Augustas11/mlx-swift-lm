import Foundation
import MLX
import XCTest

@testable import MLXVLM

private func signedHadamardReference(
    _ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool
) -> MLXArray {
    var value = input.asType(.float32)
    if !inverse { value = value * signs }
    value = hadamardTransform(value.reshaped(-1, block), scale: 1 / sqrt(Float(block)))
        .reshaped(input.shape)
    if inverse { value = value * signs }
    return value.asType(input.dtype)
}

private func signedHadamardBits(_ value: MLXArray) -> [UInt32] {
    value.asType(.float32).asArray(Float.self).map(\.bitPattern)
}

private func assertSignedHadamardBits(
    _ actual: MLXArray, _ expected: MLXArray, context: String,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(actual.shape, expected.shape, context, file: file, line: line)
    XCTAssertEqual(actual.dtype, expected.dtype, context, file: file, line: line)
    let actualValues = actual.asType(.float32).asArray(Float.self)
    let expectedValues = expected.asType(.float32).asArray(Float.self)
    XCTAssertEqual(actualValues.count, expectedValues.count, context, file: file, line: line)
    guard actualValues.count == expectedValues.count else { return }
    var differentNaNPayloads = 0
    var firstMismatch: Int?
    for index in expectedValues.indices {
        let actual = actualValues[index]
        let expected = expectedValues[index]
        if actual.isNaN && expected.isNaN {
            if actual.bitPattern != expected.bitPattern { differentNaNPayloads += 1 }
        } else if actual.bitPattern != expected.bitPattern, firstMismatch == nil {
            firstMismatch = index
        }
    }
    if differentNaNPayloads > 0 {
        print("Hadamard NaN payload diagnostic: \(context), count=\(differentNaNPayloads)")
    }
    XCTAssertNil(
        firstMismatch, "\(context), first bit mismatch=\(String(describing: firstMismatch))",
        file: file, line: line)
}

final class PrismSignedHadamardKernelTests: XCTestCase {
    private func requireGPU() throws {
        #if os(macOS)
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Signed-Hadamard Metal parity requires the GPU")
        }
        #else
        throw XCTSkip("Signed-Hadamard Metal kernel is macOS-only")
        #endif
    }

    func testFiniteBitParityAcrossBlocksRowsDirectionsAndChangedSigns() throws {
        try requireGPU()
        for block in [512, 1024] {
            for width in [block, 5 * block, 18 * block] {
                for rows in [1, 2, 8] {
                    for dtype in [DType.float16, .bfloat16, .float32] {
                        let input = MLXArray(
                            (0 ..< (rows * width)).map {
                                Float($0 % 127 - 63) / 65
                            }
                        ).reshaped(1, rows, width).asType(dtype)
                        for inverse in [false, true] {
                            for signPattern in [0, 1] {
                                let signs = MLXArray(
                                    (0 ..< width).map {
                                        Float(($0 + signPattern) % 3 == 0 ? -1 : 1)
                                    })
                                let actual = try XCTUnwrap(
                                    PrismSignedHadamardKernel.apply(
                                        input, block: block, signs: signs, inverse: inverse))
                                let expected = signedHadamardReference(
                                    input, block: block, signs: signs, inverse: inverse)
                                let context =
                                    "block=\(block), width=\(width), rows=\(rows), dtype=\(dtype), inverse=\(inverse), signs=\(signPattern)"
                                assertSignedHadamardBits(actual, expected, context: context)
                                assertSignedHadamardBits(
                                    prismHadamardTransform(
                                        input, block: block, signs: signs,
                                        inverse: inverse),
                                    expected, context: "dispatcher: " + context)
                            }
                        }
                    }
                }
            }
        }
    }

    func testSignedZerosSubnormalsCancellationAndFiniteExtremes() throws {
        try requireGPU()
        for block in [512, 1024] {
            for dtype in [DType.float16, .bfloat16, .float32] {
                let maximum: Float =
                    dtype == .float16
                    ? Float(Float16.greatestFiniteMagnitude)
                    : (dtype == .bfloat16
                        ? Float(bitPattern: 0x7F7F_0000) : .greatestFiniteMagnitude)
                let tiny: Float =
                    dtype == .float16
                    ? Float(Float16.leastNonzeroMagnitude)
                    : (dtype == .bfloat16 ? Float(bitPattern: 0x0001_0000) : .leastNonzeroMagnitude)
                let patterns: [[Float]] = [
                    [0, -Float.zero],
                    [tiny, -tiny, .leastNormalMagnitude, -.leastNormalMagnitude],
                    [1, -1, 1 + Float.ulpOfOne, -1 - Float.ulpOfOne],
                    [maximum / Float(block * 2), -maximum / Float(block * 2)],
                    [maximum, maximum, -maximum, -maximum],
                ]
                for (patternIndex, pattern) in patterns.enumerated() {
                    let input = MLXArray((0 ..< (2 * block)).map { pattern[$0 % pattern.count] })
                        .reshaped(2, block).asType(dtype)
                    let signs = MLXArray((0 ..< block).map { Float($0 % 3 == 0 ? -1 : 1) })
                    for inverse in [false, true] {
                        let actual = try XCTUnwrap(
                            PrismSignedHadamardKernel.apply(
                                input, block: block, signs: signs, inverse: inverse))
                        assertSignedHadamardBits(
                            actual,
                            signedHadamardReference(
                                input, block: block, signs: signs,
                                inverse: inverse),
                            context:
                                "IEEE block=\(block), dtype=\(dtype), pattern=\(patternIndex), inverse=\(inverse)"
                        )
                    }
                }
            }
        }
    }

    func testNoncontiguousActivationsAndSigns() throws {
        try requireGPU()
        for block in [512, 1024] {
            let width = block * 5
            let signs = MLXArray(
                (0 ..< (width * 2)).map {
                    Float(($0 / 2) % 3 == 0 ? -1 : 1)
                })[.stride(by: 2)]
            let backing = MLXArray((0 ..< (2 * width * 2)).map { Float($0 % 31 - 15) / 32 })
                .reshaped(2, width * 2).asType(.float16)
            let views = [
                backing[.ellipsis, .stride(by: 2)],
                backing.reshaped(width, 4)[0..., 0 ..< 2].transposed(),
            ]
            for input in views {
                for inverse in [false, true] {
                    let actual = try XCTUnwrap(
                        PrismSignedHadamardKernel.apply(
                            input, block: block, signs: signs, inverse: inverse))
                    assertSignedHadamardBits(
                        actual,
                        signedHadamardReference(
                            input, block: block, signs: signs,
                            inverse: inverse),
                        context: "strided block=\(block), inverse=\(inverse)")
                }
            }
        }
    }

    func testRoutingRejectsUnsupportedMetadataAndPreservesReferenceFallback() throws {
        try requireGPU()
        for block in [128, 2048, 4096] {
            let input = MLXArray.ones([2, block], dtype: .float16)
            let signs = MLXArray.ones([block])
            XCTAssertNil(
                PrismSignedHadamardKernel.apply(
                    input, block: block, signs: signs, inverse: false))
            assertSignedHadamardBits(
                prismHadamardTransform(input, block: block, signs: signs),
                signedHadamardReference(input, block: block, signs: signs, inverse: false),
                context: "fallback block=\(block)")
        }
        for block in [512, 1024] {
            let signs = MLXArray.ones([block])
            let long = MLXArray.ones([9, block], dtype: .float16)
            XCTAssertNil(
                PrismSignedHadamardKernel.apply(long, block: block, signs: signs, inverse: false))
            assertSignedHadamardBits(
                prismHadamardTransform(long, block: block, signs: signs),
                signedHadamardReference(long, block: block, signs: signs, inverse: false),
                context: "nine actual rows fall back, block=\(block)")
            for input in [
                MLXArray(Float(1)), MLXArray.zeros([0, block]),
                MLXArray.zeros([1, 0]), MLXArray.ones([1, block + 1]),
                MLXArray.ones([1, block], dtype: .int32),
            ] {
                XCTAssertNil(
                    PrismSignedHadamardKernel.apply(
                        input, block: block, signs: signs, inverse: false))
            }
            let input = MLXArray.ones([1, block], dtype: .float16)
            for invalid in [
                MLXArray.ones([block], dtype: .float16), MLXArray.ones([1, block]),
                MLXArray.ones([block - 1]),
            ] {
                XCTAssertNil(
                    PrismSignedHadamardKernel.apply(
                        input, block: block, signs: invalid, inverse: false))
            }
        }
        Device.withDefaultDevice(.cpu) {
            XCTAssertNil(
                PrismSignedHadamardKernel.apply(
                    MLXArray.ones([1, 512]), block: 512, signs: MLXArray.ones([512]), inverse: false
                ))
        }
    }

    func testConcurrentInvocationsKeepIndependentSignsAndStreams() async throws {
        try requireGPU()
        await withTaskGroup(of: (Int, [UInt32], [UInt32]).self) { group in
            for request in 0 ..< 8 {
                group.addTask {
                    Stream.withNewDefaultStream(device: .gpu) {
                        let block = request % 2 == 0 ? 512 : 1024
                        let input = MLXArray(
                            (0 ..< (block * 2)).map {
                                Float($0 % 31 - 15) / 32
                            }
                        ).reshaped(2, block).asType(.float16)
                        let signs = MLXArray(
                            (0 ..< block).map {
                                Float(($0 + request) % 3 == 0 ? -1 : 1)
                            })
                        let inverse = request % 3 == 0
                        guard
                            let actual = PrismSignedHadamardKernel.apply(
                                input, block: block, signs: signs, inverse: inverse)
                        else { return (request, [], [0]) }
                        let expected = signedHadamardReference(
                            input, block: block, signs: signs, inverse: inverse)
                        return (request, signedHadamardBits(actual), signedHadamardBits(expected))
                    }
                }
            }
            for await (request, actual, expected) in group {
                XCTAssertEqual(actual, expected, "concurrent request=\(request)")
            }
        }
    }
}
