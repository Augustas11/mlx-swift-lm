// Adapted from sawfwair/mere-run Sources/MereRunQwenModel/Q35PrismHadamard.swift
// Revision: 5c18e3c7b3de36737e5aa2ee1b4e21b6fc2bccea
// Donor SHA-256: b2b45e9d9ae37644209b79d7416d069d08edd74745a801f91b81879f72b40443
// https://github.com/sawfwair/mere-run/blob/5c18e3c7b3de36737e5aa2ee1b4e21b6fc2bccea/Sources/MereRunQwenModel/Q35PrismHadamard.swift
//
// MIT License
//
// Copyright (c) 2026 Sawfwair Inc. and mere.run contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Foundation
import MLX

/// One launch for short rows' signed FP32 butterfly and activation conversions.
enum PrismSignedHadamardKernel {
    #if os(macOS)
    private static let kernel = MLXFast.metalKernel(
        name: "prism_signed_hadamard_short_rows",
        inputNames: ["x", "signs"],
        outputNames: ["y"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint base = threadgroup_position_in_grid.x * BLOCK;
            threadgroup float values[BLOCK];
            for (uint i = lane; i < BLOCK; i += 256) {
                float value = float(x[base + i]);
                if (!INVERSE) value *= float(signs[(base + i) % WIDTH]);
                values[i] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < BLOCK; stride *= 2) {
                for (uint pair = lane; pair < BLOCK / 2; pair += 256) {
                    uint low = (pair / stride) * (2 * stride) + pair % stride;
                    float a = values[low];
                    float b = values[low + stride];
                    values[low] = a + b;
                    values[low + stride] = a - b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint i = lane; i < BLOCK; i += 256) {
                float value = values[i] * as_type<float>(uint(SCALE_BITS));
                if (INVERSE) value *= float(signs[(base + i) % WIDTH]);
                y[base + i] = T(value);
            }
            """,
        ensureRowContiguous: true
    )
    #endif

    static func apply(
        _ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool
    ) -> MLXArray? {
        #if os(macOS)
        guard Device.defaultDevice().deviceType == .gpu,
            block == 512 || block == 1024,
            input.ndim > 0,
            [DType.float16, .bfloat16, .float32].contains(input.dtype)
        else { return nil }
        let width = input.dim(-1)
        // Sign values are validated during checkpoint update; inspect only metadata here.
        guard width > 0, width % block == 0, input.size > 0,
            (1 ... 8).contains(input.size / width),
            signs.shape == [width], signs.dtype == .float32,
            input.size <= Int(UInt32.max),
            input.size / block <= Int(Int32.max) / 256
        else { return nil }
        // Preserve the reference's host rounding, rather than using Metal rsqrt.
        let scale = 1 / sqrt(Float(block))
        return kernel(
            [input, signs],
            template: [
                ("T", input.dtype), ("WIDTH", width), ("INVERSE", inverse),
                ("BLOCK", block), ("SCALE_BITS", Int(scale.bitPattern)),
            ],
            grid: (input.size / block * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }
}
