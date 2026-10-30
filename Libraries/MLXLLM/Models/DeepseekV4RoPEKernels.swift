import MLX

// Float32 phase evaluation follows NumPy's Cody-Waite/FMA algorithm so
// cancellation before the bf16 output cast agrees with the reference.
// Adapted from NumPy v2.3.4, numpy/_core/src/umath/loops_trigonometric.dispatch.cpp.
/*
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
private let deepseekV4RoPEPhasesKernel = MLXFast.metalKernel(
    name: "deepseek_v4_rope_phases_f32",
    inputNames: ["angles"], outputNames: ["cosine", "sine"],
    source: """
        const uint i = thread_position_in_grid.x;
        if (i >= N) { return; }
        const float x = angles[i];
        if (!isfinite(x) || abs(x) > 117435.992f) {
            cosine[i] = metal::precise::cos(x);
            sine[i] = metal::precise::sin(x);
            return;
        }
        // Explicit boundaries keep multiplication and rounding separate.
        volatile float quadrant = x * 0x1.45f306p-1f;
        quadrant = quadrant + 0x1.800000p+23f;
        quadrant = quadrant - 0x1.800000p+23f;
        float reduced = fma(float(quadrant), -0x1.921fb0p+00f, x);
        reduced = fma(float(quadrant), -0x1.5110b4p-22f, reduced);
        reduced = fma(float(quadrant), -0x1.846988p-48f, reduced);
        const float squared = reduced * reduced;
        float c = fma(0x1.98e616p-16f, squared, -0x1.6c06dcp-10f);
        c = fma(c, squared, 0x1.55553cp-05f);
        c = fma(c, squared, -0.5f);
        c = fma(c, squared, 1.0f);
        float s = fma(0x1.7d3bbcp-19f, squared, -0x1.a06bbap-13f);
        s = fma(s, squared, 0x1.11119ap-07f);
        s = fma(s, squared, -0x1.555556p-03f);
        s = fma(s, squared, 0.0f);
        s = fma(s, reduced, reduced);
        const int q = int(float(quadrant));
        const float sn = (q & 1) == 0 ? s : c;
        const float cs = ((q + 1) & 1) == 0 ? s : c;
        sine[i] = (q & 2) == 0 ? sn : -sn;
        cosine[i] = abs(x) > 71476.0625f ? metal::precise::cos(x)
            : (((q + 1) & 2) == 0 ? cs : -cs);
        """
)

func deepseekV4RoPEPhases(_ angles: MLXArray) -> (MLXArray, MLXArray) {
    precondition(angles.dtype == .float32)
    guard Device.defaultDevice().deviceType == .gpu else {
        return (cos(angles), sin(angles))
    }
    let output = deepseekV4RoPEPhasesKernel(
        [angles], template: [("N", angles.size)],
        grid: (angles.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [angles.shape, angles.shape], outputDTypes: [.float32, .float32])
    return (output[0], output[1])
}
