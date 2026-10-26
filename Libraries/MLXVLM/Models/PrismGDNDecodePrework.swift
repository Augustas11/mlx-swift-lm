import MLX
import MLXNN

// The default custom-kernel math mode is Safe, matching the reference build.
private let prismConv4Kernel = MLXFast.metalKernel(
    name: "prism_conv4_state", inputNames: ["qkv", "state", "weight"],
    outputNames: ["conv", "next_state"],
    source: """
        uint c = thread_position_in_grid.x;
        if (c >= C) return;
        float acc = 0.0;
        for (int tap = 0; tap < 4; ++tap) {
            float value = tap < 3
                ? static_cast<float>(state[tap * state_strides[1] + c * state_strides[2]])
                : static_cast<float>(qkv[c * qkv_strides[2]]);
            acc += value * weight[c * weight_strides[0] + tap * weight_strides[1]];
        }
        conv[c] = acc;
        next_state[c] = static_cast<StateT>(state[state_strides[1] + c * state_strides[2]]);
        next_state[C + c] = static_cast<StateT>(state[2 * state_strides[1] + c * state_strides[2]]);
        next_state[2 * C + c] = static_cast<StateT>(qkv[c * qkv_strides[2]]);
        """, ensureRowContiguous: false)

func prismConv4Decode(qkv: MLXArray, state: MLXArray, weight: MLXArray) -> (
    conv: MLXArray, activated: MLXArray, state: MLXArray
) {
    precondition(weight.dtype == .float32)
    precondition([DType.float16, .float32].contains(qkv.dtype))
    precondition([DType.float16, .float32].contains(state.dtype))
    let stateType: DType = qkv.dtype == .float32 || state.dtype == .float32 ? .float32 : .float16
    let channels = qkv.dim(2)
    let result = prismConv4Kernel(
        [qkv, state, weight], template: [("C", channels), ("StateT", stateType)],
        grid: (channels, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, channels], [1, 3, channels]], outputDTypes: [.float32, stateType])
    return (result[0], silu(result[0]), result[1])
}

#if DEBUG
    struct PrismConvDecodeObservation {
        let admitted: Bool
        let inputType: DType
        let qkvType: DType
        let stateType: DType
        let weightType: DType
    }
#endif
