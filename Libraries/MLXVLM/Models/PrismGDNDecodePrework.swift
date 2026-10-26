import MLX
import MLXLMCommon
import MLXNN

// Every operand is an input, including weights replaced after the first trace.
private let compiledPrismConv4 = MLXLMCommon.compile { (arguments: [MLXArray]) in
    let qkv = arguments[0]
    let state = arguments[1]
    let weight = arguments[2]
    var acc = state[0..., 0, 0...].asType(.float32) * weight[0..., 0, 0].asType(.float32)
    for tap in 1..<4 {
        let row = tap < 3 ? state[0..., tap, 0...] : qkv[0..., 0, 0...]
        acc = acc + row.asType(.float32) * weight[0..., tap, 0].asType(.float32)
    }
    // Convolution starts its accumulator at +0. Keep negative nonzero FP32
    // values intact so underflow during the activation cast still produces -0.
    acc = MLX.where(acc .== 0, MLXArray.zeros(acc.shape, dtype: .float32), acc)
    return [
        silu(acc.asType(qkv.dtype).reshaped(1, 1, qkv.dim(2))),
        contiguous(concatenated([state[0..., 1..., 0...], qkv], axis: 1)),
    ]
}

func prismConv4Decode(qkv: MLXArray, state: MLXArray, weight: MLXArray) -> (
    activated: MLXArray, state: MLXArray
) {
    let result = compiledPrismConv4([qkv, state, weight])
    return (result[0], result[1])
}
