import MLX
import MLXNN

// The same operations as `silu(gate) * x` (silu is `x * sigmoid(x)`), traced
// into one kernel as mlx-lm's `swiglu` is, so decode launches one kernel per
// MLP instead of two (#2373).
private let compiledSwiGLU = compile(shapeless: true) { (gate: MLXArray, x: MLXArray) in
    gate * sigmoid(gate) * x
}

/// `silu(gate) * x` as one compiled kernel. Mixed dtypes keep the two-kernel
/// form, so type promotion never happens inside the trace.
package func swiGLU(_ gate: MLXArray, _ x: MLXArray) -> MLXArray {
    guard gate.dtype == x.dtype else { return silu(gate) * x }
    return compiledSwiGLU(gate, x)
}
