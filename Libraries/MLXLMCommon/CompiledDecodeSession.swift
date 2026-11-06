// Copyright © 2026 Apple Inc.
//
// Port of ml-explore/mlx-swift-lm#550 (MIT). One deliberate difference: the
// model's weights are declared as compile inputs through a one-time snapshot
// of `innerState()`, as `CompiledTrace` does, instead of passing the module
// itself, which would re-flatten the module tree on every decode step.

import Foundation
import MLX

/// Errors raised before a compiled decode step can issue an invalid cache write.
public enum CompiledDecodeSessionError: Error, LocalizedError, Equatable {
    case invalidCapacity(Int)
    case invalidPromptShape([Int])
    case emptyPrompt
    case promptExceedsCapacity(promptTokens: Int, capacity: Int)
    case invalidTokenShape([Int], expectedBatchSize: Int)
    case capacityExceeded(capacity: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidCapacity(let capacity):
            "Compiled decode capacity must be positive and fit in Int32; got \(capacity)."
        case .invalidPromptShape(let shape):
            "Compiled decode requires prompt tokens shaped [batch, sequence]; got \(shape)."
        case .emptyPrompt:
            "Compiled decode requires at least one prompt token."
        case .promptExceedsCapacity(let promptTokens, let capacity):
            "Prompt length \(promptTokens) exceeds compiled decode capacity \(capacity)."
        case .invalidTokenShape(let shape, let expectedBatchSize):
            "Compiled decode requires one token per row with shape [\(expectedBatchSize), 1]; got \(shape)."
        case .capacityExceeded(let capacity):
            "Compiled decode capacity \(capacity) is exhausted."
        }
    }
}

/// Build the compiled single-token decode step over `caches`.
///
/// Weights are compile inputs, so parameter updates written through
/// `update(parameters:)` are read on every call rather than frozen as trace
/// constants. Replacing modules (loading or fusing adapters, quantizing)
/// requires a new step. Each call returns a new compiled function: dropping
/// it erases its traces from every thread's compile cache, so a step that is
/// abandoned (after a failure, or to grow capacity) never leaves a trace a
/// later step could replay.
func makeFixedCapacityDecodeStep(
    model: any FixedCapacityKVCacheProviding, caches: [FixedCapacityKVCache]
) -> (MLXArray) -> MLXArray {
    let weights: [MLXArray] = model.innerState()
    var inputs: [any Updatable] = [weights]
    inputs.append(contentsOf: caches.map { $0 as any Updatable })
    let outputs: [any Updatable] = caches.map { $0 as any Updatable }
    return MLX.compile(inputs: inputs, outputs: outputs) { (token: MLXArray) -> MLXArray in
        model(token, cache: caches)
    }
}

/// Owns a fixed-capacity cache and a compiled, capacity-safe decode step.
///
/// Construction performs prefill eagerly, evaluates the allocated cache state,
/// and only then creates the compiled closure. ``step(_:)`` accepts exactly one
/// token per batch row and checks capacity on the host before dispatching the
/// graph. This prevents out-of-range `putAlong` writes without synchronizing the
/// graph-valued cache position on every decode step.
///
/// This session supports text-only, same-length batches. Keep it within the same
/// serialized model access that owns `model`; it is intentionally not `Sendable`.
/// Finish the session before replacing modules (for example, loading or fusing
/// LoRA adapters or quantizing), then create a new session for the updated model.
///
/// ``TokenIterator`` uses the same cache and compiled step for its generic
/// serial decode when ``GenerateParameters/compiledDecode`` is set; see
/// <doc:compiled-kv-cache>.
public final class CompiledDecodeSession {
    public let capacity: Int
    public let batchSize: Int
    public let prefillLogits: MLXArray

    /// Number of prompt and decode-input tokens consumed by the model.
    public private(set) var processedTokenCount: Int

    /// Number of additional single-token decode calls that are safe.
    public var remainingCapacity: Int {
        capacity - processedTokenCount
    }

    /// Per-layer positions for diagnostics. Reading this synchronizes the
    /// graph-valued positions to the host; do not poll it in a decode loop.
    public var cacheOffsets: [Int] {
        caches.map(\.offset)
    }

    private let caches: [FixedCapacityKVCache]
    private let decode: (MLXArray) -> MLXArray

    /// Prefill `model` and prepare a compiled single-token decode step.
    ///
    /// - Parameters:
    ///   - model: A model audited for graph-valued cache positions.
    ///   - prompt: Token IDs shaped `[batch, sequence]`. All rows share one
    ///     sequence length and cache timeline.
    ///   - capacity: Total sequence capacity, including the prompt and every
    ///     token subsequently passed to ``step(_:)``.
    public init(
        model: any FixedCapacityKVCacheProviding,
        prompt: MLXArray,
        capacity: Int
    ) throws {
        guard capacity > 0, capacity <= Int(Int32.max) else {
            throw CompiledDecodeSessionError.invalidCapacity(capacity)
        }
        guard prompt.ndim == 2 else {
            throw CompiledDecodeSessionError.invalidPromptShape(prompt.shape)
        }
        let promptTokenCount = prompt.dim(1)
        guard prompt.dim(0) > 0, promptTokenCount > 0 else {
            throw CompiledDecodeSessionError.emptyPrompt
        }
        guard promptTokenCount <= capacity else {
            throw CompiledDecodeSessionError.promptExceedsCapacity(
                promptTokens: promptTokenCount, capacity: capacity)
        }

        let caches = try model.newFixedCapacityCache(maxTokens: capacity)
        let prefillLogits = model(prompt, cache: caches)
        eval(prefillLogits, caches)

        self.capacity = capacity
        self.batchSize = prompt.dim(0)
        self.caches = caches
        self.prefillLogits = prefillLogits
        self.processedTokenCount = promptTokenCount
        self.decode = makeFixedCapacityDecodeStep(model: model, caches: caches)
    }

    /// Run one compiled decode step after validating shape and capacity.
    public func step(_ token: MLXArray) throws -> MLXArray {
        guard token.ndim == 2, token.dim(0) == batchSize, token.dim(1) == 1 else {
            throw CompiledDecodeSessionError.invalidTokenShape(
                token.shape, expectedBatchSize: batchSize)
        }
        guard processedTokenCount < capacity else {
            throw CompiledDecodeSessionError.capacityExceeded(capacity: capacity)
        }

        let logits = decode(token)
        processedTokenCount += 1
        return logits
    }
}
