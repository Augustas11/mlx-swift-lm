// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Protocol for Multi-Token Prediction (MTP) speculative drafter models.
///
/// Mirrors `EmbeddingModel`'s relationship to `BaseLanguageModel`: this
/// protocol refines `BaseLanguageModel` with drafter-specific surface, so
/// implementations inherit weight loading and `sanitize` hooks while defining
/// their own forward signature.
///
/// MTP drafters do **not** conform to `LanguageModel` — their I/O contract is
/// different: a drafter consumes the target's last hidden state and per
/// layer-type shared K/V, produces a block of K-1 candidate tokens in a
/// single call, and holds no transient round-state between calls. The
/// `MTPSpeculativeTokenIterator` extracts the shared K/V from the target's
/// `LMOutput.state` and threads it to the drafter as a method argument.
///
/// Conforming types are expected to be stateless with respect to the target
/// model: every per-round input — including the target itself — flows through
/// `draftBlock(...)` as a method parameter. This makes drafter instances safe
/// to share across iterators without per-iterator mutable state.
public protocol MTPDrafterModel: BaseLanguageModel {
    /// Largest total verification block the drafter can produce efficiently.
    /// `nil` leaves the choice to the caller.
    var maximumBlockSize: Int? { get }

    /// Whether the drafter consumes target-emitted shared K/V state.
    var requiresSharedTargetKV: Bool { get }

    /// Whether the drafter needs a shifted-prompt prefill before proposing.
    var requiresPromptPrefill: Bool { get }

    /// Whether only greedy sampling currently has exact acceptance semantics.
    var requiresGreedySampling: Bool { get }

    /// K-step drafting from a constant position.
    ///
    /// Returns the proposed tokens as a `[B, blockSize - 1]` MLXArray. The
    /// drafter holds no transient round-state between calls — every per-round
    /// input is threaded as a method argument.
    ///
    /// - Parameters:
    ///   - target: The main language model this drafter is speculating for.
    ///     Used to look up target-derived constants (input embeddings, embed
    ///     scale, etc.) inline per round; conformers must not retain or
    ///     mutate references derived from `target`.
    ///   - lastToken: Bonus token from the target's last verify pass, shape `[B]`.
    ///   - lastHidden: Target's last hidden state, shape `[B, 1, backbone_hidden_size]`.
    ///   - sharedKV: Dict keyed by `layer_type` (`"full_attention"` /
    ///     `"sliding_attention"`) mapping to `(keys, values)` `MLXArray`s for
    ///     the last layer of that layer-type in the target.
    ///   - queryOffset: Constant absolute position for the round (the
    ///     position the bonus token sits at in the target's KV cache).
    ///     Passed as a Swift `Int` rather than an `MLXArray` to avoid the
    ///     `.item()` round-trip that would otherwise stall the GPU per
    ///     speculation round.
    ///   - blockSize: Total tokens in the round (the drafter returns
    ///     `blockSize - 1`; the bonus token is implicit).
    ///   - sampler: `LogitSampler` to apply to each step's logits.
    /// - Returns: `[B, blockSize - 1]` token array.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray
}

extension MTPDrafterModel {
    public var maximumBlockSize: Int? { nil }
    public var requiresSharedTargetKV: Bool { true }
    public var requiresPromptPrefill: Bool { false }
    public var requiresGreedySampling: Bool { false }

    /// Position-aware drafting entry point used by packed schedulers.
    ///
    /// The compatibility default preserves the 3.31.4 stateless-drafter
    /// behavior. Architectures with target-specific continuation positions can
    /// override it without changing existing conformers.
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler)
    }
}

/// Target capability for an exact in-place speculative cache rewind.
public protocol SpeculativeCacheRewindModel {
    var maximumNativeTargetCacheRewind: Int { get }
}

/// Iterator- or scheduler-owned transient state for a stateful MTP drafter.
public struct MTPDrafterState {
    public var cache: [KVCache]
    public var nextPosition: Int
    public var seedToken: MLXArray?
    public var seedHidden: MLXArray?
    public var proposalAppended: Int

    public init(
        cache: [KVCache],
        nextPosition: Int = 0,
        seedToken: MLXArray? = nil,
        seedHidden: MLXArray? = nil,
        proposalAppended: Int = 0
    ) {
        self.cache = cache
        self.nextPosition = nextPosition
        self.seedToken = seedToken
        self.seedHidden = seedHidden
        self.proposalAppended = proposalAppended
    }
}

/// MTP drafter whose mutable row state is owned outside the model instance.
public protocol StatefulMTPDrafterModel: MTPDrafterModel {
    func makeState(parameters: GenerateParameters?) -> MTPDrafterState

    func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray

    func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )
}

extension StatefulMTPDrafterModel {
    public func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {}

    public func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        let rejected = draftTokens.dim(-1) - acceptedCount
        if rejected > 0 {
            trimPromptCache(state.cache, numTokens: rejected)
        }
    }
}

/// Lightweight context for an MTP drafter — simpler than `ModelContext`
/// because drafters have no tokenizer, no user input processor, no chat
/// template.
///
/// Not `Sendable`; cross-domain access goes through ``MTPDrafterContainer``.
public struct MTPDrafterContext {
    public var configuration: ModelConfiguration
    public var model: any MTPDrafterModel

    public init(configuration: ModelConfiguration, model: any MTPDrafterModel) {
        self.configuration = configuration
        self.model = model
    }
}

/// Sendable container for an ``MTPDrafterContext``.
///
/// Mirrors the ``ModelContainer`` pattern: a `final class : Sendable` that
/// wraps the non-Sendable context in a `SerialAccessContainer` and exposes
/// async `perform(_:)` closures for serialized access.
public final class MTPDrafterContainer: Sendable {
    private let context: SerialAccessContainer<MTPDrafterContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public init(context: consuming MTPDrafterContext) {
        self.context = .init(context)
    }

    /// Perform an action on the ``MTPDrafterContext``. Callers _must_ eval
    /// any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    public func perform<R: Sendable>(
        _ action: @Sendable (MTPDrafterContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    /// Perform serialized access while moving row-owned non-Sendable state
    /// into and out of the isolation domain.
    public func perform<V, R>(
        nonSendable values: consuming V,
        _ action: @Sendable (MTPDrafterContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            SendableBox(try await action($0, values.consume()))
        }.consume()
    }
}

// MARK: - Cross-model state keys
//
// Public ``LMOutput/Key`` declarations for MTP speculative decoding. The
// target model (e.g. ``Gemma4`` in MLXVLM) writes these into its
// ``LMOutput/state`` when the iterator opts in via ``mtpEmitFlagKey``; the
// ``MTPSpeculativeTokenIterator`` reads them and threads them to the drafter
// as method arguments. Public scope is required because writer and reader
// live in different modules.

/// Target writes its post-final-norm hidden state here (pre-lm_head,
/// pre-softcap). ``MTPSpeculativeTokenIterator`` reads it and threads it to
/// the drafter as `lastHidden`.
public let mtpLastHiddenStatesKey =
    LMOutput.Key<MLXArray>("mtp.lastHiddenStates")

/// Target writes one `(keys, values)` tuple per `layer_type`
/// (`"full_attention"`, `"sliding_attention"`) here, drawn from the last
/// layer of each type. ``MTPSpeculativeTokenIterator`` reads it and threads
/// it to the drafter as `sharedKV`.
public let mtpSharedKVStatesKey =
    LMOutput.Key<[String: (MLXArray, MLXArray)]>("mtp.sharedKVStates")

/// Absolute target cache offsets for each emitted shared-K/V layer type.
public let mtpSharedKVOffsetsKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVOffsets")

/// Optional target-specific position delta state for continuation.
public let mtpPositionDeltasKey =
    LMOutput.Key<MLXArray>("mtp.positionDeltas")

/// The MTP iterator sets this key on the ``LMOutput/State`` it passes into
/// the main model on each call to opt the target into emitting
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey``. An absent key
/// reads as `false` (no emit), so non-MTP callers are unaffected.
public let mtpEmitFlagKey = LMOutput.Key<Bool>("mtp.emitDrafterState")

/// Requests a recurrent checkpoint after this many verification inputs.
public let mtpCacheCheckpointIndexKey =
    LMOutput.Key<Int>("mtp.cacheCheckpointIndex")

/// Cache entry that supplied each shared-K/V tuple.
public let mtpSharedKVSourceIndicesKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVSourceIndices")

// MARK: - Iterator stats surface

/// Introspection surface for token iterators that perform MTP speculative
/// decoding.
///
/// The iterator value lives inside `generateLoopTask` and never escapes the
/// stream, so its per-stream draft proposal and acceptance counters are not
/// reachable through the high-level `generate(...)` API. Conforming the
/// iterator to this protocol lets `generateLoopTask` downcast and thread the
/// counters through the emitted `.info` event's
/// ``GenerateCompletionInfo/proposedDraftTokens``,
/// ``GenerateCompletionInfo/acceptedDraftTokens``, and
/// ``GenerateCompletionInfo/passthroughReason`` fields. Non-MTP iterators do
/// not conform; the downcast returns nil and the fields default to nil for
/// non-MTP streams.
public protocol MTPStatsCollecting {
    /// Total tokens proposed across all speculation rounds in the stream.
    var proposedDraftTokens: Int { get }

    /// Total tokens accepted by the target across all speculation rounds.
    var acceptedDraftTokens: Int { get }

    /// nil if the iterator stayed in speculative mode for the full stream;
    /// non-nil if sticky-passthrough engaged, with the reason string captured
    /// at the moment of engagement.
    var passthroughReason: String? { get }
}
