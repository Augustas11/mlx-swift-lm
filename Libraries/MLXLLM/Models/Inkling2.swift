// Inkling 2 Mini (`inkling2_loop`) text model.
//
// Port of mlx-vlm's `models/inkling2` (Blaizzy/mlx-vlm@f1618d6d, which contains
// "Add Inkling 2 Mini model." 111ac668; Apache-2.0, adapted from Transformers' Inkling2) for
// thinkingmachines/Inkling-2-Mini (Apache-2.0).
//
// A 24-layer MoE decoder whose middle layers [core_start, core_end) run `num_passes` times on
// shared weights: Mini makes 48 layer visits per token. Every visit has its own attention cache
// and convolution states. Passes after the first start again from the prelude's state and read
// the previous pass's completed residual blocks as "messages" that low-rank controllers blend
// into each sublayer input.
//
// This file holds the configuration, the visit schedule, the loop driver, the module tree, the
// layer arithmetic and the checkpoint layouts. The module tree follows the oQ layout of
// mlx-community/Inkling-2-Mini-oQ4e-mtp (oMLX's runtime names, jundot/omlx@a0ed2f59
// `omlx/patches/inkling2/oq_layout.py`, Apache-2.0); `Inkling2WeightLayout` maps the source
// checkpoint (`model.llm.*`) onto it.
//
// The arithmetic follows mlx-vlm's portable (non-Metal-kernel) path operation by operation:
// each part of a fused projection is computed from its own rows, as the reference computes the
// separate projection; the short convolutions, residual mixes and controllers keep float32
// where the reference does. Not ported yet: padded batches (the reference's conv padding mask),
// speculative cache recording, and the reference's Metal kernels (band mask, decode short
// convolution, masked expert products), whose float32 accumulation order may differ in the last
// bit.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Inkling2LoopConfiguration: Codable, Sendable, Equatable {
    /// First looped layer.
    public let coreStart: Int
    /// One past the last looped layer.
    public let coreEnd: Int
    public let passes: Int
    /// Rank of the message controllers.
    public let gateRank: Int
    public let isVarloop: Bool

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
        case isVarloop = "is_varloop"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        coreStart = try c.decode(Int.self, forKey: .coreStart)
        coreEnd = try c.decode(Int.self, forKey: .coreEnd)
        passes = try c.decode(Int.self, forKey: .passes)
        gateRank = try c.decode(Int.self, forKey: .gateRank)
        isVarloop = try c.decodeIfPresent(Bool.self, forKey: .isVarloop) ?? false
    }
}

public struct Inkling2TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    /// Logits past this id are padding and are dropped.
    public let unpaddedVocabularySize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let slidingAttentionHeads: Int
    public let slidingKVHeads: Int
    public let slidingHeadDim: Int
    public let slidingWindow: Int
    /// Layers with local (windowed) attention; the others attend to every key.
    public let localLayerIds: [Int]
    public let relativeDim: Int
    public let relativeExtent: Int
    public let logScalingFloor: Int?
    public let logScalingAlpha: Float
    public let rmsNormEps: Float
    public let useValueNorm: Bool
    public let qkClipThreshold: Float
    public let relativeClipThreshold: Float
    public let convolutionKernelSize: Int
    public let layersPerBlock: Int
    public let attentionResidualNormalizesValues: Bool
    public let experts: Int
    public let expertsPerToken: Int
    public let sharedExperts: Int
    public let intermediateSize: Int
    public let gateSoftcap: Float
    public let upSoftcap: Float
    public let routeScale: Float
    public let useGateBias: Bool
    public let finalLogitSoftcap: Float
    public let loop: Inkling2LoopConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingAttentionHeads = "swa_num_attention_heads"
        case slidingKVHeads = "swa_num_key_value_heads"
        case slidingHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case relativeDim = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case useValueNorm = "use_value_norm_no_gamma"
        case qkClipThreshold = "qk_l2_clip_threshold"
        case relativeClipThreshold = "rel_l2_clip_threshold"
        case convolutionKernelSize = "sconv_kernel_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case attentionResidualNormalizesValues = "attn_res_normalize_values"
        case experts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts"
        case intermediateSize = "intermediate_size"
        case gateSoftcap = "mlp_gate_softcap"
        case upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case useGateBias = "use_gate_bias"
        case finalLogitSoftcap = "final_logit_softcapping"
        case loop = "loop_config"
    }

    private enum LayerTypesKey: String, CodingKey {
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        unpaddedVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize) ?? vocabularySize
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        slidingAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .slidingAttentionHeads) ?? attentionHeads
        slidingKVHeads = try c.decodeIfPresent(Int.self, forKey: .slidingKVHeads) ?? kvHeads
        slidingHeadDim = try c.decodeIfPresent(Int.self, forKey: .slidingHeadDim) ?? headDim
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        // The reference order of precedence: layer_types, then local_layer_ids, then every
        // sixth layer global.
        let types = try decoder.container(keyedBy: LayerTypesKey.self)
            .decodeIfPresent([String].self, forKey: .layerTypes)
        if let types {
            localLayerIds = types.indices.filter { types[$0] == "hybrid_sliding" }
        } else if let ids = try c.decodeIfPresent([Int].self, forKey: .localLayerIds) {
            localLayerIds = ids
        } else {
            localLayerIds = (0 ..< hiddenLayers).filter { ($0 + 1) % 6 != 0 }
        }
        relativeDim = try c.decode(Int.self, forKey: .relativeDim)
        relativeExtent = try c.decode(Int.self, forKey: .relativeExtent)
        logScalingFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingFloor)
        logScalingAlpha = try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        useValueNorm = try c.decodeIfPresent(Bool.self, forKey: .useValueNorm) ?? true
        qkClipThreshold =
            try c.decodeIfPresent(Float.self, forKey: .qkClipThreshold) ?? 22.627416997969522
        relativeClipThreshold =
            try c.decodeIfPresent(Float.self, forKey: .relativeClipThreshold) ?? 8.0
        convolutionKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .convolutionKernelSize) ?? 4
        layersPerBlock = try c.decodeIfPresent(Int.self, forKey: .layersPerBlock) ?? 6
        attentionResidualNormalizesValues =
            try c.decodeIfPresent(Bool.self, forKey: .attentionResidualNormalizesValues) ?? true
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        gateSoftcap = try c.decodeIfPresent(Float.self, forKey: .gateSoftcap) ?? 4.0
        upSoftcap = try c.decodeIfPresent(Float.self, forKey: .upSoftcap) ?? 25.0
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8.0
        useGateBias = try c.decodeIfPresent(Bool.self, forKey: .useGateBias) ?? true
        finalLogitSoftcap = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 30.0
        loop = try c.decodeIfPresent(Inkling2LoopConfiguration.self, forKey: .loop)
    }

    public func isSliding(_ layer: Int) -> Bool { localLayerIds.contains(layer) }

    /// A layer that closes an attention-residual block appends its input to the block list
    /// instead of adding the attention residual.
    public func closesBlock(_ layer: Int) -> Bool { layer % layersPerBlock == 0 }

    public func isCore(_ layer: Int) -> Bool {
        guard let loop else { return false }
        return loop.coreStart <= layer && layer < loop.coreEnd
    }

    /// The layer each visit runs, in order: the reference's `layer_indices`.
    public var layerVisits: [Int] {
        let layers = Array(0 ..< hiddenLayers)
        guard let loop else { return layers }
        return Array(layers[..<loop.coreStart])
            + Array(repeating: Array(layers[loop.coreStart ..< loop.coreEnd]), count: loop.passes)
            .flatMap { $0 }
            + Array(layers[loop.coreEnd...])
    }
}

/// The bundled DSpark draft (`mtp_config`).
public struct Inkling2DSparkConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let localLayerIds: [Int]
    public let slidingWindow: Int
    /// One anchor plus `blockSize - 1` proposals.
    public let blockSize: Int
    public let markovRank: Int
    public let markovHeadType: String
    /// Target layer visits whose input the draft reads; the last is the final norm.
    public let targetLayerIds: [Int]
    public let queryBias: Bool
    public let outputBias: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case localLayerIds = "local_layer_ids"
        case slidingWindow = "sliding_window_size"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case targetLayerIds = "target_layer_ids"
        case queryBias = "q_bias"
        case outputBias = "o_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        localLayerIds = try c.decode([Int].self, forKey: .localLayerIds)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        markovRank = try c.decode(Int.self, forKey: .markovRank)
        markovHeadType = try c.decode(String.self, forKey: .markovHeadType)
        targetLayerIds = try c.decode([Int].self, forKey: .targetLayerIds)
        queryBias = try c.decodeIfPresent(Bool.self, forKey: .queryBias) ?? false
        outputBias = try c.decodeIfPresent(Bool.self, forKey: .outputBias) ?? false
    }
}

public struct Inkling2Configuration: Codable, Sendable {
    public let modelType: String
    public let text: Inkling2TextConfiguration
    public let dspark: Inkling2DSparkConfiguration?
    public let eosTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case dspark = "mtp_config"
        case eosTokenId = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        text = try c.decode(Inkling2TextConfiguration.self, forKey: .text)
        dspark = try c.decodeIfPresent(Inkling2DSparkConfiguration.self, forKey: .dspark)
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId)
    }
}

public struct Inkling2ConfigurationError: LocalizedError, Equatable {
    public let message: String
    public var errorDescription: String? { "Inkling 2 configuration: \(message)" }
}

extension Inkling2Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ message: String) throws {
            if !condition { throw Inkling2ConfigurationError(message: message) }
        }
        let t = text
        try require(
            modelType == "inkling2_loop" && t.modelType == "inkling2_loop_text",
            "only inkling2_loop (Inkling 2 Mini) is supported, got \(modelType)/\(t.modelType)")
        // The reference's loop checks (`TextConfig.__post_init__`).
        guard let loop = t.loop else { throw Inkling2ConfigurationError(message: "missing loop_config") }
        try require(
            0 < loop.coreStart && loop.coreStart < loop.coreEnd && loop.coreEnd <= t.hiddenLayers
                && t.layersPerBlock > 0 && loop.coreStart % t.layersPerBlock == 0
                && loop.coreEnd % t.layersPerBlock == 0 && loop.passes >= 1
                && loop.gateRank >= 1 && !loop.isVarloop,
            "unsupported loop_config")
        try require(
            Set(t.localLayerIds).count == t.localLayerIds.count
                && t.localLayerIds.allSatisfy { 0 <= $0 && $0 < t.hiddenLayers },
            "local_layer_ids must be distinct layer indices")
        try require(
            t.kvHeads > 0 && t.attentionHeads % t.kvHeads == 0 && t.slidingKVHeads > 0
                && t.slidingAttentionHeads % t.slidingKVHeads == 0,
            "attention heads must be a multiple of key/value heads")
        try require(
            t.hiddenSize > 0 && t.headDim > 0 && t.slidingHeadDim > 0 && t.relativeDim > 0,
            "dimensions must be positive")
        try require(
            t.slidingWindow > 0 && t.relativeExtent > 0, "window and relative extent must be positive")
        try require(t.convolutionKernelSize >= 1, "sconv_kernel_size must be positive")
        try require(
            t.experts > 0 && 0 < t.expertsPerToken && t.expertsPerToken <= t.experts,
            "num_experts_per_tok must be in 1...n_routed_experts")
        try require(t.sharedExperts == 0, "shared experts are not supported")
        try require(
            0 < t.unpaddedVocabularySize && t.unpaddedVocabularySize <= t.vocabularySize,
            "unpadded_vocab_size must not exceed vocab_size")
        if let d = dspark {
            let visits = Inkling2LoopSchedule(t).count
            try require(
                d.markovHeadType == "rnn_hidden" && !d.queryBias && !d.outputBias,
                "unsupported mtp_config head")
            try require(d.hiddenSize == t.hiddenSize, "mtp_config hidden_size must match the target")
            try require(d.blockSize >= 2 && d.hiddenLayers >= 1 && d.markovRank >= 1, "unsupported mtp_config")
            try require(
                d.kvHeads > 0 && d.attentionHeads % d.kvHeads == 0 && d.headDim % 2 == 0
                    && d.slidingWindow >= 1,
                "unsupported mtp_config attention")
            try require(
                d.localLayerIds.allSatisfy { 0 <= $0 && $0 < d.hiddenLayers },
                "mtp_config local_layer_ids out of range")
            let ids = d.targetLayerIds
            try require(
                ids.count >= 2 && ids == Array(Set(ids)).sorted() && ids[0] >= 0
                    && ids.last == visits,
                "mtp_config target_layer_ids must rise strictly and end at the visit count \(visits)")
        }
    }
}

// MARK: - Visit schedule

/// The order in which the decoder visits its layers, with each visit's role in the loop.
public struct Inkling2LoopSchedule: Sendable, Equatable {
    public struct Visit: Sendable, Equatable {
        /// Position in the schedule; the index of this visit's cache.
        public let index: Int
        public let layer: Int
        /// Pass through the looped core, from 0; 0 outside the core.
        public let pass: Int
        public let inCore: Bool
        public let isSliding: Bool
        public let closesBlock: Bool
    }

    public let visits: [Visit]
    public let coreStart: Int
    public let coreEnd: Int
    public let passes: Int

    public init(_ text: Inkling2TextConfiguration) {
        // Locals, not the stored properties: the closure below may not capture `self` before
        // `visits` is set.
        let start = text.loop?.coreStart ?? text.hiddenLayers
        let end = text.loop?.coreEnd ?? text.hiddenLayers
        coreStart = start
        coreEnd = end
        passes = text.loop?.passes ?? 1
        let coreLength = end - start
        visits = text.layerVisits.enumerated().map { index, layer in
            let inCore = text.isCore(layer)
            return Visit(
                index: index, layer: layer,
                pass: inCore ? (index - start) / coreLength : 0,
                inCore: inCore, isSliding: text.isSliding(layer),
                closesBlock: text.closesBlock(layer))
        }
    }

    public var count: Int { visits.count }

    /// The visits that run `layer`, in order.
    public func visits(ofLayer layer: Int) -> [Int] {
        visits.filter { $0.layer == layer }.map(\.index)
    }
}

// MARK: - Loop driver

/// Runs the visit schedule as the reference `TextModel.__call__` does, with the layer itself
/// supplied by the caller.
///
/// - At the first visit of the core the prelude's state and completed blocks are kept.
/// - At the start of each later pass the previous pass's core blocks (all but the first, which
///   is the prelude state the core's first layer appended) and its final state become the
///   message bank, RMS-normalised in float32; state and blocks go back to the prelude.
/// - Every core visit after the first pass gets one message per sublayer input
///   (`softmax(query . bank / sqrt(H)) . bank`, queries in the order attn, k, v, mlp).
public enum Inkling2LoopDriver {
    public struct Result {
        public let hidden: MLXArray
        public let blocks: [MLXArray]
        /// The state entering each captured visit; index `count` is not captured here.
        public let captures: [Int: MLXArray]
    }

    /// - Parameters:
    ///   - messageQuery: the `[4, H]` message queries of a core layer.
    ///   - visit: runs one visit: the visit, its input, the block list it may append to, and its
    ///     messages `[B, L, 4, H]` (nil in the first pass and outside the core).
    public static func run(
        schedule: Inkling2LoopSchedule, hidden: MLXArray, eps: Float = 1e-6,
        capturing: Set<Int> = [],
        messageQuery: (Int) -> MLXArray,
        visit: (Inkling2LoopSchedule.Visit, MLXArray, inout [MLXArray], MLXArray?) -> MLXArray
    ) -> Result {
        var x = hidden
        var blocks = [MLXArray]()
        var prelude: (hidden: MLXArray, blocks: [MLXArray])?
        var bank: MLXArray?
        var captures = [Int: MLXArray]()
        for step in schedule.visits {
            if step.layer == schedule.coreStart && step.inCore {
                if let saved = prelude {
                    let states = Array(blocks[(saved.blocks.count + 1)...]) + [x]
                    bank = Inkling2PhaseProfiler.measure(.messages) { messageBank(states, eps: eps) }
                    x = saved.hidden
                    blocks = saved.blocks
                } else {
                    prelude = (x, blocks)
                }
            }
            var messages: MLXArray?
            if let bank, step.inCore {
                messages = Inkling2PhaseProfiler.measure(.messages) {
                    Self.messages(query: messageQuery(step.layer), bank: bank)
                }
            }
            if capturing.contains(step.index) { captures[step.index] = x }
            x = visit(step, x, &blocks, messages)
        }
        return Result(hidden: x, blocks: blocks, captures: captures)
    }

    /// `rms_norm(stack(states, axis: -2), None, eps)` in float32: `[..., n, H]`.
    static func messageBank(_ states: [MLXArray], eps: Float) -> MLXArray {
        let states = MLX.stacked(states, axis: -2).asType(.float32)
        return MLXFast.rmsNorm(
            states, weight: MLXArray.ones([states.dim(-1)], dtype: .float32), eps: eps)
    }

    /// `softmax(query @ bank^T / sqrt(H)) @ bank` for `[4, H]` queries: `[..., 4, H]`.
    static func messages(query: MLXArray, bank: MLXArray) -> MLXArray {
        // `x.shape[-1] ** -0.5`: a double rounded once to float32.
        let scale = Float(pow(Double(bank.dim(-1)), -0.5))
        let scores = matmul(query, bank.swappedAxes(-1, -2)) * scale
        return matmul(softmax(scores, axis: -1), bank)
    }
}

// MARK: - Diagnostic phase timing

/// Wall time per phase of Inkling 2 forward passes: the decode time split of slice 5. Off unless
/// ``active`` is set. When it is set, every phase ends with an `eval` barrier, which removes the
/// overlap between phases and between host graph building and GPU work, so read the shares, not
/// the sum, against an unprofiled step. A phase's time includes building its graph on the host.
/// A nested phase (the K/V convolutions inside attention) is subtracted from the phase that
/// encloses it; its inputs are evaluated first, on the enclosing phase's clock. Lazy work outside
/// every phase (slices, casts, the residual add) runs in whichever barrier comes next.
/// Diagnostic only: never set it while serving.
public final class Inkling2PhaseProfiler {
    public enum Phase: String, CaseIterable, Sendable {
        case embed
        /// Attention residual mixes (sublayer inputs and the final input).
        case residualMix
        /// Sublayer input RMS norms.
        case norms
        /// Projections, clips, band mask, cache update and SDPA, output gate and projection.
        case attention
        /// Static and dynamic K/V convolutions and the two residual-path convolutions.
        case shortConvolutions
        case router
        /// Expert products, activation and route combine.
        case experts
        /// Message bank, message queries and the controllers that blend messages in.
        case messages
        /// Final norm, head product, softcap and vocabulary slice.
        case head
    }

    nonisolated(unsafe) public static var active: Inkling2PhaseProfiler?

    public private(set) var seconds = [Phase: Double]()
    public private(set) var calls = [Phase: Int]()
    private var nested = [Double]()

    public init() {}

    public func reset() {
        seconds = [:]
        calls = [:]
        nested = []
    }

    /// `inputs` is an autoclosure so that the unprofiled path allocates nothing.
    static func measure(
        _ phase: Phase, inputs: @autoclosure () -> [MLXArray] = [], _ body: () -> MLXArray
    ) -> MLXArray {
        measure(phase, inputs: inputs(), outputs: { [$0] }, body)
    }

    static func measure<T>(
        _ phase: Phase, inputs: @autoclosure () -> [MLXArray] = [], outputs: (T) -> [MLXArray],
        _ body: () -> T
    ) -> T {
        guard let profiler = active else { return body() }
        return profiler.run(phase, inputs: inputs(), outputs: outputs, body)
    }

    private func run<T>(
        _ phase: Phase, inputs: [MLXArray], outputs: (T) -> [MLXArray], _ body: () -> T
    ) -> T {
        if !inputs.isEmpty { eval(inputs) }
        nested.append(0)
        let start = DispatchTime.now().uptimeNanoseconds
        let result = body()
        eval(outputs(result))
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let inner = nested.removeLast()
        seconds[phase, default: 0] += elapsed - inner
        calls[phase, default: 0] += 1
        if !nested.isEmpty { nested[nested.count - 1] += elapsed }
        return result
    }
}

// MARK: - Expert formats

/// Storage of one stacked expert projection.
public struct Inkling2ExpertFormat: Equatable, Sendable {
    public let mode: QuantizationMode
    public let bits: Int
    public let groupSize: Int

    public init(mode: QuantizationMode, bits: Int, groupSize: Int) {
        self.mode = mode
        self.bits = bits
        self.groupSize = groupSize
    }

    /// NVFP4 with a per-expert float32 tensor scale (layers 0-19 of every Mini build).
    public static let nvfp4 = Inkling2ExpertFormat(mode: .nvfp4, bits: 4, groupSize: 16)
    /// MXFP8 (layers 20-23 of the source checkpoint).
    public static let mxfp8 = Inkling2ExpertFormat(mode: .mxfp8, bits: 8, groupSize: 32)

    public var hasBiases: Bool { mode == .affine }
    public var hasGlobalScale: Bool { mode == .nvfp4 }

    /// The format a checkpoint's stacked projection `prefix` holds, read from which tensors exist
    /// and their shapes; nil when `prefix.weight` is absent.
    public static func detect(prefix: String, in weights: [String: MLXArray], inputDims: Int)
        throws -> Inkling2ExpertFormat?
    {
        guard let weight = weights["\(prefix).weight"] else { return nil }
        guard let scales = weights["\(prefix).scales"], weight.dtype == .uint32,
            weight.ndim == 3, scales.ndim == 3, inputDims % scales.dim(-1) == 0,
            (weight.dim(-1) * 32) % inputDims == 0
        else {
            throw Inkling2ConfigurationError(message: "\(prefix) is not a packed expert projection")
        }
        let bits = weight.dim(-1) * 32 / inputDims
        let groupSize = inputDims / scales.dim(-1)
        let hasBiases = weights["\(prefix).biases"] != nil
        let hasGlobalScale = weights["\(prefix).weight_scale_2"] != nil
        let format: Inkling2ExpertFormat
        if hasBiases {
            format = Inkling2ExpertFormat(mode: .affine, bits: bits, groupSize: groupSize)
        } else if hasGlobalScale {
            format = Inkling2ExpertFormat(mode: .nvfp4, bits: bits, groupSize: groupSize)
        } else if bits == 8 {
            format = Inkling2ExpertFormat(mode: .mxfp8, bits: bits, groupSize: groupSize)
        } else {
            format = Inkling2ExpertFormat(mode: .mxfp4, bits: bits, groupSize: groupSize)
        }
        guard format.hasBiases == hasBiases, format.hasGlobalScale == hasGlobalScale,
            format.mode == .affine || scales.dtype == .uint8,
            format.mode != .nvfp4 || (bits == 4 && groupSize == 16),
            format.mode != .mxfp8 || groupSize == 32, format.mode != .mxfp4 || groupSize == 32
        else {
            throw Inkling2ConfigurationError(
                message: "\(prefix): unsupported expert storage (bits \(bits), group \(groupSize))")
        }
        return format
    }
}

/// The source checkpoint's `hf_quant_config.json` (`quant_method` "inkling2"): parameter groups
/// with a `weight_transform` each. `nvfp4_nearest` and `mxfp8_nearest` name expert tensors
/// (`...layers.N.mlp.experts.w13_weight` / `w2_weight`); `cast` groups keep their stored dtype.
/// Read as mlx-vlm's `Model.quantization_from_metadata` reads it: a layer the file names for one
/// projection only keeps NVFP4 for the other.
public enum Inkling2QuantizationMetadata {
    public struct LayerFormats: Equatable, Sendable {
        public let gateUp: Inkling2ExpertFormat
        public let down: Inkling2ExpertFormat
    }

    /// Declared formats by layer, or nil when the file is not Inkling 2 metadata.
    public static func expertFormats(from data: Data, layers: Int) throws -> [Int: LayerFormats]? {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            root["quant_method"] as? String == "inkling2"
        else { return nil }
        guard let model = root["model"] as? [String: Any],
            let groups = model["parameter_groups"] as? [[String: Any]]
        else {
            throw Inkling2ConfigurationError(message: "hf_quant_config.json has no model.parameter_groups")
        }
        var gateUps = [Int: Inkling2ExpertFormat]()
        var downs = [Int: Inkling2ExpertFormat]()
        for group in groups {
            let transform = group["weight_transform"] as? String ?? "(none)"
            let format: Inkling2ExpertFormat
            switch transform {
            case "cast": continue
            case "nvfp4_nearest": format = .nvfp4
            case "mxfp8_nearest": format = .mxfp8
            default:
                throw Inkling2ConfigurationError(message: "unsupported Inkling 2 weight transform \(transform)")
            }
            for tensor in group["tensors"] as? [String] ?? [] {
                guard let marker = tensor.range(of: "layers."),
                    let layer = Int(tensor[marker.upperBound...].prefix { $0 != "." }),
                    (0 ..< layers).contains(layer)
                else {
                    throw Inkling2ConfigurationError(message: "hf_quant_config.json names \(tensor)")
                }
                if tensor.hasSuffix("w13_weight") {
                    gateUps[layer] = format
                } else {
                    downs[layer] = format
                }
            }
        }
        var result = [Int: LayerFormats]()
        for layer in Set(gateUps.keys).union(downs.keys) {
            result[layer] = LayerFormats(gateUp: gateUps[layer] ?? .nvfp4, down: downs[layer] ?? .nvfp4)
        }
        return result
    }
}

// MARK: - Arithmetic helpers

/// `mx.fast.rms_norm(x, None, eps)`: RMS normalisation without a gain.
private func inkling2UnitRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
}

/// `l2_clip`: scale rows whose L2 norm exceeds `threshold` down to it, in float32.
private func inkling2L2Clip(_ x: MLXArray, threshold: Float, scale: MLXArray? = nil) -> MLXArray {
    var v = x.asType(.float32)
    if let scale { v = v * scale.asType(.float32) }
    let norm = v.square().sum(axis: -1, keepDims: true).sqrt()
    return (v * minimum(MLXArray(threshold) / (norm + Float(1e-6)), Float(1))).asType(x.dtype)
}

/// `x ** -0.5` for an integer `x`: a double rounded once to float32, as a Python scalar is.
private func inkling2InverseSquareRoot(_ x: Int) -> Float {
    Float(pow(Double(x), -0.5))
}

/// Consecutive row ranges of a fused projection with outputs of the given widths.
private func inkling2Ranges(_ widths: [Int]) -> [Range<Int>] {
    var start = 0
    return widths.map { width in
        defer { start += width }
        return start ..< (start + width)
    }
}

/// Rows `rows` of a (possibly quantized) projection applied to `x`: one part of a fused
/// projection, computed from that part's rows alone as the reference computes the separate
/// projection. Quantization groups run along the input dimension, so row slices are exact.
private func inkling2Project(_ linear: Linear, _ rows: Range<Int>, _ x: MLXArray) -> MLXArray {
    if rows.lowerBound == 0 && rows.upperBound == linear.weight.dim(0) {
        return linear(x)
    }
    let r = rows.lowerBound ..< rows.upperBound
    if let q = linear as? QuantizedLinear {
        return quantizedMM(
            x, q.weight[r], scales: q.scales[r], biases: q.biases?[r], transpose: true,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode)
    }
    return matmul(x, linear.weight[r].transposed())
}

/// The reference `ShortConvolution`: a depthwise causal convolution over the previous
/// `kernel - 1` inputs, with the residual added before the `1 / sqrt(kernel + 1)` scale, all in
/// float32. `state[slot]` holds the last `kernel - 1` inputs in float32.
private func inkling2ShortConvolution(
    _ x: MLXArray, taps: MLXArray, state: ArraysCache?, slot: Int
) -> MLXArray {
    if let out = inkling2KernelShortConvolution(x, taps: taps, state: state, slot: slot) {
        return out
    }
    let (batch, channels) = (x.dim(0), x.dim(2))
    let kernel = taps.dim(-1)
    let xf = x.asType(.float32)
    let previous = state?[slot] ?? MLXArray.zeros([batch, kernel - 1, channels], dtype: .float32)
    let context = concatenated([previous, xf], axis: 1)
    state?[slot] = context[0..., (context.dim(1) - (kernel - 1))..., 0...]
    // `mx.conv1d(context, weight [C, K, 1], groups=C)`: tap K - 1 multiplies the current input.
    let weight = taps.asType(.float32).reshaped(channels, kernel, 1)
    let out = conv1d(context, weight, groups: channels)
    return ((out + xf) / Float(Double(kernel + 1).squareRoot())).asType(x.dtype)
}

/// The reference `DynamicShortConvolution`: static taps plus taps projected from the
/// sublayer input, one set per key/value head; `1 / sqrt(2 kernel + 1)` scale, float32.
///
/// - Parameters:
///   - dynamicTaps: the tap projection of the sublayer input, `[B, L, heads * kernel]`.
///   - staticTaps: `[kernel, channels]`.
private func inkling2DynamicShortConvolution(
    _ x: MLXArray, dynamicTaps: MLXArray, staticTaps: MLXArray, heads: Int,
    state: ArraysCache?, slot: Int
) -> MLXArray {
    let (batch, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
    let kernel = staticTaps.dim(0)
    let xf = x.asType(.float32)
    let previous = state?[slot] ?? MLXArray.zeros([batch, kernel - 1, channels], dtype: .float32)
    let context = concatenated([previous, xf], axis: 1)
    state?[slot] = context[0..., (context.dim(1) - (kernel - 1))..., 0...]
    let dynamic = dynamicTaps.asType(.float32).reshaped(batch, length, heads, 1, kernel)
    let weights = staticTaps.transposed().reshaped(heads, -1, kernel) + dynamic
    var out = MLXArray.zeros([batch, length, heads, channels / heads], dtype: .float32)
    for lag in 0 ..< kernel {
        let start = kernel - 1 - lag
        let values = context[0..., start ..< (start + length), 0...].reshaped(out.shape)
        out = out + values * weights[.ellipsis, lag]
    }
    return ((out.reshaped(x.shape) + xf) / Float(Double(2 * kernel + 1).squareRoot()))
        .asType(x.dtype)
}

/// The reference `banded_additive_mask` (portable path): `[B, H, L, S]` additive logits in the
/// dtype of `rel`. Backward distances below `extent` get `rel . proj[:, distance]`; farther keys
/// in context get 0; future keys, and keys at or beyond `sliding` when it is positive, get -1e30.
///
/// - Parameters:
///   - rel: `[B, L, H, d_rel]` relative queries.
///   - proj: `[d_rel, extent]`.
///   - queryOffset: position of the first query among the `keys` keys.
private func inkling2BandedMask(
    _ rel: MLXArray, _ proj: MLXArray, queryOffset: Int, keys: Int, sliding: Int, extent: Int
) -> MLXArray {
    let dtype = rel.dtype
    let (batch, length, heads) = (rel.dim(0), rel.dim(1), rel.dim(2))
    let logits = matmul(rel, proj).transposed(0, 2, 1, 3)
    let queryPositions = MLXArray((0 ..< length).map { Int32($0 + queryOffset) })
    let keyPositions = MLXArray((0 ..< keys).map { Int32($0) })
    let distance = queryPositions.reshaped(length, 1) - keyPositions.reshaped(1, keys)
    let index = broadcast(
        clip(distance, min: Int32(0), max: Int32(extent - 1)).reshaped(1, 1, length, keys),
        to: [batch, heads, length, keys])
    var bias = takeAlong(logits, index, axis: -1)
    bias = which(distance .>= Int32(extent), MLXArray(Float(0)).asType(dtype), bias)
    var masked = distance .< Int32(0)
    if sliding > 0 {
        masked = masked .|| (distance .>= Int32(sliding))
    }
    return which(masked, MLXArray(Float(-1e30)).asType(dtype), bias).asType(dtype)
}

// MARK: - Fast decode path

/// mlx-vlm f1618d6d's fast path for Inkling 2, mirrored where mlx-swift can express it:
///
/// - `mx.compile` functions: `l2_clip`, `_softcapped_swiglu`, `_route`, `_combine_routes` and
///   `_attention_residual` (non-shapeless, as `@mx.compile` traces each shape);
/// - the `inkling2_virtual_short_convolution` Metal kernel for bf16 static short convolutions
///   with kernel 4, at most 4 rows, 8 tokens and channels a multiple of 128 (the reference's
///   GPU condition);
/// - the `inkling_banded_mask_v2` Metal kernel for the band bias;
/// - the head's float32 unembedding kept once instead of promoted every step (plain `Linear`
///   heads, at most 15 rows, GPU).
///
/// Not mirrored: the NVFP4 tensor scale inside `gather_qmm` (the pinned mlx-swift binding has no
/// `global_scale` argument; the product is scaled after it, in one compiled multiply and cast),
/// and `rms_norm(x, None)` (mlx-swift needs a weight; a ones vector is passed).
///
/// The fast path is not the same op sequence as the portable path (fused kernels accumulate in
/// float32 where the portable mask runs a bf16 matmul), so the plain path stays as the parity
/// oracle: `INKLING2_COMPILED=0` in the environment, or ``enabled`` set to false, selects it.
public enum Inkling2FastPath {
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["INKLING2_COMPILED"] != "0"

    static var onGPU: Bool { Device.defaultDevice().deviceType == .gpu }
}

/// `l2_clip` without a scale, compiled.
private func inkling2CompiledL2Clip(threshold: Float) -> @Sendable (MLXArray) -> MLXArray {
    compile(shapeless: false) { x in
        let v = x.asType(.float32)
        let norm = v.square().sum(axis: -1, keepDims: true).sqrt()
        return (v * minimum(MLXArray(threshold) / (norm + Float(1e-6)), Float(1))).asType(x.dtype)
    }
}

/// `l2_clip` with a per-head scale, compiled.
private func inkling2CompiledScaledL2Clip(threshold: Float) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    compile(shapeless: false) { x, scale in
        let v = x.asType(.float32) * scale.asType(.float32)
        let norm = v.square().sum(axis: -1, keepDims: true).sqrt()
        return (v * minimum(MLXArray(threshold) / (norm + Float(1e-6)), Float(1))).asType(x.dtype)
    }
}

/// `_softcapped_swiglu(up, gate)`, compiled; returns the gate's dtype.
private func inkling2CompiledSoftcappedSwiGLU(gateCap: Float, upCap: Float)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: false) { gate, up in
        let g = gate.asType(.float32)
        let u = up.asType(.float32)
        return (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap)
            .asType(gate.dtype)
    }
}

/// `_route(x, weight, bias, top_k, route_scale)`, compiled: (indices, weights).
private func inkling2CompiledRoute(topK: Int, routeScale: Float)
    -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
{
    compile(shapeless: false) { x, weight, bias in
        let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).transposed()))
        let indices = argPartition(-(scores + bias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let chosen = takeAlong(scores, indices, axis: -1)
        let weights = chosen / (chosen.sum(axis: -1, keepDims: true) + Float(1e-20)) * routeScale
        return (indices, weights)
    }
}

/// `_combine_routes(out, weights)`, compiled.
private let inkling2CompiledCombineRoutes: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: false) { out, weights in
        (out.asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(out.dtype)
    }

/// `_attention_residual(states, query, normalize_values)`, compiled; `states` is float32.
private func inkling2CompiledAttentionResidual(normalizesValues: Bool)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: false) { states, query in
        let width = states.dim(-1)
        let keys = MLXFast.rmsNorm(
            states, weight: MLXArray.ones([width], dtype: .float32), eps: 1e-6)
        let scores =
            matmul(keys, query.reshaped(-1, width).transposed()) * inkling2InverseSquareRoot(width)
        let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
        return matmul(weights, normalizesValues ? keys : states)
    }
}

/// The NVFP4 tensor scale after a float32 `gather_qmm`, compiled: `(product * factor)` cast to
/// the dtype of `like`.
private let inkling2CompiledScaleProduct: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: false) { product, factor, like in
        (product * factor).asType(like.dtype)
    }

private final class Inkling2Kernels: @unchecked Sendable {
    static let shared = Inkling2Kernels()

    /// mlx-vlm `models/inkling2/kernels.py` `_short_convolution` (math mode safe, MLX's default).
    let shortConvolution = MLXFast.metalKernel(
        name: "inkling2_virtual_short_convolution",
        inputNames: ["state", "weight", "x", "mask", "divisor"],
        outputNames: ["out"],
        source: """
                uint i = thread_position_in_grid.x;
                int B = x_shape[0], L = x_shape[1], C = x_shape[2];
                if (i >= B * L * C) return;
                int c = i % C, t = (i / C) % L, b = i / (C * L);
                float value = 0;
                for (int j = 0; j < W; ++j) {
                    int p = t + j - W + 1;
                    float xv = p < 0
                        ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                                      + c * state_strides[2]])
                        : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                            ? 0.0f
                            : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
                    float w = weight[c * weight_strides[0] + j * weight_strides[1]];
                    value = value + xv * w;
                }
                out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                           + c * x_strides[2]])) / divisor);
            """,
        ensureRowContiguous: false)

    /// mlx-vlm `models/inkling/language.py` `_mask_v2_kernel`.
    let bandedMask = MLXFast.metalKernel(
        name: "inkling_banded_mask_v2",
        inputNames: ["rel", "proj", "kshape"],
        outputNames: ["out"],
        source: """
                const int B  = rel_shape[0];
                const int LQ = rel_shape[1];
                const int H  = rel_shape[2];
                const int S  = kshape_shape[2];
                uint j  = thread_position_in_grid.x;
                uint i  = thread_position_in_grid.y;
                uint bh = thread_position_in_grid.z;
                if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
                uint b = bh / H, h = bh % H;
                int dist = ((int)i + (S - LQ)) - (int)j;
                T val;
                if (dist < 0) {
                    val = (T)(-1e30f);
                } else if (SLIDING > 0 && dist >= (int)SLIDING) {
                    val = (T)(-1e30f);
                } else if (dist < (int)REL_EXTENT) {
                    float acc = 0.0f;
                    const size_t rbase = (size_t)b * rel_strides[0]
                        + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
                    for (uint d = 0; d < D_REL; ++d)
                        acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                             * (float)proj[(size_t)d * proj_strides[0]
                                           + (size_t)dist * proj_strides[1]];
                    val = (T)acc;
                } else {
                    val = (T)0;
                }
                out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
            """,
        ensureRowContiguous: false)
}

/// The reference's GPU short convolution: one kernel over the virtual state/input concatenation;
/// the state keeps the last `kernel - 1` inputs in float32. Nil when the reference's kernel
/// condition does not hold, so the caller takes the portable path.
private func inkling2KernelShortConvolution(
    _ x: MLXArray, taps: MLXArray, state: ArraysCache?, slot: Int
) -> MLXArray? {
    let kernel = taps.dim(-1)
    guard Inkling2FastPath.enabled, Inkling2FastPath.onGPU, x.dtype == .bfloat16, kernel == 4,
        x.ndim == 3, x.dim(0) <= 4, x.dim(1) <= 8, x.dim(2) % 128 == 0
    else { return nil }
    let (batch, channels) = (x.dim(0), x.dim(2))
    let previous = state?[slot] ?? MLXArray.zeros([batch, kernel - 1, channels], dtype: .float32)
    if let state {
        let context = concatenated([previous, x.asType(.float32)], axis: 1)
        state[slot] = context[0..., (context.dim(1) - (kernel - 1))..., 0...]
    }
    return Inkling2Kernels.shared.shortConvolution(
        [
            previous, taps, x, MLXArray.zeros([1, 1], dtype: .bool),
            MLXArray(Float(Double(kernel + 1).squareRoot())),
        ],
        template: [("T", x.dtype), ("W", kernel), ("HAS_MASK", false)],
        grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

/// The reference's GPU band bias (`_mask_v2_kernel`): `[B, H, L, S]` in the dtype of `rel`, for
/// queries that are the last `L` of the `S` keys. Nil off the fast path.
private func inkling2KernelBandedMask(
    _ rel: MLXArray, _ proj: MLXArray, keys: MLXArray, sliding: Int, extent: Int
) -> MLXArray? {
    guard Inkling2FastPath.enabled, Inkling2FastPath.onGPU else { return nil }
    let (batch, length, heads, relativeDim) = (rel.dim(0), rel.dim(1), rel.dim(2), rel.dim(3))
    let size = keys.dim(2)
    func roundUp(_ a: Int) -> Int { (a + 7) / 8 * 8 }
    return Inkling2Kernels.shared.bandedMask(
        [rel, proj, keys],
        template: [
            ("T", rel.dtype), ("D_REL", relativeDim), ("REL_EXTENT", extent), ("SLIDING", sliding),
        ],
        grid: (roundUp(size), roundUp(length), batch * heads), threadGroup: (8, 8, 1),
        outputShapes: [[batch, heads, length, size]], outputDTypes: [rel.dtype])[0]
}

/// The head's float32 unembedding, converted once (mlx-vlm `_head_projection_cache`). Not a
/// module parameter.
final class Inkling2HeadProjectionCache {
    private var source: MLXArray?
    private var projected: MLXArray?

    func projection(of weight: MLXArray) -> MLXArray {
        if let source, source === weight, let projected { return projected }
        let converted = weight.asType(.float32)
        eval(converted)
        source = weight
        projected = converted
        return converted
    }
}

// MARK: - Modules

/// Softmax mix over completed blocks and the current partial state ("attention residual").
/// One query per output: 1 for most sublayer inputs, 3 for a global layer's Q/K/V inputs.
final class Inkling2AttentionResidual: Module {
    let normalizesValues: Bool
    private let compiled: @Sendable (MLXArray, MLXArray) -> MLXArray

    @ParameterInfo(key: "query") var query: MLXArray

    init(queries: Int, dimensions: Int, normalizesValues: Bool = true) {
        self.normalizesValues = normalizesValues
        compiled = inkling2CompiledAttentionResidual(normalizesValues: normalizesValues)
        _query.wrappedValue = MLXArray.zeros([queries, dimensions], dtype: .float32)
        super.init()
    }

    /// `_attention_residual(stack([*blocks, partial], axis=-2), query)`: `[B, L, queries, H]`,
    /// float32.
    func callAsFunction(_ blocks: [MLXArray], _ partial: MLXArray) -> MLXArray {
        let states = stacked((blocks + [partial]).map { $0.asType(.float32) }, axis: -2)
        if Inkling2FastPath.enabled { return compiled(states, query) }
        let width = states.dim(-1)
        let keys = inkling2UnitRMSNorm(states, eps: 1e-6)
        let scores =
            matmul(keys, query.reshaped(-1, width).transposed()) * inkling2InverseSquareRoot(width)
        let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
        return matmul(weights, normalizesValues ? keys : states)
    }
}

/// A core layer's four message queries (attn, k, v, mlp).
final class Inkling2LoopMessage: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(dimensions: Int) {
        _query.wrappedValue = MLXArray.zeros([4, dimensions], dtype: .float32)
        super.init()
    }
}

/// Stacked, packed weights of one expert projection, `[experts, outputDims, packed inputs]`.
final class Inkling2ExpertProjection: Module {
    let format: Inkling2ExpertFormat
    let inputDims: Int
    let outputDims: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?
    /// NVFP4 only: the per-expert tensor scale the product is multiplied by, mlx-vlm's
    /// `weight_scale_2` (source: `amax / outer_scale_denominator`; oQ: `global_scale / 2688`).
    @ParameterInfo(key: "weight_scale_2") var tensorScale: MLXArray?

    init(experts: Int, inputDims: Int, outputDims: Int, format: Inkling2ExpertFormat) {
        self.format = format
        self.inputDims = inputDims
        self.outputDims = outputDims
        let groups = inputDims / format.groupSize
        _weight.wrappedValue = MLXArray.zeros(
            [experts, outputDims, inputDims * format.bits / 32], dtype: .uint32)
        let scaleType: DType = format.mode == .affine ? .bfloat16 : .uint8
        _scales.wrappedValue = MLXArray.zeros([experts, outputDims, groups], dtype: scaleType)
        _biases.wrappedValue =
            format.hasBiases
            ? MLXArray.zeros([experts, outputDims, groups], dtype: .bfloat16) : nil
        _tensorScale.wrappedValue =
            format.hasGlobalScale ? MLXArray.zeros([experts], dtype: .float32) : nil
        super.init()
    }

    /// `x [..., 1, inputDims]` through the experts `indices` names: `[..., 1, outputDims]`.
    ///
    /// NVFP4 follows the reference's `ScaledQuantizedSwitchLinear` portable path: the product is
    /// taken in float32, multiplied by the routed expert's `weight_scale_2` after the product,
    /// and rounded to the input dtype once. The pinned `gatherQuantizedMM` has no global-scale
    /// argument, so the reference's in-kernel scaling on the GPU is not reproduced.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        if let tensorScale {
            let product = gatherQuantizedMM(
                x.asType(.float32), weight, scales: scales, biases: nil, rhsIndices: indices,
                transpose: true, groupSize: format.groupSize, bits: format.bits,
                mode: format.mode, sortedIndices: sortedIndices)
            let factor = tensorScale.asType(.float32)[indices].reshaped(indices.shape + [1, 1])
            if Inkling2FastPath.enabled { return inkling2CompiledScaleProduct(product, factor, x) }
            return (product * factor).asType(x.dtype)
        }
        return gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices, transpose: true,
            groupSize: format.groupSize, bits: format.bits, mode: format.mode,
            sortedIndices: sortedIndices)
    }
}

/// `w13` holds gate and up interleaved by output row (gate on even rows); `w2` is down.
final class Inkling2Experts: Module {
    let intermediateSize: Int
    let gateSoftcap: Float
    let upSoftcap: Float
    private let compiledActivation: @Sendable (MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "w13") var gateUp: Inkling2ExpertProjection
    @ModuleInfo(key: "w2") var down: Inkling2ExpertProjection

    init(_ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat) {
        intermediateSize = config.intermediateSize
        gateSoftcap = config.gateSoftcap
        upSoftcap = config.upSoftcap
        compiledActivation = inkling2CompiledSoftcappedSwiGLU(
            gateCap: config.gateSoftcap, upCap: config.upSoftcap)
        _gateUp.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: config.hiddenSize,
            outputDims: 2 * config.intermediateSize, format: gateUp)
        _down.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: config.intermediateSize,
            outputDims: config.hiddenSize, format: down)
        super.init()
    }

    /// The reference `SwitchGLU` with `SoftcappedSwiGLU`: `[B, L, H]` and `[B, L, k]` routes to
    /// `[B, L, k, H]`. Gate and up come from one product over the interleaved `w13` rows; each
    /// output is the same dot product the reference's separate gate and up products take.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axes: [-2, -3])
        let sorts = indices.size >= 64
        var routes = indices
        var inverseOrder: MLXArray?
        if sorts {
            let (sortedInput, sortedRoutes, inverse) = gatherSort(x: h, indices: indices)
            h = sortedInput
            routes = sortedRoutes
            inverseOrder = inverse
        }
        let fused = gateUp(h, routes, sortedIndices: sorts)
        let pairs = fused.reshaped(Array(fused.shape.dropLast()) + [intermediateSize, 2])
        let activated: MLXArray
        if Inkling2FastPath.enabled {
            activated = compiledActivation(pairs[.ellipsis, 0], pairs[.ellipsis, 1])
        } else {
            let gate = pairs[.ellipsis, 0].asType(.float32)
            let up = pairs[.ellipsis, 1].asType(.float32)
            // `_softcapped_swiglu`, in the reference's order of operations.
            activated =
                (tanh(gate / gateSoftcap) * gateSoftcap * sigmoid(gate) * tanh(up / upSoftcap)
                * upSoftcap).asType(fused.dtype)
        }
        var out = down(activated, routes, sortedIndices: sorts)
        if let inverseOrder {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }
        return out.squeezed(axis: -2)
    }
}

final class Inkling2SparseMoE: Module {
    let topK: Int
    let routeScale: Float
    private let compiledRoute: @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)

    @ParameterInfo(key: "router_weight") var routerWeight: MLXArray
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    init(_ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat) {
        topK = config.expertsPerToken
        routeScale = config.routeScale
        compiledRoute = inkling2CompiledRoute(topK: config.expertsPerToken, routeScale: config.routeScale)
        _routerWeight.wrappedValue = MLXArray.zeros(
            [config.experts, config.hiddenSize], dtype: .float32)
        _routerBias.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
        _experts.wrappedValue = Inkling2Experts(config, gateUp: gateUp, down: down)
        super.init()
    }

    /// `_route`: sigmoid scores in float32, top-k of the bias-corrected scores, weights
    /// normalised over the chosen experts and multiplied by `route_scale`.
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        if Inkling2FastPath.enabled {
            let (indices, weights) = compiledRoute(x, routerWeight, routerBias)
            return (indices, weights)
        }
        let scores = sigmoid(
            matmul(x.asType(.float32), routerWeight.asType(.float32).transposed()))
        let indices = argPartition(-(scores + routerBias), kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK]
        let chosen = takeAlong(scores, indices, axis: -1)
        let weights = chosen / (chosen.sum(axis: -1, keepDims: true) + Float(1e-20)) * routeScale
        return (indices, weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = Inkling2PhaseProfiler.measure(
            .router, outputs: { (r: (indices: MLXArray, weights: MLXArray)) in [r.indices, r.weights] }
        ) { route(x) }
        return Inkling2PhaseProfiler.measure(.experts) {
            let out = experts(x, indices)
            if Inkling2FastPath.enabled { return inkling2CompiledCombineRoutes(out, weights) }
            // `_combine_routes`.
            return (out.asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2)
                .asType(out.dtype)
        }
    }
}

/// Relative-position attention with a sink, an output gate and short convolutions on K and V.
/// Local layers fuse every projection of `x` into `in_proj` and use static K/V taps; global
/// layers read separate Q, K and V inputs and add input-dependent taps to static ones.
final class Inkling2Attention: Module {
    let configuration: Inkling2TextConfiguration
    let isSliding: Bool
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    /// Output widths of the fused projections, in row order: local `in_proj` is
    /// (q, k, v, r, gate, sink); global `q_proj` is (q, r, gate, sink) and `k_proj` / `v_proj`
    /// are (k or v, taps).
    let querySplits: [Int]
    let keyValueSplits: [Int]
    let queryRanges: [Range<Int>]
    let keyValueRanges: [Range<Int>]
    private let queryClip: @Sendable (MLXArray, MLXArray) -> MLXArray
    private let keyClip: @Sendable (MLXArray) -> MLXArray
    private let relativeClip: @Sendable (MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "in_proj") var inProj: Linear?
    @ModuleInfo(key: "q_proj") var queryProj: Linear?
    @ModuleInfo(key: "k_proj") var keyProj: Linear?
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ParameterInfo(key: "q_scale") var queryScale: MLXArray
    @ParameterInfo(key: "r_scale") var relativeScale: MLXArray
    @ParameterInfo(key: "rel_proj") var relativeProj: MLXArray
    /// Local layers: depthwise taps `[channels, kernel]`.
    @ParameterInfo(key: "k_conv") var keyConv: MLXArray?
    @ParameterInfo(key: "v_conv") var valueConv: MLXArray?
    /// Global layers: static taps `[kernel, channels]`.
    @ParameterInfo(key: "k_static") var keyStatic: MLXArray?
    @ParameterInfo(key: "v_static") var valueStatic: MLXArray?

    init(_ config: Inkling2TextConfiguration, layer: Int) {
        configuration = config
        let sliding = config.isSliding(layer)
        let heads = sliding ? config.slidingAttentionHeads : config.attentionHeads
        let kvHeads = sliding ? config.slidingKVHeads : config.kvHeads
        let headDim = sliding ? config.slidingHeadDim : config.headDim
        isSliding = sliding
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        let h = config.hiddenSize
        let kernel = config.convolutionKernelSize
        let q = heads * headDim
        let kv = kvHeads * headDim
        let r = heads * config.relativeDim
        let querySplits = sliding ? [q, kv, kv, r, q, heads] : [q, r, q, heads]
        let keyValueSplits = sliding ? [] : [kv, kvHeads * kernel]
        self.querySplits = querySplits
        self.keyValueSplits = keyValueSplits
        queryRanges = inkling2Ranges(querySplits)
        keyValueRanges = inkling2Ranges(keyValueSplits)
        queryClip = inkling2CompiledScaledL2Clip(threshold: config.qkClipThreshold)
        keyClip = inkling2CompiledL2Clip(threshold: config.qkClipThreshold)
        relativeClip = inkling2CompiledScaledL2Clip(threshold: config.relativeClipThreshold)
        if sliding {
            _inProj.wrappedValue = Linear(h, querySplits.reduce(0, +), bias: false)
            _keyConv.wrappedValue = MLXArray.zeros([kv, kernel], dtype: .bfloat16)
            _valueConv.wrappedValue = MLXArray.zeros([kv, kernel], dtype: .bfloat16)
        } else {
            _queryProj.wrappedValue = Linear(h, querySplits.reduce(0, +), bias: false)
            _keyProj.wrappedValue = Linear(h, keyValueSplits.reduce(0, +), bias: false)
            _valueProj.wrappedValue = Linear(h, keyValueSplits.reduce(0, +), bias: false)
            _keyStatic.wrappedValue = MLXArray.zeros([kernel, kv], dtype: .bfloat16)
            _valueStatic.wrappedValue = MLXArray.zeros([kernel, kv], dtype: .bfloat16)
        }
        _outputProj.wrappedValue = Linear(q, h, bias: false)
        _queryScale.wrappedValue = MLXArray.ones([heads, headDim], dtype: .bfloat16)
        _relativeScale.wrappedValue = MLXArray.ones([heads, config.relativeDim], dtype: .bfloat16)
        let extent = sliding ? config.slidingWindow : config.relativeExtent
        _relativeProj.wrappedValue = MLXArray.zeros([config.relativeDim, extent], dtype: .bfloat16)
        super.init()
    }

    /// The reference `Attention.__call__(x, key_input, value_input, cache)`: queries, relative
    /// queries, the gate and the sink read `query`; keys read `key` and values `value`.
    ///
    /// - Parameter cache: this visit's `(attention cache, ArraysCache(4))`; conv slots 0 and 1.
    func callAsFunction(
        query x: MLXArray, key keyInput: MLXArray, value valueInput: MLXArray, cache: CacheList?
    ) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))
        let dtype = x.dtype
        let attentionCache = cache?[0]
        let conv = cache?[1] as? ArraysCache

        var keys: MLXArray
        var values: MLXArray
        let projection: Linear
        if let inProj, let keyConv, let valueConv {
            projection = inProj
            let k = inkling2Project(inProj, queryRanges[1], keyInput)
            let v = inkling2Project(inProj, queryRanges[2], valueInput)
            keys = Inkling2PhaseProfiler.measure(.shortConvolutions, inputs: [k]) {
                inkling2ShortConvolution(k, taps: keyConv, state: conv, slot: 0)
            }
            values = Inkling2PhaseProfiler.measure(.shortConvolutions, inputs: [v]) {
                inkling2ShortConvolution(v, taps: valueConv, state: conv, slot: 1)
            }
        } else if let queryProj, let keyProj, let valueProj, let keyStatic, let valueStatic {
            projection = queryProj
            let k = inkling2Project(keyProj, keyValueRanges[0], keyInput)
            let kTaps = inkling2Project(keyProj, keyValueRanges[1], keyInput)
            let v = inkling2Project(valueProj, keyValueRanges[0], valueInput)
            let vTaps = inkling2Project(valueProj, keyValueRanges[1], valueInput)
            keys = Inkling2PhaseProfiler.measure(.shortConvolutions, inputs: [k, kTaps]) {
                inkling2DynamicShortConvolution(
                    k, dynamicTaps: kTaps, staticTaps: keyStatic, heads: kvHeads, state: conv,
                    slot: 0)
            }
            values = Inkling2PhaseProfiler.measure(.shortConvolutions, inputs: [v, vTaps]) {
                inkling2DynamicShortConvolution(
                    v, dynamicTaps: vTaps, staticTaps: valueStatic, heads: kvHeads, state: conv,
                    slot: 1)
            }
        } else {
            fatalError("Inkling 2 attention has neither the local nor the global projections")
        }
        // Row parts of `in_proj` (local) or `q_proj` (global) that read the query input.
        let (q, r, g, s) = isSliding ? (0, 3, 4, 5) : (0, 1, 2, 3)

        let fast = Inkling2FastPath.enabled
        let queryInput =
            inkling2Project(projection, queryRanges[q], x).reshaped(batch, length, heads, headDim)
        var queries =
            fast
            ? queryClip(queryInput, queryScale)
            : inkling2L2Clip(queryInput, threshold: c.qkClipThreshold, scale: queryScale)
        keys = keys.reshaped(batch, length, kvHeads, headDim)
        keys = fast ? keyClip(keys) : inkling2L2Clip(keys, threshold: c.qkClipThreshold)
        values = values.reshaped(batch, length, kvHeads, headDim)
        if c.useValueNorm {
            values = inkling2UnitRMSNorm(values.asType(.float32), eps: c.rmsNormEps).asType(dtype)
        }
        let relativeInput = inkling2Project(projection, queryRanges[r], x)
            .reshaped(batch, length, heads, c.relativeDim)
        var relative =
            fast
            ? relativeClip(relativeInput, relativeScale)
            : inkling2L2Clip(
                relativeInput, threshold: c.relativeClipThreshold, scale: relativeScale)
        if !isSliding, let floor = c.logScalingFloor {
            // tau = 1 + alpha log(max(position / floor, 1)), positions counted from 1.
            let offset = attentionCache?.offset ?? 0
            let positions = MLXArray((0 ..< length).map { Int32(offset + 1 + $0) })
                .asType(.float32)
            let tau =
                (Float(1)
                + c.logScalingAlpha * log(maximum(positions / Float(floor), Float(1))))
                .reshaped(length, 1, 1)
            queries = (queries.asType(.float32) * tau).asType(dtype)
            relative = (relative.asType(.float32) * tau).asType(dtype)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        if let attentionCache {
            (keys, values) = attentionCache.update(keys: keys, values: values)
            // Relative logits index backward distance, so a ring is read in chronological
            // order. A multi-token update already returns it so; a one-token update can return
            // the ring as stored.
            if length == 1, let ring = attentionCache as? RotatingKVCache,
                let window = ring.maxSize, let view = ring.logicalView(tail: window)
            {
                (keys, values) = view
            }
        }
        let size = keys.dim(2)
        let sliding = isSliding ? c.slidingWindow : 0
        let extent = isSliding ? c.slidingWindow : c.relativeExtent
        var mask =
            inkling2KernelBandedMask(
                relative, relativeProj.asType(dtype), keys: keys, sliding: sliding, extent: extent)
            ?? inkling2BandedMask(
                relative, relativeProj.asType(dtype), queryOffset: size - length, keys: size,
                sliding: sliding, extent: extent)
        // A query-dependent sink: its own logit against a zero key and value.
        let sinks = inkling2Project(projection, queryRanges[s], x).transposed(0, 2, 1)
            .expandedDimensions(axis: -1)
        mask = concatenated([mask, sinks], axis: -1)
        keys = concatenated(
            [keys, MLXArray.zeros([batch, kvHeads, 1, headDim], dtype: keys.dtype)], axis: 2)
        values = concatenated(
            [values, MLXArray.zeros([batch, kvHeads, 1, headDim], dtype: values.dtype)], axis: 2)
        var out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: inkling2InverseSquareRoot(headDim), mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        let gate = inkling2Project(projection, queryRanges[g], x)
        out = (out.asType(.float32) * sigmoid(gate.asType(.float32))).asType(dtype)
        return outputProj(out)
    }
}

final class Inkling2DecoderLayer: Module {
    let layer: Int
    let isSliding: Bool
    let closesBlock: Bool
    let eps: Float

    @ModuleInfo(key: "attn") var attention: Inkling2Attention
    @ModuleInfo(key: "mlp") var mlp: Inkling2SparseMoE
    /// Depthwise taps `[hidden, kernel]` after attention and after the MLP.
    @ParameterInfo(key: "attn_conv") var attentionConv: MLXArray
    @ParameterInfo(key: "mlp_conv") var mlpConv: MLXArray
    /// Absent on layer 0, which reads the embeddings directly.
    @ModuleInfo(key: "attn_input") var attentionInput: Inkling2AttentionResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2AttentionResidual
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm?
    @ModuleInfo(key: "q_input_norm") var queryInputNorm: RMSNorm?
    @ModuleInfo(key: "k_input_norm") var keyInputNorm: RMSNorm?
    @ModuleInfo(key: "v_input_norm") var valueInputNorm: RMSNorm?
    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    /// Core layers: message controllers for (attn, k, v, mlp). `loop_down` is
    /// `[4, 2 (input, message), H, rank]`; `loop_gain_forget` is `[4, rank, 2H]` (gain, then forget).
    @ParameterInfo(key: "loop_down") var loopDown: MLXArray?
    @ParameterInfo(key: "loop_gain_forget") var loopGainForget: MLXArray?
    @ModuleInfo(key: "loop_message") var loopMessage: Inkling2LoopMessage?

    init(
        _ config: Inkling2TextConfiguration, layer: Int,
        gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat
    ) {
        self.layer = layer
        let sliding = config.isSliding(layer)
        isSliding = sliding
        closesBlock = config.closesBlock(layer)
        eps = config.rmsNormEps
        let h = config.hiddenSize
        let normalizes = config.attentionResidualNormalizesValues
        _attention.wrappedValue = Inkling2Attention(config, layer: layer)
        _mlp.wrappedValue = Inkling2SparseMoE(config, gateUp: gateUp, down: down)
        _attentionConv.wrappedValue = MLXArray.zeros([h, config.convolutionKernelSize], dtype: .bfloat16)
        _mlpConv.wrappedValue = MLXArray.zeros([h, config.convolutionKernelSize], dtype: .bfloat16)
        _attentionInput.wrappedValue =
            layer == 0
            ? nil
            : Inkling2AttentionResidual(
                queries: sliding ? 1 : 3, dimensions: h, normalizesValues: normalizes)
        _mlpInput.wrappedValue = Inkling2AttentionResidual(
            queries: 1, dimensions: h, normalizesValues: normalizes)
        if sliding {
            _attentionNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        } else {
            _queryInputNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
            _keyInputNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
            _valueInputNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        }
        _mlpNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        if config.isCore(layer), let rank = config.loop?.gateRank {
            _loopDown.wrappedValue = MLXArray.zeros([4, 2, h, rank], dtype: .float32)
            _loopGainForget.wrappedValue = MLXArray.zeros([4, rank, 2 * h], dtype: .float32)
            _loopMessage.wrappedValue = Inkling2LoopMessage(dimensions: h)
        }
        super.init()
    }

    /// The reference `MessageController` `m` (0 attn, 1 k, 2 v, 3 mlp): blends the previous
    /// pass's message into one float32 sublayer input.
    func blend(_ x: MLXArray, _ message: MLXArray, controller m: Int) -> MLXArray {
        guard let loopDown, let loopGainForget else {
            fatalError("Inkling 2 layer \(layer) has messages but no controllers")
        }
        let width = x.dim(-1)
        let pair = concatenated(
            [inkling2UnitRMSNorm(x, eps: eps), inkling2UnitRMSNorm(message, eps: eps)], axis: -1)
        // `[2, H, rank]` reshaped to `[2H, rank]` is `down_weight.T`.
        let down = loopDown[m].reshaped(2 * width, -1)
        let hidden = matmul(pair.asType(down.dtype), down).asType(.float32)
        let gainForget = loopGainForget[m]
        let gain = tanh(matmul(hidden, gainForget[0..., ..<width]))
        let forget = tanh(matmul(hidden, gainForget[0..., width...]))
        return (Float(1) - forget) * x + gain * message
    }

    /// One visit of this layer (the reference `DecoderLayer.__call__`).
    ///
    /// - Parameters:
    ///   - blocks: completed residual blocks; a block-closing layer appends its input.
    ///   - cache: this visit's `(attention cache, ArraysCache(4))`.
    ///   - messages: `[B, L, 4, H]` in passes after the first through the core, else nil.
    func callAsFunction(
        _ x: MLXArray, blocks: inout [MLXArray], cache: CacheList?, messages: MLXArray?
    ) -> MLXArray {
        let dtype = x.dtype
        let conv = cache?[1] as? ArraysCache
        let mixed = Inkling2PhaseProfiler.measure(.residualMix) {
            attentionInput.map { $0(blocks, x) } ?? x.expandedDimensions(axis: -2)
        }
        let attended: MLXArray
        if isSliding, messages == nil, let attentionNorm {
            let input = Inkling2PhaseProfiler.measure(.norms) {
                attentionNorm(mixed[.ellipsis, 0, 0...]).asType(dtype)
            }
            attended = Inkling2PhaseProfiler.measure(.attention) {
                attention(query: input, key: input, value: input, cache: cache)
            }
        } else {
            let count = mixed.dim(-2)
            var parts = (0 ..< 3).map { mixed[.ellipsis, Swift.min($0, count - 1), 0...] }
            if let messages {
                parts = parts.enumerated().map { (i, part) -> MLXArray in
                    Inkling2PhaseProfiler.measure(.messages) {
                        blend(part, messages[.ellipsis, i, 0...], controller: i)
                    }
                }
            }
            let norms: [RMSNorm?] =
                isSliding
                ? [attentionNorm, attentionNorm, attentionNorm]
                : [queryInputNorm, keyInputNorm, valueInputNorm]
            parts = zip(norms, parts).map { (norm, part) -> MLXArray in
                guard let norm else { fatalError("Inkling 2 layer \(layer) has no input norm") }
                return Inkling2PhaseProfiler.measure(.norms) { norm(part).asType(dtype) }
            }
            attended = Inkling2PhaseProfiler.measure(.attention) {
                attention(query: parts[0], key: parts[1], value: parts[2], cache: cache)
            }
        }
        var residual = Inkling2PhaseProfiler.measure(.shortConvolutions) {
            inkling2ShortConvolution(attended, taps: attentionConv, state: conv, slot: 2)
        }.asType(.float32)
        if closesBlock {
            blocks.append(x)
        } else {
            residual = residual + x.asType(.float32)
        }
        var hidden = Inkling2PhaseProfiler.measure(.residualMix) {
            mlpInput(blocks, residual)[.ellipsis, 0, 0...]
        }
        if let messages {
            hidden = Inkling2PhaseProfiler.measure(.messages) {
                blend(hidden, messages[.ellipsis, 3, 0...], controller: 3)
            }
        }
        let normalized = Inkling2PhaseProfiler.measure(.norms) { mlpNorm(hidden).asType(dtype) }
        hidden = mlp(normalized)
        let convolved = Inkling2PhaseProfiler.measure(.shortConvolutions) {
            inkling2ShortConvolution(hidden, taps: mlpConv, state: conv, slot: 3)
        }
        let out = (residual + convolved.asType(.float32)).asType(dtype)
        conv?.advance(x.dim(1))
        return out
    }
}

public final class Inkling2TextModel: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2AttentionResidual
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Inkling2TextConfiguration, expertFormats: [(Inkling2ExpertFormat, Inkling2ExpertFormat)]) {
        _embed.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Inkling2DecoderLayer(
                config, layer: $0, gateUp: expertFormats[$0].0, down: expertFormats[$0].1)
        }
        _finalInput.wrappedValue = Inkling2AttentionResidual(
            queries: 1, dimensions: config.hiddenSize,
            normalizesValues: config.attentionResidualNormalizesValues)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// The reference `TextModel.__call__(..., skip_final_norm=True)`: the final mix before the
    /// norm, `[B, L, H]` float32.
    ///
    /// - Parameter cache: one `CacheList` per visit (48 for Mini), or nil.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, schedule: Inkling2LoopSchedule
    ) -> MLXArray {
        precondition(
            cache == nil || cache?.count == schedule.count,
            "Inkling 2 requires a separate cache for every layer visit")
        let x = Inkling2PhaseProfiler.measure(.embed) { embedNorm(embed(inputs)) }
        let result = Inkling2LoopDriver.run(
            schedule: schedule, hidden: x,
            messageQuery: { (layer: Int) -> MLXArray in
                guard let message = layers[layer].loopMessage else {
                    fatalError("Inkling 2 core layer \(layer) has no message queries")
                }
                return message.query
            }
        ) { step, hidden, blocks, messages in
            let visitCache = cache.map { caches -> CacheList in
                guard let list = caches[step.index] as? CacheList else {
                    fatalError("Inkling 2 visit \(step.index) needs a CacheList cache")
                }
                return list
            }
            return layers[step.layer](
                hidden, blocks: &blocks, cache: visitCache, messages: messages)
        }
        return Inkling2PhaseProfiler.measure(.residualMix) {
            finalInput(result.blocks, result.hidden)[.ellipsis, 0, 0...]
        }
    }
}

/// The text model and head.
public final class Inkling2Model: Module, LLMModel {
    public let configuration: Inkling2Configuration
    public let schedule: Inkling2LoopSchedule

    @ModuleInfo(key: "model") var model: Inkling2TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    /// Fast path: the float32 unembedding of a plain `Linear` head, converted once.
    private let headProjection = Inkling2HeadProjectionCache()

    /// - Parameter quantization: the checkpoint's per-layer quantization map. It names the
    ///   expert format of each layer (`model.layers.N.mlp.experts.w13` / `.w2`); layers it does
    ///   not name take its default, and without a map every layer starts as NVFP4.
    ///   ``sanitize(weights:)`` sets each layer to the format the checkpoint holds.
    public init(
        _ configuration: Inkling2Configuration,
        quantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.configuration = configuration
        let text = configuration.text
        schedule = Inkling2LoopSchedule(text)
        func format(_ path: String) -> Inkling2ExpertFormat {
            guard let q = quantization?.quantization(layer: path) else { return .nvfp4 }
            return Inkling2ExpertFormat(mode: q.mode, bits: q.bits, groupSize: q.groupSize)
        }
        let formats = (0 ..< text.hiddenLayers).map {
            (format("model.layers.\($0).mlp.experts.w13"), format("model.layers.\($0).mlp.experts.w2"))
        }
        _model.wrappedValue = Inkling2TextModel(text, expertFormats: formats)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
        super.init()
    }

    public var vocabularySize: Int { configuration.text.unpaddedVocabularySize }

    /// One entry per layer visit (48 for Mini), not per layer.
    public var kvHeads: [Int] {
        schedule.visits.map { $0.isSliding ? configuration.text.slidingKVHeads : configuration.text.kvHeads }
    }

    public var loraLayers: [Module] { model.layers }

    /// Logits `[B, L, unpadded vocabulary]` for token ids `[B, L]`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache, schedule: schedule)
        return Inkling2PhaseProfiler.measure(.head) { logits(fromNormalized: model.norm(hidden)) }
    }

    /// The reference `_logits_from_norm`: the normalised state rounded to the embedding dtype,
    /// a float32 head product, the final softcap, and the padding vocabulary dropped.
    func logits(fromNormalized normalized: MLXArray) -> MLXArray {
        let h = normalized.asType(model.embedNorm.weight.dtype).asType(.float32)
        var logits: MLXArray
        if Inkling2FastPath.enabled, Inkling2FastPath.onGPU, type(of: lmHead) == Linear.self,
            lmHead.bias == nil, h.ndim == 3, h.dim(0) * h.dim(1) <= 15
        {
            // mlx-vlm `_head_projection_cache`: the same float32 values, read without converting
            // the bf16 matrix again on every decode step.
            logits = matmul(h, headProjection.projection(of: lmHead.weight).transposed())
        } else {
            logits = lmHead(h)
        }
        let cap = configuration.text.finalLogitSoftcap
        if cap != 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits[.ellipsis, ..<configuration.text.unpaddedVocabularySize]
    }

    /// One `(attention cache, ArraysCache(4))` per layer visit: nothing is shared between
    /// passes through the core, only weights. Local visits keep the architecture's ring (window
    /// 1024 for Mini); global visits keep every key.
    public func newCache(parameters: GenerateParameters? = nil) throws -> [KVCache] {
        let window = configuration.text.slidingWindow
        return schedule.visits.map { visit in
            let attention: KVCache =
                visit.isSliding ? RotatingKVCache(maxSize: window, keep: 0) : KVCacheSimple()
            return CacheList(attention, ArraysCache(size: 4))
        }
    }

    /// Fills the cache through the decoder only, so the 201K-wide head runs for the last token
    /// alone. The default chunk is the reference's 2048; the chunk size changes greedy output,
    /// so parity runs fix it on both sides.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State? = nil,
        prefill: PrefillParameters = .init()
    ) throws -> PrepareResult {
        let y = input.text
        guard y.tokens.size > 0 else {
            return .tokens(y)
        }
        let processed = try prefill.forEachChunk(
            total: y.tokens.size,
            defaultStepSize: Self.defaultPrefillChunkSize
        ) { range in
            _ = model(y[.newAxis, range].tokens, cache: cache, schedule: schedule)
            asyncEval(cache)
        }
        return .tokens(y[processed...])
    }

    private static let defaultPrefillChunkSize = 2048

    /// Text weights in the oQ layout. A source checkpoint is converted; vision and draft tensors
    /// are dropped. A checkpoint the converter rejects is returned unchanged, so the loader's
    /// key check reports it.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let converted = try? Inkling2WeightLayout.convert(weights, configuration: configuration.text)
        else { return weights }
        // A layer whose tensors contradict the declared formats leaves the checkpoint unconverted,
        // so the loader's key check reports it.
        do { try setExpertFormats(from: converted) } catch { return weights }
        return converted
    }

    /// The expert formats the source checkpoint's `hf_quant_config.json` declares, by layer.
    private(set) var declaredExpertFormats = [Int: Inkling2QuantizationMetadata.LayerFormats]()

    /// Reads `hf_quant_config.json` from `directory` when it is there (the source checkpoint) and
    /// sets each layer it names to the declared expert formats, as mlx-vlm's
    /// `quantization_from_metadata` does. ``sanitize(weights:)`` then refuses tensors that hold
    /// another format. Without the file the formats come from the tensors alone.
    public func applyQuantizationMetadata(directory: URL) throws {
        let url = directory.appendingPathComponent("hf_quant_config.json")
        guard FileManager.default.fileExists(atPath: url.path),
            let formats = try Inkling2QuantizationMetadata.expertFormats(
                from: Data(contentsOf: url), layers: configuration.text.hiddenLayers)
        else { return }
        declaredExpertFormats = formats
        try rebuildExperts(formats)
    }

    /// Rebuilds each layer's experts whose checkpoint storage differs from the module's.
    func setExpertFormats(from weights: [String: MLXArray]) throws {
        let text = configuration.text
        var detected = [Int: Inkling2QuantizationMetadata.LayerFormats]()
        for index in model.layers.indices {
            let prefix = "model.layers.\(index).mlp.experts"
            guard
                let gateUp = try Inkling2ExpertFormat.detect(
                    prefix: "\(prefix).w13", in: weights, inputDims: text.hiddenSize),
                let down = try Inkling2ExpertFormat.detect(
                    prefix: "\(prefix).w2", in: weights, inputDims: text.intermediateSize)
            else { continue }
            if let declared = declaredExpertFormats[index],
                declared.gateUp != gateUp || declared.down != down
            {
                throw Inkling2ConfigurationError(
                    message: "layer \(index): hf_quant_config.json declares other expert formats than its tensors hold")
            }
            detected[index] = Inkling2QuantizationMetadata.LayerFormats(gateUp: gateUp, down: down)
        }
        try rebuildExperts(detected)
    }

    private func rebuildExperts(_ formats: [Int: Inkling2QuantizationMetadata.LayerFormats]) throws {
        let text = configuration.text
        for (index, wanted) in formats {
            let layer = model.layers[index]
            let experts = layer.mlp.experts
            if experts.gateUp.format != wanted.gateUp || experts.down.format != wanted.down {
                try layer.mlp.update(
                    modules: ModuleChildren(values: [
                        "experts": .value(
                            Inkling2Experts(text, gateUp: wanted.gateUp, down: wanted.down))
                    ]), verify: [])
            }
        }
    }

    /// The expert format each layer currently holds: (w13, w2).
    public var expertFormats: [(gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat)] {
        model.layers.map { ($0.mlp.experts.gateUp.format, $0.mlp.experts.down.format) }
    }
}

// MARK: - Checkpoint layouts

/// Maps the source checkpoint (thinkingmachines/Inkling-2-Mini, `model.llm.*`) onto the oQ
/// layout, as oMLX's `oq_layout.sanitizer` and `loader.sanitize_weights` do. The oQ layout
/// (mlx-community/Inkling-2-Mini-oQ4e-mtp) passes through. Only key renames, row
/// concatenations, reshapes, transposes, dtype views and the NVFP4 tensor scale (mlx-vlm's
/// `weight_scale_2` expression for each layout) run.
public enum Inkling2WeightLayout {
    /// ModelOpt stores the NVFP4 tensor scale as amax / (E4M3 max x E2M1 max); the oQ layout keeps
    /// amax rescaled to this denominator.
    static let nvfp4Denominator: Float = 448.0 * 6.0
    static let sourcePrefix = "model.llm."

    public enum Kind: Equatable, Sendable {
        case oq
        case source
    }

    public static func kind(of keys: some Sequence<String>) throws -> Kind {
        var oq = false
        var source = false
        for key in keys {
            if key.hasPrefix(sourcePrefix) { source = true }
            if key.hasPrefix("model.layers.") || key == "lm_head.weight" { oq = true }
            if key.hasPrefix("model.language_model.") || key.hasPrefix("language_model.") {
                throw Inkling2ConfigurationError(
                    message: "the mlx-lm #2086 layout (\(key)) is not supported; use the source or oQ layout")
            }
        }
        switch (oq, source) {
        case (true, false): return .oq
        case (false, true): return .source
        case (true, true): throw Inkling2ConfigurationError(message: "checkpoint mixes source and oQ keys")
        case (false, false): throw Inkling2ConfigurationError(message: "no Inkling 2 text weights")
        }
    }

    /// Vision and draft tensors, which the text model does not load.
    static func isDropped(_ key: String) -> Bool {
        key.hasPrefix("vision_tower.") || key.hasPrefix("model.visual.")
            || key.hasPrefix("model.mtp.") || key.hasPrefix("mtp.")
    }

    public static func convert(_ weights: [String: MLXArray], configuration: Inkling2TextConfiguration)
        throws -> [String: MLXArray]
    {
        let kept = weights.filter { !isDropped($0.key) }
        switch try kind(of: kept.keys) {
        case .oq: return convertOQ(kept)
        case .source: return try convertSource(kept, configuration: configuration)
        }
    }

    /// The oQ layout as loaded: each NVFP4 `global_scale` (amax rescaled to 2688) becomes the
    /// tensor scale the product is multiplied by, `global_scale / 2688` in float32, the value an
    /// mlx-vlm reading of the oQ bytes uses as `weight_scale_2`.
    static func convertOQ(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasSuffix(".global_scale") {
                let prefix = String(key.dropLast(".global_scale".count))
                out[prefix + ".weight_scale_2"] = value.asType(.float32) / nvfp4Denominator
            } else {
                out[key] = value
            }
        }
        return out
    }

    static func convertSource(_ weights: [String: MLXArray], configuration c: Inkling2TextConfiguration)
        throws -> [String: MLXArray]
    {
        let h = c.hiddenSize
        var out = [String: MLXArray]()
        var perLayer = [Int: [String: MLXArray]]()
        var crossPass = [String: MLXArray]()
        for (key, value) in weights {
            guard key.hasPrefix(sourcePrefix) else {
                throw Inkling2ConfigurationError(message: "unmapped tensor \(key)")
            }
            let name = String(key.dropFirst(sourcePrefix.count))
            switch name {
            case "embed.weight": out["model.embed.weight"] = value
            case "embed_norm.weight": out["model.embed_norm.weight"] = value
            case "norm.weight": out["model.norm.weight"] = value
            case "unembed.weight": out["lm_head.weight"] = value
            case "attn_res.output_query":
                out["model.final_input.query"] = value.reshaped(-1, h).asType(.float32)
            case "middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries":
                crossPass[name] = value
            default:
                if name.hasPrefix("attn_res.attn_queries.") {
                    // attn_queries[i] belongs to layer i + 1; layer 0 has none.
                    let index = try layerIndex(name, prefix: "attn_res.attn_queries.") + 1
                    out["model.layers.\(index).attn_input.query"] = value.reshaped(-1, h).asType(.float32)
                } else if name.hasPrefix("attn_res.mlp_queries.") {
                    let index = try layerIndex(name, prefix: "attn_res.mlp_queries.")
                    out["model.layers.\(index).mlp_input.query"] = value.reshaped(-1, h).asType(.float32)
                } else if name.hasPrefix("layers.") {
                    let rest = name.dropFirst("layers.".count)
                    guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else {
                        throw Inkling2ConfigurationError(message: "unmapped tensor \(key)")
                    }
                    perLayer[index, default: [:]][String(rest[rest.index(after: dot)...])] = value
                } else {
                    throw Inkling2ConfigurationError(message: "unmapped tensor \(key)")
                }
            }
        }
        for (index, tensors) in perLayer {
            try mapLayer(tensors, layer: index, configuration: c, into: &out)
        }
        if let loop = c.loop {
            guard let q = crossPass["middle_cycle_cross_pass_queries"],
                let kv = crossPass["middle_cycle_cross_pass_kv_queries"],
                q.dim(0) == 2 * (loop.coreEnd - loop.coreStart), kv.dim(0) == q.dim(0)
            else {
                throw Inkling2ConfigurationError(message: "missing or mis-sized cross-pass queries")
            }
            // Row 2i of each table serves core layer i's attn (or k) input, row 2i + 1 its MLP
            // (or v) input; the oQ order is attn, k, v, mlp.
            for (i, layer) in (loop.coreStart ..< loop.coreEnd).enumerated() {
                out["model.layers.\(layer).loop_message.query"] = stacked(
                    [q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]].map { $0.asType(.float32) })
            }
        } else if !crossPass.isEmpty {
            throw Inkling2ConfigurationError(message: "cross-pass queries without loop_config")
        }
        return out
    }

    private static func layerIndex(_ name: String, prefix: String) throws -> Int {
        guard let index = Int(name.dropFirst(prefix.count)) else {
            throw Inkling2ConfigurationError(message: "bad layer index in \(name)")
        }
        return index
    }

    static func mapLayer(
        _ source: [String: MLXArray], layer: Int, configuration c: Inkling2TextConfiguration,
        into out: inout [String: MLXArray]
    ) throws {
        var d = source
        func take(_ name: String) throws -> MLXArray {
            guard let value = d.removeValue(forKey: name) else {
                throw Inkling2ConfigurationError(message: "layer \(layer) has no \(name)")
            }
            return value
        }
        let pre = "model.layers.\(layer)."
        let h = c.hiddenSize
        let kernel = c.convolutionKernelSize
        if c.isSliding(layer) {
            out[pre + "attn.in_proj.weight"] = concatenated(
                try ["wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj"].map {
                    try take("attn.\($0).weight")
                }, axis: 0)
            out[pre + "attn.k_conv"] = try take("attn.k_sconv.weight").reshaped(-1, kernel)
            out[pre + "attn.v_conv"] = try take("attn.v_sconv.weight").reshaped(-1, kernel)
            out[pre + "attn_norm.weight"] = try take("attn_norm.weight")
        } else {
            out[pre + "attn.q_proj.weight"] = concatenated(
                try ["wq_du", "wr_du", "gate_attn", "attn_sink_proj"].map {
                    try take("attn.\($0).weight")
                }, axis: 0)
            out[pre + "attn.k_proj.weight"] = concatenated(
                [try take("attn.wk_dv.weight"), try take("attn.k_sconv.weight_proj")], axis: 0)
            out[pre + "attn.v_proj.weight"] = concatenated(
                [try take("attn.wv_dv.weight"), try take("attn.v_sconv.weight_proj")], axis: 0)
            out[pre + "attn.k_static"] = try take("attn.k_sconv.static_w")
            out[pre + "attn.v_static"] = try take("attn.v_sconv.static_w")
            for n in ["q", "k", "v"] {
                out[pre + "\(n)_input_norm.weight"] = try take("\(n)_input_norm.weight")
            }
        }
        out[pre + "attn.o_proj.weight"] = try take("attn.wo_ud.weight")
        out[pre + "attn.q_scale"] = try take("attn.q_l2_clip.scale")
        out[pre + "attn.r_scale"] = try take("attn.rel_l2_clip.scale")
        out[pre + "attn.rel_proj"] = try take("attn.rel_logits_proj.proj")
        out[pre + "attn_conv"] = try take("attn_sconv.weight").reshaped(h, kernel)
        out[pre + "mlp_conv"] = try take("mlp_sconv.weight").reshaped(h, kernel)
        out[pre + "mlp_norm.weight"] = try take("mlp_norm.weight")
        out[pre + "mlp.router_weight"] = try take("mlp.gate.weight").asType(.float32)
        out[pre + "mlp.router_bias"] = try take("mlp.gate.bias").asType(.float32)
        for which in ["w13", "w2"] {
            let base = pre + "mlp.experts.\(which)"
            // Packed bytes (FP4 pairs or FP8 values) become uint32 words; scales keep their bytes.
            out[base + ".weight"] = try take("mlp.experts.\(which)_weight").view(dtype: .uint32)
            out[base + ".scales"] = try take("mlp.experts.\(which)_weight_scale")
            if let amax = d.removeValue(forKey: "mlp.experts.\(which)_weight_amax") {
                // mlx-vlm's `amax / outer_scale_denominator`, both float32 in the checkpoint.
                // Going through the oQ `global_scale` (x 2688, then / 2688) can be one ulp off.
                let denominator = try take("mlp.experts.\(which)_weight_outer_scale_denominator")
                out[base + ".weight_scale_2"] = amax.asType(.float32) / denominator.asType(.float32)
            }
        }
        if c.isCore(layer) {
            var downs = [MLXArray]()
            var gates = [MLXArray]()
            for m in ["attn", "k", "v", "mlp"] {
                // The reference multiplies in the stored dtype (`pair.astype(down_weight.dtype)`),
                // so the down projection keeps it; gain and forget multiply a float32 input,
                // which promotes them exactly.
                let down = try take("controllers.\(m).down_weight")
                let rank = down.dim(0)
                downs.append(down.reshaped(rank, 2, -1).transposed(1, 2, 0))
                let gain = try take("controllers.\(m).gain_weight").asType(.float32)
                let forget = try take("controllers.\(m).forget_weight").asType(.float32)
                gates.append(concatenated([gain.transposed(), forget.transposed()], axis: 1))
            }
            out[pre + "loop_down"] = stacked(downs)
            out[pre + "loop_gain_forget"] = stacked(gates)
        }
        guard d.isEmpty else {
            throw Inkling2ConfigurationError(
                message: "unmapped tensors in layer \(layer): \(d.keys.sorted())")
        }
    }
}
