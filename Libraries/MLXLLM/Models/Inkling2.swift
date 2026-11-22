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
// This file holds the configuration, the visit schedule, the loop driver, the module tree and the
// checkpoint layouts. The module tree follows the oQ layout of
// mlx-community/Inkling-2-Mini-oQ4e-mtp (oMLX's runtime names, jundot/omlx@a0ed2f59
// `omlx/patches/inkling2/oq_layout.py`, Apache-2.0); `Inkling2WeightLayout` maps the source
// checkpoint (`model.llm.*`) onto it. Layer arithmetic is not ported yet, so the model is not
// registered with the factory.

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
        coreStart = text.loop?.coreStart ?? text.hiddenLayers
        coreEnd = text.loop?.coreEnd ?? text.hiddenLayers
        passes = text.loop?.passes ?? 1
        let coreLength = coreEnd - coreStart
        visits = text.layerVisits.enumerated().map { index, layer in
            let inCore = text.isCore(layer)
            return Visit(
                index: index, layer: layer,
                pass: inCore ? (index - coreStart) / coreLength : 0,
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
                    bank = messageBank(states, eps: eps)
                    x = saved.hidden
                    blocks = saved.blocks
                } else {
                    prelude = (x, blocks)
                }
            }
            var messages: MLXArray?
            if let bank, step.inCore {
                messages = Self.messages(query: messageQuery(step.layer), bank: bank)
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
        let scale = 1 / Float(bank.dim(-1)).squareRoot()
        let scores = matmul(query, bank.swappedAxes(-1, -2)) * scale
        return matmul(softmax(scores, axis: -1), bank)
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

    /// NVFP4 with a per-expert float32 `global_scale` (layers 0-19 of every Mini build).
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
        let hasGlobalScale = weights["\(prefix).global_scale"] != nil
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

// MARK: - Modules

/// Softmax mix over completed blocks and the current partial state ("attention residual").
/// One query per output: 1 for most sublayer inputs, 3 for a global layer's Q/K/V inputs.
final class Inkling2AttentionResidual: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(queries: Int, dimensions: Int) {
        _query.wrappedValue = MLXArray.zeros([queries, dimensions], dtype: .float32)
        super.init()
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
    /// NVFP4 only: the per-expert amax in MLX's convention; the product is multiplied by
    /// `global_scale / (6 * 448)`.
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?

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
        _globalScale.wrappedValue =
            format.hasGlobalScale ? MLXArray.zeros([experts], dtype: .float32) : nil
        super.init()
    }
}

/// `w13` holds gate and up interleaved by output row (gate on even rows); `w2` is down.
final class Inkling2Experts: Module {
    @ModuleInfo(key: "w13") var gateUp: Inkling2ExpertProjection
    @ModuleInfo(key: "w2") var down: Inkling2ExpertProjection

    init(_ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat) {
        _gateUp.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: config.hiddenSize,
            outputDims: 2 * config.intermediateSize, format: gateUp)
        _down.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: config.intermediateSize,
            outputDims: config.hiddenSize, format: down)
        super.init()
    }
}

final class Inkling2SparseMoE: Module {
    @ParameterInfo(key: "router_weight") var routerWeight: MLXArray
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    init(_ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat, down: Inkling2ExpertFormat) {
        _routerWeight.wrappedValue = MLXArray.zeros(
            [config.experts, config.hiddenSize], dtype: .float32)
        _routerBias.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
        _experts.wrappedValue = Inkling2Experts(config, gateUp: gateUp, down: down)
        super.init()
    }
}

/// Relative-position attention with a sink, an output gate and short convolutions on K and V.
/// Local layers fuse every projection of `x` into `in_proj` and use static K/V taps; global
/// layers read separate Q, K and V inputs and add input-dependent taps to static ones.
final class Inkling2Attention: Module {
    let isSliding: Bool
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    /// Output widths of the fused projections, in row order: local `in_proj` is
    /// (q, k, v, r, gate, sink); global `q_proj` is (q, r, gate, sink) and `k_proj` / `v_proj`
    /// are (k or v, taps).
    let querySplits: [Int]
    let keyValueSplits: [Int]

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
        isSliding = config.isSliding(layer)
        heads = isSliding ? config.slidingAttentionHeads : config.attentionHeads
        kvHeads = isSliding ? config.slidingKVHeads : config.kvHeads
        headDim = isSliding ? config.slidingHeadDim : config.headDim
        let h = config.hiddenSize
        let kernel = config.convolutionKernelSize
        let q = heads * headDim
        let kv = kvHeads * headDim
        let r = heads * config.relativeDim
        if isSliding {
            querySplits = [q, kv, kv, r, q, heads]
            keyValueSplits = []
            _inProj.wrappedValue = Linear(h, querySplits.reduce(0, +), bias: false)
            _keyConv.wrappedValue = MLXArray.zeros([kv, kernel], dtype: .bfloat16)
            _valueConv.wrappedValue = MLXArray.zeros([kv, kernel], dtype: .bfloat16)
        } else {
            querySplits = [q, r, q, heads]
            keyValueSplits = [kv, kvHeads * kernel]
            _queryProj.wrappedValue = Linear(h, querySplits.reduce(0, +), bias: false)
            _keyProj.wrappedValue = Linear(h, keyValueSplits.reduce(0, +), bias: false)
            _valueProj.wrappedValue = Linear(h, keyValueSplits.reduce(0, +), bias: false)
            _keyStatic.wrappedValue = MLXArray.zeros([kernel, kv], dtype: .bfloat16)
            _valueStatic.wrappedValue = MLXArray.zeros([kernel, kv], dtype: .bfloat16)
        }
        _outputProj.wrappedValue = Linear(q, h, bias: false)
        _queryScale.wrappedValue = MLXArray.ones([heads, headDim], dtype: .bfloat16)
        _relativeScale.wrappedValue = MLXArray.ones([heads, config.relativeDim], dtype: .bfloat16)
        let extent = isSliding ? config.slidingWindow : config.relativeExtent
        _relativeProj.wrappedValue = MLXArray.zeros([config.relativeDim, extent], dtype: .bfloat16)
        super.init()
    }
}

final class Inkling2DecoderLayer: Module {
    let layer: Int
    let isSliding: Bool
    let closesBlock: Bool

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
        isSliding = config.isSliding(layer)
        closesBlock = config.closesBlock(layer)
        let h = config.hiddenSize
        _attention.wrappedValue = Inkling2Attention(config, layer: layer)
        _mlp.wrappedValue = Inkling2SparseMoE(config, gateUp: gateUp, down: down)
        _attentionConv.wrappedValue = MLXArray.zeros([h, config.convolutionKernelSize], dtype: .bfloat16)
        _mlpConv.wrappedValue = MLXArray.zeros([h, config.convolutionKernelSize], dtype: .bfloat16)
        _attentionInput.wrappedValue =
            layer == 0
            ? nil : Inkling2AttentionResidual(queries: isSliding ? 1 : 3, dimensions: h)
        _mlpInput.wrappedValue = Inkling2AttentionResidual(queries: 1, dimensions: h)
        if isSliding {
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
        _finalInput.wrappedValue = Inkling2AttentionResidual(queries: 1, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// The text model and head. Loading fills the module tree; the forward pass is not ported yet.
public final class Inkling2Model: Module, BaseLanguageModel {
    public let configuration: Inkling2Configuration
    public let schedule: Inkling2LoopSchedule

    @ModuleInfo(key: "model") var model: Inkling2TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

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

    /// Text weights in the oQ layout. A source checkpoint is converted; vision and draft tensors
    /// are dropped. A checkpoint the converter rejects is returned unchanged, so the loader's
    /// key check reports it.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let converted = try? Inkling2WeightLayout.convert(weights, configuration: configuration.text)
        else { return weights }
        try? setExpertFormats(from: converted)
        return converted
    }

    /// Rebuilds each layer's experts whose checkpoint storage differs from the module's.
    func setExpertFormats(from weights: [String: MLXArray]) throws {
        let text = configuration.text
        for (index, layer) in model.layers.enumerated() {
            let prefix = "model.layers.\(index).mlp.experts"
            guard
                let gateUp = try Inkling2ExpertFormat.detect(
                    prefix: "\(prefix).w13", in: weights, inputDims: text.hiddenSize),
                let down = try Inkling2ExpertFormat.detect(
                    prefix: "\(prefix).w2", in: weights, inputDims: text.intermediateSize)
            else { continue }
            let experts = layer.mlp.experts
            if experts.gateUp.format != gateUp || experts.down.format != down {
                try layer.mlp.update(
                    modules: ModuleChildren(values: [
                        "experts": .value(Inkling2Experts(text, gateUp: gateUp, down: down))
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
/// concatenations, reshapes, transposes, dtype views and the global-scale rescale run, so the
/// conversion is exact.
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
        case .oq: return kept
        case .source: return try convertSource(kept, configuration: configuration)
        }
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
                let denominator = try take("mlp.experts.\(which)_weight_outer_scale_denominator")
                out[base + ".global_scale"] =
                    amax.asType(.float32) * (nvfp4Denominator / denominator.asType(.float32))
            }
        }
        if c.isCore(layer) {
            var downs = [MLXArray]()
            var gates = [MLXArray]()
            for m in ["attn", "k", "v", "mlp"] {
                let down = try take("controllers.\(m).down_weight").asType(.float32)
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
