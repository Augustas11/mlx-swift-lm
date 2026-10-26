// Port of mlx-vlm's prism_hadamard_qwen35 schema 2 runtime.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum PrismHadamardError: Error, Equatable {
    case invalidConfiguration(String)
    case invalidModule(String)
    case invalidSigns(String)
    case invalidPackedTensor(String)
}

/// A schema 2 pack keeps ordinary affine tensors in a signed Hadamard basis.
public struct PrismHadamardQwen35Configuration: Decodable, Sendable {
    public struct PackedModule: Decodable, Sendable {
        public let path: String
        public let block: Int
        public let embedding: Bool
        public let dtype: String
    }

    public let base: Qwen35Configuration
    public let modules: [PackedModule]

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case schemaVersion = "schema_version"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case quantization, modules
    }

    private enum TopologyKeys: String, CodingKey { case text = "text_config" }
    private enum TextKeys: String, CodingKey { case heads = "num_attention_heads" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(String.self, forKey: .modelType) == "prism_hadamard_qwen35",
            try c.decode(Int.self, forKey: .schemaVersion) == 2,
            try c.decode(String.self, forKey: .baseModelType) == "qwen3_5",
            try c.decode(String.self, forKey: .tensorNamespace) == "mlx-vlm-qwen3_5",
            try c.decode(String.self, forKey: .gdnActivationLayout) == "grouped"
        else { throw PrismHadamardError.invalidConfiguration("unsupported pack schema") }
        struct Quantization: Decodable {
            let bits: Int
            let groupSize: Int
            let mode: String
            enum CodingKeys: String, CodingKey {
                case bits, mode
                case groupSize = "group_size"
            }
        }
        let q = try c.decode(Quantization.self, forKey: .quantization)
        guard q.bits == 2, q.groupSize == 128, q.mode == "affine" else {
            throw PrismHadamardError.invalidConfiguration("expected affine 2-bit/group-128")
        }
        // Qwen's decoder derives head_dim by division before module construction.
        let raw = try decoder.container(keyedBy: TopologyKeys.self)
        let text = try raw.nestedContainer(keyedBy: TextKeys.self, forKey: .text)
        guard try text.decode(Int.self, forKey: .heads) > 0 else {
            throw PrismHadamardError.invalidConfiguration("invalid attention head count")
        }
        base = try Qwen35Configuration(from: decoder)
        modules = try c.decode([PackedModule].self, forKey: .modules)
        guard !modules.isEmpty else {
            throw PrismHadamardError.invalidConfiguration("missing packed module manifest")
        }
        var seen = Set<String>()
        for record in modules {
            guard seen.insert(record.path).inserted,
                record.dtype == "float16",
                [0, 512, 1024, 2048, 4096].contains(record.block)
            else { throw PrismHadamardError.invalidModule(record.path) }
        }
        let t = base.textConfiguration
        guard t.modelType == "qwen3_5_text", t.numExperts == 0,
            t.mtpNumHiddenLayers == 0, t.tieWordEmbeddings,
            t.hiddenLayers > 0, t.fullAttentionInterval > 0,
            t.fullAttentionInterval <= t.hiddenLayers,
            t.attentionHeads > 0, t.kvHeads > 0,
            t.linearNumKeyHeads > 0, t.linearNumValueHeads > 0,
            t.linearNumValueHeads % t.linearNumKeyHeads == 0,
            t.hiddenSize > 0, t.intermediateSize > 0, t.vocabularySize > 0,
            t.linearKeyHeadDim > 0, t.linearValueHeadDim > 0,
            t.linearConvKernelDim > 0, (t.headDim ?? 0) > 0,
            t.attentionHeads % t.kvHeads == 0, !t.attentionBias
        else { throw PrismHadamardError.invalidConfiguration("unsupported text architecture") }
        let v = base.visionConfiguration
        guard v.depth > 0, v.hiddenSize > 0, v.intermediateSize > 0,
            v.outHiddenSize == t.hiddenSize, v.numHeads > 0,
            v.hiddenSize % v.numHeads == 0, v.patchSize > 0,
            v.spatialMergeSize > 0, v.temporalPatchSize > 0,
            v.numPositionEmbeddings > 0, v.inChannels > 0,
            v.deepstackVisualIndexes.isEmpty
        else { throw PrismHadamardError.invalidConfiguration("unsupported vision architecture") }
    }
}

// Explicit arguments keep every loaded sign vector live when the pointwise trace is reused.
private let prismSignedFloat32Input: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { input, signs in
    input.asType(.float32) * signs
}

/// The reference calibrates in float32, then restores the activation dtype.
func prismHadamardTransform(
    _ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false
) -> MLXArray {
    var x = inverse ? input.asType(.float32) : prismSignedFloat32Input(input, signs)
    x = hadamardTransform(x.reshaped(-1, block), scale: 1 / sqrt(Float(block)))
        .reshaped(input.shape)
    if inverse { x = x * signs }
    return x.asType(input.dtype)
}

private func validatePrismDimensions(width: Int, rows: Int, block: Int) throws {
    guard width > 0, rows > 0, width % 128 == 0,
        [0, 512, 1024, 2048, 4096].contains(block),
        block == 0 || width % block == 0
    else { throw PrismHadamardError.invalidModule("incompatible packed dimensions") }
}

private func validatePrismParameters(_ parameters: ModuleParameters, width: Int, block: Int)
    throws
{
    for (key, value) in parameters.flattened() {
        switch key {
        case "signs":
            guard block != 0, value.shape == [width], value.dtype == .float32,
                ((value .== 1) .|| (value .== -1)).all().item(Bool.self)
            else { throw PrismHadamardError.invalidSigns(key) }
        case "weight":
            guard value.dtype == .uint32 else {
                throw PrismHadamardError.invalidPackedTensor(key)
            }
        case "scales", "biases":
            guard value.dtype == .float16 else {
                throw PrismHadamardError.invalidPackedTensor(key)
            }
        default: break
        }
    }
}

/// Allocate packed placeholders directly; never quantize temporary dense weights.
final class PrismHadamardLinear: Linear, Quantized {
    let groupSize = 128
    let bits = 2
    let mode = QuantizationMode.affine
    let block: Int
    let width: Int
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?

    override var shape: (Int, Int) { (weight.dim(0), width) }

    init(width: Int, rows: Int, block: Int) throws {
        try validatePrismDimensions(width: width, rows: rows, block: block)
        self.block = block
        self.width = width
        scales = .zeros([rows, width / 128], dtype: .float16)
        biases = .zeros([rows, width / 128], dtype: .float16)
        signs = block == 0 ? nil : .ones([width], dtype: .float32)
        super.init(weight: .zeros([rows, width / 16], dtype: .uint32))
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = signs.map { prismHadamardTransform(x, block: block, signs: $0) } ?? x
        return quantizedMM(
            input, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        try validatePrismParameters(parameters, width: width, block: block)
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

final class PrismHadamardEmbedding: Embedding, Quantized {
    let groupSize = 128
    let bits = 2
    let mode = QuantizationMode.affine
    let block: Int
    let width: Int
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?

    override var shape: (Int, Int) { (weight.dim(0), width) }

    init(width: Int, rows: Int, block: Int) throws {
        try validatePrismDimensions(width: width, rows: rows, block: block)
        self.block = block
        self.width = width
        scales = .zeros([rows, width / 128], dtype: .float16)
        biases = .zeros([rows, width / 128], dtype: .float16)
        signs = block == 0 ? nil : .ones([width], dtype: .float32)
        super.init(weight: .zeros([rows, width / 16], dtype: .uint32))
        freeze()
    }

    override func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let flat = indices.flattened()
        let rows = dequantized(
            weight[flat], scales: scales[flat], biases: biases[flat],
            groupSize: groupSize, bits: bits, mode: mode
        )
        .reshaped(indices.shape + [width]).asType(.float16)
        return signs.map {
            prismHadamardTransform(rows, block: block, signs: $0, inverse: true)
        } ?? rows
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        let input = signs.map { prismHadamardTransform(x, block: block, signs: $0) } ?? x
        return quantizedMM(
            input, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        try validatePrismParameters(parameters, width: width, block: block)
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

/// Full-artifact model: the vision tower and unlisted FP16 projections stay intact.
public final class PrismHadamardQwen35: Qwen35 {
    // Retains isolated checkpoint backing for deferred vision arrays.
    var ownedSnapshot: OwnedPrismArtifactSnapshot?

    public init(_ config: PrismHadamardQwen35Configuration) throws {
        super.init(config.base)
        let original = Dictionary(uniqueKeysWithValues: languageModel.leafModules().flattened())
        var replacements = [(String, Module)]()
        for record in config.modules {
            guard let module = original[record.path] else {
                throw PrismHadamardError.invalidModule(record.path)
            }
            let replacement: Module
            if let embedding = module as? Embedding, record.embedding {
                replacement = try PrismHadamardEmbedding(
                    width: embedding.shape.1, rows: embedding.shape.0, block: record.block)
            } else if let linear = module as? Linear, !record.embedding, linear.bias == nil {
                replacement = try PrismHadamardLinear(
                    width: linear.shape.1, rows: linear.shape.0, block: record.block)
            } else {
                throw PrismHadamardError.invalidModule(record.path)
            }
            replacements.append((record.path, replacement))
        }
        try languageModel.update(
            modules: ModuleChildren.unflattened(replacements), verify: [.noUnusedKeys])
        for case let layer as Qwen35Language.DecoderLayer in loraLayers {
            layer.linearAttn?.prismConvDecodeEnabled = true
        }
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Schema 2 already has MLX conv layouts, grouped GDN heads and offset norms.
        weights
    }

    public override func sanitize(
        weights: [String: MLXArray], metadata: [String: String]
    ) -> [String: MLXArray] {
        weights
    }
}

extension PrismHadamardQwen35: BatchInvariantDecodeStateModel {
    public func batchDecodeState(_ states: [LMOutput.State?], cacheOffsets: [Int]) throws
        -> LMOutput.State
    {
        try Qwen35BatchDecodeState.merge(states, cacheOffsets: cacheOffsets)
    }

    public func validateBatchDecodeState(_ output: LMOutput.State?, against input: LMOutput.State)
        throws
    {
        try Qwen35BatchDecodeState.validate(output, against: input)
    }
}

extension PrismHadamardQwen35: InitialInferenceMaterializationProviding {
    package var initialInferenceModules: [String: Module]? {
        ownedSnapshot == nil ? nil : ["language_model": languageModel]
    }
}
