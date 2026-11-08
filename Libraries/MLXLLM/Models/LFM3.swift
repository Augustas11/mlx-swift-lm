//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py
//
// The configuration and backbone are in MLXLMCommon (`Models/LFM3.swift`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == .conv ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(inputs, cache: cache))
    }

    /// MLX conversions nest the text model under `language_model`. Keep only the text
    /// model (`model.*` and a bare `lm_head`), which drops any vision or audio tensors.
    static func textModuleName(_ name: String) -> String? {
        var name = name
        if name.hasPrefix("language_model.") {
            name.removeFirst("language_model.".count)
        }
        let root = name.prefix { $0 != "." }
        return root == "model" || root == "lm_head" ? name : nil
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        for (name, value) in weights {
            if let mapped = Self.textModuleName(name) {
                renamed[mapped] = value
            }
        }
        var weights = filterLMHeadWeights(
            from: renamed, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        // PyTorch stores the depthwise conv as [C, 1, K]; MLX expects [C, K, 1].
        for (name, value) in weights where name.hasSuffix("conv.conv.weight") {
            if value.dim(-1) > value.dim(1) {
                weights[name] = value.transposed(0, 2, 1)
            }
        }
        return weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.newCache(parameters: parameters)
    }
}

extension LFM3Model {
    /// Vocabulary logits for final hidden states, tied or untied.
    func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        model.projectLogits(hiddenStates, lmHead: lmHead)
    }
}

/// The checkpoint's per-layer quantization map names layers as the checkpoint does
/// (`language_model.model.*`); sanitize renames tensors, so rename the map the same way.
extension LFM3Model: PerLayerQuantizationSanitizing {
    public func sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization)
        -> BaseConfiguration.PerLayerQuantization
    {
        var result = perLayerQuantization
        result.perLayerQuantization = [:]
        for (name, option) in perLayerQuantization.perLayerQuantization {
            result.perLayerQuantization[Self.textModuleName(name) ?? name] = option
        }
        return result
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension LFM3Model {
    /// LFM3's template writes calls as `<tool_call><function=name><parameter=key>`.
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }

    /// `<think>` tags toggled by `enable_thinking` (default on). With thinking off
    /// the template prefills an empty `<think></think>`. A tool call can also end
    /// the thinking block.
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
