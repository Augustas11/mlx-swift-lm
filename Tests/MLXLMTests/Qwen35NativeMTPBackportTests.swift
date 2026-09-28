// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Test
func qwen35MTPConfigurationAndStateFactoryAreAvailable() throws {
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(qwen35NativeMTPTextConfiguration().utf8))
    #expect(configuration.mtpNumHiddenLayers == 1)
    #expect(!configuration.mtpUseDedicatedEmbeddings)

    let drafter = Qwen35MTPDraftModel(configuration)
    #expect(drafter.maximumBlockSize == 2)
    #expect(!drafter.requiresSharedTargetKV)
    #expect(drafter.requiresPromptPrefill)
    #expect(drafter.requiresGreedySampling)
    #expect(drafter.makeState(parameters: nil).cache.count == 1)
}

@Test
func qwen35MTPSanitizeKeepsOnlyMTPWeightsAndShiftsNorms() throws {
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(qwen35NativeMTPTextConfiguration().utf8))
    let drafter = Qwen35MTPDraftModel(configuration)
    let sanitized = drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_hidden.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    let norm = try #require(sanitized["mtp.norm.weight"])
    let hiddenNorm = try #require(sanitized["mtp.pre_fc_norm_hidden.weight"])
    eval(norm, hiddenNorm)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(hiddenNorm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test
func qwen35TextMTPRegistrationCreatesNativeDrafters() async throws {
    await Qwen35TextMTPRegistration.register()

    let text = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: Data(qwen35NativeMTPTextConfiguration().utf8),
        modelType: "qwen3_5_text")
    #expect(text is Qwen35MTPDraftModel)

    let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: Data(qwen35NativeMTPStandaloneConfiguration().utf8),
        modelType: "qwen3_5_mtp")
    #expect(standalone is Qwen35MTPDraftModel)
}

@Test
func qwen35OrdinaryForwardRemainsEquivalentWhenMTPStateIsNotRequested() throws {
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(qwen35NativeMTPTextConfiguration().utf8))
    let model = Qwen35TextModel(configuration)
    let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])

    let direct = model(tokens, cache: model.newCache(parameters: nil))
    let ordinary = model(
        LMInput.Text(tokens: tokens),
        cache: model.newCache(parameters: nil),
        state: nil)
    eval(direct, ordinary.logits)

    #expect(ordinary.state == nil)
    #expect(allClose(direct, ordinary.logits, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func qwen35TargetEmitsStrictMTPContinuationMetadataOnRequest() throws {
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(qwen35NativeMTPTextConfiguration().utf8))
    let model = Qwen35TextModel(configuration)
    let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])
    var request = LMOutput.State()
    request[mtpEmitFlagKey] = true

    let output = model(
        LMInput.Text(tokens: tokens),
        cache: model.newCache(parameters: nil),
        state: request)
    let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
    let shared = try #require(output.state?[mtpSharedKVStatesKey])
    let sources = try #require(output.state?[mtpSharedKVSourceIndicesKey])
    let offsets = try #require(output.state?[mtpSharedKVOffsetsKey])
    eval(hidden, shared["full_attention"]!.0, shared["full_attention"]!.1)

    #expect(hidden.shape == [1, 2, 16])
    #expect(sources["full_attention"] == 0)
    #expect(offsets["full_attention"] == 2)
}

private func qwen35NativeMTPTextConfiguration() -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": 1,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": 1,
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private func qwen35NativeMTPStandaloneConfiguration() -> String {
    """
    {
      "model_type": "qwen3_5_mtp",
      "block_size": 2,
      "text_config": \(qwen35NativeMTPTextConfiguration()),
      "vision_config": {}
    }
    """
}
