// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package func qwenMTPSanitizeWeights(
    weights: [String: MLXArray],
    mtpNumHiddenLayers: Int,
    numExperts: Int,
    shiftNormWeights: Bool,
    standaloneCheckpoint: Bool = false
) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights {
        if key.hasPrefix("mtp.") {
            sanitized[key] = value
        } else if standaloneCheckpoint {
            sanitized["mtp.\(key)"] = value
        }
    }

    for layer in 0 ..< max(mtpNumHiddenLayers, 1) {
        let prefix = "mtp.layers.\(layer).mlp"
        if numExperts > 0,
            sanitized["\(prefix).experts.0.up_proj.weight"] != nil
        {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let toJoin = (0 ..< numExperts).map { expert in
                    sanitized.removeValue(
                        forKey: "\(prefix).experts.\(expert).\(projection).weight")!
                }
                sanitized["\(prefix).switch_mlp.\(projection).weight"] = MLX.stacked(toJoin)
            }
        }

        let gateUpKey = "\(prefix).experts.gate_up_proj"
        if let gateUp = sanitized.removeValue(forKey: gateUpKey) {
            let mid = gateUp.dim(-2) / 2
            sanitized["\(prefix).switch_mlp.gate_proj.weight"] =
                gateUp[.ellipsis, ..<mid, 0...]
            sanitized["\(prefix).switch_mlp.up_proj.weight"] =
                gateUp[.ellipsis, mid..., 0...]
        }

        let downProjKey = "\(prefix).experts.down_proj"
        if let downProj = sanitized.removeValue(forKey: downProjKey) {
            sanitized["\(prefix).switch_mlp.down_proj.weight"] = downProj
        }
    }

    let normKeys = [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ]

    for key in Array(sanitized.keys) {
        guard let value = sanitized[key] else { continue }
        if shiftNormWeights,
            normKeys.contains(where: { key.hasSuffix($0) }), value.ndim == 1
        {
            sanitized[key] = value + MLXArray(1, dtype: value.dtype)
        }
    }

    return sanitized
}

package func draftMTPTokenBlock(
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    inputEmbedding: Embedding,
    lastToken: MLXArray,
    lastHidden: MLXArray,
    queryOffset: Int,
    blockSize: Int,
    sampler: any LogitSampler,
    cache: [KVCache],
    draftVocabulary: DraftVocabularyProjection? = nil,
    forward: (
        _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
        _ positionOffset: Int
    ) -> MLXArray
) -> MLXArray {
    precondition(blockSize >= 2, "blockSize must be >= 2")

    var tok = lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
    var hidden = lastHidden
    precondition(!cache.isEmpty, "Qwen MTP drafter cache must not be empty")
    var tokens: [MLXArray] = []
    tokens.reserveCapacity(blockSize - 1)

    for stepIndex in 0 ..< (blockSize - 1) {
        let mtpHidden = forward(
            inputEmbedding(tok),
            hidden,
            cache,
            queryOffset + stepIndex
        )
        hidden = mtpHidden

        let next: MLXArray
        if let draftVocabulary {
            next = draftVocabulary.sample(
                subsetLogits: draftVocabulary.logits(mtpHidden)[0..., -1, 0...], sampler: sampler)
        } else {
            let logits: MLXArray
            if let lmHead {
                logits = lmHead(mtpHidden)
            } else {
                logits = targetEmbedTokens.asLinear(mtpHidden)
            }
            next = sampler.sample(logits: logits[0..., -1, 0...])
        }
        tok = next.ndim == 1 ? next.reshaped([next.dim(0), 1]) : next
        tokens.append(tok)
    }

    return concatenated(tokens, axis: 1)
}

package func normalizedMTPTokenBatch(_ tokens: MLXArray) -> MLXArray {
    switch tokens.ndim {
    case 1:
        return tokens[.newAxis, 0...]
    default:
        return tokens
    }
}

package func normalizedMTPColumn(_ tokens: MLXArray) -> MLXArray {
    let tokens = normalizedMTPTokenBatch(tokens)
    return tokens.dim(-1) == 1 ? tokens : tokens[0..., (-1)...]
}

package func sampleMTPSeed(
    hidden: MLXArray,
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    sampler: any LogitSampler,
    draftVocabulary: DraftVocabularyProjection? = nil
) -> MLXArray {
    normalizedMTPColumn(
        sampleMTPLastColumn(
            hidden: hidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
            sampler: sampler, draftVocabulary: draftVocabulary))
}

/// One draft per row from the last column of `hidden`: the full head, or the
/// gathered subset head mapped back to token ids.
package func sampleMTPLastColumn(
    hidden: MLXArray,
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    sampler: any LogitSampler,
    draftVocabulary: DraftVocabularyProjection?
) -> MLXArray {
    if let draftVocabulary {
        return draftVocabulary.sample(
            subsetLogits: draftVocabulary.logits(hidden)[0..., -1, 0...], sampler: sampler)
    }
    let logits = lmHead.map { $0(hidden) } ?? targetEmbedTokens.asLinear(hidden)
    return sampler.sample(logits: logits[0..., -1, 0...])
}
