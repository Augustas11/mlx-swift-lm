// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5/Qwen3.6 text MTP drafter model types.
///
/// Callers should invoke this once before loading a Qwen text drafter through
/// `MTPDrafterModelFactory`.
public enum Qwen35TextMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_text",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35TextConfiguration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_mtp",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(
                    config,
                    preconvertedNorms: true,
                    standaloneCheckpoint: true)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_moe",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
        )
    }
}
