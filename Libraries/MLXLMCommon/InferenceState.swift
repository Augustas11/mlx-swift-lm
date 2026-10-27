// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Opt-in initial evaluation roots in the checkpoint's unchanged namespace.
/// All weights remain installed and validated. Omitted modules evaluate naturally
/// when used; their checkpoint files must remain immutable for the model lifetime.
package protocol InitialInferenceMaterializationProviding: BaseLanguageModel {
    var initialInferenceModules: [String: Module]? { get }
}

/// Match a module subtree, never a similarly named sibling.
func initiallyMaterializesWeight(_ name: String, prefixes: [String]?) -> Bool {
    guard let prefixes else { return true }
    return prefixes.contains { name.hasPrefix($0 + ".") }
}

package struct InferenceStatePreparationFailure {
    package let modelType: String
    package let error: any Error
}

package struct InferenceStatePreparationReport {
    package let failures: [InferenceStatePreparationFailure]

    package var succeeded: Bool { failures.isEmpty }
}

private let inferenceStateLogger = Logger(
    subsystem: "mlx-swift-lm", category: "inference-state")

/// Prepare a language model after checkpoint loading or an explicit topology
/// update.
///
/// A failed optional optimization is logged and reported while the model
/// remains usable through its unfused path. `BaseLanguageModel` values outside
/// the inference lifecycle, such as rerankers, require no preparation.
@discardableResult
package func prepareInferenceState(
    in model: BaseLanguageModel
) -> InferenceStatePreparationReport {
    guard let languageModel = model as? any LanguageModel else {
        return InferenceStatePreparationReport(failures: [])
    }

    do {
        try languageModel.prepare()
        return InferenceStatePreparationReport(failures: [])
    } catch {
        let modelType = String(reflecting: type(of: languageModel))
        inferenceStateLogger.error(
            "Failed to prepare inference state for \(modelType): \(String(describing: error))")
        return InferenceStatePreparationReport(failures: [
            .init(modelType: modelType, error: error)
        ])
    }
}

/// Prepare derived state and realize the initial inference roots before publication.
///
/// All custom checkpoint loaders should finalize through this function so
/// inference-only optimizations are applied consistently.
@discardableResult
package func materializeModelForInference(
    _ model: BaseLanguageModel
) -> InferenceStatePreparationReport {
    let report = prepareInferenceState(in: model)
    if let modules = (model as? any InitialInferenceMaterializationProviding)?
        .initialInferenceModules
    {
        for module in modules.values {
            eval(module)
        }
    } else {
        eval(model)
    }
    return report
}
