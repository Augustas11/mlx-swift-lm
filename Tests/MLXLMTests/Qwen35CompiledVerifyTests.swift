// Copyright © 2026 Apple Inc.
//
// Pins the compiled short multi-token step (MTP verification) bit-for-bit
// against the general path: hidden states, every cache entry it writes, and
// the recurrent checkpoint a packed verification restores on rejection.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35CompiledVerifyTests: XCTestCase {

    /// GDN, attention, GDN, attention: two segments with a full-attention
    /// tail and head between them, and a MoE mlp in every layer.
    private func tinyConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        guard got.shape == want.shape else { return }
        let a = got.asType(.float32).asArray(Float.self)
        let b = want.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).lazy.filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    private func assertCachesBitIdentical(
        _ got: [KVCache], _ want: [KVCache], _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        for (index, (g, w)) in zip(got, want).enumerated() {
            XCTAssertEqual(g.offset, w.offset, "\(label) cache \(index) offset", file: file, line: line)
            let gs = g.state
            let ws = w.state
            XCTAssertEqual(gs.count, ws.count, "\(label) cache \(index) state count")
            for (slot, (a, b)) in zip(gs, ws).enumerated() {
                assertBitIdentical(a, b, "\(label) cache \(index) state \(slot)", file: file, line: line)
            }
        }
    }

    /// Prefill both cache sets identically, then run verification steps of
    /// `width` columns: one set through the compiled step
    /// (`applyFinalNorm: true`), the other through the general path
    /// (`applyFinalNorm: false`, then the same final norm).
    private func runSteps(
        dtype: DType, batch: Int, width: Int, checkpointAfter: Int?, seed: UInt64,
        quantized: Bool = false
    ) throws {
        let config = try tinyConfiguration()
        try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35TextModel(config)
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            if quantized {
                quantize(model: model, groupSize: 32, bits: 4)
            }
            eval(model)
            try verifySteps(
                model, dtype: dtype, batch: batch, width: width,
                checkpointAfter: checkpointAfter, label: quantized ? " q4" : "")
        }
    }

    private func verifySteps(
        _ model: Qwen35TextModel, dtype: DType, batch: Int, width: Int,
        checkpointAfter: Int?, label variant: String
    ) throws {
        let inner = model.model

        let compiled = try model.newCache(parameters: nil)
        let general = try model.newCache(parameters: nil)
        let prompt = MLXRandom.randInt(0 ..< 32, [batch, 7]).asType(.int32)
        for cache in [compiled, general] {
            eval(inner.forward(prompt, cache: cache, applyFinalNorm: true))
        }
        assertCachesBitIdentical(compiled, general, "after prefill")

        for step in 0 ..< 3 {
            let tokens = MLXRandom.randInt(0 ..< 32, [batch, width]).asType(.int32)
            let label =
                "\(dtype)\(variant) B=\(batch) T=\(width) checkpoint=\(checkpointAfter.map(String.init) ?? "nil") step \(step)"

            let fast = inner.forward(
                tokens, cache: compiled, applyFinalNorm: true,
                checkpointAfter: checkpointAfter)
            let slow = inner.norm(
                inner.forward(
                    tokens, cache: general, applyFinalNorm: false,
                    checkpointAfter: checkpointAfter))
            eval(fast, slow)

            assertBitIdentical(fast, slow, "\(label) hidden")
            assertCachesBitIdentical(compiled, general, label)

            if let split = checkpointAfter, split > 0, split < width {
                if step % 2 == 1 {
                    // Reject the proposals: restore each GDN layer's
                    // checkpoint and trim the attention KV to match, so
                    // the next step continues from the restored state.
                    for cache in compiled + general {
                        if let mamba = cache as? MambaCache {
                            XCTAssertTrue(mamba.restoreSpeculativeCheckpoint(), "\(label) checkpoint")
                        } else {
                            XCTAssertEqual(cache.trim(width - split), width - split)
                        }
                    }
                    assertCachesBitIdentical(compiled, general, "\(label) restored checkpoint")
                } else {
                    for cache in compiled + general {
                        (cache as? MambaCache)?.discardSpeculativeCheckpoint()
                    }
                }
            }
        }
        XCTAssertGreaterThan(
            inner.compiledVerifySegmentCount, 0, "the compiled verify step never ran")
    }

    /// A served bf16 q4 model after `prepare()`: every GDN layer reads its
    /// input projections through the fused projection, which is not a
    /// registered child, so the trace sees it only as declared compile state.
    private func preparedModel(seed: UInt64) throws -> Qwen35TextModel {
        try XCTSkipUnless(qwen35FourGDNEnabled, "fused GDN projection kill switch is set")
        let model = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35TextModel(try tinyConfiguration())
            model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
            quantize(model: model, groupSize: 32, bits: 4)
            eval(model)
            return model
        }
        try model.prepare()
        try assertFusedProjectionsPrepared(model)
        return model
    }

    /// GDN layers, ordered by module path so two models pair layer by layer.
    private func gdnLayers(_ model: Qwen35TextModel) -> [Qwen35GatedDeltaNet] {
        model.namedModules().sorted { $0.0 < $1.0 }.compactMap { $0.1 as? Qwen35GatedDeltaNet }
    }

    private func assertFusedProjectionsPrepared(
        _ model: Qwen35TextModel, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let gdn = gdnLayers(model)
        XCTAssertEqual(gdn.count, 2, "expected two GDN layers", file: file, line: line)
        for (index, layer) in gdn.enumerated() {
            XCTAssertTrue(
                layer.hasFusedInputProjection, "GDN \(index) fused projection not prepared",
                file: file, line: line)
            XCTAssertEqual(
                layer.fusedProjectionTraceState.count, 1,
                "GDN \(index) fused projection not declared as trace state",
                file: file, line: line)
        }
    }

    /// The prepared (served) model: the compiled step reads the fused GDN
    /// projection and stays bit-identical to the general path.
    func testCompiledVerifyWithPreparedFusedProjectionMatchesGeneralPath() throws {
        let model = try preparedModel(seed: 31)
        try withRandomState(MLXRandom.RandomState(seed: 32)) {
            try verifySteps(
                model, dtype: .bfloat16, batch: 1, width: 2, checkpointAfter: 1,
                label: " q4 prepared")
            try verifySteps(
                model, dtype: .bfloat16, batch: 2, width: 3, checkpointAfter: 2,
                label: " q4 prepared")
        }
    }

    /// Lifetime and reload. The trace snapshots its declared state's array
    /// objects, and a parameter update writes through those same objects, so
    /// new weights loaded into the fused projection must reach the next
    /// replay. A trace that captured the fused projection as a tape constant
    /// would keep replaying the weights it was traced with and disagree with
    /// the general path. Then models are freed and rebuilt with different
    /// weights: a replay of a freed model's trace would do the same.
    func testCompiledVerifyFollowsReloadedFusedProjection() throws {
        let model = try preparedModel(seed: 41)
        try withRandomState(MLXRandom.RandomState(seed: 42)) {
            try verifySteps(
                model, dtype: .bfloat16, batch: 1, width: 2, checkpointAfter: 1,
                label: " q4 before reload")
        }
        let traced = model.model.compiledVerifySegmentCount
        XCTAssertGreaterThan(traced, 0)

        // Load different weights into each traced fused projection in place.
        let donor = try preparedModel(seed: 43)
        let targets = gdnLayers(model)
        let sources = gdnLayers(donor)
        XCTAssertEqual(targets.count, sources.count)
        for (target, source) in zip(targets, sources) {
            let fused = try XCTUnwrap(target.fusedProjectionTraceState.first)
            let replacement = try XCTUnwrap(source.fusedProjectionTraceState.first)
            let before = fused.parameters().flattened().map { $0.1 }
            try fused.update(parameters: replacement.parameters(), verify: .all)
            eval(fused)
            // Written through the same array objects the trace snapshot holds.
            for (old, new) in zip(before, fused.parameters().flattened().map { $0.1 }) {
                XCTAssertTrue(old === new, "parameter update replaced an array object")
            }
        }
        try withRandomState(MLXRandom.RandomState(seed: 44)) {
            try verifySteps(
                model, dtype: .bfloat16, batch: 1, width: 2, checkpointAfter: 1,
                label: " q4 after in-place reload")
        }
        XCTAssertEqual(
            model.model.compiledVerifySegmentCount, traced,
            "the reload must replay the existing trace, not retrace")

        // Free and rebuild: each model runs its compiled step and is
        // released before the next one, with different weights, is built.
        for seed in UInt64(51) ... 53 {
            let fresh = try preparedModel(seed: seed)
            try withRandomState(MLXRandom.RandomState(seed: seed + 100)) {
                try verifySteps(
                    fresh, dtype: .bfloat16, batch: 1, width: 2, checkpointAfter: 1,
                    label: " q4 rebuilt seed \(seed)")
            }
        }
    }

    func testCompiledVerifyMatchesGeneralPathBitwise() throws {
        for dtype in [DType.float16, DType.bfloat16] {
            try runSteps(dtype: dtype, batch: 1, width: 2, checkpointAfter: 1, seed: 21)
            try runSteps(dtype: dtype, batch: 2, width: 2, checkpointAfter: 1, seed: 22)
            try runSteps(dtype: dtype, batch: 1, width: 3, checkpointAfter: 2, seed: 23)
        }
        // Quantized weights, as served: the step's matmuls run the quantized
        // kernels inside the trace.
        try runSteps(
            dtype: .bfloat16, batch: 1, width: 2, checkpointAfter: 1, seed: 25, quantized: true)
        try runSteps(
            dtype: .bfloat16, batch: 2, width: 2, checkpointAfter: 1, seed: 26, quantized: true)
    }

    /// Short steps without a checkpoint (a prompt tail, a short follow-up
    /// turn) and steps wider than the cap keep the general path.
    func testOtherMultiTokenStepsKeepTheGeneralPath() throws {
        let model = Qwen35TextModel(try tinyConfiguration())
        eval(model)
        let inner = model.model
        let cache = try model.newCache(parameters: nil)
        func tokens(_ count: Int) -> MLXArray {
            MLXArray(Int32(0) ..< Int32(count)).reshaped(1, count)
        }
        eval(inner.forward(tokens(7), cache: cache, applyFinalNorm: true))
        eval(inner.forward(tokens(3), cache: cache, applyFinalNorm: true))
        eval(inner.forward(tokens(3), cache: cache, applyFinalNorm: true, checkpointAfter: 3))
        let wide = Qwen35TextModelInner.maxCompiledVerifyWidth + 1
        eval(inner.forward(tokens(wide), cache: cache, applyFinalNorm: true, checkpointAfter: 1))
        XCTAssertEqual(inner.compiledVerifySegmentCount, 0)
    }
}
