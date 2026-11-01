// Copyright © 2026 Apple Inc.
//
// Depth-D (D >= 2) MTP verification on the Qwen3.5 hybrid target: every
// kept prefix's GDN state, rebuilt from the recorded tape, must equal the
// state single-token decode reaches over the same tokens, bit for bit, on
// both the compiled verify step and the general path, and through the packed
// row transactions the provider commits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35TapeRollbackTests: XCTestCase {

    /// Dense, as the 27B rows: GDN, attention, GDN, attention.
    private func tinyDenseConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_text",
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
                "full_attention_interval": 2
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func quantizedModel(seed: UInt64) throws -> Qwen35TextModel {
        try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35TextModel(try tinyDenseConfiguration())
            model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
            quantize(model: model, groupSize: 32, bits: 4)
            eval(model)
            return model
        }
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

    private func gdnStates(_ cache: [KVCache]) -> [[MLXArray]] {
        cache.compactMap { ($0 as? MambaCache).map { [$0[0]!, $0[1]!] } }
    }

    // MARK: - Kernel

    /// The replayed state after any kept prefix equals both the in-kernel
    /// checkpoint path and single-step kernel calls, bit for bit.
    func testTapeReplayMatchesSerialKernelSteps() {
        withRandomState(MLXRandom.RandomState(seed: 2199)) {
            let (B, T, Hk, Dk, Hv, Dv) = (2, 4, 2, 32, 4, 32)
            let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
            let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16) * 0.2
            let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.bfloat16)
            let a = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
            let b = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
            let aLog = MLXRandom.normal([Hv]).asType(.float32)
            let dtBias = MLXRandom.normal([Hv]).asType(.float32)
            let state0 = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.float32) * 0.1

            let (_, checkpoint, final, g, beta) = gatedDeltaUpdateCheckpointedWithGates(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state0,
                checkpointAfter: 1)

            var serial = [MLXArray]()
            var state = state0
            for t in 0 ..< T {
                state = gatedDeltaUpdate(
                    q: q[0..., t ..< (t + 1)], k: k[0..., t ..< (t + 1)],
                    v: v[0..., t ..< (t + 1)], a: a[0..., t ..< (t + 1)],
                    b: b[0..., t ..< (t + 1)], aLog: aLog, dtBias: dtBias, state: state
                ).1
                eval(state)
                serial.append(state)
            }
            assertBitIdentical(checkpoint, serial[0], "checkpoint after 1")
            assertBitIdentical(final, serial[T - 1], "final")
            for keep in 2 ... T {
                let replay = gatedDeltaReplayState(
                    k: k[0..., 1 ..< keep], v: v[0..., 1 ..< keep], g: g[0..., 1 ..< keep],
                    beta: beta[0..., 1 ..< keep], state: checkpoint)
                assertBitIdentical(replay, serial[keep - 1], "replay keep \(keep)")
            }
        }
    }

    // MARK: - Model

    /// One verification of `width` columns from a prefilled cache, then each
    /// kept prefix restored in turn (longest first) and compared with the
    /// GDN state single-token decode reached after the same tokens.
    private func assertPrefixesMatchSerialDecode(
        _ model: Qwen35TextModel, width: Int, compiled: Bool, seed: UInt64
    ) throws {
        let inner = model.model
        let label = "width \(width) \(compiled ? "compiled" : "general")"
        let (prompt, tokens) = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.randInt(0 ..< 32, [1, 9]).asType(.int32),
                MLXRandom.randInt(0 ..< 32, [1, width]).asType(.int32)
            )
        }
        let verify = try model.newCache(parameters: nil)
        let serial = try model.newCache(parameters: nil)
        for cache in [verify, serial] {
            eval(inner.forward(prompt, cache: cache, applyFinalNorm: true))
        }

        var serialStates = [[[MLXArray]]]()
        for c in 0 ..< width {
            eval(inner.forward(tokens[0..., c ..< (c + 1)], cache: serial, applyFinalNorm: true))
            serialStates.append(gdnStates(serial))
        }

        let before = inner.compiledVerifySegmentCount
        let hidden =
            compiled
            ? inner.forward(tokens, cache: verify, applyFinalNorm: true, checkpointAfter: 1)
            : inner.norm(
                inner.forward(tokens, cache: verify, applyFinalNorm: false, checkpointAfter: 1))
        eval(hidden)
        if compiled {
            XCTAssertGreaterThan(inner.compiledVerifySegmentCount, before, "\(label): not compiled")
        }
        for (layer, state) in gdnStates(verify).enumerated() {
            assertBitIdentical(state[0], serialStates[width - 1][layer][0], "\(label) full conv \(layer)")
            assertBitIdentical(state[1], serialStates[width - 1][layer][1], "\(label) full rec \(layer)")
        }
        for keep in stride(from: width - 1, through: 1, by: -1) {
            for cache in verify {
                if let mamba = cache as? MambaCache {
                    XCTAssertTrue(mamba.canRewindSpeculative(1), "\(label) keep \(keep) rewindable")
                    XCTAssertTrue(mamba.restoreSpeculativeCheckpoint(rewinding: 1), "\(label) keep \(keep)")
                } else {
                    XCTAssertEqual(cache.trim(1), 1)
                }
            }
            for (layer, state) in gdnStates(verify).enumerated() {
                assertBitIdentical(
                    state[0], serialStates[keep - 1][layer][0], "\(label) keep \(keep) conv \(layer)")
                assertBitIdentical(
                    state[1], serialStates[keep - 1][layer][1], "\(label) keep \(keep) rec \(layer)")
            }
        }
        // Decoding on from the shortest restored prefix equals serial decode.
        let restoredSerial = try model.newCache(parameters: nil)
        eval(inner.forward(prompt, cache: restoredSerial, applyFinalNorm: true))
        eval(inner.forward(tokens[0..., ..<1], cache: restoredSerial, applyFinalNorm: true))
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        assertBitIdentical(
            model(next, cache: verify), model(next, cache: restoredSerial),
            "\(label) decode after rollback")
    }

    func testDepthTwoAndThreeVerifyPrefixesMatchSerialDecode() throws {
        let model = try quantizedModel(seed: 2199)
        for width in [3, 4] {
            for compiled in [true, false] {
                try assertPrefixesMatchSerialDecode(
                    model, width: width, compiled: compiled, seed: UInt64(30 + width))
            }
        }
    }

    /// Depth one keeps the single in-kernel checkpoint; no tape is recorded.
    func testDepthOneKeepsSingleCheckpoint() throws {
        XCTAssertFalse(Qwen35GatedDeltaNet.recordsVerifyTape(width: 2, checkpointAfter: 1))
        XCTAssertTrue(Qwen35GatedDeltaNet.recordsVerifyTape(width: 3, checkpointAfter: 1))
        XCTAssertFalse(Qwen35GatedDeltaNet.recordsVerifyTape(width: 4, checkpointAfter: nil))
        let model = try quantizedModel(seed: 7)
        let cache = try model.newCache(parameters: nil)
        eval(model.model.forward(
            MLXArray((0 ..< 6).map(Int32.init)).reshaped(1, 6), cache: cache, applyFinalNorm: true))
        eval(model.model.forward(
            MLXArray([Int32(3), 4]).reshaped(1, 2), cache: cache, applyFinalNorm: true,
            checkpointAfter: 1))
        for case let mamba as MambaCache in cache {
            XCTAssertTrue(mamba.speculativeCheckpointIsSingle)
        }
    }

    // MARK: - Packed transactions

    /// The provider's packed path through one GDN layer in
    /// `MTPPackedMambaBatchCache`: a depth-3 native row beside a right-padded
    /// zero-proposal row. Every retained prefix of the native row, and the
    /// padded row's single column, commit the state serial decode reaches.
    func testPackedTransactionsCommitEveryKeptPrefix() throws {
        let model = try quantizedModel(seed: 4199)
        let gdn = try XCTUnwrap(model.model.layers[0].linearAttn)
        let width = 4
        let (prompts, columns) = withRandomState(MLXRandom.RandomState(seed: 4200)) {
            (
                [MLXRandom.normal([1, 5, 64]), MLXRandom.normal([1, 9, 64])].map {
                    $0.asType(.bfloat16)
                },
                MLXRandom.normal([2, width, 64]).asType(.bfloat16)
            )
        }
        func prefilled(_ row: Int) -> MambaCache {
            let cache = MambaCache()
            eval(gdn(prompts[row], cache: cache))
            return cache
        }
        func states(_ cache: MambaCache) -> [MLXArray] { [cache[0]!, cache[1]!] }

        var nativeSerial = [[MLXArray]]()
        let nativeCache = prefilled(0)
        for c in 0 ..< width {
            eval(gdn(columns[0 ..< 1, c ..< (c + 1)], cache: nativeCache))
            nativeSerial.append(states(nativeCache))
        }
        let paddedCache = prefilled(1)
        eval(gdn(columns[1 ..< 2, 0 ..< 1], cache: paddedCache))
        let paddedSerial = states(paddedCache)

        for retain in 1 ... width {
            let rows = [prefilled(0), prefilled(1)]
            let batch = try MTPPackedMambaBatchCache(rowCaches: rows)
            try batch.prepareMTPPackedVerification(rowMaps: [
                MTPPackedVerificationRowMap(
                    rowIndex: 0, queryOffset: 5, inputCount: width, proposalCount: width - 1),
                MTPPackedVerificationRowMap(
                    rowIndex: 1, queryOffset: 9, inputCount: 1, proposalCount: 0),
            ])
            batch.prepare(lengths: [width, 1])
            eval(gdn(columns, mask: batch.makeMask(N: width), cache: batch, checkpointAfter: 1))
            let transactions = try batch.rowTransactions()
            try transactions[0].commit(retaining: retain)
            try transactions[1].commit(retaining: 1)
            assertBitIdentical(rows[0][0]!, nativeSerial[retain - 1][0], "retain \(retain) conv")
            assertBitIdentical(rows[0][1]!, nativeSerial[retain - 1][1], "retain \(retain) rec")
            assertBitIdentical(rows[1][0]!, paddedSerial[0], "retain \(retain) padded conv")
            assertBitIdentical(rows[1][1]!, paddedSerial[1], "retain \(retain) padded rec")
        }

        // One native row alone: no padding, so no mask.
        for retain in 1 ... width {
            let row = prefilled(0)
            let batch = try MTPPackedMambaBatchCache(rowCaches: [row])
            try batch.prepareMTPPackedVerification(rowMaps: [
                MTPPackedVerificationRowMap(
                    rowIndex: 0, queryOffset: 5, inputCount: width, proposalCount: width - 1)
            ])
            eval(gdn(columns[0 ..< 1], cache: batch, checkpointAfter: 1))
            try batch.rowTransactions()[0].commit(retaining: retain)
            assertBitIdentical(row[0]!, nativeSerial[retain - 1][0], "alone retain \(retain) conv")
            assertBitIdentical(row[1]!, nativeSerial[retain - 1][1], "alone retain \(retain) rec")
        }
    }
}
