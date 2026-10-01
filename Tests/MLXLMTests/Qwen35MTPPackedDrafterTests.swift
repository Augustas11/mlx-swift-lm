// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Row inputs for one drafter round: verification hidden states for
/// `[lastCommitted, proposals...]`, the accepted proposals, and the target's
/// final token.
private struct DrafterRoundRow {
    let hidden: MLXArray
    let proposals: [Int]
    let acceptedCount: Int
    let finalToken: Int
}

@Suite("Qwen35 packed MTP drafter", .serialized)
struct Qwen35MTPPackedDrafterTests {
    private let target: Qwen35TextModel
    private let drafter: Qwen35MTPDraftModel
    private let sampler = GenerateParameters(temperature: 0).sampler()

    init() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(packedDrafterConfiguration.utf8))
        MLXRandom.seed(1770)
        target = Qwen35TextModel(configuration)
        drafter = Qwen35MTPDraftModel(configuration)
        eval(target, drafter)
    }

    private func preparedState(promptLength: Int, seed: UInt64) -> MTPDrafterState {
        let key = MLXRandom.key(seed)
        let prompt = MLXRandom.randInt(low: 0, high: 16, [1, promptLength], key: key)
            .asType(.int32)
        let hidden = MLXRandom.normal([1, promptLength, 16], key: MLXRandom.key(seed + 1))
        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: target,
            promptTokens: prompt,
            targetHidden: hidden,
            firstBonus: MLXArray([Int32(seed % 16)]),
            positionDeltas: nil,
            state: &state,
            sampler: sampler)
        eval(state.cache.flatMap(\.state), state.seedToken!)
        return state
    }

    private func roundRow(
        proposals: [Int], acceptedCount: Int, finalToken: Int, seed: UInt64
    ) -> DrafterRoundRow {
        DrafterRoundRow(
            hidden: MLXRandom.normal(
                [1, proposals.count + 1, 16], key: MLXRandom.key(seed)),
            proposals: proposals,
            acceptedCount: acceptedCount,
            finalToken: finalToken)
    }

    private func copied(_ state: MTPDrafterState) -> MTPDrafterState {
        var copy = state
        copy.cache = state.cache.map { $0.copy() }
        return copy
    }

    /// The per-row reference: one `commitDrafterState` call per row.
    private func perRowAdvance(
        _ states: [MTPDrafterState], _ rows: [DrafterRoundRow]
    ) -> [MTPDrafterState] {
        zip(states, rows).map { state, row in
            var state = copied(state)
            drafter.commitDrafterState(
                target: target,
                targetHidden: row.hidden,
                draftTokens: MLXArray(row.proposals.map(Int32.init), [1, row.proposals.count]),
                acceptedCount: row.acceptedCount,
                finalToken: MLXArray([Int32(row.finalToken)]),
                positionDeltas: nil,
                state: &state,
                sampler: sampler)
            eval(state.cache.flatMap(\.state), state.seedToken!, state.seedHidden!)
            return state
        }
    }

    private func packedAdvance(
        _ states: [MTPDrafterState], _ rows: [DrafterRoundRow]
    ) throws -> MTPPackedDrafterAdvanceResult {
        let result = try drafter.advanceAndProposePacked(
            target: target,
            rows: zip(states, rows).map { state, row in
                MTPPackedDrafterAdvanceRow(
                    targetHidden: row.hidden,
                    acceptedTokens: Array(row.proposals.prefix(row.acceptedCount)),
                    finalToken: row.finalToken,
                    positionDeltas: nil,
                    state: state)
            },
            sampler: sampler)
        eval(
            result.states.flatMap { $0.cache.flatMap(\.state) + [$0.seedHidden!] },
            result.proposals)
        return result
    }

    /// Largest absolute difference between two float arrays of equal shape.
    private func maxDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    /// Compare two advanced drafter states. Positions, shapes, and the next
    /// proposal must always match exactly. Float contents must match bit for
    /// bit when `tolerance` is zero, else within it.
    private func expectSameState(
        _ lhs: MTPDrafterState, _ rhs: MTPDrafterState, row: Int, tolerance: Float = 0
    ) {
        #expect(lhs.nextPosition == rhs.nextPosition, "row \(row) position")
        #expect(lhs.proposalAppended == rhs.proposalAppended, "row \(row) appended")
        #expect(lhs.seedToken!.shape == rhs.seedToken!.shape, "row \(row) seed shape")
        #expect(
            lhs.seedToken!.asArray(Int32.self) == rhs.seedToken!.asArray(Int32.self),
            "row \(row) seed token")
        var arrays = [(lhs.seedHidden!, rhs.seedHidden!)]
        #expect(lhs.cache.count == rhs.cache.count)
        for (left, right) in zip(lhs.cache, rhs.cache) {
            #expect(left.offset == right.offset, "row \(row) cache offset")
            #expect(left.state.count == 2 && right.state.count == 2)
            arrays += zip(left.state, right.state).map { ($0, $1) }
        }
        for (l, r) in arrays {
            #expect(l.shape == r.shape, "row \(row) shape")
            guard l.shape == r.shape else { continue }
            if tolerance == 0 {
                #expect(l.asArray(Float.self) == r.asArray(Float.self), "row \(row) bytes")
            } else {
                #expect(maxDifference(l, r) <= tolerance, "row \(row) drift")
            }
        }
    }

    /// Whenever the packed forward runs the same matmul shapes as the per-row
    /// commit, it reproduces the per-row state bit for bit: every row of an
    /// equal-width batch, and every kind of row (accepted, rejected, depth
    /// zero) advanced on its own.
    @Test func packedAdvanceIsBitExactWhenShapesMatchPerRowCommit() throws {
        let states = [
            preparedState(promptLength: 5, seed: 11),
            preparedState(promptLength: 2, seed: 21),
            preparedState(promptLength: 9, seed: 31),
        ]
        let equalWidth = [
            roundRow(proposals: [3], acceptedCount: 1, finalToken: 7, seed: 101),
            roundRow(proposals: [4], acceptedCount: 1, finalToken: 9, seed: 102),
            roundRow(proposals: [5], acceptedCount: 1, finalToken: 1, seed: 103),
        ]
        let reference = perRowAdvance(states, equalWidth)
        let packed = try packedAdvance(states, equalWidth)
        for row in equalWidth.indices {
            expectSameState(packed.states[row], reference[row], row: row)
        }

        let single = [
            roundRow(proposals: [3], acceptedCount: 1, finalToken: 7, seed: 111),  // accepted
            roundRow(proposals: [4], acceptedCount: 0, finalToken: 9, seed: 112),  // rejected
            roundRow(proposals: [], acceptedCount: 0, finalToken: 2, seed: 113),  // depth zero
        ]
        for (index, row) in single.enumerated() {
            let alone = try packedAdvance([states[index]], [row])
            expectSameState(
                alone.states[0], perRowAdvance([states[index]], [row])[0], row: index)
            #expect(alone.proposals.shape == [1, 1])
        }
    }

    /// Mixed batches: accepted, rejected, and depth-zero rows with different
    /// prompt lengths advance in one call. Full-width rows stay bit-exact.
    /// Narrower rows ran a one-token forward per row (matrix-vector kernels)
    /// but share a multi-token packed forward (matrix-matrix kernels), so
    /// their floats differ only by kernel accumulation order; positions,
    /// shapes, and proposals stay exact.
    @Test func mixedWidthPackedAdvanceMatchesPerRowCommit() throws {
        let states = [
            preparedState(promptLength: 5, seed: 11),
            preparedState(promptLength: 2, seed: 21),
            preparedState(promptLength: 9, seed: 31),
            preparedState(promptLength: 1, seed: 41),
        ]
        let rows = [
            roundRow(proposals: [3], acceptedCount: 1, finalToken: 7, seed: 101),  // accepted
            roundRow(proposals: [4], acceptedCount: 0, finalToken: 9, seed: 102),  // rejected
            roundRow(proposals: [], acceptedCount: 0, finalToken: 2, seed: 103),  // depth zero
            roundRow(proposals: [5], acceptedCount: 1, finalToken: 1, seed: 104),  // accepted
        ]

        let reference = perRowAdvance(states, rows)
        let packed = try packedAdvance(states, rows)

        #expect(packed.proposals.shape == [4, 1])
        #expect(
            packed.proposals.asArray(Int32.self)
                == reference.flatMap { $0.seedToken!.asArray(Int32.self) })
        expectSameState(packed.states[0], reference[0], row: 0)
        expectSameState(packed.states[1], reference[1], row: 1, tolerance: 5e-3)
        expectSameState(packed.states[2], reference[2], row: 2, tolerance: 5e-3)
        expectSameState(packed.states[3], reference[3], row: 3)
        #expect(packed.states.map(\.nextPosition) == [7, 3, 10, 3])
        // Inputs are not mutated.
        #expect(states.map(\.nextPosition) == [5, 2, 9, 1])
        #expect(states.map { $0.cache[0].offset } == [5, 2, 9, 1])
    }

    /// Rows join and leave between rounds: a second packed round over a
    /// different row set still matches per-row commits, so a row's state
    /// depends only on its own history.
    @Test func rowsJoiningAndLeavingBetweenRoundsMatchPerRowCommits() throws {
        let initial = [
            preparedState(promptLength: 4, seed: 51),
            preparedState(promptLength: 6, seed: 61),
            preparedState(promptLength: 3, seed: 71),
        ]
        let first = [
            roundRow(proposals: [8], acceptedCount: 1, finalToken: 6, seed: 201),
            roundRow(proposals: [2], acceptedCount: 1, finalToken: 3, seed: 202),
            roundRow(proposals: [7], acceptedCount: 1, finalToken: 12, seed: 203),
        ]
        var referenceStates = perRowAdvance(initial, first)
        let firstPacked = try packedAdvance(initial, first)
        for row in first.indices {
            expectSameState(firstPacked.states[row], referenceStates[row], row: row)
        }

        // Row 1 leaves, a fresh row joins, rows 0 and 2 continue.
        let joined = preparedState(promptLength: 7, seed: 81)
        let second = [
            roundRow(proposals: [1], acceptedCount: 1, finalToken: 4, seed: 301),
            roundRow(proposals: [9], acceptedCount: 1, finalToken: 5, seed: 302),
            roundRow(proposals: [14], acceptedCount: 1, finalToken: 0, seed: 303),
        ]
        referenceStates = perRowAdvance(
            [referenceStates[2], joined, referenceStates[0]], second)
        let secondPacked = try packedAdvance(
            [firstPacked.states[2], joined, firstPacked.states[0]], second)
        for row in second.indices {
            expectSameState(secondPacked.states[row], referenceStates[row], row: row)
        }
        #expect(secondPacked.states.map(\.nextPosition) == [7, 9, 8])

        // A third round mixes a rejection and a depth-zero row into the
        // continuing rows.
        let third = [
            roundRow(proposals: [2], acceptedCount: 0, finalToken: 3, seed: 401),
            roundRow(proposals: [], acceptedCount: 0, finalToken: 8, seed: 402),
            roundRow(proposals: [6], acceptedCount: 1, finalToken: 10, seed: 403),
        ]
        referenceStates = perRowAdvance(referenceStates, third)
        let thirdPacked = try packedAdvance(secondPacked.states, third)
        expectSameState(thirdPacked.states[0], referenceStates[0], row: 0, tolerance: 5e-3)
        expectSameState(thirdPacked.states[1], referenceStates[1], row: 1, tolerance: 5e-3)
        expectSameState(thirdPacked.states[2], referenceStates[2], row: 2)
    }

    /// Changing one row's verdict changes nothing in any other row of the
    /// same packed call.
    @Test func oneRowsVerdictNeverLeaksIntoAnotherRow() throws {
        let states = [
            preparedState(promptLength: 5, seed: 91),
            preparedState(promptLength: 8, seed: 92),
            preparedState(promptLength: 3, seed: 93),
        ]
        let base = [
            roundRow(proposals: [3], acceptedCount: 1, finalToken: 7, seed: 401),
            roundRow(proposals: [4], acceptedCount: 1, finalToken: 9, seed: 402),
            roundRow(proposals: [5], acceptedCount: 0, finalToken: 2, seed: 403),
        ]
        var perturbed = base
        perturbed[1] = roundRow(proposals: [11], acceptedCount: 0, finalToken: 15, seed: 499)

        let lhs = try packedAdvance(states, base)
        let rhs = try packedAdvance(states, perturbed)
        expectSameState(lhs.states[0], rhs.states[0], row: 0)
        expectSameState(lhs.states[2], rhs.states[2], row: 2)
        #expect(lhs.states[1].nextPosition == 10)
        #expect(rhs.states[1].nextPosition == 9)
    }

    @Test func malformedRowsFailBeforeAnyStateChanges() throws {
        let state = preparedState(promptLength: 4, seed: 7)
        let good = MTPPackedDrafterAdvanceRow(
            targetHidden: MLXRandom.normal([1, 2, 16]),
            acceptedTokens: [3], finalToken: 1, positionDeltas: nil, state: state)

        #expect(throws: MTPPackedDrafterError.invalidAcceptedCount(
            row: 1, acceptedCount: 2, inputCount: 2)
        ) {
            try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    good,
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: MLXRandom.normal([1, 2, 16]),
                        acceptedTokens: [3, 4], finalToken: 1, positionDeltas: nil,
                        state: state),
                ],
                sampler: sampler)
        }
        #expect(throws: MTPPackedDrafterError.invalidTargetHidden(row: 0, shape: [2, 16])) {
            try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: MLXRandom.normal([2, 16]),
                        acceptedTokens: [], finalToken: 1, positionDeltas: nil, state: state)
                ],
                sampler: sampler)
        }
        var skewed = state
        skewed.nextPosition += 1
        #expect(throws: MTPPackedDrafterError.positionMismatch(
            row: 0, nextPosition: 5, cacheOffset: 4)
        ) {
            try drafter.advanceAndProposePacked(
                target: target,
                rows: [
                    MTPPackedDrafterAdvanceRow(
                        targetHidden: MLXRandom.normal([1, 1, 16]),
                        acceptedTokens: [], finalToken: 1, positionDeltas: nil,
                        state: skewed)
                ],
                sampler: sampler)
        }
        #expect(state.nextPosition == 4)
        #expect(state.cache[0].offset == 4)
        #expect(
            try drafter.advanceAndProposePacked(target: target, rows: [], sampler: sampler)
                .proposals.shape == [0, 1])
    }
}

private let packedDrafterConfiguration = """
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
