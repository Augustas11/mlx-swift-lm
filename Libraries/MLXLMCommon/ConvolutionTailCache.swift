// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Causal short-convolution tails: each slot holds the last `kernel - 1` inputs of one
/// convolution, `[B, kernel - 1, C]`.
///
/// An ``ArraysCache`` whose type promises one more thing: every write to a slot keeps the newest
/// `kernel - 1` rows of `concat(previous tail, new inputs)`, and the model reports that padded
/// input through ``record(slot:previous:input:)``. That promise is what lets a speculative round
/// keep any prefix of what it wrote: the tail after keeping `n` of `L` verified rows is
/// `padded[:, n ..< n + kernel - 1]`, which is how oMLX rolls an Inkling 2 conv state back
/// (`mtp_partial_rollback`, Apache-2.0). A plain ``ArraysCache`` makes no such promise, so a
/// staged round refuses it.
///
/// Outside a round ``record(slot:previous:input:)`` returns at once: ordinary decode pays nothing.
public final class ConvolutionTailCache: ArraysCache {
    public override class var serializedClassName: String? { "ConvolutionTailCache" }

    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        let cache = ConvolutionTailCache(size: 0)
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache
    }

    /// Per slot, `concat(tail at round start, every row written this round)` in float32, and the
    /// tail width. Non-nil only on a round's staged copy.
    fileprivate var padded: [Int: (rows: MLXArray, width: Int)]?

    public override init(size: Int, leftPadding: [Int]? = nil) {
        super.init(size: size, leftPadding: leftPadding)
    }

    public override func copy() -> any KVCache {
        let new = ConvolutionTailCache(size: slotCount)
        copyContents(to: new)
        return new
    }

    /// Whether this is a round's staged copy, which wants ``record(slot:previous:input:)``.
    public var isRecording: Bool { padded != nil }

    /// Report one convolution write: `previous` is the tail it read, `input` the rows it
    /// appended. Call it beside the write to `self[slot]`; a no-op outside a staged round.
    public func record(slot: Int, previous: MLXArray, input: MLXArray) {
        guard padded != nil else { return }
        let rows = input.asType(.float32)
        if let existing = padded?[slot] {
            padded?[slot] = (concatenated([existing.rows, rows], axis: 1), existing.width)
        } else {
            padded?[slot] = (
                concatenated([previous.asType(.float32), rows], axis: 1), previous.dim(1)
            )
        }
    }

    /// A copy that records, for a staged round. The live cache stays untouched until commit.
    func makeStagedCopy() -> ConvolutionTailCache {
        // Shares the tail arrays rather than slicing them, as `copy()` does: a write rebinds the
        // slot, so sharing is safe, and identity is how commit tells an unwritten slot apart.
        let staged = ConvolutionTailCache(size: slotCount)
        for index in 0 ..< slotCount {
            staged[index] = self[index]
        }
        staged.offset = offset
        staged.leftPadding = leftPadding
        staged.lengths = lengths
        staged.padded = [:]
        return staged
    }

    /// What the staged copy recorded: per slot, the padded input and the tail width.
    var recordedInputs: [Int: (rows: MLXArray, width: Int)] { padded ?? [:] }

    /// Rows recorded for `slot` beyond the round-start tail.
    func recordedPositions(slot: Int) -> Int? {
        padded?[slot].map { $0.rows.dim(1) - $0.width }
    }

    /// Set each recorded slot to its tail after keeping the first `retaining` recorded rows.
    func cut(from recorded: [Int: (rows: MLXArray, width: Int)], retaining: Int) {
        for (slot, entry) in recorded {
            let start = Swift.min(Swift.max(retaining, 0), entry.rows.dim(1) - entry.width)
            self[slot] = entry.rows[0..., start ..< (start + entry.width), 0...]
        }
    }
}
