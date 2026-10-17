// Copyright © 2026 Apple Inc.

import Foundation

/// Per-stream draft-depth controller for MTP speculative decoding.
///
/// Each round it picks the depth `d` (drafts per round, `1 ... maxDepth`)
/// that maximizes expected tokens per second, `E(d) / t(d)`:
///
/// - `E(d) = 1 + p1 + p1 p2 + ... + p1 ... pd`, where `pj` is the measured
///   probability that draft `j` is accepted given drafts `1 ..< j` were (a
///   moving average over the rounds that reached `j`);
/// - `t(d)` is the measured wall time of a round at depth `d` (moving
///   average of rounds at exactly that depth).
///
/// Every depth is measured before any is preferred: a warm-up runs each depth
/// in turn, discarding each depth's first round (it can carry one-time
/// compilation), until every depth has `warmupRounds` kept samples. After
/// that, every `probeInterval` rounds it runs the depth measured least
/// recently, so a depth whose cost or acceptance changed is found again.
/// When even the best depth's rate stays below plain decoding
/// (`1 / plainStepTime` when the caller measured it) for `parkAfter`
/// consecutive rounds, it reports `shouldPark`.
public struct MTPAdaptiveDepth: Sendable {
    public let maxDepth: Int
    /// Smoothing of acceptance and round-time averages.
    public var alpha: Double = 0.2
    /// Kept rounds per depth before the controller chooses.
    public var warmupRounds = 2
    /// Rounds between probes of the least recently measured depth.
    public var probeInterval = 24
    /// Consecutive rounds below plain decoding before parking.
    public var parkAfter = 24
    /// Measured seconds of one pipelined plain decode step, when known; the
    /// controller parks only against a measured plain step.
    public var plainStepTime: Double?

    private var acceptance: [Double]
    private var reached: [Int]
    private var roundTime: [Double?]
    private var samples: [Int]
    private var seen: [Int]
    private var lastMeasured: [Int]
    private var rounds = 0
    private var belowPlain = 0

    /// - Parameters:
    ///   - maxDepth: deepest depth allowed (at most the drafter's block size - 1).
    ///   - initialAcceptance: per-position acceptance assumed before measurement.
    public init(maxDepth: Int, initialAcceptance: Double = 0.7) {
        precondition(maxDepth >= 1)
        self.maxDepth = maxDepth
        self.acceptance = Array(repeating: initialAcceptance, count: maxDepth)
        self.reached = Array(repeating: 0, count: maxDepth)
        self.roundTime = Array(repeating: nil, count: maxDepth + 1)
        self.samples = Array(repeating: 0, count: maxDepth + 1)
        self.seen = Array(repeating: 0, count: maxDepth + 1)
        self.lastMeasured = Array(repeating: 0, count: maxDepth + 1)
    }

    /// Expected tokens emitted by a round at depth `d`.
    public func expectedTokens(_ d: Int) -> Double {
        var total = 1.0
        var product = 1.0
        for j in 0 ..< d {
            product *= acceptance[j]
            total += product
        }
        return total
    }

    /// Measured wall time of a round at depth `d`, `nil` before it is measured.
    public func measuredTime(_ d: Int) -> Double? { roundTime[d] }

    /// Expected tokens per second at depth `d`, `nil` before it is measured.
    public func rate(_ d: Int) -> Double? {
        roundTime[d].map { expectedTokens(d) / max($0, 1e-6) }
    }

    private var warmedUp: Bool {
        (1 ... maxDepth).allSatisfy { samples[$0] >= warmupRounds }
    }

    /// Depth for the next round.
    public func depth() -> Int {
        if !warmedUp {
            return (1 ... maxDepth).first { samples[$0] < warmupRounds } ?? 1
        }
        if maxDepth > 1, rounds > 0, rounds % probeInterval == 0 {
            return (1 ... maxDepth).min { lastMeasured[$0] < lastMeasured[$1] } ?? 1
        }
        return bestDepth()
    }

    /// The measured depth with the highest expected rate.
    public func bestDepth() -> Int {
        var best = 1
        var bestRate = -Double.infinity
        for d in 1 ... maxDepth {
            if let r = rate(d), r > bestRate {
                best = d
                bestRate = r
            }
        }
        return best
    }

    /// Whether plain decoding is expected to beat every depth.
    public var shouldPark: Bool { belowPlain >= parkAfter }

    /// Record a round: drafts proposed, the accepted prefix, and its wall time.
    public mutating func record(drafted: Int, accepted: Int, seconds: Double) {
        guard drafted >= 1, drafted <= maxDepth else { return }
        rounds += 1
        for j in 0 ..< min(accepted + 1, drafted) {
            let hit = j < accepted ? 1.0 : 0.0
            acceptance[j] = reached[j] == 0 ? hit : (1 - alpha) * acceptance[j] + alpha * hit
            reached[j] += 1
        }
        seen[drafted] += 1
        // A depth's first round can carry one-time compilation: not timed.
        guard seen[drafted] > 1 else { return }
        roundTime[drafted] =
            roundTime[drafted].map { (1 - alpha) * $0 + alpha * seconds } ?? seconds
        samples[drafted] += 1
        lastMeasured[drafted] = rounds
        if warmedUp, let plainStepTime, let bestRate = rate(bestDepth()) {
            belowPlain = bestRate < 1 / max(plainStepTime, 1e-6) ? belowPlain + 1 : 0
        }
    }
}
