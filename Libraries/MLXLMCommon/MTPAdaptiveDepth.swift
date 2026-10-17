// Copyright © 2026 Apple Inc.

import Foundation

/// Per-stream draft-depth controller for MTP speculative decoding.
///
/// Each round it picks the depth `d` (drafts per round, `1 ... maxDepth`)
/// that maximizes expected tokens per second, `E(d) / t(d)`:
///
/// - `E(d) = 1 + p1 + p1 p2 + ... + p1 ... pd`, where `pj` is the measured
///   probability that draft `j` is accepted given drafts `1 ..< j` were
///   (an exponential moving average over the rounds that reached `j`);
/// - `t(d)` is the measured wall time of a round at depth `d` (moving
///   average), extrapolated for depths not yet measured as
///   `t(d0) + (d - d0) * columnCost` from the nearest measured depth.
///
/// Every `exploreInterval` rounds it tries one depth above the current best
/// so a depth that improved is found again. When even the best depth's rate
/// stays below plain decoding (`1 / plainStepTime` when the caller measured
/// it, else `1 / (t(1) - columnCost)`, one round minus its draft column) for
/// `parkAfter` consecutive rounds, it reports `shouldPark`: the stream is
/// better off without speculation.
public struct MTPAdaptiveDepth: Sendable {
    public let maxDepth: Int
    /// Smoothing of acceptance and round-time averages.
    public var alpha: Double = 0.2
    /// Rounds between exploration of the next deeper depth.
    public var exploreInterval = 16
    /// Consecutive rounds below the plain-decode estimate before parking.
    public var parkAfter = 24
    /// Prior cost of one extra verify column (seconds), used until two depths
    /// have been measured.
    public var columnCostPrior: Double
    /// Measured seconds of one pipelined plain decode step, when known.
    public var plainStepTime: Double?

    private var acceptance: [Double]
    private var reached: [Int]
    private var roundTime: [Double?]
    private var rounds = 0
    private var belowPlain = 0
    private var exploring: Int?

    /// - Parameters:
    ///   - maxDepth: deepest depth allowed (at most the drafter's block size - 1).
    ///   - initialAcceptance: prior per-position acceptance before measurement.
    ///   - columnCostPrior: prior seconds per extra verify column.
    public init(maxDepth: Int, initialAcceptance: Double = 0.7, columnCostPrior: Double = 0.012) {
        precondition(maxDepth >= 1)
        self.maxDepth = maxDepth
        self.columnCostPrior = columnCostPrior
        self.acceptance = Array(repeating: initialAcceptance, count: maxDepth)
        self.reached = Array(repeating: 0, count: maxDepth)
        self.roundTime = Array(repeating: nil, count: maxDepth + 1)
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

    /// The marginal cost of one extra verify column (seconds).
    public var columnCost: Double {
        let measured = (1 ... maxDepth).compactMap { d in roundTime[d].map { (d, $0) } }
        guard measured.count >= 2, let first = measured.first, let last = measured.last,
            last.0 > first.0
        else { return columnCostPrior }
        return max(0, (last.1 - first.1) / Double(last.0 - first.0))
    }

    /// Estimated wall time of a round at depth `d`, or `nil` before any round.
    public func estimatedTime(_ d: Int) -> Double? {
        if let t = roundTime[d] { return t }
        let measured = (1 ... maxDepth).compactMap { k in roundTime[k].map { (k, $0) } }
        guard let nearest = measured.min(by: { abs($0.0 - d) < abs($1.0 - d) }) else {
            return nil
        }
        return nearest.1 + Double(d - nearest.0) * columnCost
    }

    /// Expected tokens per second at depth `d`.
    public func rate(_ d: Int) -> Double? {
        estimatedTime(d).map { expectedTokens(d) / max($0, 1e-6) }
    }

    /// Depth for the next round.
    public func depth() -> Int {
        if let exploring { return exploring }
        return bestDepth()
    }

    private func bestDepth() -> Int {
        var best = 1
        var bestRate = -Double.infinity
        for d in 1 ... maxDepth {
            guard let r = rate(d) else { return 1 }
            if r > bestRate {
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
        for j in 0 ..< min(accepted + 1, drafted) {
            let hit = j < accepted ? 1.0 : 0.0
            acceptance[j] = reached[j] == 0 ? hit : (1 - alpha) * acceptance[j] + alpha * hit
            reached[j] += 1
        }
        roundTime[drafted] =
            roundTime[drafted].map { (1 - alpha) * $0 + alpha * seconds } ?? seconds
        rounds += 1
        exploring = nil
        let best = bestDepth()
        if rounds % exploreInterval == 0, best < maxDepth {
            exploring = best + 1
        }
        if let bestRate = rate(best), let t1 = estimatedTime(1) {
            let plain = 1 / max(plainStepTime ?? (t1 - columnCost), 1e-6)
            belowPlain = bestRate < plain ? belowPlain + 1 : 0
        }
    }
}
