// DeepInfer's schedule_mtp_varlen: rank cumulative survival probabilities,
// retaining a deterministic prefix for each row under a shared verify budget.

import Foundation

public enum MTPConfidenceSchedule {
    /// Returns proposal counts. `verificationBudget` includes one mandatory
    /// target anchor per row; confidence never decides token acceptance.
    public static func proposalCounts(
        conditionalRates: [[Float]], verificationBudget: Int
    ) -> [Int] {
        let rows = conditionalRates.count
        precondition(verificationBudget >= rows)
        var ranked = [(row: Int, position: Int, survival: Float)]()
        for (row, rates) in conditionalRates.enumerated() {
            var survival: Float = 1
            for (position, rate) in rates.enumerated() {
                // Invalid predictors receive no preference over zero confidence.
                let probability = rate.isFinite ? min(max(rate, 0), 1) : 0
                survival *= probability
                ranked.append((row, position, survival))
            }
        }
        ranked.sort {
            if $0.survival != $1.survival { return $0.survival > $1.survival }
            if $0.row != $1.row { return $0.row < $1.row }
            return $0.position < $1.position
        }
        var counts = Array(repeating: 0, count: rows)
        for item in ranked.prefix(verificationBudget - rows) {
            counts[item.row] += 1
        }
        return counts
    }
}
