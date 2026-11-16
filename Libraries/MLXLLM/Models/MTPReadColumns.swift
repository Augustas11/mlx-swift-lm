// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Kill switch for running the last MTP layer's post-attention norm and MLP
/// (and, for Qwen, the final norm) on the read column only:
/// `MLX_LM_MTP_LAST_COLUMN=0` runs them on every column again.
let mtpLastColumnEnabled: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_LM_MTP_LAST_COLUMN"] {
    case "0", "false", "FALSE", "no", "NO", "off", "OFF":
        return false
    default:
        return true
    }
}()

/// Row `r`'s column `columns[r]` of `x` `[B, L, D]`, as `[B, 1, D]`. `nil`
/// reads every row's last column.
func mtpReadColumns(_ x: MLXArray, _ columns: [Int]?) -> MLXArray {
    let last = x.dim(1) - 1
    guard let columns, columns.contains(where: { $0 != last }) else {
        return x.dim(1) == 1 ? x : x[0..., last..., 0...]
    }
    precondition(columns.count == x.dim(0), "one read column per row")
    if let first = columns.first, columns.allSatisfy({ $0 == first }) {
        return x[0..., first ..< (first + 1), 0...]
    }
    return concatenated(
        columns.enumerated().map { row, column in
            x[row ..< (row + 1), column ..< (column + 1), 0...]
        }, axis: 0)
}
