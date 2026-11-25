//
//  LabSectionTimer.swift
//  mlx-swift-lm
//
//  Lab-only section timing for profiling a model forward (macprovider #2409).
//  Off unless `MLX_LM_SECTION_TIMING=1` is set or a harness sets `enabled`.
//  Off, a section is one Bool read and a direct call of its body: no
//  evaluation and no change to the computed values or the graph.
//
//  On, a section evaluates its inputs before its clock starts and its outputs
//  before it stops, so the section's GPU work falls inside its own wall time.
//  That serializes CPU graph building and GPU execution: the times are shares
//  of a synchronized forward, not the latency of the pipelined one. Values
//  are unchanged either way (evaluation order does not change results).
//
//  Sections nest; each records its exclusive time (its own time minus that of
//  the sections inside it). `forward(tag:)` brackets one model forward: it
//  groups the sections under a tag (for example `rows=4 L=1`) and, with
//  `emitLines`, prints one `section_timing` line per forward to stderr.
//

import Foundation
import MLX

public enum LabSectionTimer {

    /// Read once from `MLX_LM_SECTION_TIMING`; a harness may set it.
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["MLX_LM_SECTION_TIMING"] == "1"

    /// Print one stderr line per `forward(tag:)` (the serve has no other
    /// reader). Defaults to `enabled` at load; a harness that drains the
    /// totals itself turns it off.
    nonisolated(unsafe) public static var emitLines: Bool = enabled

    /// One section's totals: calls and exclusive milliseconds.
    public struct Total: Sendable, Equatable {
        public var calls: Int
        public var ms: Double
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var totals: [String: [String: Total]] = [:]
    nonisolated(unsafe) private static var childMs: [Double] = []
    nonisolated(unsafe) private static var currentTag = "-"
    nonisolated(unsafe) private static var forwardTotals: [String: Total] = [:]

    /// Run `body` as the section `name`, `inputs` evaluated before the clock.
    @inlinable
    public static func run(
        _ name: String, _ inputs: [MLXArray] = [], _ body: () -> MLXArray
    ) -> MLXArray {
        guard enabled else { return body() }
        return timed(name, inputs, outputs: { [$0] }, body)
    }

    /// As `run`, for a body returning any value; `outputs` names the arrays
    /// evaluated before the clock stops.
    @inlinable
    public static func run<T>(
        _ name: String, _ inputs: [MLXArray] = [], outputs: (T) -> [MLXArray], _ body: () -> T
    ) -> T {
        guard enabled else { return body() }
        return timed(name, inputs, outputs: outputs, body)
    }

    /// Bracket one model forward under `tag`: its sections are grouped under
    /// the tag, and the forward's own exclusive time is `forward`.
    @inlinable
    public static func forward<T>(
        tag: @autoclosure () -> String, _ inputs: [MLXArray] = [], outputs: (T) -> [MLXArray],
        _ body: () -> T
    ) -> T {
        guard enabled else { return body() }
        return timedForward(tag(), inputs, outputs: outputs, body)
    }

    /// The totals since the last drain, by tag then section; resets them.
    public static func drain() -> [String: [String: Total]] {
        lock.lock()
        defer { lock.unlock() }
        let out = totals
        totals = [:]
        return out
    }

    @usableFromInline
    static func timed<T>(
        _ name: String, _ inputs: [MLXArray], outputs: (T) -> [MLXArray], _ body: () -> T
    ) -> T {
        if !inputs.isEmpty { eval(inputs) }
        lock.lock()
        childMs.append(0)
        lock.unlock()
        let start = DispatchTime.now().uptimeNanoseconds
        let result = body()
        let out = outputs(result)
        if !out.isEmpty { eval(out) }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        lock.lock()
        let children = childMs.popLast() ?? 0
        if !childMs.isEmpty { childMs[childMs.count - 1] += ms }
        let tag = currentTag
        var section = totals[tag, default: [:]][name, default: Total(calls: 0, ms: 0)]
        section.calls += 1
        section.ms += max(0, ms - children)
        totals[tag, default: [:]][name] = section
        var perForward = forwardTotals[name, default: Total(calls: 0, ms: 0)]
        perForward.calls += 1
        perForward.ms += max(0, ms - children)
        forwardTotals[name] = perForward
        lock.unlock()
        return result
    }

    @usableFromInline
    static func timedForward<T>(
        _ tag: String, _ inputs: [MLXArray], outputs: (T) -> [MLXArray], _ body: () -> T
    ) -> T {
        lock.lock()
        let outerTag = currentTag
        let outerForward = forwardTotals
        currentTag = tag
        forwardTotals = [:]
        lock.unlock()
        let start = DispatchTime.now().uptimeNanoseconds
        let result = timed("forward", inputs, outputs: outputs, body)
        let wallMs = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        lock.lock()
        let sections = forwardTotals
        currentTag = outerTag
        forwardTotals = outerForward
        lock.unlock()
        if emitLines {
            let fields = sections.keys.sorted().map {
                "\($0)=\(String(format: "%.3f", sections[$0]!.ms))/\(sections[$0]!.calls)"
            }
            let line = "section_timing tag=\(tag) wall_ms=\(String(format: "%.3f", wallMs)) "
                + fields.joined(separator: " ") + "\n"
            try? FileHandle.standardError.write(contentsOf: Data(line.utf8))
        }
        return result
    }
}
