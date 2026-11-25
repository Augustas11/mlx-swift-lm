import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// A reference ``KVCacheRowAttentionProtocol`` batch: each row keeps the one-row cache the model
/// builds (`RotatingKVCache` or `KVCacheSimple`) and hands attention exactly what that cache
/// returns, as a paged batch does.
private final class RowCachesBatch: BaseKVCache, KVCacheRowAttentionProtocol {
    let rows: [KVCache]

    init(rows: [KVCache]) {
        self.rows = rows
        super.init()
    }

    var rowOffsets: [Int] { rows.map(\.offset) }

    override var state: [MLXArray] {
        get { [] }
        set {}
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("Inkling 2 attends batched rows through updateAndAttendRows")
    }

    func updateAndAttendRows(
        keys: MLXArray, values: MLXArray,
        attend: (Int, KVCacheRowAttentionView) -> MLXArray
    ) -> MLXArray {
        let length = keys.dim(2)
        let outputs = rows.enumerated().map { row, cache -> MLXArray in
            let r = row ..< row + 1
            let (k, v) = cache.update(keys: keys[r], values: values[r])
            let rotation = length == 1 ? (cache as? RotatingKVCache)?.ringRotation ?? 0 : 0
            return attend(row, KVCacheRowAttentionView(keys: k, values: v, ringRotation: rotation))
        }
        return concatenated(outputs, axis: 0)
    }
}

/// Continuous batching (#2505): rows at their own offsets, batched through a row-attention cache
/// with packed convolution tails, give each row the logits it gets alone. Tiny fixture (see
/// `Inkling2ParityTests`): an 8-token window and a 6-token global band, shorter than the prompts,
/// so the rows' rings wrap at different columns and the global band clips. CPU by default;
/// `INKLING2_PARITY_DEVICE=gpu` runs the GPU path (ring-as-stored decode, fused inputs, per-row tau).
final class Inkling2BatchRowTests: XCTestCase {

    private var device: Device {
        ProcessInfo.processInfo.environment["INKLING2_PARITY_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func load(_ name: String) throws -> Inkling2Model {
        let root = try XCTUnwrap(Bundle.module.url(forResource: "inkling2", withExtension: nil))
        let directory = root.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path)
        else { throw XCTSkip("\(name) fixture absent (tools/inkling2_parity/make_tiny_fixture.py)") }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
        try configuration.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = Inkling2Model(configuration, quantization: base.perLayerQuantization)
        try model.applyQuantizationMetadata(directory: directory)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        return model
    }

    /// The row's cache after prefilling `prompt` alone in chunks of `step`.
    private func prefill(_ model: Inkling2Model, _ prompt: [Int], step: Int) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while !rest.isEmpty {
            let n = min(step, rest.count)
            eval(model(MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n), cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        return cache
    }

    func testRaggedRowsMatchEachRowAlone() throws {
        for fixture in ["tiny-oq", "tiny-source"] {
            try Device.withDefaultDevice(device) {
                let model = try load(fixture)
                let window = model.configuration.text.slidingWindow
                // Lengths past the window at different ring columns, and one inside it.
                let prompts: [[Int]] = [
                    (0 ..< window + 5).map { ($0 * 7 + 3) % 97 + 10 },
                    (0 ..< 2 * window + 3).map { ($0 * 11 + 5) % 89 + 12 },
                    (0 ..< window - 3).map { ($0 * 13 + 1) % 83 + 7 },
                ]
                let steps = 2 * window + 2
                let first = prompts.map { $0.last! }

                // Alone: every row decodes its own greedy tokens.
                var alone: [[Int]] = []
                var aloneLogits: [[[Float]]] = []
                for prompt in prompts {
                    let cache = try prefill(model, Array(prompt.dropLast()), step: 5)
                    var token = prompt.last!
                    var tokens: [Int] = []
                    var logits: [[Float]] = []
                    for _ in 0 ..< steps {
                        let row = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0, -1]
                        eval(row)
                        token = argMax(row, axis: -1).item(Int.self)
                        tokens.append(token)
                        logits.append(row.asType(.float32).asArray(Float.self))
                    }
                    alone.append(tokens)
                    aloneLogits.append(logits)
                }

                // Batched: the same prefilled rows, flat layout, attention per row, tails packed.
                let rowCaches = try prompts.map { try prefill(model, Array($0.dropLast()), step: 5) }
                let flatRows = rowCaches.map { Inkling2Model.flatCacheLayout($0) }
                let batch: [KVCache] = (0 ..< flatRows[0].count).map { index in
                    if index % 2 == 0 {
                        return RowCachesBatch(rows: flatRows.map { $0[index] })
                    }
                    let tails = flatRows.map { $0[index] as! ArraysCache }
                    let packed = ArraysCache(size: 4)
                    for slot in 0 ..< 4 {
                        packed[slot] = concatenated(tails.map { $0[slot]! }, axis: 0)
                    }
                    return packed
                }
                var current = first
                var batched = Array(repeating: [Int](), count: prompts.count)
                var worst: Float = 0
                for step in 0 ..< steps {
                    let input = MLXArray(current.map(Int32.init)).reshaped(prompts.count, 1)
                    let logits = model(input, cache: batch)[0..., -1]
                    eval(logits)
                    current = argMax(logits, axis: -1).asArray(Int.self)
                    for row in prompts.indices {
                        batched[row].append(current[row])
                        let mine = logits[row].asType(.float32).asArray(Float.self)
                        let d = zip(mine, aloneLogits[row][step]).map { abs($0 - $1) }.max() ?? 0
                        worst = max(worst, d)
                    }
                }
                print("inkling2 batch rows \(fixture) \(device): max |dlogit| = \(worst)")
                XCTAssertEqual(batched, alone, "\(fixture): batched rows' greedy tokens")
                XCTAssertLessThan(worst, 1e-3, "\(fixture): batched rows' logits")
            }
        }
    }

    /// The flat layout unwraps each visit's (attention, tail) and pairs back to the same caches.
    func testFlatLayoutRoundTrips() throws {
        let caches: [KVCache] = [
            CacheList(KVCacheSimple(), ArraysCache(size: 4)),
            CacheList(RotatingKVCache(maxSize: 8, keep: 0), ArraysCache(size: 4)),
        ]
        let flat = Inkling2Model.flatCacheLayout(caches)
        XCTAssertEqual(flat.count, 4)
        XCTAssertTrue(flat[0] is KVCacheSimple)
        XCTAssertTrue(flat[1] is ArraysCache)
        XCTAssertTrue(flat[2] is RotatingKVCache)
        let paired = Inkling2Model.visitCaches(flat, visits: 2)
        XCTAssertEqual(paired.count, 2)
        XCTAssertTrue(((paired[1] as? CacheList)?[0] as AnyObject?) === (flat[2] as AnyObject))
        XCTAssertTrue(
            (Inkling2Model.visitCaches(caches, visits: 2)[0] as AnyObject) === (caches[0] as AnyObject))
    }
}
