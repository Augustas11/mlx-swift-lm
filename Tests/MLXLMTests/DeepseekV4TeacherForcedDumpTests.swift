import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Lab tool for a Studio window, skipped unless its environment is set: a
/// teacher-forced pass of a fixed token sequence through the full V4-Flash
/// model, dumping the per-layer state that `scripts/lab/dsv4/
/// dsv4_tf_reference.py dump` writes for DeepSeek's reference, so the two
/// can be diffed layer by layer (`... compare`).
///
/// - `DSV4_ARTIFACT_DIR`: the V4-Flash-0731 MXFP4 directory.
/// - `DSV4_DUMP_TOKENS`: JSON `{"tokens": [...], "positions": [...]}`.
/// - `DSV4_DUMP_OUT`: the safetensors file to write.
///
/// The tokens prefill in the serving plan's 2048-token chunks. At each listed
/// position the dump holds, per layer, the attention and expert inputs, the
/// hidden state after the layer (`h.L`, all hyper-connection streams), the
/// chosen experts (sorted) and, on ratio-4 layers, the indexer's top-k over
/// the entries visible at that position (sorted, `-1` padded); per
/// compressed layer the final pooled entries; and the float32 logits.
final class DeepseekV4TeacherForcedDumpTests: XCTestCase {
    func testFullModelTeacherForcedDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let artifact = env["DSV4_ARTIFACT_DIR"], let tokenPath = env["DSV4_DUMP_TOKENS"],
            let outPath = env["DSV4_DUMP_OUT"]
        else { throw XCTSkip("DSV4_ARTIFACT_DIR, DSV4_DUMP_TOKENS and DSV4_DUMP_OUT are not set") }

        struct Request: Decodable {
            let tokens: [Int]
            let positions: [Int]
        }
        let request = try JSONDecoder().decode(
            Request.self, from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
        let tokens = request.tokens
        let positions = request.positions
        XCTAssertFalse(positions.isEmpty)
        XCTAssertTrue(positions.allSatisfy { (0 ..< tokens.count).contains($0) })
        XCTAssertEqual(positions, positions.sorted())

        let dir = URL(fileURLWithPath: artifact)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = DeepseekV4Model(config)
        try loadWeights(
            modelDirectory: dir, model: model, perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let caches = try model.newCache(parameters: nil)
        var captured = [String: [MLXArray]]()
        func keep(_ key: String, _ value: MLXArray) { captured[key, default: []].append(value) }

        let step = DeepseekV4Model.prefillStepSize
        for start in stride(from: 0, to: tokens.count, by: step) {
            let end = min(start + step, tokens.count)
            let rows = positions.filter { (start ..< end).contains($0) }.map { Int32($0 - start) }
            let ids = MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, end - start)
            let layerCaches = zip(caches, model.layers).map {
                DeepseekV4LayerCache($0, ratio: $1.attn.ratio)
            }
            var mask: MLXArray?
            if end - start > 1,
                case .array(let array) = layerCaches[0].local.makeMask(
                    n: end - start, windowSize: config.slidingWindow, returnArray: true)
            {
                mask = array
            }
            let at = MLXArray(rows)
            var h = repeated(
                expandedDimensions(model.embed(ids), axis: 2), count: config.hcMult, axis: 2)
            for (l, (layer, cache)) in zip(model.layers, layerCaches).enumerated() {
                let (attnInput, attnPost, attnComb) = layer.attnHC.normalizedBranchInput(
                    h, norm: layer.attnNorm)
                h = glm5NextHCExpand(
                    layer.attn(attnInput, mask: mask, cache: cache), h, attnPost, attnComb)
                let (ffnInput, ffnPost, ffnComb) = layer.ffnHC.normalizedBranchInput(
                    h, norm: layer.ffnNorm)
                let experts = layer.ffn.gate(ffnInput, inputIDs: ids).0
                h = glm5NextHCExpand(layer.ffn(ffnInput, inputIDs: ids), h, ffnPost, ffnComb)
                if !rows.isEmpty {
                    keep("attn_in.\(l)", take(attnInput[0], at, axis: 0))
                    keep("ffn_in.\(l)", take(ffnInput[0], at, axis: 0))
                    keep("experts.\(l)", sorted(take(experts[0], at, axis: 0).asType(.int32), axis: -1))
                    keep("h.\(l)", take(h[0], at, axis: 0).asType(.float32))
                }
            }
            eval(captured.values.flatMap { $0 } + caches.flatMap { $0.innerState() })
        }

        var out = captured.mapValues { concatenated($0, axis: 0) }
        out["positions"] = MLXArray(positions.map(Int32.init))
        let last = out["h.\(model.layers.count - 1)"]!
        out["logits"] = deepseekV4LinearF32(
            model.head, model.norm(model.hcHead(expandedDimensions(last.asType(.bfloat16), axis: 0))))[0]

        for (l, (layer, cache)) in zip(model.layers, caches).enumerated() where layer.attn.ratio > 0 {
            let layerCache = DeepseekV4LayerCache(cache, ratio: layer.attn.ratio)
            if let pooled = layerCache.pool?.pooled { out["pool.\(l)"] = pooled[0].asType(.float32) }
            guard let indexer = layer.attn.indexer, let indexPool = layerCache.indexPool?.pooled
            else { continue }
            out["index_pool.\(l)"] = indexPool[0].asType(.float32)
            let inputs = out["attn_in.\(l)"]!
            var selections = [MLXArray]()
            for (i, p) in positions.enumerated() {
                let visible = (p + 1) / layer.attn.ratio
                let k = min(indexer.topK, visible)
                guard k > 0 else {
                    selections.append(MLXArray.full([indexer.topK], values: MLXArray(Int32(-1))))
                    continue
                }
                let x = inputs[i ..< (i + 1)].reshaped(1, 1, -1)
                let qResidual = layer.attn.qNorm(layer.attn.wqA(x))
                let q = layer.attn.rope(
                    indexer.wqB(qResidual).reshaped(1, 1, indexer.heads, indexer.headDim)
                        .transposed(0, 2, 1, 3), offset: p
                ).asType(.float32)
                let w = indexer.weightsProj(x).asType(.float32).reshaped(indexer.heads, 1, 1)
                    * pow(Float(indexer.heads), -0.5)
                let keys = indexPool[0, ..<visible].asType(.float32)
                let scores = (maximum(matmul(q[0], keys.T), 0) * indexer.scale * w).sum(axis: 0)
                    .reshaped(-1)
                let top = sorted(argSort(-scores)[..<k].asType(.int32))
                selections.append(
                    k == indexer.topK
                        ? top
                        : concatenated([top, MLXArray.full([indexer.topK - k], values: MLXArray(Int32(-1)))]))
            }
            out["topk.\(l)"] = stacked(selections)
        }
        out = out.mapValues { $0.dtype == .bfloat16 ? $0.asType(.float32) : $0 }
        eval(Array(out.values))
        try save(arrays: out, url: URL(fileURLWithPath: outPath))
    }
}
