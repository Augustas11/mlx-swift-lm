import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Inkling 2 Mini configuration, visit schedule, loop driver and checkpoint layouts. CPU only:
/// no test evaluates a weight-sized array.
///
/// Fixtures (`Resources/inkling2`): the `config.json` of thinkingmachines/Inkling-2-Mini @ b85c3cfa
/// (`source-*`) and mlx-community/Inkling-2-Mini-oQ4e-mtp @ 3eea0427 (`oq4e-*`), and the dtype and
/// shape of every tensor their `model.safetensors.index.json` lists, read from the safetensors headers.
final class Inkling2Tests: XCTestCase {

    private struct TensorList: Decodable {
        let tensors: [String: Entry]

        struct Entry: Decodable {
            let dtype: String
            let shape: [Int]

            init(from decoder: Decoder) throws {
                var c = try decoder.unkeyedContainer()
                dtype = try c.decode(String.self)
                shape = try c.decode([Int].self)
            }
        }
    }

    private func resource(_ name: String) throws -> Data {
        let directory = try XCTUnwrap(Bundle.module.url(forResource: "inkling2", withExtension: nil))
        return try Data(contentsOf: directory.appendingPathComponent(name))
    }

    private func configuration(_ layout: String) throws -> Inkling2Configuration {
        try JSONDecoder().decode(Inkling2Configuration.self, from: resource("\(layout)-config.json"))
    }

    /// Lazy zero arrays with each checkpoint tensor's dtype and shape. FP8 bytes load as uint8.
    private func placeholders(_ layout: String) throws -> [String: MLXArray] {
        let list = try JSONDecoder().decode(TensorList.self, from: resource("\(layout)-tensors.json"))
        let types: [String: DType] = [
            "BF16": .bfloat16, "F16": .float16, "F32": .float32, "U32": .uint32, "U8": .uint8,
            "F8_E4M3": .uint8,
        ]
        return try list.tensors.mapValues { entry in
            MLXArray.zeros(entry.shape, dtype: try XCTUnwrap(types[entry.dtype], entry.dtype))
        }
    }

    // MARK: Configuration

    func testConfigurationDecodesBothLayouts() throws {
        for layout in ["oq4e", "source"] {
            let config = try configuration(layout)
            try config.validateModelConfiguration()
            let t = config.text
            XCTAssertEqual(config.modelType, "inkling2_loop")
            XCTAssertEqual(config.eosTokenId, 200006)
            XCTAssertEqual(t.hiddenSize, 2304)
            XCTAssertEqual(t.hiddenLayers, 24)
            XCTAssertEqual(t.vocabularySize, 201024)
            XCTAssertEqual(t.unpaddedVocabularySize, 200058)
            XCTAssertEqual([t.attentionHeads, t.kvHeads, t.headDim], [24, 8, 128])
            XCTAssertEqual([t.slidingAttentionHeads, t.slidingKVHeads, t.slidingHeadDim], [24, 8, 128])
            XCTAssertEqual(t.slidingWindow, 1024)
            XCTAssertEqual([t.relativeDim, t.relativeExtent], [16, 1024])
            XCTAssertEqual(t.logScalingFloor, 8192)
            XCTAssertEqual(t.logScalingAlpha, 0.1)
            XCTAssertEqual([t.experts, t.expertsPerToken, t.sharedExperts, t.intermediateSize], [128, 8, 0, 1152])
            XCTAssertEqual([t.gateSoftcap, t.upSoftcap, t.routeScale, t.finalLogitSoftcap], [4, 25, 8, 30])
            XCTAssertEqual(t.convolutionKernelSize, 4)
            XCTAssertEqual(t.layersPerBlock, 6)
            XCTAssertTrue(t.useValueNorm)
            XCTAssertTrue(t.attentionResidualNormalizesValues)
            XCTAssertEqual(
                (0 ..< 24).filter { !t.isSliding($0) }, [5, 11, 17, 23], "global layers in \(layout)")
            let loop = try XCTUnwrap(t.loop)
            XCTAssertEqual([loop.coreStart, loop.coreEnd, loop.passes, loop.gateRank], [6, 18, 3, 64])
            XCTAssertFalse(loop.isVarloop)
            let d = try XCTUnwrap(config.dspark)
            XCTAssertEqual(d.blockSize, 8)
            XCTAssertEqual(d.hiddenLayers, 3)
            XCTAssertEqual(d.localLayerIds, [0, 2])
            XCTAssertEqual(d.targetLayerIds, [5, 23, 41, 48])
            XCTAssertEqual(d.markovRank, 256)
            XCTAssertEqual(d.intermediateSize, 9216)
            XCTAssertEqual(d.ropeTheta, 1_000_000)
        }
    }

    func testValidationRejectsUnsupportedConfigurations() throws {
        let base = try JSONSerialization.jsonObject(with: resource("oq4e-config.json")) as! [String: Any]

        func decode(_ edit: (inout [String: Any]) -> Void) throws -> Inkling2Configuration {
            var json = base
            edit(&json)
            return try JSONDecoder().decode(
                Inkling2Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        }
        func editText(_ json: inout [String: Any], _ edit: (inout [String: Any]) -> Void) {
            var text = json["text_config"] as! [String: Any]
            edit(&text)
            json["text_config"] = text
        }
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("core end off a block boundary", { j in
                editText(&j) { t in
                    var loop = t["loop_config"] as! [String: Any]
                    loop["core_end"] = 17
                    t["loop_config"] = loop
                }
            }),
            ("variable loop", { j in
                editText(&j) { t in
                    var loop = t["loop_config"] as! [String: Any]
                    loop["is_varloop"] = true
                    t["loop_config"] = loop
                }
            }),
            ("no loop", { j in editText(&j) { $0.removeValue(forKey: "loop_config") } }),
            ("shared experts", { j in editText(&j) { $0["n_shared_experts"] = 1 } }),
            ("top-k above experts", { j in editText(&j) { $0["num_experts_per_tok"] = 129 } }),
            ("draft target past the visits", { j in
                var mtp = j["mtp_config"] as! [String: Any]
                mtp["target_layer_ids"] = [5, 23, 41, 47]
                j["mtp_config"] = mtp
            }),
            ("other model type", { j in j["model_type"] = "inkling2" }),
        ]
        for (name, edit) in cases {
            XCTAssertThrowsError(try decode(edit).validateModelConfiguration(), name)
        }
        XCTAssertNoThrow(try decode { _ in }.validateModelConfiguration())
    }

    // MARK: Schedule

    func testLoopScheduleVisitsTheCoreThreeTimes() throws {
        let schedule = Inkling2LoopSchedule(try configuration("oq4e").text)
        XCTAssertEqual(schedule.count, 48)
        XCTAssertEqual(
            schedule.visits.map(\.layer),
            Array(0 ..< 6) + Array(6 ..< 18) + Array(6 ..< 18) + Array(6 ..< 18) + Array(18 ..< 24))
        XCTAssertEqual(schedule.visits.map(\.index), Array(0 ..< 48))
        XCTAssertEqual(
            schedule.visits.map(\.pass),
            Array(repeating: 0, count: 18) + Array(repeating: 1, count: 12)
                + Array(repeating: 2, count: 12) + Array(repeating: 0, count: 6))
        XCTAssertEqual(schedule.visits.filter(\.inCore).map(\.index), Array(6 ..< 42))
        // Eight global visits carry unbounded K/V; the other forty are 1024-slot rings.
        XCTAssertEqual(schedule.visits.filter { !$0.isSliding }.map(\.index), [5, 11, 17, 23, 29, 35, 41, 47])
        XCTAssertEqual(schedule.visits(ofLayer: 11), [11, 23, 35])
        XCTAssertEqual(schedule.visits(ofLayer: 5), [5])
        XCTAssertEqual(
            schedule.visits.filter(\.closesBlock).map(\.index), [0, 6, 12, 18, 24, 30, 36, 42])
        // DSpark reads the inputs of visits 5, 23 and 41 (layer 5; layer 11, pass 2; layer 17,
        // pass 3) and the final norm (48).
        let v = schedule.visits
        XCTAssertEqual([v[5].layer, v[5].pass], [5, 0])
        XCTAssertEqual([v[23].layer, v[23].pass], [11, 1])
        XCTAssertEqual([v[41].layer, v[41].pass], [17, 2])
        let model = try Device.withDefaultDevice(.cpu) { Inkling2Model(try configuration("oq4e")) }
        XCTAssertEqual(model.kvHeads.count, 48)
        XCTAssertEqual(model.vocabularySize, 200058)
    }

    /// The driver against a toy layer (`x + c(layer)`, appending its input when it closes a
    /// block, ignoring messages) whose trajectory is known in closed form.
    func testLoopDriverRestartsEachPassFromThePreludeAndReadsThePreviousPass() throws {
        try Device.withDefaultDevice(.cpu) {
            let schedule = Inkling2LoopSchedule(try configuration("oq4e").text)
            func c(_ layer: Int) -> [Float] { [Float(layer + 1), -0.25 * Float(layer + 1)] }
            func plus(_ a: [Float], _ layers: Range<Int>) -> [Float] {
                layers.reduce(a) { acc, l in zip(acc, c(l)).map { $0 + $1 } }
            }
            let x0: [Float] = [1, 2]
            let prelude = plus(x0, 0 ..< 6)
            let midPass = plus(prelude, 6 ..< 12)
            let endPass = plus(prelude, 6 ..< 18)

            // Message queries `[4, 2]`: row i is (i, -i) / 2.
            let query = MLXArray((0 ..< 4).flatMap { [Float($0) / 2, -Float($0) / 2] }, [4, 2])
            var entering = [Int: [Float]]()
            var messages = [Int: [Float]]()
            var blockCounts = [Int: Int]()
            let result = Inkling2LoopDriver.run(
                schedule: schedule, hidden: MLXArray(x0, [1, 1, 2]), capturing: [5, 23, 41],
                messageQuery: { _ in query }
            ) { step, x, blocks, message in
                entering[step.index] = x.asArray(Float.self)
                blockCounts[step.index] = blocks.count
                if let message {
                    XCTAssertEqual(message.shape, [1, 1, 4, 2])
                    messages[step.index] = message.asArray(Float.self)
                }
                if step.closesBlock { blocks.append(x) }
                return x + MLXArray(c(step.layer), [1, 1, 2])
            }

            func close(_ a: [Float]?, _ b: [Float], _ what: String) {
                guard let a else { return XCTFail("missing \(what)") }
                XCTAssertEqual(a.count, b.count, what)
                for (p, q) in zip(a, b) { XCTAssertEqual(p, q, accuracy: 1e-4, what) }
            }
            for start in [6, 18, 30] {
                close(entering[start], prelude, "pass entering visit \(start) starts at the prelude")
                XCTAssertEqual(blockCounts[start], 1, "blocks reset to the prelude's at visit \(start)")
            }
            XCTAssertEqual(Set(messages.keys), Set(18 ..< 42), "messages only after the first pass")

            // Bank: the previous pass's second core block (entering layer 12) and its final state.
            func unitRMS(_ v: [Float]) -> [Float] {
                let scale = 1 / (v.map { $0 * $0 }.reduce(0, +) / Float(v.count) + 1e-6).squareRoot()
                return v.map { $0 * scale }
            }
            let bank = [unitRMS(midPass), unitRMS(endPass)]
            var expected = [Float]()
            for i in 0 ..< 4 {
                let q: [Float] = [Float(i) / 2, -Float(i) / 2]
                let scores = bank.map { b in zip(q, b).map { $0 * $1 }.reduce(0, +) / Float(2).squareRoot() }
                let top = scores.max()!
                let weights = scores.map { Foundation.exp($0 - top) }
                let total = weights.reduce(0, +)
                expected += (0 ..< 2).map { d in
                    zip(weights, bank).map { $0 / total * $1[d] }.reduce(0, +)
                }
            }
            for index in [18, 29, 30, 41] { close(messages[index], expected, "messages at visit \(index)") }

            close(result.captures[5]?.asArray(Float.self), plus(x0, 0 ..< 5), "capture 5")
            close(result.captures[23]?.asArray(Float.self), plus(prelude, 6 ..< 11), "capture 23")
            close(result.captures[41]?.asArray(Float.self), plus(prelude, 6 ..< 17), "capture 41")
            XCTAssertEqual(Set(result.captures.keys), [5, 23, 41])
            // Final blocks: the embeddings, the third pass's two core blocks, and layer 18's input.
            XCTAssertEqual(result.blocks.count, 4)
            for (block, value) in zip(result.blocks, [x0, prelude, midPass, endPass]) {
                close(block.asArray(Float.self), value, "final blocks")
            }
            close(result.hidden.asArray(Float.self), plus(endPass, 18 ..< 24), "final state")
        }
    }

    // MARK: Checkpoint layouts

    /// Loads the tensor list into the module tree the way `loadWeights` does, without weights:
    /// sanitize, quantize the modules whose `.scales` the checkpoint carries, then compare the
    /// parameter keys and shapes with the checkpoint's.
    private func mapCheckpoint(_ layout: String) throws -> (Inkling2Model, [String: MLXArray]) {
        let config = try configuration(layout)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: resource("\(layout)-config.json"))
        let model = Inkling2Model(config, quantization: base.perLayerQuantization)
        let checkpoint = try placeholders(layout)
        let weights = try Inkling2WeightLayout.convert(checkpoint, configuration: config.text)
        try model.setExpertFormats(from: weights)
        if let perLayer = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                if weights["\(path).scales"] != nil {
                    return perLayer.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(
            Set(parameters.keys).subtracting(weights.keys).sorted(), [], "\(layout): parameters with no tensor")
        XCTAssertEqual(
            Set(weights.keys).subtracting(parameters.keys).sorted(), [], "\(layout): tensors with no parameter")
        for (key, tensor) in weights {
            guard let parameter = parameters[key] else { continue }
            XCTAssertEqual(parameter.shape, tensor.shape, "\(layout): shape of \(key)")
            let integers: Set<DType> = [.uint8, .uint32]
            if integers.contains(parameter.dtype) || integers.contains(tensor.dtype) {
                XCTAssertEqual(parameter.dtype, tensor.dtype, "\(layout): dtype of \(key)")
            }
        }
        return (model, weights)
    }

    func testOQCheckpointMapsOntoTheModule() throws {
        try Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(try Inkling2WeightLayout.kind(of: try placeholders("oq4e").keys), .oq)
            let (model, weights) = try mapCheckpoint("oq4e")
            // Vision tensors are dropped; the draft lives in mtp.safetensors, outside the index.
            XCTAssertFalse(weights.keys.contains { $0.hasPrefix("vision_tower.") })
            XCTAssertEqual(weights.count, 1098 - 422)
            let formats = model.expertFormats
            let affine4 = Inkling2ExpertFormat(mode: .affine, bits: 4, groupSize: 64)
            for layer in 0 ..< 24 {
                let expected: Inkling2ExpertFormat = layer < 20 ? .nvfp4 : affine4
                XCTAssertEqual(formats[layer].gateUp, expected, "w13 of layer \(layer)")
                XCTAssertEqual(formats[layer].down, expected, "w2 of layer \(layer)")
            }
            // The attention projections and the head quantize to affine 8-bit, as the map says.
            let inProj = try XCTUnwrap(model.model.layers[0].attention.inProj as? QuantizedLinear)
            XCTAssertEqual([inProj.bits, inProj.groupSize], [8, 64])
            XCTAssertTrue(model.lmHead is QuantizedLinear)
        }
    }

    func testSourceCheckpointConvertsOntoTheModule() throws {
        try Device.withDefaultDevice(.cpu) {
            let checkpoint = try placeholders("source")
            XCTAssertEqual(try Inkling2WeightLayout.kind(of: checkpoint.keys), .source)
            let (model, weights) = try mapCheckpoint("source")
            XCTAssertFalse(weights.keys.contains { $0.hasPrefix("model.visual.") || $0.hasPrefix("model.llm.") })
            let formats = model.expertFormats
            for layer in 0 ..< 24 {
                let expected: Inkling2ExpertFormat = layer < 20 ? .nvfp4 : .mxfp8
                XCTAssertEqual(formats[layer].gateUp, expected, "w13 of layer \(layer)")
                XCTAssertEqual(formats[layer].down, expected, "w2 of layer \(layer)")
            }
            XCTAssertFalse(model.lmHead is QuantizedLinear, "the source head is BF16")
        }
    }

    /// `hf_quant_config.json` as mlx-vlm's `quantization_from_metadata` reads it, and a declared
    /// format the tensors contradict.
    func testQuantizationMetadataDeclaresExpertFormats() throws {
        let sidecar = """
            {"quant_method": "inkling2", "model": {"parameter_groups": [
              {"weight_transform": "nvfp4_nearest", "tensors": [
                "model.llm.layers.0.mlp.experts.w13_weight", "model.llm.layers.0.mlp.experts.w2_weight"]},
              {"weight_transform": "mxfp8_nearest", "tensors": [
                "model.llm.layers.23.mlp.experts.w13_weight", "model.llm.layers.23.mlp.experts.w2_weight"]},
              {"weight_transform": "cast", "tensors": ["model.llm.unembed.weight"]}]}}
            """
        let formats = try XCTUnwrap(
            Inkling2QuantizationMetadata.expertFormats(from: Data(sidecar.utf8), layers: 24))
        XCTAssertEqual(formats.count, 2)
        XCTAssertEqual(formats[0], Inkling2QuantizationMetadata.LayerFormats(gateUp: .nvfp4, down: .nvfp4))
        XCTAssertEqual(formats[23], Inkling2QuantizationMetadata.LayerFormats(gateUp: .mxfp8, down: .mxfp8))
        XCTAssertNil(
            try Inkling2QuantizationMetadata.expertFormats(
                from: Data(#"{"quant_method": "modelopt"}"#.utf8), layers: 24))
        XCTAssertThrowsError(
            try Inkling2QuantizationMetadata.expertFormats(
                from: Data(
                    #"{"quant_method": "inkling2", "model": {"parameter_groups": [{"weight_transform": "int4", "tensors": []}]}}"#
                        .utf8), layers: 24))

        try Device.withDefaultDevice(.cpu) {
            // Layer 23 of the source holds MXFP8; a sidecar that says NVFP4 is refused.
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("inkling2-sidecar-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try Data(sidecar.replacingOccurrences(of: "mxfp8_nearest", with: "nvfp4_nearest").utf8)
                .write(to: directory.appendingPathComponent("hf_quant_config.json"))
            let config = try configuration("source")
            let model = Inkling2Model(config)
            try model.applyQuantizationMetadata(directory: directory)
            let weights = try Inkling2WeightLayout.convert(try placeholders("source"), configuration: config.text)
            XCTAssertThrowsError(try model.setExpertFormats(from: weights))
        }
    }

    /// oMLX wrote the oQ build from the source checkpoint: the converted source must have the oQ
    /// build's keys, and every tensor the oQ build left unquantized must have its shape.
    func testSourceConversionReproducesTheOQLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration("source").text
            let source = try Inkling2WeightLayout.convert(try placeholders("source"), configuration: config)
            let oq = try Inkling2WeightLayout.convert(try placeholders("oq4e"), configuration: config)
            func quantizedPrefix(_ key: String) -> String? {
                for suffix in [".weight", ".scales", ".biases", ".weight_scale_2"] where key.hasSuffix(suffix) {
                    let prefix = String(key.dropLast(suffix.count))
                    if oq["\(prefix).scales"] != nil { return prefix }
                }
                return nil
            }
            let oqPlain = oq.filter { quantizedPrefix($0.key) == nil }
            let sourcePlain = source.filter { quantizedPrefix($0.key) == nil }
            XCTAssertEqual(Set(oqPlain.keys), Set(sourcePlain.keys))
            for (key, value) in oqPlain {
                XCTAssertEqual(sourcePlain[key]?.shape, value.shape, "shape of \(key)")
            }
            // Quantized in oQ: the source holds the same output rows.
            let prefixes = Set(oq.keys.compactMap(quantizedPrefix))
            for prefix in prefixes {
                let packed = try XCTUnwrap(oq["\(prefix).weight"])
                let dense = try XCTUnwrap(source["\(prefix).weight"], prefix)
                XCTAssertEqual(
                    Array(dense.shape.dropLast()), Array(packed.shape.dropLast()), "rows of \(prefix)")
            }
            // Layers 0-19 keep the same NVFP4 bytes in both builds.
            for layer in 0 ..< 20 {
                for which in ["w13", "w2"] {
                    let prefix = "model.layers.\(layer).mlp.experts.\(which)"
                    for suffix in ["weight", "scales", "weight_scale_2"] {
                        XCTAssertEqual(
                            source["\(prefix).\(suffix)"]?.shape, oq["\(prefix).\(suffix)"]?.shape,
                            "\(prefix).\(suffix)")
                        XCTAssertEqual(
                            source["\(prefix).\(suffix)"]?.dtype, oq["\(prefix).\(suffix)"]?.dtype,
                            "\(prefix).\(suffix)")
                    }
                }
            }
        }
    }

    func testOtherLayoutsAreRejected() throws {
        let tensor = MLXArray.zeros([1], dtype: .float32)
        XCTAssertThrowsError(
            try Inkling2WeightLayout.kind(of: ["model.language_model.layers.0.mlp.gate.weight"]))
        XCTAssertThrowsError(try Inkling2WeightLayout.kind(of: ["model.llm.norm.weight", "model.layers.0.mlp_norm.weight"]))
        XCTAssertThrowsError(try Inkling2WeightLayout.kind(of: ["vision_tower.x"]))
        let config = try configuration("source").text
        XCTAssertThrowsError(
            try Inkling2WeightLayout.convert(["model.llm.something_new": tensor], configuration: config))
    }
}
