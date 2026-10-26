import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV4DSparkTests: XCTestCase {
    private func configuration() throws -> DeepseekV4Configuration {
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "deepseek_v4_tiny", withExtension: nil))
        let data = try Data(contentsOf: fixture.appendingPathComponent("config.json"))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let layers = try XCTUnwrap(object["num_hidden_layers"] as? Int)
        let ratios = try XCTUnwrap(object["compress_ratios"] as? [Int])
        object["compress_ratios"] = Array(ratios.prefix(layers)) + [0, 0, 0]
        object["dspark_target_layer_ids"] = [layers - 3, layers - 2, layers - 1]
        object["dspark_block_size"] = 5
        object["dspark_noise_token_id"] = 9
        object["dspark_markov_rank"] = 4
        return try JSONDecoder().decode(DeepseekV4Configuration.self,
            from: JSONSerialization.data(withJSONObject: object))
    }

    private func assertStateEqual(_ first: DeepseekV4PoolCache, _ second: DeepseekV4PoolCache,
        file: StaticString = #filePath, line: UInt = #line)
    {
        XCTAssertEqual(first.metaState, second.metaState, file: file, line: line)
        XCTAssertEqual(first.state.count, second.state.count, file: file, line: line)
        for (a, b) in zip(first.state, second.state) {
            XCTAssertEqual(a.shape, b.shape, file: file, line: line)
            XCTAssertTrue((a .== b).all().item(Bool.self), file: file, line: line)
        }
    }

    func testPoolRetainsEveryExactPrefixAcrossBothCompressionBoundaries() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            for ratio in [4, 128] {
                let compressor = DeepseekV4Compressor(config, ratio: ratio, headDim: config.headDim)
                for prefix in [ratio - 1, ratio, ratio * 2 - 1] {
                    for keep in 0 ... 6 {
                        let live = DeepseekV4PoolCache(ratio: ratio)
                        for position in 0 ..< prefix {
                            let x = MLXArray.full([1, 1, config.hiddenSize], values: MLXArray(Float(position % 17)))
                            _ = compressor(x, cache: live, offset: position)
                        }
                        let ordinary = live.copy() as! DeepseekV4PoolCache
                        XCTAssertTrue(live.beginSpeculation(maximumPositions: 6))
                        for position in 0 ..< 6 {
                            let x = MLXArray.full([1, 1, config.hiddenSize], values: MLXArray(Float((prefix + position) % 17)))
                            _ = compressor(x, cache: live, offset: prefix + position)
                            if position < keep { _ = compressor(x, cache: ordinary, offset: prefix + position) }
                        }
                        XCTAssertTrue(live.commitSpeculation(retaining: keep))
                        assertStateEqual(live, ordinary)
                        let next = MLXArray.ones([1, 1, config.hiddenSize])
                        XCTAssertTrue((compressor(next, cache: live, offset: prefix + keep)
                            .== compressor(next, cache: ordinary, offset: prefix + keep)).all().item(Bool.self))
                    }
                }
            }
        }
    }

    func testPoolAbortAndInvalidCommitDoNotLeakRejectedOverlap() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            let compressor = DeepseekV4Compressor(config, ratio: 4, headDim: config.headDim)
            let cache = DeepseekV4PoolCache(ratio: 4)
            let x = MLXArray.ones([1, 1, config.hiddenSize])
            for position in 0 ..< 7 { _ = compressor(x, cache: cache, offset: position) }
            let before = cache.copy() as! DeepseekV4PoolCache
            XCTAssertTrue(cache.beginSpeculation(maximumPositions: 2))
            XCTAssertFalse(cache.beginSpeculation(maximumPositions: 2))
            _ = compressor(x, cache: cache, offset: 7)
            XCTAssertFalse(cache.commitSpeculation(retaining: 2))
            XCTAssertTrue(cache.speculationIsOpen)
            XCTAssertTrue(cache.abortSpeculation())
            assertStateEqual(cache, before)
            XCTAssertFalse(cache.abortSpeculation())
        }
    }

    func testEveryEmbeddedStageTensorIsConsumedByStrictWeightUpdate() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            let model = DeepseekV4Model(config)
            var checkpoint = [String: MLXArray]()
            for (path, tensor) in model.parameters().flattened() {
                var name = path
                for branch in ["attn", "ffn"] {
                    for parameter in ["fn", "base", "scale"] {
                        name = name.replacingOccurrences(of: ".\(branch)_hc.\(parameter)",
                            with: ".hc_\(branch)_\(parameter)")
                    }
                }
                for parameter in ["fn", "base", "scale"] {
                    name = name.replacingOccurrences(of: "hc_head.\(parameter)", with: "hc_head_\(parameter)")
                }
                let parts = name.split(separator: ".").map(String.init)
                if parts.count == 6, parts[2] == "ffn", parts[3] == "switch_mlp" {
                    let projections = ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"]
                    let projection = try XCTUnwrap(projections[parts[4]])
                    for expert in 0 ..< config.nRoutedExperts {
                        checkpoint["\(parts[0]).\(parts[1]).ffn.experts.\(expert).\(projection).\(parts[5])"] = tensor[expert]
                    }
                } else {
                    checkpoint[name] = tensor
                }
            }
            let sanitized = model.sanitize(weights: checkpoint)
            XCTAssertEqual(Set(sanitized.keys), Set(model.parameters().flattened().map(\.0)))
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            XCTAssertNotNil(sanitized["mtp.2.confidence_head.proj.weight"])
            XCTAssertNotNil(sanitized["mtp.2.markov_head.markov_w2.weight"])
        }
    }

    func testConfidenceScheduleUsesCumulativeSurvivalAndStablePrefixTies() {
        XCTAssertEqual(MTPConfidenceSchedule.proposalCounts(
            conditionalRates: [[0.9, 0.2, 1], [0.8, 0.8, 0.8]], verificationBudget: 5), [1, 2])
        XCTAssertEqual(MTPConfidenceSchedule.proposalCounts(
            conditionalRates: [[1, 1], [1, 1]], verificationBudget: 4), [2, 0])
        XCTAssertEqual(MTPConfidenceSchedule.proposalCounts(
            conditionalRates: [[.nan, 1], [0.5, 0.5]], verificationBudget: 4), [0, 2])
        XCTAssertEqual(MTPConfidenceSchedule.proposalCounts(
            conditionalRates: [[0.9], [0.8]], verificationBudget: 2), [0, 0])
        XCTAssertEqual(MTPConfidenceSchedule.proposalCounts(
            conditionalRates: [], verificationBudget: 0), [])
    }

    func testDraftDoesNotAppendTentativeContextAtAnyDepth() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            let target = DeepseekV4Model(config)
            let drafter = DeepseekV4DSparkDraftModel(config)
            var state = drafter.makeState(parameters: nil)
            let hidden = MLXArray.ones([1, 11, config.hiddenSize * config.dsparkTargetLayerIDs.count])
            drafter.prepareDrafterState(target: target, promptTokens: MLXArray.zeros([1, 11], dtype: .int32),
                targetHidden: hidden, firstBonus: MLXArray(Int32(3)), positionDeltas: nil,
                state: &state, sampler: ArgMaxSampler())
            XCTAssertEqual(state.nextPosition, 11)
            let before = state.cache.map { $0.copy() }
            for depth in 1 ... 5 {
                let tokens = drafter.draftBlock(target: target, lastToken: MLXArray([Int32(3)]),
                    lastHidden: hidden, sharedKV: [:], positionDeltas: nil, queryOffset: 11,
                    blockSize: depth + 1, state: &state, sampler: ArgMaxSampler())
                eval(tokens, state.proposalConfidence!)
                XCTAssertEqual(tokens.shape, [1, depth])
                XCTAssertEqual(state.proposalConfidence?.shape, [1, depth])
                XCTAssertEqual(state.nextPosition, 11)
                XCTAssertEqual(state.proposalAppended, 0)
                for (a, b) in zip(before, state.cache) {
                    XCTAssertEqual(a.metaState, b.metaState)
                    for (x, y) in zip(a.state, b.state) {
                        XCTAssertTrue((x .== y).all().item(Bool.self))
                    }
                }
            }
        }
    }
    func testTwoHundredMixedPrefixTransactionsMatchSerialPoolState() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            for ratio in [4, 128] {
                let compressor = DeepseekV4Compressor(config, ratio: ratio, headDim: config.headDim)
                let cache = DeepseekV4PoolCache(ratio: ratio)
                let ordinary = DeepseekV4PoolCache(ratio: ratio)
                var position = 0
                var random: UInt64 = 0x44535041524b
                for _ in 0 ..< 200 {
                    random = random &* 6364136223846793005 &+ 1
                    let depth = Int((random >> 32) % 6) + 1
                    random = random &* 6364136223846793005 &+ 1
                    let keep = Int((random >> 32) % UInt64(depth + 1))
                    XCTAssertTrue(cache.beginSpeculation(maximumPositions: depth))
                    for column in 0 ..< depth {
                        let values = MLXArray.arange(config.hiddenSize, dtype: .float32)
                            .reshaped(1, 1, config.hiddenSize) + Float((position + column) % 31)
                        _ = compressor(values, cache: cache, offset: position + column)
                        if column < keep {
                            _ = compressor(values, cache: ordinary, offset: position + column)
                        }
                    }
                    XCTAssertTrue(cache.commitSpeculation(retaining: keep))
                    position += keep
                    assertStateEqual(cache, ordinary)
                }
            }
        }
    }

    func testPackedDraftMatchesIndependentRowsAfterContextRingWrap() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            let target = DeepseekV4Model(config)
            let drafter = DeepseekV4DSparkDraftModel(config)
            var states = [MTPDrafterState]()
            for length in [config.slidingWindow + 3, config.slidingWindow + 5] {
                var state = drafter.makeState(parameters: nil)
                let hidden = MLXArray.ones([1, length, config.hiddenSize * config.dsparkTargetLayerIDs.count])
                drafter.prepareDrafterState(target: target,
                    promptTokens: MLXArray.zeros([1, length], dtype: .int32), targetHidden: hidden,
                    firstBonus: MLXArray(Int32(3)), positionDeltas: nil,
                    state: &state, sampler: ArgMaxSampler())
                states.append(state)
            }
            let anchors = [3, 7]
            for depth in [1, 2, 3, 5] {
                let packed = try drafter.proposePacked(target: target, states: states,
                    anchors: anchors, depth: depth)
                for index in states.indices {
                    let single = try drafter.proposePacked(target: target, states: [states[index]],
                        anchors: [anchors[index]], depth: depth)
                    XCTAssertTrue((packed.proposals[index ..< (index + 1)]
                        .== single.proposals).all().item(Bool.self))
                    XCTAssertEqual(packed.states[index].nextPosition, states[index].nextPosition)
                    XCTAssertEqual(packed.states[index].cache.map(\.offset), states[index].cache.map(\.offset))
                }
            }
        }
    }

}
