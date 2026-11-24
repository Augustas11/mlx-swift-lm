// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// Staged rounds over `CacheList(attention, ConvolutionTailCache)` entries (Inkling 2's layer
// visits): committing a prefix must leave exactly what writing only that prefix leaves.

private let width = 3

private func rowsKV(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
    let keys = MLXArray(positions.flatMap { [Float($0), Float($0)] }, [1, 1, positions.count, 2])
    return (keys, -keys)
}

private func rowsConv(_ positions: Range<Int>) -> MLXArray {
    MLXArray(positions.map { Float($0) }, [1, positions.count, 1])
}

/// One layer visit's write, the way Inkling 2 does it: the attention child appends, each tail
/// slot keeps the newest `width` rows of `concat(previous, input)` and reports the padded input.
private func write(_ entry: KVCache, _ positions: Range<Int>) {
    let list = entry as! CacheList
    let (k, v) = rowsKV(positions)
    _ = list[0].update(keys: k, values: v)
    let tails = list[1] as! ArraysCache
    for slot in 0 ..< 2 {
        let input = rowsConv(positions) * Float(slot + 1)
        let previous = tails[slot] ?? MLXArray.zeros([1, width, 1], dtype: .float32)
        (tails as? ConvolutionTailCache)?.record(slot: slot, previous: previous, input: input)
        let context = concatenated([previous, input], axis: 1)
        tails[slot] = context[0..., (context.dim(1) - width)..., 0...]
    }
}

private func makeStorage(attention: KVCache, prompt: Int) -> KVCacheStorage {
    let storage = KVCacheStorage(
        [CacheList(attention, ConvolutionTailCache(size: 2))], plan: .disabled)
    write(storage.cache[0], 0 ..< prompt)
    storage.commitProcessedTokens(prompt)
    return storage
}

private func expectSame(_ a: KVCacheStorage, _ b: KVCacheStorage, _ note: String) {
    #expect(a.processedTokenCount == b.processedTokenCount, "\(note): timeline")
    let (la, lb) = (a.cache[0] as! CacheList, b.cache[0] as! CacheList)
    #expect(la[0].offset == lb[0].offset, "\(note): attention offset")
    for (x, y) in zip(la[0].state, lb[0].state) {
        #expect(x.shape == y.shape && allClose(x, y).item(Bool.self), "\(note): attention K/V")
    }
    let (ta, tb) = (la[1] as! ArraysCache, lb[1] as! ArraysCache)
    for slot in 0 ..< 2 {
        let (x, y) = (ta[slot]!, tb[slot]!)
        #expect(x.shape == y.shape && allClose(x, y).item(Bool.self), "\(note): tail \(slot)")
    }
}

@Test(arguments: [0, 1, 3, 4])
func testTailRoundCommitsExactlyThePrefixOverAppendOnlyAttention(retaining: Int) throws {
    let live = makeStorage(attention: KVCacheSimple(), prompt: 5)
    let oracle = live.copy()

    let round = try #require(live.beginRound(maximumPositions: 4))
    write(round.caches[0], 5 ..< 9)
    live.commit(round, retaining: retaining)

    write(oracle.cache[0], 5 ..< (5 + retaining))
    oracle.commitProcessedTokens(retaining)
    expectSame(live, oracle, "retaining \(retaining)")
}

@Test func testTailRoundOverAWrappedRingRewindsTheLastCommit() throws {
    let live = makeStorage(attention: RotatingKVCache(maxSize: 4, keep: 0), prompt: 6)
    let oracle = live.copy()

    let round = try #require(live.beginRound(maximumPositions: 3))
    write(round.caches[0], 6 ..< 9)
    live.commit(round, retaining: 3)
    #expect(live.rewindLastRound(2) == 2)

    write(oracle.cache[0], 6 ..< 7)
    oracle.commitProcessedTokens(1)
    expectSame(live, oracle, "wrapped rewind")
}

@Test func testTailRoundLeavesTheLiveTailsAloneUntilCommit() throws {
    let live = makeStorage(attention: KVCacheSimple(), prompt: 4)
    let before = (live.cache[0] as! CacheList)[1] as! ArraysCache
    let tail = before[0]!

    let round = try #require(live.beginRound(maximumPositions: 2))
    write(round.caches[0], 4 ..< 6)
    #expect(before[0]! === tail, "the round wrote through to the live tails")
    live.rollback(round)
    #expect(allClose(before[0]!, tail).item(Bool.self))
}

@Test func testRoundRefusesACacheListWithoutConvolutionTails() {
    let plain = KVCacheStorage([CacheList(KVCacheSimple(), ArraysCache(size: 2))], plan: .disabled)
    #expect(plain.beginRound(maximumPositions: 2) == nil)
    let mamba = KVCacheStorage([CacheList(MambaCache(), KVCacheSimple())], plan: .disabled)
    #expect(mamba.beginRound(maximumPositions: 2) == nil)
}
