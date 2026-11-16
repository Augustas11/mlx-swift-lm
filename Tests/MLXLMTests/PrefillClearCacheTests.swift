// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

final class PrefillClearCacheTests: XCTestCase {

    // Buffers a chunk leaves behind, such as a KV cache the next chunk outgrows,
    // must not stay in the buffer cache for the rest of the prefill.
    func testEachChunkClearsBufferCache() throws {
        let seeded = 256 * 1024 * 1024
        var cached: [Int] = []
        let processed = try PrefillParameters(stepSize: 4).forEachChunk(total: 9) { range in
            cached.append(Memory.cacheMemory)
            do {
                let buffer = MLXArray.zeros([seeded], dtype: .uint8) + range.lowerBound + 1
                eval(buffer)
            }
            // A freed buffer reaches the cache only after the GPU releases it on another
            // thread, so wait for it before the driver clears the cache.
            let deadline = Date().addingTimeInterval(1)
            while Memory.cacheMemory < seeded, Date() < deadline {
                usleep(1000)
            }
            XCTAssertGreaterThanOrEqual(Memory.cacheMemory, seeded)
        }

        XCTAssertEqual(processed, 8)
        XCTAssertEqual(cached.count, 2)
        XCTAssertLessThan(cached[1], seeded / 4)
        XCTAssertLessThan(Memory.cacheMemory, seeded / 4)
    }

    // The switch for hosts without a Metal library: the loop still runs every chunk.
    func testChunkLoopRunsWithoutClearingWhenSwitchedOff() throws {
        var parameters = PrefillParameters(stepSize: 4)
        parameters.clearsBufferCache = false
        var chunks = 0
        let processed = try parameters.forEachChunk(total: 9) { _ in chunks += 1 }
        XCTAssertEqual(processed, 8)
        XCTAssertEqual(chunks, 2)
    }
}
