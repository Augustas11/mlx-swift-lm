import Foundation
import XCTest

@testable import MLXVLM

final class PrismArtifactSnapshotTests: XCTestCase {
    func testSnapshotPreservesCompleteTreeAndIsIndependentOfOriginalWrites() throws {
        let manager = FileManager.default
        let source = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try manager.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: source) }
        let nested = source.appendingPathComponent("assets")
        try manager.createDirectory(at: nested, withIntermediateDirectories: false)
        let files = ["config.json", "model.safetensors", "assets/tokenizer.json", ".metadata"]
        for file in files { try Data(file.utf8).write(to: source.appendingPathComponent(file)) }
        var snapshot: OwnedPrismArtifactSnapshot? = try OwnedPrismArtifactSnapshot(source: source)
        let directory = try XCTUnwrap(snapshot?.directory)
        let permissions = try manager.attributesOfItem(atPath: directory.path)[.posixPermissions]
        XCTAssertEqual((permissions as? NSNumber)?.intValue, 0o700)
        // In-place changes and atomic replacement must both leave the snapshot intact.
        try Data("overwritten".utf8).write(to: source.appendingPathComponent(files[0]))
        try Data("replaced".utf8).write(
            to: source.appendingPathComponent(files[1]), options: .atomic)
        for file in files {
            XCTAssertEqual(
                try Data(contentsOf: directory.appendingPathComponent(file)), Data(file.utf8))
        }
        snapshot = nil
        XCTAssertFalse(manager.fileExists(atPath: directory.path))
    }

    func testSnapshotRejectsLinksRatherThanCopyingEscapedOrSharedBacking() throws {
        let manager = FileManager.default
        let source = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try manager.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: source) }
        let original = source.appendingPathComponent("model.safetensors")
        try Data([1, 2, 3]).write(to: original)
        let link = source.appendingPathComponent("linked")
        try manager.createSymbolicLink(at: link, withDestinationURL: original)
        XCTAssertThrowsError(try OwnedPrismArtifactSnapshot(source: source))
        try manager.removeItem(at: link)
        try manager.linkItem(at: original, to: link)
        XCTAssertThrowsError(try OwnedPrismArtifactSnapshot(source: source))
    }
}
