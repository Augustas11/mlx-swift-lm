import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Private backing files for lazy Prism vision weights. This is a per-model
/// resource, not a shared artifact cache; provider admission hashes this directory.
final class OwnedPrismArtifactSnapshot: Sendable {
    enum SnapshotError: Error {
        case externalTokenizer
        case unsafeEntry(String)
        case copyFailed(String)
        case modelTypeChanged
    }

    let directory: URL

    init(source: URL) throws {
        let manager = FileManager.default
        directory = manager.temporaryDirectory.appendingPathComponent(
            "mlx-prism-\(UUID().uuidString)", isDirectory: true)
        do {
            try manager.createDirectory(
                at: directory, withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
            try Self.validate(source)
            var enumerationError: Error?
            guard
                let entries = manager.enumerator(
                    at: source, includingPropertiesForKeys: [.isDirectoryKey], options: [],
                    errorHandler: { _, error in
                        enumerationError = error
                        return false
                    })
            else { throw SnapshotError.unsafeEntry(source.lastPathComponent) }
            let base = source.standardizedFileURL.path + "/"
            for case let entry as URL in entries {
                try Self.validate(entry)
                let path = entry.standardizedFileURL.path
                guard path.hasPrefix(base) else { throw SnapshotError.unsafeEntry(path) }
                let relative = String(path.dropFirst(base.count))
                let destination = directory.appendingPathComponent(relative)
                if try entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true {
                    try manager.createDirectory(
                        at: destination, withIntermediateDirectories: false,
                        attributes: [.posixPermissions: 0o700])
                } else {
                    #if canImport(Darwin)
                    let cloned =
                        clonefile(entry.path, destination.path, UInt32(CLONE_NOFOLLOW)) == 0
                    #else
                    let cloned = false
                    #endif
                    if !cloned { try manager.copyItem(at: entry, to: destination) }
                    try Self.validate(destination, allowDirectory: false)
                    try manager.setAttributes(
                        [.posixPermissions: 0o400], ofItemAtPath: destination.path)
                }
            }
            if let enumerationError { throw enumerationError }
        } catch {
            try? manager.removeItem(at: directory)
            if let error = error as? SnapshotError { throw error }
            throw SnapshotError.copyFailed(String(describing: error))
        }
    }

    private static func validate(_ url: URL, allowDirectory: Bool = true) throws {
        let values = try url.resourceValues(forKeys: [
            .isSymbolicLinkKey, .isRegularFileKey, .isDirectoryKey, .linkCountKey,
        ])
        guard values.isSymbolicLink != true,
            (allowDirectory && values.isDirectory == true)
                || (values.isRegularFile == true && values.linkCount == 1)
        else { throw SnapshotError.unsafeEntry(url.lastPathComponent) }
    }

    deinit {
        try? FileManager.default.removeItem(at: directory)
    }
}
