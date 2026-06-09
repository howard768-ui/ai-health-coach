import Foundation
import Testing
@testable import Meld

// MARK: - BackendDateTests
//
// Pins BackendDate.parse against the exact wire forms the backend emits.
// Python datetime.isoformat() on a naive UTC value produces six fractional
// digits when present and NO timezone designator, e.g.:
//   "2026-05-03T08:00:00.123456"   (chat / meal created_at)
//   "2026-05-03T08:00:00"          (zero microseconds)
// A bare ISO8601DateFormatter returns nil for both (needs a tz), so coach,
// meal, and dashboard timestamps used to silently render as "now". A few
// endpoints append "Z", so those are pinned too.

@Suite("BackendDate")
struct BackendDateTests {

    /// A reference instant parsed from an unambiguous "Z" string. The naive
    /// (no-tz) backend forms must resolve to this SAME instant; if the parser
    /// read them in device-local time, the equality checks fail by the runner's
    /// UTC offset.
    private static let referenceInstant = ISO8601DateFormatter()
        .date(from: "2026-05-03T08:00:00Z")!

    @Test("Naive timestamp without fractional seconds parses as UTC")
    func naiveNoFractionalIsUTC() throws {
        let parsed = try #require(BackendDate.parse("2026-05-03T08:00:00"))
        #expect(parsed == Self.referenceInstant)
    }

    @Test("Naive timestamp with six-digit microseconds parses as UTC")
    func naiveMicrosecondsIsUTC() throws {
        let parsed = try #require(BackendDate.parse("2026-05-03T08:00:00.123456"))
        let expected = Self.referenceInstant.addingTimeInterval(0.123456)
        #expect(abs(parsed.timeIntervalSince1970 - expected.timeIntervalSince1970) < 0.01)
    }

    @Test("Zulu timestamp without fractional seconds parses")
    func zuluNoFractional() throws {
        let parsed = try #require(BackendDate.parse("2026-05-03T08:00:00Z"))
        #expect(parsed == Self.referenceInstant)
    }

    @Test("Zulu timestamp with fractional seconds parses")
    func zuluWithFractional() throws {
        let parsed = try #require(BackendDate.parse("2026-05-03T08:00:00.123Z"))
        let expected = Self.referenceInstant.addingTimeInterval(0.123)
        #expect(abs(parsed.timeIntervalSince1970 - expected.timeIntervalSince1970) < 0.01)
    }

    @Test("Unparseable strings return nil so callers fall back to now")
    func garbageReturnsNil() {
        #expect(BackendDate.parse("not-a-date") == nil)
        #expect(BackendDate.parse("") == nil)
    }
}
