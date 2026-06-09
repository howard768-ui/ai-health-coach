import Foundation

/// Parses timestamp strings emitted by the Meld backend.
///
/// The backend serializes datetimes with Python's `datetime.isoformat()` on
/// NAIVE UTC values (see backend `app/core/time.py:utcnow_naive`), so the wire
/// form is e.g. `2026-05-03T08:00:00.123456` or `2026-05-03T08:00:00`: SIX
/// fractional digits when present and NO timezone designator.
///
/// A bare `ISO8601DateFormatter` returns nil for all of these (it requires a
/// timezone and is documented around 3-digit fractional seconds). That made
/// coach message timestamps, meal timestamps, and the dashboard coach-insight
/// silently fall back to `Date()` (the current time) on every parse. A few
/// endpoints (data sources, ops) DO append `Z` / `+00:00`, so tz-bearing forms
/// are accepted too.
///
/// The naive forms are interpreted as UTC (POSIX locale, UTC time zone) because
/// that is what the backend stores; interpreting them in device-local time
/// would shift every timestamp by the device's UTC offset.
enum BackendDate {
    /// Naive (no-timezone) backend forms, ordered most-specific first.
    private static let naiveFormats = [
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSS",
        "yyyy-MM-dd'T'HH:mm:ss",
    ]

    /// Parse a backend timestamp string, or nil if it matches no known form.
    /// Callers keep their own `?? Date()` fallback for display.
    ///
    /// Formatters are created locally rather than cached in `static` storage:
    /// `DateFormatter`/`ISO8601DateFormatter` are non-`Sendable`, so a cached
    /// `static let` trips Swift 6 strict-concurrency checks, and this repo
    /// avoids `nonisolated(unsafe)`. Allocation cost is negligible at call
    /// sites (a handful of items per response).
    static func parse(_ string: String) -> Date? {
        for format in naiveFormats {
            let formatter = DateFormatter()
            formatter.dateFormat = format
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(identifier: "UTC")
            if let date = formatter.date(from: string) {
                return date
            }
        }

        // tz-bearing forms ("...Z" / "+00:00") from a few endpoints.
        let isoWithFractional = ISO8601DateFormatter()
        isoWithFractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = isoWithFractional.date(from: string) {
            return date
        }

        return ISO8601DateFormatter().date(from: string)
    }
}
