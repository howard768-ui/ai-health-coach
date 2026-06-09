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
    /// Fixed-format parsers for the naive (no-timezone) backend forms, ordered
    /// most-specific first. UTC + en_US_POSIX so the wall-clock string is read
    /// as UTC and parsing is independent of device locale/calendar settings.
    private static let naiveFormatters: [DateFormatter] = {
        ["yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
         "yyyy-MM-dd'T'HH:mm:ss.SSS",
         "yyyy-MM-dd'T'HH:mm:ss"].map { format in
            let formatter = DateFormatter()
            formatter.dateFormat = format
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(identifier: "UTC")
            return formatter
        }
    }()

    /// ISO8601 with fractional seconds, for tz-bearing strings like
    /// `2026-05-03T08:00:00.123Z`.
    private static let isoWithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    /// ISO8601 without fractional seconds, for `2026-05-03T08:00:00Z`.
    private static let isoPlain = ISO8601DateFormatter()

    /// Parse a backend timestamp string, or nil if it matches no known form.
    /// Callers keep their own `?? Date()` fallback for display.
    static func parse(_ string: String) -> Date? {
        for formatter in naiveFormatters {
            if let date = formatter.date(from: string) {
                return date
            }
        }
        if let date = isoWithFractional.date(from: string) {
            return date
        }
        return isoPlain.date(from: string)
    }
}
