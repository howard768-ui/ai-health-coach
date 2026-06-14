import Foundation

/// True when the app is being driven by UI automation (Maestro or XCUITest).
/// Single switch for behavior that must change under automation; today that
/// is perpetual animations and network timeouts.
///
/// Why animations: XCUITest's accessibility snapshots wait for the app to
/// become quiescent before they capture. A `repeatForever` animation means
/// the app never goes idle, so every tapOn/assertVisible in a Maestro flow
/// stalls until the snapshot machinery gives up. Observed on loaded macos-15
/// CI runners: 30-90s per interaction, which turned coach chat flows into
/// 4-7 minute runs (diagnosed 2026-06-12). Animation call sites treat this
/// flag exactly like `accessibilityReduceMotion`.
///
/// Deliberately mirrors `MeldApp.isUITesting` (Maestro launch argument,
/// UserDefaults fallback, env fallback) and does NOT check
/// `XCTestConfigurationFilePath`: reacting to plain XCTest would change how
/// views render under the locally-run MeldSnapshotTests and invalidate the
/// committed snapshot baselines. The auth bypass in MeldApp keeps its own
/// copy of this check for the same isolation reason.
enum UITestMode {
    static let isActive: Bool = {
        #if DEBUG
        return ProcessInfo.processInfo.arguments.contains("-uitesting-skip-auth")
            || UserDefaults.standard.bool(forKey: "uitesting-skip-auth")
            || ProcessInfo.processInfo.environment["MELD_UI_TESTING"] == "1"
        #else
        return false
        #endif
    }()
}
