import BackgroundTasks
import UIKit
import UserNotifications

/// Bridges SwiftUI to UIKit's push notification lifecycle.
/// Handles device token registration, foreground notification display,
/// and notification tap actions.
final class AppDelegate: NSObject, UIApplicationDelegate, @preconcurrency UNUserNotificationCenterDelegate {

    /// True when the app is running under any testing harness (XCTest,
    /// XCUITest, or Maestro). Skips side-effectful auto-runs (push
    /// registration, HealthKit sync, BGTaskScheduler.register) that
    /// signal-trap when the app runs unsigned.
    ///
    /// XCTest + XCUITest set `XCTestConfigurationFilePath`. Maestro does
    /// NOT, it just passes `-uitesting-skip-auth` as a launch argument,
    /// so the XCTest-only guard used to let it through. On iOS 26 sim
    /// that meant `HealthKitService.queryTodaySteps()` would crash the
    /// app back to SpringBoard before `MainTabView` ever rendered,
    /// breaking 15 of 16 Maestro flows with
    /// `Assertion is false: id: tab-home is visible` (PR #56 diagnostic).
    private static var isRunningUnderTestHarness: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
            || ProcessInfo.processInfo.arguments.contains("-uitesting-skip-auth")
            || ProcessInfo.processInfo.environment["MELD_UI_TESTING"] == "1"
    }

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        registerNotificationCategories()
        registerBackgroundTasks()

        if Self.isRunningUnderTestHarness {
            // Disable UIKit-level animations under test harnesses. SwiftUI
            // repeatForever call sites are gated individually via UITestMode;
            // this covers UIKit-backed transitions (alerts, sheet presents)
            // so XCUITest's quiescence wait settles quickly between steps.
            UIView.setAnimationsEnabled(false)
        }

        if !Self.isRunningUnderTestHarness {
            // Always re-register for remote notifications on launch
            // (token may have changed, or permission was granted in a previous install)
            Task {
                let status = await NotificationService.shared.getPermissionStatus()
                if status == .authorized {
                    await MainActor.run {
                        UIApplication.shared.registerForRemoteNotifications()
                    }
                    Log.notifications.info("Re-registering for remote notifications on launch")
                }
            }

            // Auto-sync HealthKit data on every launch
            Task {
                if HealthKitService.shared.isAvailable {
                    let steps = await HealthKitService.shared.queryTodaySteps()
                    if steps != nil {
                        // HealthKit is authorized — sync data to backend
                        HealthKitService.shared.isAuthorized = true
                        await HealthKitService.shared.syncToBackend()
                        Log.healthKit.info("Auto-synced on launch")
                    }
                }
            }
        }

        return true
    }

    // MARK: - Token Registration

    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        let token = deviceToken.map { String(format: "%02x", $0) }.joined()
        Task {
            await NotificationService.shared.registerToken(token)
        }
    }

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        Log.notifications.error("Registration failed: \(error.localizedDescription)")
    }

    // MARK: - Foreground Notification Display
    // Uses completion handler variant — the async variant deadlocks in Swift 6.

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound, .badge])
    }

    // MARK: - Notification Tap Handling
    // Uses completion handler variant — the async variant deadlocks in Swift 6.

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let userInfo = response.notification.request.content.userInfo

        // Determine which tab to open — action buttons take priority.
        // PR-H: route through `NotificationActionID` + `Tab` rawValues so
        // the registration site (registerNotificationCategories) and this
        // matching site stay in lockstep at compile time.
        let tabName: String
        switch NotificationActionID(rawValue: response.actionIdentifier) {
        case .review:
            tabName = Tab.home.rawValue
        case .askCoach, .tellMeMore:
            tabName = Tab.coach.rawValue
        case .windDown, .logNow:
            tabName = Tab.home.rawValue
        case .seeReview:
            tabName = Tab.trends.rawValue
        case .none:
            if let deepLink = userInfo["deep_link"] as? String,
               let url = URL(string: deepLink),
               let host = url.host {
                // The notification deep_link can carry "dashboard" (legacy
                // alias for home) or any Tab.rawValue. Anything else falls
                // through to home as the safe default.
                tabName = (host == "dashboard") ? Tab.home.rawValue
                    : (Tab(rawValue: host)?.rawValue ?? Tab.home.rawValue)
            } else {
                tabName = Tab.home.rawValue
            }
        }

        AppDelegate.pendingTab.set(tabName)
        Log.notifications.debug("Set pendingTab to: \(tabName)")
        completionHandler()
    }

    /// Static storage for pending navigation — written from the
    /// UNUserNotificationCenter delegate callback (runs on UN's internal
    /// queue, NOT MainActor) and read from `MainTabView.checkPendingTab()`
    /// (runs on MainActor via SwiftUI lifecycle).
    ///
    /// Pre-followup #1 this was `nonisolated(unsafe) static var pendingTab:
    /// String?` — racy under concurrent set/get, and Swift 6 only kept it
    /// because the `unsafe` opt-out silenced the compiler. Now wrapped in
    /// `PendingTabHolder` which provides lock-protected `set` and atomic
    /// `consume` (read + nil-out in one critical section). The MainTabView
    /// reader uses `consume()` which both reads and clears, eliminating the
    /// previous "read then nil-out" two-call window.
    static let pendingTab = PendingTabHolder()

    // MARK: - Notification Categories (Action Buttons)

    private func registerNotificationCategories() {
        // PR-H: identifiers are NotificationActionID rawValues. The matching
        // site in userNotificationCenter(_:didReceive:) consumes the same enum.
        let reviewAction = UNNotificationAction(
            identifier: NotificationActionID.review.rawValue,
            title: "Review",
            options: .foreground
        )
        let askCoachAction = UNNotificationAction(
            identifier: NotificationActionID.askCoach.rawValue,
            title: "Ask Coach",
            options: .foreground
        )
        let tellMeMoreAction = UNNotificationAction(
            identifier: NotificationActionID.tellMeMore.rawValue,
            title: "Tell me more",
            options: .foreground
        )
        let windDownAction = UNNotificationAction(
            identifier: NotificationActionID.windDown.rawValue,
            title: "Wind down",
            options: .foreground
        )

        let logNowAction = UNNotificationAction(
            identifier: NotificationActionID.logNow.rawValue,
            title: "Log now",
            options: .foreground
        )
        let seeReviewAction = UNNotificationAction(
            identifier: NotificationActionID.seeReview.rawValue,
            title: "See full review",
            options: .foreground
        )

        let morningBrief = UNNotificationCategory(
            identifier: "MORNING_BRIEF",
            actions: [reviewAction, askCoachAction],
            intentIdentifiers: []
        )
        let coachingNudge = UNNotificationCategory(
            identifier: "COACHING_NUDGE",
            actions: [tellMeMoreAction],
            intentIdentifiers: []
        )
        let bedtimeCoaching = UNNotificationCategory(
            identifier: "BEDTIME_COACHING",
            actions: [windDownAction],
            intentIdentifiers: []
        )
        let streakSaver = UNNotificationCategory(
            identifier: "STREAK_SAVER",
            actions: [logNowAction],
            intentIdentifiers: []
        )
        let weeklyReview = UNNotificationCategory(
            identifier: "WEEKLY_REVIEW",
            actions: [seeReviewAction],
            intentIdentifiers: []
        )
        let healthAlert = UNNotificationCategory(
            identifier: "HEALTH_ALERT",
            actions: [reviewAction],
            intentIdentifiers: []
        )

        UNUserNotificationCenter.current().setNotificationCategories([
            morningBrief, coachingNudge, bedtimeCoaching,
            streakSaver, weeklyReview, healthAlert,
        ])
    }

    // MARK: - Background Tasks (Phase 7B)

    private static let modelUpdateTaskID = "com.heymeld.ranker.modelUpdate"

    private func registerBackgroundTasks() {
        guard !Self.isRunningUnderTestHarness else { return }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.modelUpdateTaskID,
            using: nil
        ) { task in
            // PR-H: replace `as!` force-cast with guard. The system should
            // always hand us a `BGProcessingTask` for this identifier, but a
            // future BGTaskScheduler API change could surprise us — fail
            // soft (mark complete + log) rather than crashing the app.
            guard let processingTask = task as? BGProcessingTask else {
                Log.notifications.error(
                    "BGTask handler received unexpected task type: \(type(of: task))"
                )
                task.setTaskCompleted(success: false)
                return
            }
            self.handleModelUpdateTask(processingTask)
        }

        scheduleModelUpdateTask()
    }

    private func scheduleModelUpdateTask() {
        let request = BGProcessingTaskRequest(
            identifier: Self.modelUpdateTaskID
        )
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 24 * 3600)

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            // Non-fatal: background model updates are best-effort.
        }
    }

    private func handleModelUpdateTask(_ task: BGProcessingTask) {
        // Re-schedule for next check before starting work.
        scheduleModelUpdateTask()

        let updateTask = Task {
            let updated = await RankerModelManager.shared.updateIfNeeded()
            if updated {
                await SignalRanker.shared.loadCachedModel()
            }
            task.setTaskCompleted(success: true)
        }

        task.expirationHandler = {
            updateTask.cancel()
            task.setTaskCompleted(success: false)
        }
    }
}
