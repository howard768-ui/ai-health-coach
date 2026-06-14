import UserNotifications
import UIKit

/// Manages push notification permissions and device token registration.
@MainActor
final class NotificationService {

    static let shared = NotificationService()
    private init() {}

    private let lastTokenKey = "meld_last_push_token"

    /// Request notification permission and register for remote notifications.
    /// Returns true if permission was granted.
    func requestPermission() async -> Bool {
        do {
            let granted = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .badge, .sound])

            if granted {
                UIApplication.shared.registerForRemoteNotifications()
                Analytics.signal("Notifications.permissionGranted")
            } else {
                Analytics.signal("Notifications.permissionDenied")
            }

            return granted
        } catch {
            Log.notifications.error("Permission request failed: \(error.localizedDescription)")
            return false
        }
    }

    /// Check current notification authorization status.
    ///
    /// Uses the completion-handler form of `getNotificationSettings` and
    /// extracts the Sendable `authorizationStatus` inside the completion so
    /// the non-Sendable `UNNotificationSettings` never crosses an actor
    /// boundary. The async variant `notificationSettings()` returns
    /// `UNNotificationSettings` across a nonisolated context, which Swift 6
    /// rejects.
    ///
    /// `nonisolated` is load-bearing (issue #199). On the `@MainActor` class
    /// the completion closure inherited MainActor isolation, but UN invokes
    /// it on its own background XPC queue
    /// (`UNUserNotificationServiceConnection.call-out`). The runtime's
    /// `isCurrentExecutor` check then tripped `dispatch_assert_queue` and
    /// crashed with EXC_BREAKPOINT at launch (~10-30% of cold launches,
    /// confirmed by a captured DiagnosticReports trace). Marking the method
    /// `nonisolated` keeps the closure off MainActor, so no executor
    /// assertion is inserted. The body touches no isolated state, and
    /// `authorizationStatus` is Sendable, so this is safe.
    nonisolated func getPermissionStatus() async -> UNAuthorizationStatus {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().getNotificationSettings { settings in
                continuation.resume(returning: settings.authorizationStatus)
            }
        }
    }

    /// Send device token to the backend for APNs delivery.
    /// Skips the network call if the token hasn't changed since the last successful registration.
    func registerToken(_ token: String) async {
        let stored = UserDefaults.standard.string(forKey: lastTokenKey)
        guard token != stored else {
            Log.notifications.debug("Push token unchanged — skipping registration")
            return
        }
        do {
            try await APIClient.shared.registerDeviceToken(token)
            UserDefaults.standard.set(token, forKey: lastTokenKey)
            Log.notifications.info("Token registered with backend")
        } catch {
            Log.notifications.error("Token registration failed: \(error.localizedDescription)")
        }
    }
}
