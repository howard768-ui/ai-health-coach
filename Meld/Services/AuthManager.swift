import Foundation

// MARK: - AuthManager
//
// Orchestrates the session lifecycle — Sign in with Apple exchange, token
// refresh on 401, logout, and account deletion. Isolated as an actor so
// concurrent callers serialize on refresh (single-flight pattern).
//
// Design (per research in Audits/Full Codebase Audit 2026-04-10.md):
// - Single-flight refresh: if a refresh is in progress, every other 401
//   handler awaits the same Task (Donny Wals' pattern).
// - On refresh failure, wipe Keychain and broadcast `meldSessionExpired`
//   so the UI can kick to Welcome.
// - Never preemptively refresh — only refresh on 401 from the API.

@MainActor
final class AuthSessionState: ObservableObject {
    static let shared = AuthSessionState()

    @Published var isSignedIn: Bool = false
    @Published var userDisplayName: String? = nil
    @Published var userEmail: String? = nil

    func setSignedIn(_ signedIn: Bool, name: String? = nil, email: String? = nil) {
        self.isSignedIn = signedIn
        if let name { self.userDisplayName = name }
        if let email { self.userEmail = email }
    }
}

actor AuthManager {
    static let shared = AuthManager()

    private var refreshTask: Task<TokenPair, Error>?

    struct TokenPair: Sendable {
        let accessToken: String
        let refreshToken: String
        let expiresIn: Int  // seconds from issuance
        let user: UserInfo
    }

    struct UserInfo: Sendable, Codable {
        let id: String
        let name: String?
        let email: String?
        let is_private_email: Bool
    }

    /// Return a valid access token, refreshing if needed. Callers should
    /// retry the originating request once after calling this.
    func validAccessToken() async throws -> String {
        // If a refresh is already in flight, await it
        if let task = refreshTask {
            let pair = try await task.value
            return pair.accessToken
        }
        // Return the stored token as-is; we refresh only on 401
        return try await KeychainStore.shared.readAccessToken()
    }

    /// Force a refresh. Cached in-flight Task ensures concurrent callers
    /// share a single network round-trip.
    ///
    /// Followup #1: previously the task was cleared via
    /// ``defer { Task { await self.clearRefreshTask() } }`` which scheduled
    /// the clear ASYNCHRONOUSLY on the actor. That left a window where:
    ///   1. Refresh task A completes (success or fail).
    ///   2. Caller A returns its result.
    ///   3. The deferred clear-task is queued but not yet run.
    ///   4. Caller B enters refresh(), sees the still-set `refreshTask`,
    ///      and awaits its value — getting A's already-resolved value.
    /// On A's success this is harmless dedup; on A's failure B gets A's
    /// error instead of a fresh attempt. Now we clear synchronously inside
    /// the actor body before returning, so the cache reflects exactly
    /// "is there a refresh in flight right now?".
    @discardableResult
    func refresh() async throws -> TokenPair {
        if let task = refreshTask {
            return try await task.value
        }

        let task = Task<TokenPair, Error> {
            let stored = try await KeychainStore.shared.readRefreshToken()
            let pair = try await APIClient.shared.refreshSession(refreshToken: stored)
            try await self.persistTokens(pair)
            return pair
        }
        self.refreshTask = task
        do {
            let pair = try await task.value
            self.refreshTask = nil
            return pair
        } catch {
            self.refreshTask = nil
            throw error
        }
    }

    // MARK: - Sign in with Apple

    /// Called by the Sign in with Apple completion handler. Exchanges Apple's
    /// identity token for our own access + refresh tokens and persists them.
    func signInWithApple(
        identityToken: String,
        rawNonce: String,
        fullName: String?,
        email: String?,
        deviceId: String?
    ) async throws {
        let pair = try await APIClient.shared.signInWithApple(
            identityToken: identityToken,
            rawNonce: rawNonce,
            fullName: fullName,
            email: email,
            deviceId: deviceId
        )
        try await persistTokens(pair)
        let displayName = pair.user.name
        let userEmail = pair.user.email
        await MainActor.run {
            AuthSessionState.shared.setSignedIn(true, name: displayName, email: userEmail)
        }
    }

    // MARK: - Logout

    func logout() async {
        // Best-effort backend call
        do {
            let refresh = try await KeychainStore.shared.readRefreshToken()
            try? await APIClient.shared.logoutSession(refreshToken: refresh)
        } catch {
            // No refresh token stored — nothing to tell the backend about
        }
        try? await KeychainStore.shared.wipe()
        await MainActor.run {
            AuthSessionState.shared.setSignedIn(false)
        }
    }

    // MARK: - Account Deletion

    func deleteAccount() async throws {
        try await APIClient.shared.deleteAccount()
        try? await KeychainStore.shared.wipe()
        await MainActor.run {
            AuthSessionState.shared.setSignedIn(false)
        }
    }

    // MARK: - Session bootstrapping

    /// Called at app launch. If a token is stored, mark the session active
    /// and trust it until the first 401 forces a refresh.
    func bootstrapSession() async {
        let hasSession = await KeychainStore.shared.hasStoredSession()
        await MainActor.run {
            AuthSessionState.shared.setSignedIn(hasSession)
        }
    }

    /// Called when the 401-recovery path has already been exhausted: either
    /// `refresh()` just threw, or a request 401'd again WITH a freshly
    /// refreshed token. Wipes the session and signals the UI to return to
    /// login. Deliberately does NOT attempt another refresh: the old version
    /// re-called `refresh()` here, firing a second, pointless POST
    /// /auth/refresh after the first had already failed (refreshTask is
    /// cleared synchronously on failure, so it was a fresh network call,
    /// not a dedup). Audit P3/E7 (round-2 low, APIClient 113-128 /
    /// AuthManager 165-176).
    func handleUnauthorized() async {
        try? await KeychainStore.shared.wipe()
        await MainActor.run {
            AuthSessionState.shared.setSignedIn(false)
        }
    }

    // MARK: - Private

    private func persistTokens(_ pair: TokenPair) async throws {
        try await KeychainStore.shared.saveAccessToken(pair.accessToken)
        try await KeychainStore.shared.saveRefreshToken(pair.refreshToken)
        try await KeychainStore.shared.saveAppleUserId(pair.user.id)
    }
}
