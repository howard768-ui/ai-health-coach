import AuthenticationServices
import CryptoKit
import Foundation
import Security
import UIKit

// MARK: - AppleSignInCoordinator
//
// Handles the Sign in with Apple request: generates a random nonce, SHA256-
// hashes it for the request, and passes the raw nonce + identity token to
// AuthManager on success.
//
// Design (per research in Audits/Full Codebase Audit 2026-04-10.md):
// - Random nonce via SecRandomCopyBytes (CSPRNG).
// - SHA256 hash sent to Apple, raw nonce kept on device and sent to backend
//   for nonce verification in the Apple JWT claims.
// - Name/email arrive ONLY on first sign-in — persist immediately.
// - Use claims["sub"] (from the identity token) as source of truth, not the
//   iOS-supplied credential.user field. This is enforced server-side.

@MainActor
final class AppleSignInCoordinator: NSObject {
    /// Current raw nonce — kept until sign-in completes so we can send it
    /// alongside the identity token to our backend. The view reads this via
    /// `currentRawNonce` after calling `prepareRequest`.
    private(set) var currentRawNonce: String?
    private var continuation: CheckedContinuation<Void, Error>?

    /// Call this from the SignInWithAppleButton's onRequest closure.
    ///
    /// If CSPRNG nonce generation fails (vanishingly rare), `currentRawNonce`
    /// stays nil and no nonce is attached; the delegate's missing-nonce guard
    /// then fails the flow with a retryable error instead of the old
    /// `fatalError` crash. The identity token is never forwarded to the
    /// backend without a verified nonce. Audit P3/E7.
    func prepareRequest(_ request: ASAuthorizationAppleIDRequest) {
        request.requestedScopes = [.fullName, .email]
        guard let nonce = Self.randomNonceString() else {
            self.currentRawNonce = nil
            return
        }
        self.currentRawNonce = nonce
        request.nonce = Self.sha256(nonce)
    }

    /// Clear the stored nonce after a successful sign-in.
    func clearNonce() {
        self.currentRawNonce = nil
    }

    /// Awaitable sign-in entry point — used by WelcomeView.
    /// The continuation resumes when the delegate fires.
    func signIn() async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.continuation = cont

            let request = ASAuthorizationAppleIDProvider().createRequest()
            prepareRequest(request)

            // Fail fast (and retryably) if nonce generation failed, instead
            // of sending the user through Face ID for a flow that the
            // delegate would have to reject anyway. Audit P3/E7.
            guard currentRawNonce != nil else {
                self.continuation = nil
                cont.resume(throwing: AppleSignInError.nonceGenerationFailed)
                return
            }

            let controller = ASAuthorizationController(authorizationRequests: [request])
            controller.delegate = self
            controller.presentationContextProvider = self
            controller.performRequests()
        }
    }

    // MARK: - Nonce helpers

    /// Generate a CSPRNG-backed random nonce. URL-safe characters only.
    /// Returns nil if SecRandomCopyBytes fails (callers surface a retryable
    /// sign-in error; the old behavior was a fatalError crash). Audit P3/E7.
    static func randomNonceString(length: Int = 32) -> String? {
        precondition(length > 0)
        let charset: [Character] = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-._")
        var result = ""
        var remainingLength = length

        while remainingLength > 0 {
            var randoms = [UInt8](repeating: 0, count: 16)
            let status = SecRandomCopyBytes(kSecRandomDefault, randoms.count, &randoms)
            if status != errSecSuccess {
                return nil
            }

            for random in randoms {
                if remainingLength == 0 {
                    break
                }
                if random < charset.count {
                    result.append(charset[Int(random)])
                    remainingLength -= 1
                }
            }
        }
        return result
    }

    /// SHA256 hex of the input — Apple expects this in `request.nonce`.
    static func sha256(_ input: String) -> String {
        let inputData = Data(input.utf8)
        let hashed = SHA256.hash(data: inputData)
        return hashed.compactMap { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - ASAuthorizationControllerDelegate

extension AppleSignInCoordinator: ASAuthorizationControllerDelegate {
    func authorizationController(
        controller: ASAuthorizationController,
        didCompleteWithAuthorization authorization: ASAuthorization
    ) {
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else {
            continuation?.resume(throwing: AppleSignInError.invalidCredential)
            continuation = nil
            return
        }
        guard let identityTokenData = credential.identityToken,
              let identityToken = String(data: identityTokenData, encoding: .utf8)
        else {
            continuation?.resume(throwing: AppleSignInError.missingIdentityToken)
            continuation = nil
            return
        }
        guard let rawNonce = currentRawNonce else {
            continuation?.resume(throwing: AppleSignInError.missingNonce)
            continuation = nil
            return
        }

        // fullName and email are ONLY present on first sign-in — capture them now
        // or never see them again for this user.
        let nameComponents = credential.fullName
        let fullName: String? = {
            guard let nc = nameComponents else { return nil }
            let parts = [nc.givenName, nc.familyName].compactMap { $0 }
            return parts.isEmpty ? nil : parts.joined(separator: " ")
        }()
        let email = credential.email

        // Device ID for session tracking — identifierForVendor is stable per app install.
        let deviceId = UIDevice.current.identifierForVendor?.uuidString

        Task { @MainActor in
            do {
                try await AuthManager.shared.signInWithApple(
                    identityToken: identityToken,
                    rawNonce: rawNonce,
                    fullName: fullName,
                    email: email,
                    deviceId: deviceId
                )
                self.currentRawNonce = nil
                self.continuation?.resume()
                self.continuation = nil
            } catch {
                self.continuation?.resume(throwing: error)
                self.continuation = nil
            }
        }
    }

    func authorizationController(
        controller: ASAuthorizationController,
        didCompleteWithError error: Error
    ) {
        continuation?.resume(throwing: error)
        continuation = nil
    }
}

// MARK: - ASAuthorizationControllerPresentationContextProviding

extension AppleSignInCoordinator: ASAuthorizationControllerPresentationContextProviding {
    func presentationAnchor(for controller: ASAuthorizationController) -> ASPresentationAnchor {
        // Return the key window — works for SwiftUI apps too
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap { $0.windows }
            .first(where: \.isKeyWindow)
            ?? ASPresentationAnchor()
    }
}

// MARK: - Errors

enum AppleSignInError: LocalizedError {
    case invalidCredential
    case missingIdentityToken
    case missingNonce
    case nonceGenerationFailed
    case userCancelled

    var errorDescription: String? {
        switch self {
        case .invalidCredential: "Sign in with Apple returned an invalid credential."
        case .missingIdentityToken: "Sign in with Apple did not return an identity token."
        case .missingNonce: "Sign in nonce was lost before Apple responded."
        case .nonceGenerationFailed: "Couldn't start a secure sign-in. Please try again."
        case .userCancelled: "Sign in cancelled."
        }
    }
}
