import SwiftUI

// MARK: - Peloton Login View
// Username/password authentication (NOT OAuth).
// Peloton uses session cookies, not OAuth tokens.

struct PelotonLoginView: View {
    @State private var username = ""
    @State private var password = ""
    @State private var isLoading = false
    @State private var isConnected = false
    @State private var errorMessage: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: DSSpacing.xxl) {
                Spacer()

                if isConnected {
                    // Success state
                    VStack(spacing: DSSpacing.lg) {
                        MeldMascot(state: .celebrating, size: 64)
                        Text("Peloton connected")
                            .font(DSTypography.h2)
                            .foregroundStyle(DSColor.Text.primary)
                        Text("Your workouts will sync automatically.")
                            .font(DSTypography.bodySM)
                            .foregroundStyle(DSColor.Text.secondary)
                    }
                } else {
                    // Login form
                    VStack(alignment: .leading, spacing: DSSpacing.lg) {
                        Text("Connect Peloton")
                            .font(DSTypography.h2)
                            .foregroundStyle(DSColor.Text.primary)
                        Text("Sign in with your Peloton account to sync workouts.")
                            .font(DSTypography.bodySM)
                            .foregroundStyle(DSColor.Text.secondary)

                        DSTextField(
                            placeholder: "Email or username",
                            text: $username
                        )
                        .textContentType(.emailAddress)
                        .autocapitalization(.none)

                        DSTextField(
                            placeholder: "Password",
                            text: $password,
                            isSecure: true
                        )
                        .textContentType(.password)

                        if let error = errorMessage {
                            Text(error)
                                .font(DSTypography.caption)
                                .foregroundStyle(DSColor.Status.error)
                        }
                    }
                    .padding(.horizontal, DSSpacing.xl)
                }

                Spacer()

                if isConnected {
                    DSButton(title: "Done", style: .primary, size: .lg) {
                        dismiss()
                    }
                    .padding(.horizontal, DSSpacing.xl)
                } else {
                    DSButton(
                        title: isLoading ? "Connecting..." : "Connect",
                        style: .primary,
                        size: .lg,
                        isDisabled: username.isEmpty || password.isEmpty || isLoading
                    ) {
                        Task { await login() }
                    }
                    .padding(.horizontal, DSSpacing.xl)
                }

                Spacer().frame(height: DSSpacing.xxl)
            }
            .background(DSColor.Background.primary)
            .navigationTitle("Peloton")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func login() async {
        isLoading = true
        errorMessage = nil
        do {
            try await APIClient.shared.loginPeloton(username: username, password: password)
            isConnected = true
            DSHaptic.success()
        } catch {
            errorMessage = "Login failed. Check your credentials and try again."
            DSHaptic.error()
        }
        isLoading = false
    }
}
