import SwiftUI

// MARK: - Meld Design System Text Field
// Custom styled text input matching the design system.
// Supports standard and glass variants.

enum DSTextFieldStyle {
    /// Standard text field with surface background
    case standard
    /// Glassmorphic text field for overlay contexts
    case glass
}

struct DSTextField: View {
    let placeholder: String
    @Binding var text: String
    var style: DSTextFieldStyle = .standard
    /// When true, masks input (passwords/credentials). Audit P2c.
    var isSecure: Bool = false
    var onSubmit: (() -> Void)? = nil

    var body: some View {
        HStack(spacing: DSSpacing.sm) {
            Group {
                if isSecure {
                    SecureField(placeholder, text: $text)
                } else {
                    TextField(placeholder, text: $text)
                }
            }
            .font(DSTypography.body)
            .foregroundStyle(DSColor.Text.primary)
            .onSubmit {
                onSubmit?()
            }

            if !text.isEmpty {
                Button(action: {
                    DSHaptic.light()
                    onSubmit?()
                }) {
                    Circle()
                        .fill(DSColor.Green.green500)
                        .frame(width: 32, height: 32)
                        .overlay(
                            Image(systemName: "arrow.up")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.white)
                        )
                }
            }
        }
        .padding(.vertical, DSSpacing.md)
        .padding(.horizontal, DSSpacing.lg)
        .background(backgroundView)
    }

    @ViewBuilder
    private var backgroundView: some View {
        switch style {
        case .standard:
            DSColor.Surface.secondary
                .clipShape(RoundedRectangle(cornerRadius: DSRadius.full, style: .continuous))
        case .glass:
            RoundedRectangle(cornerRadius: DSRadius.full, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: DSRadius.full, style: .continuous)
                        .stroke(Color.white.opacity(0.15), lineWidth: 0.5)
                )
        }
    }
}
