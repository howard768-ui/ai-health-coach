import SwiftUI

// MARK: - Mascot Animation States
// Each state has a distinct motion personality using DS motion tokens.
// All animations respect Reduce Motion accessibility setting.

enum MascotState: String, CaseIterable {
    case idle        // Default — subtle breathing bob
    case thinking    // AI processing — gentle rock + pulse
    case celebrating // Milestone hit — bounce + scale pop
    case concerned   // Warning/attention — slight shrink
    case greeting    // App launch — wave/scale up
    case error       // Something wrong — shake

    var accessibilityLabel: String {
        switch self {
        case .idle: "resting"
        case .thinking: "thinking"
        case .celebrating: "celebrating"
        case .concerned: "concerned"
        case .greeting: "waving hello"
        case .error: "alerting an issue"
        }
    }
}

// MARK: - Animated Mascot View

struct AnimatedMascot: View {
    let state: MascotState
    var size: CGFloat = 48

    @State private var phase: Bool = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        SquatBlobIcon(isActive: true, size: size)
            .scaleEffect(scaleValue)
            .offset(x: offsetX, y: offsetY)
            .rotationEffect(.degrees(rotation))
            .opacity(opacityValue)
            .onAppear { startAnimation() }
            .onChange(of: state) { _, _ in startAnimation() }
            .accessibilityLabel("Coach mascot, \(state.accessibilityLabel)")
    }

    // MARK: - Animation Values Per State

    private var scaleValue: CGFloat {
        guard !reduceMotion else { return 1.0 }
        switch state {
        case .idle: return phase ? 1.03 : 1.0
        case .thinking: return 1.0
        case .celebrating: return phase ? 1.15 : 0.95
        case .concerned: return phase ? 0.92 : 1.0
        case .greeting: return phase ? 1.0 : 0.3
        case .error: return 1.0
        }
    }

    private var offsetX: CGFloat {
        guard !reduceMotion else { return 0 }
        switch state {
        case .error: return phase ? -4 : 4
        default: return 0
        }
    }

    private var offsetY: CGFloat {
        guard !reduceMotion else { return 0 }
        switch state {
        case .idle: return phase ? -1.5 : 1.5
        case .celebrating: return phase ? -8 : 0
        case .greeting: return phase ? 0 : 20
        default: return 0
        }
    }

    private var rotation: Double {
        guard !reduceMotion else { return 0 }
        switch state {
        case .thinking: return phase ? 3 : -3
        case .celebrating: return phase ? 5 : -5
        default: return 0
        }
    }

    private var opacityValue: Double {
        guard !reduceMotion else { return 1.0 }
        switch state {
        case .thinking: return phase ? 0.7 : 1.0
        case .greeting: return phase ? 1.0 : 0.0
        default: return 1.0
        }
    }

    // MARK: - Animation Trigger

    private func startAnimation() {
        // UITestMode: repeatForever prevents XCUITest quiescence (see UITestMode).
        guard !reduceMotion, !UITestMode.isActive else { return }

        phase = false

        let animation: Animation = switch state {
        case .idle:
            .easeInOut(duration: 2.0).repeatForever(autoreverses: true)
        case .thinking:
            .easeInOut(duration: 0.8).repeatForever(autoreverses: true)
        case .celebrating:
            DSMotion.bouncy.repeatCount(3, autoreverses: true)
        case .concerned:
            DSMotion.emphasis
        case .greeting:
            DSMotion.emphasis
        case .error:
            DSMotion.snappy.repeatCount(4, autoreverses: true)
        }

        withAnimation(animation) {
            phase = true
        }
    }
}

// MARK: - Previews

#Preview("All States") {
    VStack(spacing: DSSpacing.xxl) {
        ForEach(MascotState.allCases, id: \.self) { state in
            HStack(spacing: DSSpacing.lg) {
                AnimatedMascot(state: state, size: 48)
                    .frame(width: 60, height: 60)

                VStack(alignment: .leading, spacing: DSSpacing.xxs) {
                    Text(state.rawValue.capitalized)
                        .font(DSTypography.h3)
                        .foregroundStyle(DSColor.Text.primary)
                    Text(state.accessibilityLabel)
                        .font(DSTypography.caption)
                        .foregroundStyle(DSColor.Text.tertiary)
                }
            }
        }
    }
    .padding()
}

#Preview("Idle Large") {
    AnimatedMascot(state: .idle, size: 96)
        .padding(DSSpacing.huge)
}
