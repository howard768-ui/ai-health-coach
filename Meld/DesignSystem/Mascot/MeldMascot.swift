import SwiftUI

// MARK: - MeldMascot — the one mascot every call site uses
//
// Composes:
//   1. Base body (SquatBlobIcon)
//   2. Animation envelope (state-driven scale/offset/rotation/opacity)
//
// 36+ places in the app render the mascot. They all go through this
// component so the mascot appearance is consistent everywhere.

struct MeldMascot: View {
    let state: MascotState
    let size: CGFloat

    @State private var phase: Bool = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(state: MascotState = .idle, size: CGFloat = 48) {
        self.state = state
        self.size = size
    }

    var body: some View {
        SquatBlobIcon(isActive: true, size: size)
            .scaleEffect(scaleValue)
            .offset(x: offsetX, y: offsetY)
            .rotationEffect(.degrees(rotation))
            .opacity(opacityValue)
            .frame(width: size * 2, height: size * 2)
            .onAppear { startAnimation() }
            .onChange(of: state) { _, _ in startAnimation() }
            .accessibilityLabel("Coach mascot, \(state.accessibilityLabel)")
    }

    // MARK: - Animation envelope

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

    private func startAnimation() {
        // UITestMode: the idle/thinking repeatForever animations prevent
        // XCUITest quiescence and stall every Maestro snapshot (see UITestMode).
        guard !reduceMotion, !UITestMode.isActive else { return }
        // Reset phase synchronously so SwiftUI paints the starting state.
        phase = false
        let animation: Animation = switch state {
        case .idle:
            .easeInOut(duration: 2.0).repeatForever(autoreverses: true)
        case .thinking:
            .easeInOut(duration: 0.8).repeatForever(autoreverses: true)
        case .celebrating:
            DSMotion.bouncy.repeatCount(3, autoreverses: true)
        case .concerned, .greeting:
            DSMotion.emphasis
        case .error:
            DSMotion.snappy.repeatCount(4, autoreverses: true)
        }
        // Defer withAnimation by one frame. Without this, SwiftUI can coalesce
        // `phase = false` and `phase = true` into the same render pass, which
        // skips the starting state (opacity=0, scale=0.3 for .greeting) so the
        // animation appears to not happen. This was visible on the onboarding
        // Welcome screen: the mascot only popped in after Sign in with Apple
        // dismissed the modal and forced a re-render.
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(16))
            withAnimation(animation) {
                phase = true
            }
        }
    }
}

#Preview("Idle") {
    MeldMascot(state: .idle, size: 96)
        .padding()
}

#Preview("All states") {
    HStack(spacing: 20) {
        ForEach([MascotState.idle, .thinking, .celebrating, .concerned], id: \.self) { s in
            VStack {
                MeldMascot(state: s, size: 48)
                Text(s.rawValue).font(.caption2)
            }
        }
    }
    .padding()
}
