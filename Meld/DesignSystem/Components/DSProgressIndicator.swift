import SwiftUI

// MARK: - Meld Design System Progress Indicators
// Three variants: step dots (onboarding), progress bar (sync), circular (loading).
// All use DS tokens for colors, spacing, and animation.

// MARK: - Step Dots (Onboarding progression)

struct DSStepDots: View {
    let totalSteps: Int
    let currentStep: Int

    var body: some View {
        HStack(spacing: DSSpacing.sm) {
            ForEach(0..<totalSteps, id: \.self) { step in
                Capsule()
                    .fill(step <= currentStep ? DSColor.Purple.purple500 : DSColor.Text.disabled)
                    .frame(
                        width: step == currentStep ? 24 : 8,
                        height: 8
                    )
                    .animation(DSMotion.snappy, value: currentStep)
            }
        }
        .accessibilityLabel("Step \(currentStep + 1) of \(totalSteps)")
    }
}

// MARK: - Progress Bar (Sync, data loading)

struct DSProgressBar: View {
    let progress: Double // 0.0 to 1.0
    var color: Color = DSColor.Green.green500
    var height: CGFloat = 6

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Track
                Capsule()
                    .fill(DSColor.Surface.secondary)
                    .frame(height: height)

                // Fill
                Capsule()
                    .fill(color)
                    .frame(
                        width: max(0, geometry.size.width * min(1.0, progress)),
                        height: height
                    )
                    .animation(DSMotion.standard, value: progress)
            }
        }
        .frame(height: height)
        .accessibilityValue("\(Int(progress * 100)) percent")
    }
}

// MARK: - Circular Progress (Loading, sync)

struct DSCircularProgress: View {
    let progress: Double? // nil = indeterminate
    var size: CGFloat = 48
    var lineWidth: CGFloat = 4
    var color: Color = DSColor.Purple.purple500

    @State private var isAnimating = false

    var body: some View {
        ZStack {
            // Background track
            Circle()
                .stroke(DSColor.Surface.secondary, lineWidth: lineWidth)

            if let progress {
                // Determinate
                Circle()
                    .trim(from: 0, to: min(1.0, progress))
                    .stroke(
                        color,
                        style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .animation(DSMotion.standard, value: progress)
            } else {
                // Indeterminate — spinning
                Circle()
                    .trim(from: 0, to: 0.3)
                    .stroke(
                        color,
                        style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                    )
                    .rotationEffect(.degrees(isAnimating ? 360 : 0))
                    .animation(
                        .linear(duration: 1.0).repeatForever(autoreverses: false),
                        value: isAnimating
                    )
                    // UITestMode: keep the spinner static so XCUITest
                    // snapshots can reach quiescence (see UITestMode).
                    .onAppear {
                        if !UITestMode.isActive { isAnimating = true }
                    }
            }
        }
        .frame(width: size, height: size)
        .accessibilityLabel(progress != nil ? "\(Int((progress ?? 0) * 100)) percent complete" : "Loading")
    }
}

// MARK: - Previews

#Preview("Step Dots") {
    VStack(spacing: DSSpacing.xxl) {
        DSStepDots(totalSteps: 5, currentStep: 0)
        DSStepDots(totalSteps: 5, currentStep: 2)
        DSStepDots(totalSteps: 5, currentStep: 4)
    }
    .padding()
}

#Preview("Progress Bar") {
    VStack(spacing: DSSpacing.xxl) {
        DSProgressBar(progress: 0.0)
        DSProgressBar(progress: 0.35)
        DSProgressBar(progress: 0.7)
        DSProgressBar(progress: 1.0, color: DSColor.Status.success)
    }
    .padding()
}

#Preview("Circular") {
    HStack(spacing: DSSpacing.xxl) {
        DSCircularProgress(progress: nil) // Indeterminate
        DSCircularProgress(progress: 0.45)
        DSCircularProgress(progress: 1.0, color: DSColor.Status.success)
    }
    .padding()
}
