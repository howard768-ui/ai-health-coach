import SwiftUI
import Shimmer

// MARK: - Loading State Views
// Skeleton placeholders matching the shape of real content.
// Uses SwiftUI-Shimmer for the shimmer animation.
// Each skeleton mirrors the exact layout of its corresponding view.

// MARK: - Dashboard Skeleton

struct DashboardSkeleton: View {
    private let M: CGFloat = 20

    var body: some View {
        VStack(alignment: .leading, spacing: DSSpacing.xxl) {
            // Header skeleton
            VStack(alignment: .leading, spacing: DSSpacing.sm) {
                skeletonBar(width: 100, height: 12)
                skeletonBar(width: 200, height: 28)
            }

            // Coach insight skeleton
            VStack(alignment: .leading, spacing: DSSpacing.md) {
                HStack(spacing: DSSpacing.sm) {
                    skeletonCircle(size: 32)
                    VStack(alignment: .leading, spacing: DSSpacing.xs) {
                        skeletonBar(width: 80, height: 12)
                        skeletonBar(width: 50, height: 10)
                    }
                }
                skeletonBar(width: .infinity, height: 14)
                skeletonBar(width: .infinity, height: 14)
                skeletonBar(width: 200, height: 14)
            }
            .padding(DSSpacing.xl)
            .background(DSColor.Purple.purple50)
            .dsCornerRadius(DSRadius.xl)

            // "Today" header skeleton
            skeletonBar(width: 60, height: 22)

            // Metric grid skeleton
            LazyVGrid(
                columns: [GridItem(.flexible(), spacing: DSSpacing.md), GridItem(.flexible())],
                spacing: DSSpacing.md
            ) {
                ForEach(0..<4, id: \.self) { _ in
                    metricCardSkeleton
                }
            }

            // Recovery card skeleton
            VStack(alignment: .leading, spacing: DSSpacing.sm) {
                skeletonBar(width: 120, height: 10)
                skeletonBar(width: 80, height: 28)
            }
            .metricCard()
        }
        .padding(.horizontal, M)
        .padding(.top, DSSpacing.md)
        // UITestMode: shimmer is a perpetual animation; it prevents XCUITest
        // quiescence and stalls Maestro snapshots (see UITestMode).
        .shimmering(active: !UITestMode.isActive)
    }

    private var metricCardSkeleton: some View {
        VStack(alignment: .leading, spacing: DSSpacing.sm) {
            skeletonBar(width: 80, height: 10)
            skeletonBar(width: 50, height: 32)
            skeletonBar(width: 90, height: 10)
        }
        .metricCard()
    }
}

// MARK: - Chat Message Skeleton

struct ChatMessageSkeleton: View {
    var isCoach: Bool = true

    var body: some View {
        HStack(alignment: .top, spacing: DSSpacing.sm) {
            if isCoach {
                skeletonCircle(size: 28)
            }

            VStack(alignment: isCoach ? .leading : .trailing, spacing: DSSpacing.sm) {
                skeletonBar(width: 220, height: 14)
                skeletonBar(width: 180, height: 14)
                skeletonBar(width: 140, height: 14)
            }
            .padding(DSSpacing.md)
            .background(isCoach ? DSColor.Surface.secondary : DSColor.Purple.purple100)
            .dsCornerRadius(DSRadius.lg)

            if !isCoach {
                Spacer()
            }
        }
        .frame(maxWidth: .infinity, alignment: isCoach ? .leading : .trailing)
        .shimmering(active: !UITestMode.isActive)
    }
}

// MARK: - Typing Indicator (Coach is thinking)

struct TypingIndicator: View {
    @State private var dotIndex = 0

    var body: some View {
        HStack(alignment: .top, spacing: DSSpacing.sm) {
            MeldMascot(state: .thinking, size: 28)

            HStack(spacing: 6) {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .fill(DSColor.Purple.purple400)
                        .frame(width: 8, height: 8)
                        .scaleEffect(dotIndex == i ? 1.3 : 0.8)
                        .animation(
                            .easeInOut(duration: 0.4)
                                .repeatForever(autoreverses: true)
                                .delay(Double(i) * 0.15),
                            value: dotIndex
                        )
                }
            }
            .padding(.horizontal, DSSpacing.lg)
            .padding(.vertical, DSSpacing.md)
            .background(DSColor.Surface.secondary)
            .dsCornerRadius(DSRadius.lg)
        }
        // UITestMode: the dot bounce is repeatForever; leave the dots static
        // under UI automation (see UITestMode).
        .onAppear {
            if !UITestMode.isActive { dotIndex = 2 }
        }
    }
}

// MARK: - Sync Progress Overlay

struct SyncProgressOverlay: View {
    let progress: Double
    let message: String

    var body: some View {
        VStack(spacing: DSSpacing.xxl) {
            MeldMascot(state: .thinking, size: 56)

            VStack(spacing: DSSpacing.md) {
                Text(message)
                    .font(DSTypography.body)
                    .foregroundStyle(DSColor.Text.secondary)
                    .multilineTextAlignment(.center)

                DSProgressBar(progress: progress)
                    .padding(.horizontal, DSSpacing.huge)

                Text("\(Int(progress * 100))%")
                    .font(DSTypography.caption)
                    .foregroundStyle(DSColor.Text.tertiary)
            }
        }
        .padding(DSSpacing.xxxl)
        .background(.ultraThinMaterial)
        .dsCornerRadius(DSRadius.xxl)
    }
}

// MARK: - Skeleton Primitives

private func skeletonBar(width: CGFloat, height: CGFloat) -> some View {
    RoundedRectangle(cornerRadius: height / 2, style: .continuous)
        .fill(DSColor.Surface.secondary)
        .frame(width: width == .infinity ? nil : width, height: height)
        .frame(maxWidth: width == .infinity ? .infinity : nil, alignment: .leading)
}

private func skeletonCircle(size: CGFloat) -> some View {
    Circle()
        .fill(DSColor.Surface.secondary)
        .frame(width: size, height: size)
}

// MARK: - Previews

#Preview("Dashboard Skeleton") {
    ScrollView {
        DashboardSkeleton()
    }
}

#Preview("Chat Skeletons") {
    VStack(spacing: DSSpacing.lg) {
        ChatMessageSkeleton(isCoach: true)
        ChatMessageSkeleton(isCoach: false)
        TypingIndicator()
    }
    .padding()
}

#Preview("Sync Progress") {
    ZStack {
        Color.black.opacity(0.3)
        SyncProgressOverlay(progress: 0.65, message: "Reading your sleep patterns...")
    }
}
