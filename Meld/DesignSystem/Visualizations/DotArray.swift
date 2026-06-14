import SwiftUI

// MARK: - Dot Array (Training Consistency)
// 7-circle icon array representing Mon-Sun training days.
// Research: icon arrays have highest comprehension for small numerators
// (Garcia-Retamero, Galesic, Gigerenzer 2010).
//
// Filled = trained, Outlined = missed, Pulsing = today (not yet trained)
// Triple encoding: fill state + color + text label.

struct DotArray: View {
    let trainedDays: Set<Int>  // 0=Mon, 6=Sun
    let todayIndex: Int        // Which day is today (0-6)
    let target: Int            // Target days per week

    private let dayLabels = ["M", "T", "W", "T", "F", "S", "S"]
    private let dotSize: CGFloat = 18
    @State private var todayPulse = false

    private var completedCount: Int {
        trainedDays.count
    }

    private var statusText: String {
        let remaining = target - completedCount
        if remaining <= 0 { return "Goal reached this week" }
        if remaining == 1 { return "1 more day to hit your goal" }
        return "\(remaining) more days to go"
    }

    private var statusColor: Color {
        let remaining = target - completedCount
        if remaining <= 0 { return DSColor.Status.success }
        if remaining <= 1 { return DSColor.Status.warning }
        return DSColor.Text.tertiary
    }

    var body: some View {
        VStack(spacing: DSSpacing.sm) {
            // Dot row
            HStack(spacing: DSSpacing.sm) {
                ForEach(0..<7, id: \.self) { day in
                    VStack(spacing: DSSpacing.xs) {
                        ZStack {
                            if trainedDays.contains(day) {
                                // Trained — filled circle
                                Circle()
                                    .fill(DSColor.Green.green500)
                                    .frame(width: dotSize, height: dotSize)
                            } else if day == todayIndex && !trainedDays.contains(day) {
                                // Today, not yet trained — pulsing
                                Circle()
                                    .stroke(DSColor.Purple.purple500, lineWidth: 2)
                                    .frame(width: dotSize, height: dotSize)
                                    .scaleEffect(todayPulse ? 1.15 : 1.0)
                                    .animation(
                                        .easeInOut(duration: 1.2).repeatForever(autoreverses: true),
                                        value: todayPulse
                                    )
                            } else if day > todayIndex {
                                // Future — faded outline
                                Circle()
                                    .stroke(DSColor.Text.disabled.opacity(0.4), lineWidth: 1.5)
                                    .frame(width: dotSize, height: dotSize)
                            } else {
                                // Past, missed — outline
                                Circle()
                                    .stroke(DSColor.Text.disabled, lineWidth: 1.5)
                                    .frame(width: dotSize, height: dotSize)
                            }
                        }

                        Text(dayLabels[day])
                            .font(.system(size: 9, weight: .regular))
                            .foregroundStyle(DSColor.Text.disabled)
                    }
                }
            }

            // Fraction
            HStack(alignment: .firstTextBaseline, spacing: DSSpacing.xxs) {
                Text("\(completedCount)/\(target)")
                    .font(DSTypography.h3)
                    .foregroundStyle(DSColor.Text.primary)
                Text("days")
                    .font(DSTypography.caption)
                    .foregroundStyle(DSColor.Text.secondary)
            }

            // Status
            Text(statusText)
                .font(DSTypography.caption)
                .foregroundStyle(statusColor)
        }
        // UITestMode: the today-dot pulse is repeatForever; keep it static
        // under UI automation (see UITestMode).
        .onAppear {
            if !UITestMode.isActive { todayPulse = true }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Training: \(completedCount) of \(target) days this week. \(statusText).")
    }
}

// MARK: - Calendar Heatmap (Expanded View)

struct TrainingCalendarHeatmap: View {
    let weeks: [[Bool?]] // 4 weeks x 7 days, nil = future, true = trained, false = missed
    var currentStreak: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: DSSpacing.lg) {
            // 4-week grid
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: 7), spacing: 4) {
                ForEach(0..<28, id: \.self) { index in
                    let week = index / 7
                    let day = index % 7
                    let value = weeks.indices.contains(week) && weeks[week].indices.contains(day) ? weeks[week][day] : nil

                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(cellColor(value))
                        .frame(height: 36)
                }
            }

            // Day labels
            HStack(spacing: 0) {
                ForEach(["M", "T", "W", "T", "F", "S", "S"], id: \.self) { day in
                    Text(day)
                        .font(.system(size: 9))
                        .foregroundStyle(DSColor.Text.disabled)
                        .frame(maxWidth: .infinity)
                }
            }

            // Streak
            if currentStreak > 0 {
                HStack(spacing: DSSpacing.xs) {
                    Image(systemName: "flame.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(DSColor.Status.warning)
                    Text("\(currentStreak)-day streak")
                        .font(DSTypography.h3)
                        .foregroundStyle(DSColor.Text.primary)
                }
            }
        }
    }

    private func cellColor(_ value: Bool?) -> Color {
        // Use .some/.none patterns explicitly so Swift 6 sees the switch over
        // Bool? as exhaustive. The shorthand (case true/false/nil) is legal
        // Swift but trips "switch must be exhaustive" under strict checking.
        switch value {
        case .some(true): DSColor.Green.green500
        case .some(false): DSColor.Surface.secondary
        case .none: DSColor.Surface.secondary.opacity(0.4)
        }
    }
}

// MARK: - Previews

#Preview("Dot Array") {
    DotArray(trainedDays: [0, 1, 2, 3, 4], todayIndex: 5, target: 5)
        .padding()
}

#Preview("Calendar Heatmap") {
    TrainingCalendarHeatmap(
        weeks: [
            [true, true, false, true, true, false, false],
            [true, true, true, false, true, true, false],
            [true, true, false, true, true, false, false],
            [true, true, true, true, true, nil, nil],
        ],
        currentStreak: 5
    )
    .padding()
}
