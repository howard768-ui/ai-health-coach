import SwiftUI

// MARK: - Data Source Detail View
// Shows connection status, last sync time, data types being synced,
// and option to disconnect. Anti-dark-pattern: disconnect is same
// friction as connect (one tap + confirm).

struct DataSourceDetailView: View {
    let source: DataSourceType
    let isConnected: Bool
    let lastSynced: String?
    @Environment(\.dismiss) private var dismiss
    @State private var showDisconnectConfirm = false
    @State private var isSyncing = false
    @State private var syncMessage: String?
    @State private var displaySyncTime: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: DSSpacing.xxl) {

                    // Status card
                    VStack(spacing: DSSpacing.lg) {
                        dataSourceIcon
                            .frame(width: 64, height: 64)

                        Text(source.rawValue)
                            .font(DSTypography.h2)
                            .foregroundStyle(DSColor.Text.primary)

                        HStack(spacing: DSSpacing.sm) {
                            Circle()
                                .fill(isConnected ? DSColor.Green.green500 : DSColor.Status.error)
                                .frame(width: 8, height: 8)
                            Text(isConnected ? "Connected" : "Not connected")
                                .font(DSTypography.bodySM)
                                .foregroundStyle(DSColor.Text.secondary)
                        }

                        if let syncTime = displaySyncTime ?? lastSynced {
                            Text("Last synced: \(formatSyncTime(syncTime))")
                                .font(DSTypography.caption)
                                .foregroundStyle(DSColor.Text.tertiary)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, DSSpacing.xxl)

                    // Data types
                    VStack(alignment: .leading, spacing: DSSpacing.sm) {
                        DSSectionHeader(title: "DATA WE READ")

                        VStack(alignment: .leading, spacing: DSSpacing.md) {
                            ForEach(dataTypes, id: \.self) { dataType in
                                HStack(spacing: DSSpacing.md) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(DSColor.Green.green500)
                                        .font(.system(size: 16))
                                    Text(dataType)
                                        .font(DSTypography.body)
                                        .foregroundStyle(DSColor.Text.primary)
                                }
                            }
                        }
                        .padding(DSSpacing.lg)
                        .background(DSColor.Background.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: DSRadius.md))
                    }

                    if isConnected {
                        // Sync button
                        DSButton(
                            title: isSyncing ? "Syncing..." : (syncMessage ?? "Sync now"),
                            style: .secondary,
                            size: .lg,
                            isDisabled: isSyncing
                        ) {
                            Task { await syncData() }
                        }

                        Spacer().frame(height: DSSpacing.xxl)

                        // Disconnect
                        Button {
                            showDisconnectConfirm = true
                        } label: {
                            Text("Disconnect \(source.rawValue)")
                                .font(DSTypography.body)
                                .foregroundStyle(DSColor.Status.error)
                                .frame(maxWidth: .infinity)
                        }
                    } else {
                        // Connect button
                        DSButton(
                            title: "Connect \(source.rawValue)",
                            style: .primary,
                            size: .lg
                        ) {
                            connectSource()
                        }
                    }
                }
                .padding(.horizontal, DSSpacing.xl)
                .padding(.bottom, 120)
            }
            .background(DSColor.Background.primary)
            .navigationTitle(source.rawValue)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .alert("Disconnect \(source.rawValue)?", isPresented: $showDisconnectConfirm) {
                Button("Disconnect", role: .destructive) {
                    Task {
                        if source == .oura {
                            try? await APIClient.shared.disconnectOura()
                        }
                        dismiss()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Your data stays in Meld. You can reconnect any time.")
            }
        }
    }

    private var dataTypes: [String] {
        switch source {
        case .oura: ["Sleep stages and duration", "Heart rate variability (HRV)", "Resting heart rate", "Readiness score", "Body temperature"]
        case .appleHealth: ["Sleep analysis", "Steps and activity", "Heart rate", "HRV", "Workouts", "Body measurements"]
        case .peloton: ["Cycling workouts", "Running workouts", "Strength sessions", "Yoga sessions", "Calories burned"]
        case .garmin: ["Daily steps", "Heart rate", "Sleep data", "Stress levels", "Body Battery", "Activities"]
        }
    }

    @ViewBuilder
    private var dataSourceIcon: some View {
        let imageName: String = switch source {
        case .oura: "oura"
        case .appleHealth: "apple-health"
        case .peloton: "peloton"
        case .garmin: "garmin"
        }

        Image(imageName)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func formatSyncTime(_ rawString: String) -> String {
        // Backend timestamps are naive UTC (Python isoformat, see BackendDate).
        // Use the shared parser so this screen reads them as UTC instead of
        // device-local time — the previous inline DateFormatter array never set
        // timeZone = UTC, so "last synced" was off by the device's UTC offset.
        guard let parsed = BackendDate.parse(rawString) else { return rawString }

        let relative = RelativeDateTimeFormatter()
        relative.unitsStyle = .full
        return relative.localizedString(for: parsed, relativeTo: Date())
    }

    private func connectSource() {
        switch source {
        case .oura:
            // Audit P2d: fetch a CSRF-safe authorize URL from the authenticated
            // /auth/oura/start (signed, single-user state), then open it in
            // Safari. Replaces opening /auth/oura with no state (which 422'd)
            // / a client-supplied apple_user_id (CSRF-prone).
            Task {
                do {
                    let url = try await APIClient.shared.startOuraConnect()
                    _ = await UIApplication.shared.open(url)
                } catch {
                    syncMessage = "Couldn't start Oura connection. Try again."
                    DSHaptic.error()
                }
            }
        case .appleHealth:
            Task {
                await HealthKitService.shared.requestAuthorization()
            }
        case .peloton, .garmin:
            // These use login sheets — handled by ProfileSettingsView
            break
        }
    }

    private func syncData() async {
        isSyncing = true
        syncMessage = nil

        do {
            switch source {
            case .oura:
                try await APIClient.shared.syncOura()
                syncMessage = "Synced"
                // Set to current time in ISO format so formatSyncTime shows "just now"
                let formatter = DateFormatter()
                formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
                displaySyncTime = formatter.string(from: Date())
                DSHaptic.success()
            case .appleHealth:
                await HealthKitService.shared.syncToBackend()
                syncMessage = "Synced"
                let formatter = DateFormatter()
                formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
                displaySyncTime = formatter.string(from: Date())
                DSHaptic.success()
            default:
                syncMessage = "Sync not available yet"
            }
        } catch {
            syncMessage = "Sync failed"
            DSHaptic.error()
        }

        isSyncing = false
    }
}
