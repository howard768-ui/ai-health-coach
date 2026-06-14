import Foundation

// MARK: - Coach Chat View Model
// Manages chat state, message history, and AI coach responses via backend.

@Observable @MainActor
final class CoachViewModel {

    var messages: [ChatMessage] = []
    var inputText: String = ""
    var isTyping: Bool = false
    var quickActions: [QuickAction] = QuickAction.defaults

    init() {
        // Start empty — loadHistory() fills from API on appear
        messages = []
        Task { await loadHistory() }
    }

    // MARK: - History Persistence

    func loadHistory() async {
        do {
            let history = try await APIClient.shared.fetchChatHistory()
            if !history.isEmpty {
                // P2-12: Preserve the backend message_id so thumbs up/down still
                // work after a reload. Previously dropped, leaving feedback
                // buttons inert for any historical message.
                messages = history.map { msg in
                    ChatMessage(
                        role: msg.role == "coach" ? .coach : .user,
                        content: Self.content(from: msg.blocks, fallback: msg.content),
                        timestamp: BackendDate.parse(msg.createdAt) ?? Date(),
                        messageId: msg.id
                    )
                }
            } else {
                // No history — show a welcome message (not fake conversation)
                messages = [
                    ChatMessage(
                        role: .coach,
                        content: [.text("Hey! I'm your health coach. Ask me anything about your sleep, recovery, workouts, or nutrition.")],
                        timestamp: Date()
                    )
                ]
            }
        } catch {
            // Backend unavailable — show welcome message
            messages = [
                ChatMessage(
                    role: .coach,
                    content: [.text("Hey! I'm your health coach. I'm having trouble connecting right now, but ask me anything.")],
                    timestamp: Date()
                )
            ]
        }
    }

    // MARK: - Actions

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        // Add user message
        let userMsg = ChatMessage(role: .user, text: text)
        messages.append(userMsg)
        inputText = ""
        DSHaptic.light()
        Analytics.Coach.messageSent()

        fetchCoachResponse(to: text)
    }

    func prefill(_ text: String) {
        inputText = text
    }

    func sendQuickAction(_ action: QuickAction) {
        let userMsg = ChatMessage(role: .user, text: action.title)
        messages.append(userMsg)
        DSHaptic.light()
        Analytics.Coach.quickActionTapped(action: action.title)

        fetchCoachResponse(to: action.prompt)
    }

    // MARK: - Feedback

    func submitFeedback(for message: ChatMessage, feedback: String) {
        guard let messageId = message.messageId else { return }
        guard let index = messages.firstIndex(where: { $0.id == message.id }) else { return }
        messages[index].feedback = feedback
        DSHaptic.light()

        Task {
            try? await APIClient.shared.submitFeedback(messageId: messageId, feedback: feedback)
        }
    }

    // MARK: - Coach Response
    //
    // IMPORTANT: No mock/fake fallback. If the backend is unreachable, we show
    // an honest error — never fabricated health data. This is enforced by the
    // eval suite (evidence-bound coaching rule) and was fixed as P0-7 of the
    // 2026-04-10 audit.

    private func fetchCoachResponse(to prompt: String) {
        isTyping = true

        // Defense in depth: the backend already surfaces crisis resources when
        // its Claude call fails, but that safety net only catches
        // anthropic.APIError. If the request fails further up the stack (no
        // network, connection dropped, 500/502 from the host, unexpected
        // exception) iOS never hears from the backend at all, which would
        // silently route crisis language through the generic
        // "trouble connecting" reply. Detect the phrase on the client and
        // override every error branch below so the user always sees 988/741741
        // when they need to.
        let isCrisis = CrisisKeywords.detect(in: prompt)

        Task {
            // P2-10: Distinguish offline from server error so we give the
            // user a useful nudge instead of a generic "trouble connecting"
            // message. The OfflineBanner at the root is also visible.
            if !NetworkMonitor.shared.isOnline {
                let text = isCrisis
                    ? CrisisKeywords.fallbackMessage
                    : "You're offline right now. I'll be here as soon as you're back."
                messages.append(ChatMessage(role: .coach, text: text))
                isTyping = false
                return
            }

            do {
                let response = try await APIClient.shared.sendMessage(prompt)
                let coachMsg = ChatMessage(
                    role: .coach,
                    content: Self.content(from: response.blocks, fallback: response.content),
                    messageId: response.messageId
                )
                messages.append(coachMsg)
            } catch APIError.networkError {
                let text = isCrisis
                    ? CrisisKeywords.fallbackMessage
                    : "Looks like the connection dropped. Try again once you're back online."
                messages.append(ChatMessage(role: .coach, text: text))
            } catch {
                // Honest error, no fabricated data.
                let text = isCrisis
                    ? CrisisKeywords.fallbackMessage
                    : "I'm having trouble connecting right now. Check your internet connection and try again in a moment."
                messages.append(ChatMessage(role: .coach, text: text))
            }
            isTyping = false
        }
    }

    // MARK: - Block Mapping
    //
    // Convert the backend's structured `blocks` array into the UI's ChatContent
    // model. Falls back to a single text block when `blocks` is nil (older
    // server builds or an empty response), using the plain `content` string.

    private static func content(from blocks: [APIContentBlock]?, fallback: String) -> [ChatContent] {
        guard let blocks, !blocks.isEmpty else {
            return [.text(fallback)]
        }
        return blocks.map { block in
            switch block {
            case .text(let value):
                return .text(value)
            case .dataCard(let metric, let value, let unit, let subtitle):
                return .dataCard(ChatDataCard(
                    metricKey: metric,
                    value: value,
                    unit: unit,
                    subtitle: subtitle
                ))
            }
        }
    }
}
