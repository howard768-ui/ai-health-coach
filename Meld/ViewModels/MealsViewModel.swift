import Foundation

// MARK: - Meals View Model
// Manages daily meal data, logging, and food search.
// Wired to backend API for real food data (USDA + OFF + AI).

@Observable @MainActor
final class MealsViewModel {

    var dailyNutrition: DailyNutrition
    var searchText: String = ""
    var searchResults: [FoodItem] = []
    var isSearching: Bool = false
    var showInputSheet: Bool = false
    var showCamera: Bool = false
    var showBarcodeScanner: Bool = false
    var showVoiceCapture: Bool = false
    var isLoading: Bool = false
    var loadError: Bool = false
    private var searchTask: Task<Void, Never>?

    init() {
        // Start with empty nutrition — loadMeals() fills from API
        self.dailyNutrition = DailyNutrition(
            date: Date(),
            meals: [],
            calorieTarget: 2200,
            proteinTarget: 150,
            carbTarget: 220,
            fatTarget: 60,
            streak: 0
        )
    }

    // MARK: - Data Loading

    func loadMeals() async {
        isLoading = true
        loadError = false
        let dateString = Self.dateFormatter.string(from: Date())
        do {
            let response = try await APIClient.shared.fetchMeals(date: dateString)
            dailyNutrition = DailyNutrition(
                date: Date(),
                meals: response.meals.map { apiMeal in
                    Meal(
                        date: BackendDate.parse(apiMeal.created_at) ?? Date(),
                        mealType: MealType(rawValue: apiMeal.meal_type.capitalized) ?? .fromTime(Date()),
                        items: apiMeal.items.map { $0.toFoodItem() },
                        source: InputSource(rawValue: apiMeal.source) ?? .manual
                    )
                },
                calorieTarget: 2200,
                proteinTarget: 150,
                carbTarget: 220,
                fatTarget: 60,
                streak: 0
            )
        } catch {
            loadError = true
            Log.meals.error("Failed to load meals: \(error.localizedDescription)")
        }
        isLoading = false
    }

    // MARK: - Search (DOVA: USDA → OFF → AI)

    func searchFood(_ query: String) {
        searchTask?.cancel()
        guard !query.isEmpty else {
            searchResults = []
            isSearching = false
            return
        }
        isSearching = true

        searchTask = Task {
            // 300ms debounce
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }

            do {
                let results = try await APIClient.shared.searchFood(query)
                guard !Task.isCancelled else { return }
                searchResults = results
            } catch {
                guard !Task.isCancelled else { return }
                searchResults = []
                Log.meals.error("Search failed: \(error.localizedDescription)")
            }
            isSearching = false
        }
    }

    // MARK: - Logging

    func logMeal(_ meal: Meal) {
        dailyNutrition.meals.append(meal)
        DSHaptic.success()
    }

    func logFoodItem(_ item: FoodItem) {
        let meal = Meal(items: [item], source: .search)
        Task {
            do {
                try await saveMeal(meal)
            } catch {
                // Still add locally even if API fails
                logMeal(meal)
            }
        }
        showInputSheet = false
        searchText = ""
        searchResults = []
    }

    func saveMeal(_ meal: Meal) async throws {
        try await APIClient.shared.logMeal(meal)
        dailyNutrition.meals.append(meal)
    }

    // MARK: - Computed

    var recentMeals: [String] {
        let names = dailyNutrition.meals.flatMap { $0.items.map(\.name) }
        return Array(Set(names)).prefix(4).map { $0 }
    }

    var timeBasedSuggestion: String? {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 5..<11: return "Log breakfast"
        case 11..<15: return "Log lunch"
        case 17..<22: return "Log dinner"
        default: return nil
        }
    }

    // MARK: - Helpers

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        // en_US_POSIX pins the Gregorian calendar + ASCII digits: without it,
        // a device set to a non-Gregorian calendar (Buddhist, Japanese) makes
        // "yyyy" render e.g. 2569 and the backend stores meals on the wrong
        // date. Time zone stays device-local on purpose: the meal "date" is
        // the user's local calendar day. Audit P3/E7.
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}
