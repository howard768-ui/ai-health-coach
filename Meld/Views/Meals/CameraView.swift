import SwiftUI
import PhotosUI

// MARK: - Camera / Photo Capture View
// Uses PhotosPicker for MVP (simpler than AVFoundation).
// 3-tap flow: open sheet, pick/capture photo, confirm.
// Photo stored locally, base64 sent to backend for recognition.

struct CameraView: View {
    @Bindable var viewModel: MealsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedItem: PhotosPickerItem?
    @State private var capturedImage: UIImage?
    @State private var isRecognizing = false
    @State private var recognizedItems: [FoodItem] = []
    @State private var showConfirmation = false
    @State private var recognitionTimedOut = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let image = capturedImage {
                    // Photo captured — show preview
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: DSRadius.md))
                        .padding(DSSpacing.xl)

                    if isRecognizing {
                        VStack(spacing: DSSpacing.md) {
                            MeldMascot(state: .thinking, size: 48)
                            Text(recognitionTimedOut ? "Taking longer than expected..." : "Identifying your food...")
                                .font(DSTypography.body)
                                .foregroundStyle(DSColor.Text.secondary)
                            if recognitionTimedOut {
                                DSButton(title: "Skip and enter manually", style: .secondary, size: .md) {
                                    isRecognizing = false
                                    recognizedItems = []
                                    showConfirmation = true
                                }
                            }
                        }
                        .padding(.top, DSSpacing.xxl)
                    }
                } else {
                    // No photo yet — show picker prompt
                    VStack(spacing: DSSpacing.xxl) {
                        Spacer()

                        MeldMascot(state: .idle, size: 64)

                        Text("Take a photo of your meal")
                            .font(DSTypography.h2)
                            .foregroundStyle(DSColor.Text.primary)

                        Text("Your coach will identify the food\nand estimate nutrition.")
                            .font(DSTypography.bodySM)
                            .foregroundStyle(DSColor.Text.secondary)
                            .multilineTextAlignment(.center)

                        PhotosPicker(
                            selection: $selectedItem,
                            matching: .images,
                            photoLibrary: .shared()
                        ) {
                            HStack(spacing: DSSpacing.sm) {
                                Image(systemName: "camera")
                                Text("Choose Photo")
                            }
                            .font(DSTypography.body.weight(.medium))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 48)
                            .background(DSColor.Purple.purple500)
                            .clipShape(RoundedRectangle(cornerRadius: DSRadius.md))
                        }
                        .padding(.horizontal, DSSpacing.xl)

                        Spacer()
                    }
                }

                Spacer()
            }
            .background(DSColor.Background.primary)
            .navigationTitle("Log Food")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onChange(of: selectedItem) { _, newItem in
                guard let newItem else { return }
                Task {
                    if let data = try? await newItem.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        capturedImage = image
                        // Audit P2d: PhotosPicker returns the original asset
                        // bytes, which carry EXIF including GPS coordinates
                        // (where the meal was eaten). Re-encode through UIImage
                        // to strip all metadata before the photo leaves the
                        // device — the backend and Claude Vision never receive
                        // location data. If re-encoding fails, do NOT fall back
                        // to the raw bytes; skip rather than leak.
                        guard let stripped = image.jpegData(compressionQuality: 0.85) else { return }
                        await recognizeFood(imageData: stripped)
                    }
                }
            }
            .sheet(isPresented: $showConfirmation) {
                FoodConfirmationView(
                    items: $recognizedItems,
                    mealType: .fromTime(Date()),
                    viewModel: viewModel,
                    image: capturedImage
                )
            }
        }
    }

    private func recognizeFood(imageData: Data) async {
        isRecognizing = true
        recognitionTimedOut = false

        // Show timeout option after 15 seconds
        let timeoutTask = Task {
            try? await Task.sleep(for: .seconds(15))
            if isRecognizing { recognitionTimedOut = true }
        }

        do {
            let items = try await APIClient.shared.recognizeFood(imageData: imageData)
            timeoutTask.cancel()
            recognizedItems = items
            isRecognizing = false
            showConfirmation = true
        } catch {
            timeoutTask.cancel()
            isRecognizing = false
            recognizedItems = []
            showConfirmation = true
        }
    }
}
