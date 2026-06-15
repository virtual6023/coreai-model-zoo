import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var engine = DiffusionEngine()

    // Optional: the iOS build ships an empty hosted catalog (load via "Local…").
    @State private var selectedModel = DiffusionEngine.catalog.first
    @State private var prompt = "a watercolor painting of a red fox reading a book by candlelight, cozy, detailed"
    @State private var negativePrompt = ""
    @State private var steps = DiffusionEngine.catalog.first?.defaultSteps ?? 4
    @State private var guidance = DiffusionEngine.catalog.first?.defaultGuidance ?? 1.0
    @State private var seedText = "42"
    @State private var showingFolderImporter = false

    var body: some View {
        #if os(macOS)
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    group("Model") { modelControls }
                    group("Prompt") { promptControls(maxWidth: nil) }
                    group("Settings") { settingsControls }
                    generateButton
                }
                .padding(18)
            }
            .frame(minWidth: 320, idealWidth: 350, maxWidth: 440)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
            canvas.frame(minWidth: 460)
        }
        #else
        NavigationStack {
            // GeometryReader gives a CONCRETE content width. A `TextField(axis: .vertical)`
            // reports its full single-line intrinsic width upward, which inside a Form/List
            // blows the whole column wider than the screen (content overflows both edges).
            // Capping the text fields at this measured width — not `.infinity` — keeps the
            // reported width bounded, so everything stays inside the device.
            GeometryReader { geo in
                let contentWidth = geo.size.width - 32
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        canvas
                            .frame(maxWidth: .infinity)
                            .frame(height: 230)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        card("Model") { modelControls }
                        card("Prompt") { promptControls(maxWidth: contentWidth) }
                        card("Settings") { settingsControls }
                    }
                    .padding(16)
                    .frame(width: geo.size.width, alignment: .leading)
                }
                .scrollDismissesKeyboard(.interactively)
                .safeAreaInset(edge: .bottom) {
                    generateButton
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.bar)
                }
            }
            .navigationTitle("CoreAI Image Gen")
            .navigationBarTitleDisplayMode(.inline)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
        }
        #endif
    }

    // MARK: - Control content (shared; wrapped in `group`/`card` per platform)

    @ViewBuilder private var modelControls: some View {
        if !DiffusionEngine.catalog.isEmpty {
            Picker("Model", selection: $selectedModel) {
                ForEach(DiffusionEngine.catalog) { Text($0.title).tag(Optional($0)) }
            }
            #if os(macOS)
            .labelsHidden()
            #else
            .pickerStyle(.menu)
            #endif
            .disabled(engine.status.isBusy)
        }

        HStack {
            if let model = selectedModel {
                Button {
                    steps = model.defaultSteps
                    guidance = model.defaultGuidance
                    engine.loadFromHub(model)
                } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle")
                }
            }
            Button { showingFolderImporter = true } label: {
                Label("Local…", systemImage: "folder")
            }
        }
        .disabled(engine.status.isBusy)

        statusLine
    }

    @ViewBuilder private func promptControls(maxWidth: CGFloat?) -> some View {
        TextField("Prompt", text: $prompt, axis: .vertical)
            .lineLimit(2...6)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
        TextField("Negative prompt (optional)", text: $negativePrompt, axis: .vertical)
            .lineLimit(1...3)
            .foregroundStyle(.secondary)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
    }

    @ViewBuilder private var settingsControls: some View {
        Stepper("Steps: \(steps)", value: $steps, in: 1...50)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Guidance")
                Spacer()
                Text(String(format: "%.1f", guidance)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $guidance, in: 0...10)
        }
        HStack {
            Text("Seed")
            TextField("seed", text: $seedText)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif
            Button { seedText = String(UInt32.random(in: 0 ... .max)) } label: {
                Image(systemName: "die.face.5")
            }
            .buttonStyle(.borderless)
        }
    }

    private var statusLine: some View {
        HStack(spacing: 8) {
            if engine.status.isBusy { ProgressView().controlSize(.small) }
            Text(engine.status.label)
                .font(.caption).foregroundStyle(statusColor).lineLimit(2)
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusColor: Color {
        if case .error = engine.status { return .red }
        if case .ready = engine.status { return .green }
        return .secondary
    }

    private var generateButton: some View {
        Group {
            if case .generating = engine.status {
                Button(role: .destructive) { engine.cancel() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
            } else {
                Button {
                    engine.generate(
                        prompt: prompt,
                        negativePrompt: negativePrompt,
                        steps: steps,
                        guidance: guidance,
                        seed: UInt32(seedText) ?? 42)
                } label: {
                    Label("Generate", systemImage: "sparkles").frame(maxWidth: .infinity)
                }
                .disabled(!engine.canGenerate || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .controlSize(.large)
        .buttonStyle(.borderedProminent)
    }

    // MARK: - Canvas (shared)

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let cg = engine.image {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .padding(12)
            } else {
                placeholder
            }

            if case .generating(let s, let t) = engine.status {
                progressOverlay(value: Double(s), total: Double(max(t, 1)))
            } else if case .downloading = engine.status {
                VStack {
                    Spacer()
                    DownloadBar(downloader: engine.downloader)
                        .padding(12)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
                        .padding(20)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 10) {
                if let secs = engine.generateSeconds, engine.image != nil {
                    Text("\(engine.imageSize) · \(String(format: "%.1fs", secs))")
                        .font(.caption).foregroundStyle(.white.opacity(0.7))
                }
                if let url = engine.exportURL {
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
            }
            .padding(10)
        }
    }

    private func progressOverlay(value: Double, total: Double) -> some View {
        VStack {
            Spacer()
            ProgressView(value: value, total: total) {
                Text(engine.status.label).font(.caption)
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
            .padding(20)
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "photo.artframe")
                .font(.system(size: 46)).foregroundStyle(.tertiary)
            Text(placeholderText)
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle:
            return DiffusionEngine.catalog.isEmpty
                ? "Tap Local… to load a Core AI diffusion bundle (e.g. Stable Diffusion)."
                : "Pick a model and tap Download & Load to begin."
        case .downloading: return "Downloading the converted bundle from Hugging Face — a few GB, cached after the first run."
        case .loading: return "Loading the model into the Core AI runtime…"
        case .ready: return "Enter a prompt and tap Generate."
        case .error(let m): return m
        case .generating: return ""
        }
    }

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
    }

    #if os(iOS)
    @ViewBuilder
    private func card<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 14))
    }
    #endif
}

/// Download progress, observing the shared downloader directly so the bar and byte
/// counter advance as chunks land (the engine only owns the high-level phase).
private struct DownloadBar: View {
    @ObservedObject var downloader: ModelDownloader

    var body: some View {
        VStack(spacing: 6) {
            ProgressView(value: downloader.fraction)
            Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}
