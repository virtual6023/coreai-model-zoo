import SwiftUI

struct Turn: Identifiable {
    let id = UUID()
    let role: String      // "user" | "model"
    var text: String
    var stats: String = ""
}

struct ChatView: View {
    @ObservedObject var engine: Gemma4ChatEngine
    @StateObject private var downloader = ModelDownloader()
    @State private var input = ""
    @State private var turns: [Turn] = []
    @State private var repoURL = ProcessInfo.processInfo.environment["GEMMA_REPO"]
        ?? Gemma4ChatEngine.defaultRepo

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            // Model delivery: the load failed (or never ran) and files for this mode are missing.
            if !engine.ready && !engine.loading {
                let missing = engine.missingDownloads()
                if !missing.isEmpty { downloadPanel(missing) }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(turns) { turn in bubble(turn) }
                        // Live model output while generating.
                        if engine.busy {
                            bubble(Turn(role: "model", text: engine.output.isEmpty ? "…" : engine.output,
                                        stats: engine.stats))
                                .id("live")
                        }
                    }
                    .padding()
                }
                .onChange(of: engine.output) { _, _ in proxy.scrollTo("live", anchor: .bottom) }
            }
            Divider()
            inputBar
        }
        // Mode switch (GPU monolith <-> ANE chunks): reload the engine for the selected compute unit.
        .onChange(of: engine.mode) { _, _ in Task { await engine.load() } }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("CoreAIChat · Gemma 4 E2B").font(.headline)
            Spacer()
            Picker("Compute", selection: $engine.mode) {
                ForEach(GemmaMode.allCases) { m in Text(m.rawValue).tag(m) }
            }
            .pickerStyle(.segmented)
            .frame(width: 130)
            .disabled(engine.busy || engine.loading)
            Text(engine.status)
                .font(.caption).foregroundStyle(engine.ready ? .green : .secondary)
                .lineLimit(1)
        }
        .padding(.horizontal).padding(.vertical, 8)
    }

    private func bubble(_ turn: Turn) -> some View {
        HStack {
            if turn.role == "user" { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 4) {
                Text(turn.text.isEmpty ? " " : turn.text)
                    .textSelection(.enabled)
                    .padding(10)
                    .background(turn.role == "user" ? Color.accentColor.opacity(0.18) : Color.gray.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                if !turn.stats.isEmpty {
                    Text(turn.stats).font(.caption2).foregroundStyle(.secondary)
                }
            }
            if turn.role == "model" { Spacer(minLength: 40) }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
                .disabled(!engine.ready || engine.busy)
            Button {
                Task { await send() }
            } label: {
                Image(systemName: "arrow.up.circle.fill").font(.title2)
            }
            .disabled(!engine.ready || engine.busy || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal).padding(.vertical, 8)
    }

    private func send() async {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        input = ""
        turns.append(Turn(role: "user", text: prompt))
        await engine.generate(prompt)
        turns.append(Turn(role: "model", text: engine.output, stats: engine.stats))
    }

    // MARK: model delivery (published artifact set from the Hugging Face repo)

    private func downloadPanel(_ items: [ModelDownloader.Item]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(engine.mode.rawValue) model files not on device (\(items.count))")
                .font(.subheadline).fontWeight(.semibold)
            TextField("Hugging Face repo URL", text: $repoURL)
                .textFieldStyle(.roundedBorder).font(.caption)
                .autocorrectionDisabled().textInputAutocapitalization(.never)
                .disabled(downloader.busy)
            if downloader.busy {
                ProgressView(value: downloader.fraction)
                Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Button {
                    Task {
                        await downloader.fetch(
                            repo: repoURL, items: items,
                            into: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                                .appendingPathComponent("models"))
                        if downloader.phase == .done { await engine.load() }
                    }
                } label: {
                    Label("Download \(engine.mode.rawValue) set (\(engine.mode == .gpu ? "~4.1" : "~2.1–4.7") GB)",
                          systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            }
            if case .failed(let msg) = downloader.phase {
                Text(msg).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(10)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal).padding(.top, 8)
    }
}
