import SwiftUI

struct ContentView: View {
    @StateObject private var engine = FastEngine()
    @StateObject private var downloader = ModelDownloader()
    // QWEN_PROMPT overrides the default prompt (headless TTFT probes feed a long prompt here).
    @State private var prompt = ProcessInfo.processInfo.environment["QWEN_PROMPT"]
        ?? "The capital of France is"
    @State private var maxTokens = 48
    @State private var computeUnit = ProcessInfo.processInfo.environment["QWEN_CU"] ?? "gpu"
    @State private var loadError: String?
    @State private var autoRan = false
    @State private var modelMissing = false
    // Where the published bundle lives; QWEN_REPO overrides, and the field is editable in the UI.
    @State private var repoURL = ProcessInfo.processInfo.environment["QWEN_REPO"]
        ?? "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI"

    // host-cache chunks qwen3_5_0_8b_ios_hc0..N-1.aimodel in app Documents. Default = the release
    // config: monolith (hc0 only, GPU — the 24-layer graph OOMs the ANE compiler; chunking is an
    // ANEC-capacity workaround, and on GPU it costs 4x the per-token dispatches). QWEN_CHUNKS=4
    // selects the chunked export (requires re-pushing hc0..3 from a --num-chunks 4 export).
    private let numChunks = Int(ProcessInfo.processInfo.environment["QWEN_CHUNKS"] ?? "1") ?? 1
    private let computeUnits = ["ane", "gpu", "cpu"]
    // QWEN_KIND selects the decode artifact family (default int8v3 = the fused-kernel release
    // config; QWEN_KIND=fp16 = the previous fp16 monolith).
    private let kindSuffix: String = {
        let k = ProcessInfo.processInfo.environment["QWEN_KIND"] ?? "int8v3"
        return (k.isEmpty || k == "fp16") ? "" : "_\(k)"
    }()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Circle().fill(statusColor).frame(width: 10, height: 10)
                    Text(engine.status.label).font(.subheadline).foregroundStyle(.secondary)
                    Spacer()
                    Picker("", selection: $computeUnit) {
                        ForEach(computeUnits, id: \.self) { Text($0.uppercased()).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 180)
                    .onChange(of: computeUnit) { _, new in Task { await reload(cu: new) } }
                }

                TextField("Prompt", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...3)

                HStack {
                    Stepper("Max tokens: \(maxTokens)", value: $maxTokens, in: 4...200, step: 4)
                    Button {
                        Task { await engine.generate(prompt: prompt, maxTokens: maxTokens) }
                    } label: {
                        Label("Generate", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(engine.status != .ready)
                }

                ScrollView {
                    Text(prompt).fontWeight(.semibold)
                        + Text(engine.output).foregroundColor(.accentColor)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(10)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                if let err = loadError {
                    Text(err).font(.caption).foregroundStyle(.red)
                }
                if modelMissing { downloadPanel }
                Text(engine.stats.isEmpty ? "Core AI · Qwen3.5-0.8B int8 kernels · static ctx-2048 · GPU monolith + q16 prefill" : engine.stats)
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding()
            .navigationTitle("Qwen3.5 · Static Fast")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { await reload(cu: computeUnit) }
    }

    private var statusColor: Color {
        switch engine.status {
        case .ready: return .green
        case .generating, .loading: return .orange
        case .error: return .red
        case .idle: return .gray
        }
    }

    private func reload(cu: String) async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard FileManager.default.fileExists(atPath: docs.appendingPathComponent("qwen3_5_0_8b_ios_hc0\(kindSuffix).aimodel").path),
              let tokFolder = tokenizerFolder() else {
            modelMissing = true
            loadError = "Model not on device — download it below (or push with devicectl)."
            return
        }
        modelMissing = false
        loadError = nil
        await engine.load(modelDir: docs, tokenizerFolder: tokFolder, cu: cu, numChunks: numChunks)
        // One-shot auto-generate after first load so a headless `devicectl … --console` launch
        // captures a real on-device run + tok/s without a manual tap.
        if engine.status == .ready && !autoRan {
            autoRan = true
            await engine.generate(prompt: prompt, maxTokens: maxTokens)
        }
    }

    private func tokenizerFolder() -> URL? {
        Bundle.main.url(forResource: "tokenizer", withExtension: nil)
    }

    // MARK: model delivery (release monolith from the Hugging Face repo)

    private var downloadPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
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
                    Task { await downloadModel() }
                } label: {
                    Label("Download model (~2.3 GB)", systemImage: "arrow.down.circle")
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
    }

    private func downloadModel() async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        // The release set: int8v3 fused-kernel decode monolith (42.5-45.4 tok/s) + the q16-int8
        // chunked-prefill companion (147 tok/s prefill). The engine runs without the prefill
        // bundle (q=1 prefill) if its download is removed.
        await downloader.fetch(
            repo: repoURL,
            items: [
                .init(remote: "ios-gpu/qwen3_5_0_8b_ios_hc0_int8v3.aimodel",
                      local: "qwen3_5_0_8b_ios_hc0_int8v3.aimodel"),
                .init(remote: "ios-gpu/qwen3_5_0_8b_ios_hc_prefill_q16_b2048_int8.aimodel",
                      local: "qwen3_5_0_8b_ios_hc_prefill_q16_b2048_int8.aimodel"),
            ],
            into: docs)
        if downloader.phase == .done { await reload(cu: computeUnit) }
    }
}
