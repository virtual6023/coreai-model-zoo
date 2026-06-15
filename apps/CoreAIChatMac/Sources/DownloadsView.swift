// DownloadsView — in-app catalog of the large Mac models. Streams a bundle from
// Hugging Face into the app models directory (atomic staging, see ModelDownloader),
// then rescans so the new model appears in the sidebar.
import SwiftUI

struct DownloadsView: View {
    @ObservedObject var engine: ChatEngine
    @StateObject private var downloader = ModelDownloader()
    @State private var active: String?          // id currently downloading
    @State private var pendingDelete: DownloadableModel?   // awaiting delete confirmation
    @Environment(\.dismiss) private var dismiss

    private var modelsDir: URL { ChatEngine.appModelsDir }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Download models").font(.headline)
                Spacer()
                Button("Done") { dismiss() }.disabled(downloader.busy)
            }
            .padding()
            Divider()
            List(ModelCatalog.macModels) { model in
                row(model)
            }
            .listStyle(.inset)
        }
        .frame(width: 580, height: 440)
        .confirmationDialog(
            "Delete \(pendingDelete?.name ?? "model")?",
            isPresented: Binding(get: { pendingDelete != nil },
                                 set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { m in
            Button("Delete (~\(m.approxSizeGB) GB)", role: .destructive) {
                engine.deleteModel(at: modelsDir.appendingPathComponent(m.local))
            }
            Button("Cancel", role: .cancel) {}
        } message: { m in
            Text("Removes it from this Mac. You can download it again later.")
        }
    }

    private func installed(_ m: DownloadableModel) -> Bool {
        FileManager.default.fileExists(
            atPath: modelsDir.appendingPathComponent(m.local)
                .appendingPathComponent("metadata.json").path)
    }

    @ViewBuilder private func row(_ m: DownloadableModel) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(m.name).fontWeight(.medium)
                Text("\(m.detail) · ~\(m.approxSizeGB) GB")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if installed(m) {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .labelStyle(.iconOnly).foregroundStyle(.green)
                Button(role: .destructive) { pendingDelete = m } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("Delete from this Mac")
                .disabled(downloader.busy)
            } else if active == m.id {
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView(value: downloader.fraction).frame(width: 130)
                    Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                        .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
            } else {
                Button("Download") { start(m) }
                    .buttonStyle(.borderedProminent)
                    .disabled(downloader.busy)
            }
        }
        .padding(.vertical, 4)
    }

    private func start(_ m: DownloadableModel) {
        active = m.id
        Task {
            await downloader.fetch(repo: m.repo, items: [m.item], into: modelsDir)
            if downloader.phase == .done { engine.scanFolder(modelsDir) }
            active = nil
        }
    }
}
