// ModelCatalog — the large, Mac-only Core AI bundles published on Hugging Face,
// downloadable in-app. Each `remote` is the bundle directory inside its repo (it
// holds metadata.json + the .aimodel + tokenizer), `local` is the folder created
// under the app's models directory — ChatEngine.scanFolder then picks it up.
import Foundation

struct DownloadableModel: Identifiable, Hashable {
    let id: String
    let name: String          // display name
    let detail: String        // short architecture note
    let repo: String          // Hugging Face repo URL
    let remote: String        // bundle directory path inside the repo
    let approxSizeGB: Int

    var local: String { (remote as NSString).lastPathComponent }
    var item: ModelDownloader.Item { .init(remote: remote, local: local) }
}

enum ModelCatalog {
    // Verified repo subpaths (HF tree, 2026-06-14). All are `gpu-pipelined/<bundle>`.
    static let macModels: [DownloadableModel] = [
        DownloadableModel(
            id: "qwen36-35b", name: "Qwen3.6-35B-A3B", detail: "MoE · 35B/~3B active · gather_qmm",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI",
            remote: "gpu-pipelined/qwen3_6_35b_a3b_decode_sym8_gather", approxSizeGB: 35),
        DownloadableModel(
            id: "qwen36-27b", name: "Qwen3.6-27B", detail: "dense · int8 == fp16 quality",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_6_27b_decode_int8hu_block32_sym", approxSizeGB: 28),
        DownloadableModel(
            id: "glm47", name: "GLM-4.7-Flash", detail: "MoE + MLA · 30B/~3B active",
            repo: "https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI",
            remote: "gpu-pipelined/glm_4_7_flash_decode_sym8_gather", approxSizeGB: 30),
        DownloadableModel(
            id: "gemma4-12b", name: "Gemma 4 12B", detail: "dense · int8 · flash-decode kernel",
            repo: "https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI",
            remote: "gpu-pipelined/gemma4_12b_qat_decode_int8lin_msdpa_g8", approxSizeGB: 13),
        DownloadableModel(
            id: "gemma4-31b", name: "Gemma 4 31B", detail: "dense · int4 QAT · flash-decode kernel",
            repo: "https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI",
            remote: "gpu-pipelined/gemma4_31b_qat_decode_int4linsym_msdpa_g8", approxSizeGB: 18),
    ]
}

extension ChatEngine {
    // App-managed download location (macOS-appropriate): ~/Library/Application Support/CoreAIChatMac/models
    static var appModelsDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreAIChatMac/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
}
