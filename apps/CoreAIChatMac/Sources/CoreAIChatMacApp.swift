// CoreAIChatMac — a minimal macOS chat app for Core AI language bundles on
// Apple's official runtime. Pick a folder of exported bundles, click a model,
// chat — with live load / TTFT / tok/s / memory stats in the footer.

import Foundation
import SwiftUI

@main
struct CoreAIChatMacApp: App {
    init() {
        // The zoo catalog bundles are q=1 (single-token) decode graphs built around custom Metal
        // kernels: they have no multi-token prefill shape. The sequential engine's default prefill
        // batches the whole prompt (`.wholeBatch` for prompts ≤ chunkThreshold=1024), whose shape
        // the bundle can't resolve → assert in resolvingDynamicDimensions (SIGTRAP on first prompt).
        // Forcing the runtime's chunk threshold to 1 makes prefill feed one [1] token at a time,
        // the shape every decode bundle supports. Set before any engine reads ModelConfig.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
    }

    var body: some Scene {
        WindowGroup {
            ChatView()
                .frame(minWidth: 980, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }
}
