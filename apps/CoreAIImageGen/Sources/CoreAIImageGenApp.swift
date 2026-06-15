// CoreAIImageGen — a minimal cross-platform (macOS + iOS) image-generation app for
// Core AI diffusion bundles (FLUX.2 / Stable Diffusion) running on Apple's official
// CoreAIDiffusionPipeline runtime (apple/coreai-models).

import SwiftUI

@main
struct CoreAIImageGenApp: App {
    var body: some Scene {
        WindowGroup {
            #if os(macOS)
            // A minimum window size for the desktop split-view layout. This MUST be
            // macOS-only: applied on iOS it forces ContentView to 820 pt wide on a
            // ~393 pt phone screen, so the whole UI overflows both edges.
            ContentView()
                .frame(minWidth: 820, minHeight: 640)
            #else
            ContentView()
            #endif
        }
        #if os(macOS)
        .windowResizability(.contentMinSize)
        #endif
    }
}
