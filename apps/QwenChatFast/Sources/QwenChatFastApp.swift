import SwiftUI

// Qwen3.5-0.8B on-device chat — STATIC-shape (ANE) FAST PATH (Core AI, iOS 27).
//
// Separate app (bundle id com.coreai.qwenchat.fast) from the shipped DYNAMIC ondevice/QwenChat
// (com.coreai.qwenchat, 14.7 tok/s ANE). This one drives the iOS STATIC-shape decode graph
// (fixed-capacity KV written at `in_step` + `causal_mask`, NO growing position_ids) — every
// tensor shape is constant across steps, which is the path the ANE accelerates.
@main
struct QwenChatFastApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
