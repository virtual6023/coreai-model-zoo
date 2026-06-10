// CoreAIChat — on-device Core AI LLM chat for iPhone (iOS 27).
// Single app, one picker, four engines: gemma 4 E2B GPU (kernel monolith) / gemma 4 E2B ANE
// (host-cache chunks) / Qwen3.5-0.8B ⚡pipelined (50.3-51.5 tok/s on iPhone 17 Pro) /
// LFM2.5-1.2B ⚡pipelined (38.0-39.6 tok/s) — the pipelined pair rides Apple's
// coreai-pipelined engine on decode-only loop-free int8lin bundles.

import SwiftUI

@main
struct CoreAIChatApp: App {
    @StateObject private var engine = Gemma4ChatEngine()
    var body: some Scene {
        WindowGroup {
            ChatView(engine: engine)
                .task {
                    // De-risk path: GEMMA_SLICE_TEST=1 runs the ISOLATED dual-KV ANE slice self-test
                    // (does the gemma4 iOS port lower+run on the ANE?) and skips the Phase-1 engine.
                    if ProcessInfo.processInfo.environment["GEMMA_SLICE_TEST"] == "1" {
                        await SliceANETest.run()
                        return
                    }
                    // Session B's host-cache core (no state / no in-graph indexed write) on device.
                    if ProcessInfo.processInfo.environment["GEMMA_HOSTCACHE_TEST"] == "1" {
                        await HostCacheTest.run()
                        return
                    }
                    // CHUNKED host-cache ANE engine: chain the 6 chunks (host-managed KV) -> head -> argmax.
                    // GEMMA_CHUNK_VERIFY=1 = device 8/8 vs the oracle; else generate + tok/s.
                    if ProcessInfo.processInfo.environment["GEMMA_CHUNK_TEST"] == "1" {
                        await Gemma4ChunkEngine.run()
                        return
                    }
                    // GPU MONOLITH engine (fused-int8 Metal-kernel FFN core): gather -> 1 core -> head.
                    if ProcessInfo.processInfo.environment["GEMMA_MONO_TEST"] == "1" {
                        await Gemma4MonolithEngine.run()
                        return
                    }
                    await engine.load()
                    // Headless self-test for `devicectl process launch --console
                    // --environment-variables '{"GEMMA_PROMPT":"..."}'` — prints result + tok/s.
                    await engine.autoTestIfRequested()
                }
        }
    }
}
