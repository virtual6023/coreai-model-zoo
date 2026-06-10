# CoreAIChat — on-device Core AI LLM chat (iOS 27)

SwiftUI chat apps that run LLMs **on device** via Core AI, built on the
[`CoreAIRunner`](../../swift/) package. Two model engines, both verified greedy-exact (top-1 vs
the HF eager reference) on an iPhone 17 Pro running the iOS 27 beta:

- **Qwen3.5-0.8B** (hybrid linear+full attention) — all-in-one stateful graph (4 states:
  KV + SSM conv/rec), loop-free single-step decode. **GPU 27.7 tok/s (static, ctx 2048) /
  ANE 14.7 (dynamic).**
- **Gemma 4 E2B** (text decoder) — 3-stage flow: mmap embedding/PLE gather front-end → decode
  core → head. **GPU 22 tok/s (int4-k-means custom-kernel monolith) / ANE 6 (int8 chunks).**

Measured numbers, bundle sizes, and per-config caveats live in the zoo cards:
[`zoo/qwen3.5.md`](../../zoo/qwen3.5.md) · [`zoo/gemma4-e2b.md`](../../zoo/gemma4-e2b.md).

## How it works

1. **Engine first (Mac CLI)** — validate with `swift run coreai-run` against a conversion oracle
   (raw token ids in, ids out). Fast loop, no UI. See [`../../swift/`](../../swift/).
2. **App** — minimal SwiftUI: model picker, chat transcript, streaming generation.
   - Runner: `CoreAIRunner.HybridCoreAIEngine` driving the stateful `.aimodel`
     (states are function-local `inout` — class-property state trips MutableViews lifetime checks).
   - Tokenizer: swift-transformers, bundled offline.
   - Model delivery: **in-app download from the Hugging Face repos above** (editable URL field;
     files stream into a staging dir and each bundle is renamed into place only when complete —
     Core AI's specialization cache is content-keyed, and a partially-present bundle poisons it).
     Weights are NOT bundled into the .app (GB-class; Apple's guidance for large models is
     download-then-specialize off the interactive flow — WWDC26 326). For development iteration
     you can still sideload: `xcrun devicectl device copy to --domain-type appDataContainer`.
3. **Build + deploy** (Xcode 27 beta):
   ```bash
   export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer
   xcodegen generate
   xcodebuild -scheme CoreAIChat -sdk iphoneos -allowProvisioningUpdates build
   xcrun devicectl device install app <built .app>
   ```
   ⚠️ Benchmark **Release** builds only — Debug-build tok/s under-reads by 2–3× (host-side work
   dominates in Debug; e.g. 10.3 vs 30.4 tok/s on the same artifact).

## Status

The apps run on device today (numbers above). **Source consolidation into this directory is in
progress** — the engines currently live in the working workspace alongside the conversion
harnesses; they land here as a clean xcodegen project. The conversion side (how the `.aimodel`
bundles are produced) is [`../../conversion/`](../../conversion/); the gotchas they hit are all in
[`../../knowledge/`](../../knowledge/).
