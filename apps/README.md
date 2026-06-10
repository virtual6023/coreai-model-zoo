# On-device Core AI chat apps (iOS 27)

SwiftUI sample apps that run LLMs **on device** via Core AI, verified greedy-exact (top-1 vs the
HF eager reference) on an iPhone 17 Pro running the iOS 27 beta:

| App | Model | Decode (iPhone 17 Pro) |
|---|---|---|
| [`CoreAIChat/`](CoreAIChat/) | **Gemma 4 E2B** (text) — mmap embedding/PLE gather front-end → 35-layer core → head — **plus Qwen3.5-0.8B ⚡pipelined**, one Gemma-GPU / Gemma-ANE / Qwen segmented picker | **Gemma GPU 22 tok/s** (int4-k-means kernels) / **ANE 6** (int8 chunks) / **Qwen ⚡ 50.3–51.5 tok/s** (benchmark; ~48 chat-surface — int8lin on Apple's `coreai-pipelined` engine, zero custom kernels) |
| [`QwenChatFast/`](QwenChatFast/) | **Qwen3.5-0.8B** (hybrid linear+full attention) — static-shape loop-free decode, fused int8 Metal kernels + GPU argmax head, q16 chunked prefill, host-managed KV + SSM conv/rec state | **GPU 42.5–45.4 tok/s** decode · **147 tok/s** prefill (int8 kernels, ctx 2048; `QWEN_KIND=fp16` selects the previous fp16 path, 27.7) |

Measured numbers, bundle sizes, and per-config caveats live in the zoo cards:
[`zoo/gemma4-e2b.md`](../zoo/gemma4-e2b.md) · [`zoo/qwen3.5.md`](../zoo/qwen3.5.md).

## Model delivery

On first launch each app offers an **in-app download** of the published `.aimodel` set from the
Hugging Face repos (editable URL field, defaults to the zoo's repos). Files stream into a staging
directory and a bundle is renamed into place only when ALL of its files are complete — Core AI's
specialization cache is content-keyed, and a partially-present bundle poisons it
([`knowledge/swift-runtime.md`](../knowledge/swift-runtime.md)). Weights are NOT bundled into the
.app (GB-class; Apple's guidance for large models is download-then-specialize off the interactive
flow — WWDC26 session 326). For development iteration you can still sideload with
`xcrun devicectl device copy to --domain-type appDataContainer …` — the apps check the same
locations. Shared implementation: [`AppShared/ModelDownloader.swift`](AppShared/ModelDownloader.swift).

## Build

Requires the **Xcode 27 beta** and [xcodegen](https://github.com/yonaskolb/XcodeGen).

```bash
# 1. The apps build against Apple's `CoreAIShared` Swift library (AIModel / InferenceFunction /
#    NDArray) and, for CoreAIChat's Qwen ⚡pipelined mode, the `CoreAILM` engine stack.
#    Clone coreai-models AT THIS REPO'S ROOT and apply both patches (CoreAIShared product
#    export + pipelined-engine extra states for the SSM conv/rec caches):
git clone https://github.com/apple/coreai-models
git -C coreai-models apply ../apps/coreai-shared-product.patch \
                           ../apps/coreai-pipelined-extra-states.patch

# 2. tokenizer.json is not committed (tens of MB). Fetch it from the upstream model repo into
#    the app's Resources/tokenizer/ (tokenizer_config + chat template are already there):
#      CoreAIChat:   https://huggingface.co/google/gemma-4-E2B-it  (accept the Gemma terms)
#      QwenChatFast: https://huggingface.co/Qwen/Qwen3.5-0.8B

# 3. Generate + build (set DEVELOPMENT_TEAM in project.yml, or pick a team in Xcode > Signing):
export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer
cd apps/CoreAIChat            # or apps/QwenChatFast
xcodegen generate
xcodebuild -project CoreAIChat.xcodeproj -scheme CoreAIChat -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> \
  build/Build/Products/Release-iphoneos/CoreAIChat.app
```

⚠️ Benchmark **Release** builds only — Debug-build tok/s under-reads by 2–3× (host-side work
dominates in Debug; e.g. 10.3 vs 30.4 tok/s on the same artifact).

## Engine notes

- The engines are plain Swift over `CoreAIShared`: gemma = `Gemma4ChatEngine` (GPU
  metal-kernel monolith / ANE 6-chunk host-cache backends behind a `GemmaMode` picker; mode
  switch frees one model set before loading the other), qwen = `FastEngine` (static-shape
  single-step graph; the 4 state arrays are host-managed, and NDArray views are function-local
  `inout` — class-property state trips MutableViews lifetime checks).
- CoreAIChat's **Qwen ⚡ mode** is different: `QwenPipelinedBackend` hands the whole generation
  to Apple's `coreai-pipelined` engine (`EngineFactory` over the `gpu-pipelined/` LanguageBundle
  — async non-blocking encode, on-GPU argmax, on-device KV growth), so it consumes a token
  stream instead of stepping per token. Contract for the S=1 bundle: set
  `COREAI_CHUNK_THRESHOLD=1` before engine creation (prefill = pipelined S=1 steps) and never
  call `engine.warmup()` (it warms query length 256, which the static `[1,1]` graph rejects — a
  1-token generate after load is the warmup).
- Headless device-probe hooks ride env vars (`GEMMA_*` / `QWEN_*` — see `CoreAIChatApp.swift` and
  the engine sources); launching from the home screen uses the published release configuration.
- Engine-first workflow: validate the graph on the Mac CLI first (`swift run coreai-run`, raw
  token ids in/out vs a conversion oracle), UI second — see [`../swift/`](../swift/).
- The conversion side (how the `.aimodel` bundles are produced) is
  [`../conversion/`](../conversion/); the gotchas it hit are in [`../knowledge/`](../knowledge/).
