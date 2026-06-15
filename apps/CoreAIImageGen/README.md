# CoreAIImageGen

A minimal **image-generation** app for Core AI diffusion bundles running on Apple's
official `CoreAIDiffusionPipeline` runtime
([apple/coreai-models](https://github.com/apple/coreai-models)). One SwiftUI codebase,
two targets:

| Target | Platform | Bundle | Notes |
|---|---|---|---|
| `CoreAIImageGen` | iOS 27 | 512 / half-VAE (~4 GB int4) | iPhone 17 Pro (12 GB), `increased-memory-limit` entitlement |
| `CoreAIImageGenMac` | macOS 27 | 1024 full | desktop split-view UI |

Default model: **[FLUX.2 klein 4B](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)**
(4-step distilled, guidance 1.0). The pipeline type (FLUX.2 / SD3 / SD) is auto-detected
from the bundle's `metadata.json`, mirroring the zoo's `diffusion-runner` reference tool —
so any `coreai.diffusion.export` bundle drops in via **Local…**.

## Build & run

```bash
brew install xcodegen
cd apps/CoreAIImageGen
xcodegen generate
# macOS:
open CoreAIImageGen.xcodeproj          # run the CoreAIImageGenMac scheme (Release)
# iOS (device):
xcodebuild -project CoreAIImageGen.xcodeproj -scheme CoreAIImageGen -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build-ios \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> \
  build-ios/Build/Products/Release-iphoneos/CoreAIImageGen.app
```

Unlike the LLM apps, this one needs **no `coreai-models` patch stack** — the diffusion
runtime runs unmodified. `project.yml` pulls `apple/coreai-models` straight from GitHub
(pinned to the metadata.json-v0.2 revision) rather than the patched repo-root clone.

## Model delivery

First launch offers an **in-app download** of the published bundle from Hugging Face.
The `.aimodel` directory bundles + tokenizer stream via the shared
[`AppShared/ModelDownloader`](../AppShared/ModelDownloader.swift) (range-chunked parallel
download, cross-launch resume, atomic placement — a partial bundle never poisons the
content-keyed coreai-cache); the few tiny root files (`metadata.json`, `vae_bn_*.npy`)
come down with a plain resolve GET (the HF tree API only enumerates directories). Only the
subset the platform needs is fetched (iOS: `Transformer_512` / `*_half` VAEs). The screen
is kept awake during the multi-GB transfer so an auto-lock can't suspend it mid-download;
for a big set, stay on Wi-Fi and keep the app foregrounded.

## Notes

- Generation runs Apple's `CoreAIDiffusionPipeline` (`Flux2Pipeline` / `SD3Pipeline` /
  `StableDiffusionPipeline`) — there is **no model-code port**: image generation is already
  supported by the stock Core AI stack.
- FLUX.2 / SD3 use the `discreteFlow` scheduler; classic SD uses `dpmSolverMultistep`.
- Reference timings (diffusion-runner): macOS 1024 ≈ 17.4 s, iOS 512/half ≈ 6.55 s
  @ 4 steps. See the model card for details.
