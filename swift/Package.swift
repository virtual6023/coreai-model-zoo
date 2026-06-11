// swift-tools-version: 6.0
// Two libraries:
//
// * `CoreAIRunner` (DRAFT) — self-contained N-state engine on the low-level `CoreAI`
//   system framework only. ⚠️ authored on macOS 26.6, compile/verify on macOS 27 + Xcode 27.
// * `ZooFMProvider` — zoo LanguageBundles behind FoundationModels' `LanguageModelSession`
//   with tool calling + streaming. Depends on Apple's `coreai-models` package (product
//   `CoreAILM`): clone it AT THIS REPO'S ROOT and apply the patch stack, exactly like
//   `apps/` (see apps/README.md step 1). Verified on macOS 27 beta + Xcode 27 beta.
//
// export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer; swift build -c release
import PackageDescription

let package = Package(
    name: "CoreAIRunner",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "CoreAIRunner", targets: ["CoreAIRunner"]),
        .library(name: "ZooFMProvider", targets: ["ZooFMProvider"]),
        .executable(name: "coreai-run", targets: ["coreai-run"]),
        .executable(name: "zoo-fm-gate", targets: ["zoo-fm-gate"]),
    ],
    dependencies: [
        // The coreai-models clone at the repo root (../ relative to swift/).
        // Patch stack required for hybrid/SSM/per-layer-embedding bundles — same
        // prerequisite as apps/ (apply all four apps/*.patch in order).
        .package(path: "../coreai-models")
    ],
    targets: [
        .target(
            name: "CoreAIRunner",
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "coreai-run",
            dependencies: ["CoreAIRunner"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(
            name: "ZooFMProvider",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models")
            ]
        ),
        .executableTarget(
            name: "zoo-fm-gate",
            dependencies: [
                "ZooFMProvider",
                .product(name: "CoreAILM", package: "coreai-models"),
            ]
        ),
    ]
)
