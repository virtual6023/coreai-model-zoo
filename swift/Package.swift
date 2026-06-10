// swift-tools-version: 6.0
// ⚠️ DRAFT — authored on macOS 26.6, compile/verify on macOS 27 + Xcode 27
// (export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer; swift build).
// `CoreAI` is a system framework in the iOS 27 / macOS 27 SDK.
import PackageDescription

let package = Package(
    name: "CoreAIRunner",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "CoreAIRunner", targets: ["CoreAIRunner"]),
        .executable(name: "coreai-run", targets: ["coreai-run"]),
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
    ]
)
