# Your own model behind system Visual Intelligence (WWDC26)

> Verified against the 27 beta SDK swiftinterface, 2026-06-13. WWDC26 297 "Best practices for
> integrating visual intelligence in your app." Working example: `coreai-kit/Examples/VisualIntel`.
> Device surfacing (camera/screenshot → system UI) is USER device-pending; the API surface and
> the engine path are SDK- and Mac-verified.

Visual Intelligence (camera on iOS, screenshots on iPad/Mac) lets the system query your app and
show your results in its own UI — with your app closed. Apple's samples compute similarity with
Vision's `GenerateImageFeaturePrintRequest`. You can replace that with **any model you want**:
the integration is pure App Intents and never inspects what produced the results. We ran our own
converted **CLIP** (feature-print similarity) and **RF-DETR** (object detection) Core AI bundles
behind it.

## The integration is model-agnostic by construction

The whole surface is three App Intents pieces in the **main app target** (no extension):

```swift
import AppIntents
import VisualIntelligence

// 1. The query the system calls with the captured pixels.
struct VisualSearchValueQuery: IntentValueQuery {
    @Dependency var engine: VisualIntelEngine                 // your models
    func values(for input: SemanticContentDescriptor) async throws -> [VisualSearchResult] {
        guard let pb = input.pixelBuffer, let cg = cgImage(from: pb) else { return [] }
        return try await engine.analyze(cg)                   // RF-DETR + CLIP → entities
    }
}

// 2. An OpenIntent PER entity type — REQUIRED, or the app never surfaces in VI.
struct OpenDetectedObjectIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open Detected Object"
    @Parameter(title: "Object") var target: DetectedObjectEntity
    func perform() async throws -> some IntentResult { /* route in app */ .result() }
}

// 3. Optional "Continue in app".
@AppIntent(schema: .visualIntelligence.semanticContentSearch)
struct ContinueVisualSearchInAppIntent { var semanticContent: SemanticContentDescriptor; /* … */ }
```

`SemanticContentDescriptor` (in `VisualIntelligence`) carries `labels: [String]` and
`pixelBuffer: CVReadOnlyPixelBuffer?`. There is **no model parameter and no capability** anywhere
— so whatever you run inside `values(for:)` is invisible to the system. The OS discovers your app
from **App Intents metadata extracted at build time**; no Info.plist key or entitlement is needed
for the visual-search participation itself. (Contrast SpotlightSearchTool, which at least needs a
`.toolCalling` model — Visual Intelligence has *no* model gate at all.)

```swift
let cg = pixelBuffer.withUnsafeBuffer { (pb: CVPixelBuffer) -> CGImage? in
    var out: CGImage?; _ = VTCreateCGImageFromCVPixelBuffer(pb, options: nil, imageOut: &out); return out
}
```

## The real gate: running a model in the query's execution context

Surfacing is free; the engineering risk is that **the query runs in a background launch of your
app** (App Intents), with a tighter memory budget than the foreground. Putting a CV model there:

- Use the **main app target, not an App Intents extension** — a background app launch gets the
  app's memory budget (extensions are too tight for a model) and shares the app container.
- **Precompute** everything you can. Build your feature-print index (embeddings + cached display
  thumbnails) at foreground time; in the query, only encode the single incoming frame and read
  cached thumbnails — never touch PhotoKit or rebuild an index out-of-process.
- Default to your **lightest** model (we use RF-DETR nano, 103 MB, one forward pass).
- Return only `Sendable` values from your engine (JPEG `Data`, not `CGImage`) so results travel
  cleanly into the App Intents types.

## Returning more than one kind of result

`@UnionValue` (iOS 27 / macOS 27) lets one query vend multiple entity types — detections *and*
similar photos:

```swift
@UnionValue enum VisualSearchResult {
    case detectedObject(DetectedObjectEntity)   // RF-DETR
    case photo(PhotoMatchEntity)                // CLIP nearest neighbors
}
```

Each `AppEntity` provides a `DisplayRepresentation(title:subtitle:image:)` — keep it to a few
lines + a thumbnail. Each needs an `OpenIntent`. Platforms share code: iOS adds a camera entry
point, iPad/Mac use screenshots (and deliver much larger pixel buffers — resize before the model).
