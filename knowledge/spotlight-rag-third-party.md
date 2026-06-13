# Local RAG with `SpotlightSearchTool` behind a third-party model

> Verified 2026-06-13 (macOS 27 beta, M4 Max). WWDC26 246 "LLM search using Core Spotlight".
> Apple's `SpotlightSearchTool` turns the Core Spotlight index into a retrieval tool for a
> `LanguageModelSession`. It is a plain `FoundationModels.Tool`, so it works behind ANY
> `LanguageModel` — we ran it behind a Core AI zoo bundle (`KitLanguageModel`), not the system
> model. Full round trip (model writes a query → Spotlight searches → grounded answer) passes
> on the zoo qwen3.5-0.8B and on qwen3-4B. Working example: `coreai-kit/Examples/SpotlightChat`.

## The API (cross-import overlay)

`SpotlightSearchTool` lives in the `_CoreSpotlight_FoundationModels` overlay — it materializes
when a file imports BOTH `CoreSpotlight` and `FoundationModels`. Shape:

```swift
import CoreSpotlight
import FoundationModels

let tool = SpotlightSearchTool(configuration: .init(
    sources: [.coreSpotlight(CoreSpotlightSource(searchableIndexDelegate: delegate))],
    guide: SpotlightSearchTool.Guide(level: .focused(.items), format: .compact),
    contactResolver: nil,
    customStages: []))

// Behind YOUR model instead of the system one:
let session = LanguageModelSession(model: kitModel, tools: [tool], instructions: …)
let answer = try await session.respond(to: "What did I write about the night hike?")
```

- `Configuration.sources`: `.coreSpotlight` (your app's index) and/or `.files` (indexed files).
- `Guide.level`: `.complete` | `.focused(ContentDomain = .items)` | `.dynamic(GuidanceProfile)`.
  `.format`: `.structured` | `.compact`.
- `GuidanceProfile(textMatch:similarityMatch:numericMatch:dates:people:contentType:attributes:)`.
- `tool.searchResults` is an `AsyncSequence<SearchReply, Never>` — observe results live
  (items/scoredItems/groupedItems/count/table/statistic/text + label + queryToken + status).
- `CustomStage: Generable & Codable & Sendable` — pipeline stages with `inputTypes`/`outputTypes`
  and `execute(items:/scoredItems:/count:/table:/text:…)`.

## Does it work behind a third-party model? YES.

The only capability required is **`.toolCalling`** — declared by `KitLanguageModel` for ChatML
tokenizers (qwen3 family). The tool's query `GenerationSchema` is rendered into the tool prompt;
the model emits a parseable tool call; the framework runs the tool and feeds results back.
**`.guidedGeneration` is NOT required** (the tool does not constrain decoding on the model side),
so this works on the GPU-pipelined engine that cannot expose logits. Transcript:

```
prompt → reasoning → toolCall spotlight_search({"searchTerms":["night hike"]})
       → toolOutput (items) → toolCall fetch_note({"id":"note-003"})
       → toolOutput (body) → grounded answer
```

## The central gotcha: the tool returns metadata, not the body

Even with `CoreSpotlightSource(fetchAttributes: [.title, .contentDescription, .keywords])`, the
toolOutput handed to the model carries only identity attributes — `uniqueIdentifier`, `title`,
`contentType`, `contentCreationDate`, `domainIdentifier`. **`contentDescription` and `keywords`
do not appear** (in `.compact` or `.structured`). This is not a Spotlight limitation: a raw
`CSSearchQuery` with the same `fetchAttributes` returns `contentDescription` (full body) fine
(`textContent` is index-only — write-only for full-text search, returns nil on read).

Consequence: a model answering from search results alone sees only TITLES and will hallucinate
bodies (the system model, asked about a night hike, invented "rained heavily / pack a waterproof
jacket"; the real note said the headlamp died — pack spare batteries).

## The working pattern: retrieve with Spotlight, hydrate with your own tool

Give the model a second plain `Tool` that reads the full content from your store by identifier:

```swift
struct FetchNoteTool: Tool {
    let name = "fetch_note"
    let description = "Read the full saved text of a note by its identifier."
    @Generable struct Arguments {
        @Guide(description: "The note id from spotlight_search, like note-002.") var id: String
    }
    func call(arguments: Arguments) async throws -> String { store[arguments.id] ?? "not found" }
}
let session = LanguageModelSession(model: kitModel, tools: [spotlightTool, FetchNoteTool()], …)
```

The model chains `spotlight_search` → ids/titles → `fetch_note(id)` → body → grounded answer.
This mirrors a real app (Spotlight index = lightweight finding aid; full content = your store)
and doubles as a multi-tool-orchestration demo on a third-party model. Verified on the system
model, zoo qwen3.5-0.8B, and qwen3-4B.

## Guidance level is a token gate

`.complete` guidance injects ~13 k tokens of tool instructions → instant `contextSizeExceeded`
on any 4 k-context model (system or zoo). Ship `.focused(.items)` + `format: .compact` for local
models. `.dynamic(GuidanceProfile)` was prompt-sensitive in testing (a model skipped the search
and hallucinated) — use deliberately.

## Model-choice constraints (Core AI / kit)

- Tool calling via the kit needs a ChatML tokenizer (`<|im_start|>`). In the public catalog that
  is qwen3-0.6b / qwen3-4b; mistral (`[INST]`) and gemma do not get `.toolCalling`.
- qwen3-0.6b is too small for the rich SpotlightSearchTool schema (loops on `<think>` → framework
  reports "ended without producing a response"). Use qwen3-4B or larger.
- qwen3 is a thinking model; with this big tool schema its reasoning can run to the token cap
  → intermittent "ended without producing a response" on the stock engine. Append **`/no_think`**
  to the instructions to disable qwen3 reasoning — the search→fetch chain then completes reliably
  (5/5 on stock qwen3-4B) and is ignored harmlessly by non-qwen models. (This is the D1
  EOS-overshoot interaction surfacing at the app level; the engine-side fix is the pipelined
  yield-check patch.)
- Hybrid zoo bundles (qwen3.5/3.6, LFM2.5, granite) need a `coreai-models` engine with hybrid
  KV-state support; the stock public engine asserts "Expected 2 states, got 4".

## CustomStage and the delegate, in this beta

- A `CustomStage` conforms and is accepted in `Configuration.customStages` (the session builds and
  the tool round trip still passes), but neither an `items→text` nor `items→scoredItems` stage
  was routed through by the 27.0-beta pipeline for our queries — including under
  SystemLanguageModel, so it is a tool/beta behavior, not a third-party-model limitation. Docs
  note stages "run independently" (isolated execution). Prefer the companion-tool hydration above.
- `CSSearchableIndexDelegate` conforms and wires via `CoreSpotlightSource(searchableIndexDelegate:)`;
  `searchableItems(forIdentifiers:)` (macOS 15.4+, with a new protectionClass overload in 27.0)
  is the index-recovery hydration API — not the search-time body path.
