# Zoo models behind Foundation Models' `LanguageModelSession`

> Verified 2026-06-11 (macOS 27 beta, M4 Max): a zoo pipelined bundle backs Apple's standard
> `LanguageModelSession` **with zero new code** — `CoreAILanguageModel(resourcesAt: bundleDir)`
> is the entire integration (Qwen3.5-0.8B int8: load 3.7 s, first turn 0.41 s, multi-turn OK).
> The one capability Apple's adapter lacks is **tool calling**; a ~200-line own `LanguageModel`
> conformance added it, and the full round trip — model emits a call, the framework runs the
> Swift `Tool`, the model answers grounded on the result — worked on the first run, hybrid
> 4-state Qwen included. Source session: WWDC26 339 "Bring an LLM provider to the Foundation
> Models framework". Apple's adapter source ships in `coreai-models`
> (`swift/Sources/CoreAILanguageModels/LanguageModel/`).

Why this matters: every app written against the Foundation Models framework — the same
`session.respond(to:)` / `streamResponse` / `Tool` / `@Generable` API as Apple's built-in model
— can switch to a zoo model by changing one line. Anthropic and Google announced FM provider
packages for Claude/Gemini; MLX has `MLXLanguageModel` (`ml-explore/mlx-swift-lm`); Hugging Face
ships `AnyLanguageModel`. Local Core AI models join that ecosystem through the path below.

## The protocol in 60 seconds (macOS/iOS 27)

Two pieces (verified against the 27-beta `FoundationModels.swiftinterface`):

```swift
protocol LanguageModel: Sendable {
    associatedtype Executor: LanguageModelExecutor where Self == Executor.Model
    var capabilities: LanguageModelCapabilities { get }   // .vision/.guidedGeneration/.reasoning/.toolCalling
    var executorConfiguration: Executor.Configuration { get }
}
protocol LanguageModelExecutor: Sendable {
    associatedtype Configuration: Hashable, Sendable      // per-session executor cache KEY
    init(configuration: Configuration) throws
    func prewarm(model: Model, transcript: Transcript)    // careful: default no-op exists
    nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: Model,
        streamingInto channel: LanguageModelExecutorGenerationChannel) async throws
}
```

The session hands the executor the **full transcript on every `respond`** (entries:
`instructions / prompt / toolCalls / toolOutput / response / reasoning`), plus
`enabledToolDefinitions`, an optional `schema`, and `generationOptions`. The executor streams
events back: `.response(action: .appendText(...))`, `.reasoning(...)`,
`.toolCalls(action: .toolCall(id:name:action: .appendArguments(json)))`, `.updateUsage`,
`.updateMetadata`. One-shot `respond` is just collected streaming. KV reuse across turns is
the executor's job (diff the new transcript against the one you saved; invalidate at the
divergence point) — nobody does it for you.

## Quick start — Apple's adapter, zoo bundle

```swift
import CoreAILanguageModels   // product "CoreAILM" of the coreai-models package
import FoundationModels

setenv("COREAI_CHUNK_THRESHOLD", "1", 1)   // BEFORE engine creation (decode-only S=1 bundles)

let model = try await CoreAILanguageModel(resourcesAt: bundleDirURL)  // LanguageBundle dir
let session = LanguageModelSession(model: model, instructions: "You are a helpful assistant.")
let answer = try await session.respond(to: "Why is the sky blue?")
```

Requirements:

- A **LanguageBundle dir** (`metadata.json` + `.aimodel` + `tokenizer/`) — every bundle this zoo
  publishes is one.
- The **patched `coreai-models` package** for the non-standard architectures: the same
  extra-states / per-token-inputs / static-inputs patch stack the chat app already needs (see
  [`pipelined-engine.md`](pipelined-engine.md) and `apps/*.patch`). Plain-attention bundles run
  on the unpatched upstream; Qwen3.5 (hybrid GDN), LFM2.5, Granite (SSM), Gemma 4 tbl do not.
- `EngineFactory` auto-picks the engine from the bundle structure — pipelined for dynamic GPU
  bundles, sequential, or static-shape/ANE.

What you get for free from Apple's adapter: UTF-8-safe incremental detokenization, `<think>` /
`<|reasoning_start|>` auto-detection routed to `.reasoning` transcript entries, chat templating
via the bundle tokenizer, greedy default + temperature override.

## What works / what doesn't (today's beta, verified)

| Surface | Status |
|---|---|
| Plain chat, streaming, multi-turn | ✅ via Apple's adapter, zero code |
| Reasoning models (`<think>`) | ✅ routed to `Transcript.reasoning` entries automatically |
| Tool calling | ❌ in Apple's adapter (tool entries skipped, capability never declared) → ✅ with the own conformance below |
| Guided generation (`@Generable`, schema) | ⚠️ only when `engine.supportsLogits` — **GPU-pipelined engines sample on-GPU and return `false`**, so every zoo pipelined bundle lacks `.guidedGeneration`; the sequential engine has it |
| `session.prewarm()` | ❌ silent no-op for Core AI models (see trap 1) |
| Usage accounting (`.updateUsage`) | ❌ not emitted yet (placeholder in Apple's adapter) |
| KV reuse across turns | ❌ nobody implements it: every `respond` resets the engine and re-prefills the whole transcript — at S=1 ≈ decode speed on decode-only bundles, so multi-turn latency grows linearly with history |

## Tool calling with an own conformance

Apple's adapter doesn't do tools, but the protocol + the zoo engines have everything needed.
A minimal conformance that reuses the same public pieces (`CoreAIRunner(from:).makeInferenceEngine()`
+ `LanguageBundle.loadTokenizer()`) and speaks the Qwen/Hermes tool dialect:

1. Declare `capabilities = [.toolCalling]`.
2. In `respond`, render the transcript to ChatML yourself. Advertise
   `request.enabledToolDefinitions` in the system message inside `<tools>…</tools>` (each as
   `{"type":"function","function":{name, description, parameters: <JSONEncoder'd
   GenerationSchema>}}`); replay past `toolCalls` entries as assistant
   `<tool_call>{json}</tool_call>` turns and `toolOutput` entries as user-role
   `<tool_response>…</tool_response>` turns.
3. Generate (`engine.generate(with:samplingConfiguration:inferenceOptions:)`, break on
   `tokenizer.eosTokenId`), split a leading `<think>` block into a `.reasoning` event, then
   if the output contains `<tool_call>` parse `{"name", "arguments"}` and send
   `.toolCalls(action: .toolCall(id: UUID().uuidString, name: name,
   action: .appendArguments(argsJSON, tokenCount: n)))`; otherwise send the text as
   `.response`.
4. That's all — the **framework** parses the arguments against the tool's `@Generable` schema,
   executes the Swift `Tool`, appends the `toolOutput` entry, and calls `respond` again; your
   executor replays it and the model answers grounded.

Verified transcript on Qwen3.5-0.8B int8 (greedy, one shot):

```
instructions → prompt → reasoning → toolCall get_weather({"city":"Tokyo"})
             → toolOutput get_weather → response   (turn 4.6 s incl. both respond calls)
```

A packaged SPM target for this conformance (streaming `<tool_call>` parse, multi-call,
usage events) is planned; until then the recipe above is complete.

## Traps

1. **`prewarm` has a default no-op extension.** Implement `prewarm(model:transcript:)`
   *exactly* — implement `prewarm(transcript:)` and it compiles but is never called. Apple's
   own adapter has this today, which is why `session.prewarm()` does nothing for Core AI
   models: do your own warm-up (a 1-token generate after load).
2. **`request.enabledToolDefinitions`** is the property; `enabledTools` is only the
   memberwise-init label.
3. **`Configuration` is the executor cache key.** The session stores executors keyed by your
   Hashable `Configuration` — key it by bundle identity (+ anything that changes behavior).
   Apple keys by `(modelIdentifier, samplingConfig)`.
4. **`COREAI_CHUNK_THRESHOLD=1`** before engine creation for decode-only S=1 bundles, and
   never call `engine.warmup()` with the default query length on them (warms S=256, which the
   S=1 graph rejects) — same contract as [`pipelined-engine.md`](pipelined-engine.md).
5. **Pipelined ⇒ no `.guidedGeneration`.** Don't declare it without logits; schema requests
   on a pipelined bundle can't be honored (approximate-or-throw rule: throw
   `LanguageModelError.unsupportedCapability`).
6. **Multi-turn re-prefill tax.** Until an executor implements transcript diffing, budget
   ~decode-speed × history-tokens per turn on decode-only bundles (measured: turn 1 = 0.41 s,
   turn 2 = 2.8 s on the 0.8B with a 3-entry history + hidden thinking).
7. **Thinking is invisible in `response.content`** — it lands as `.reasoning` transcript
   entries. A "hanging" first response is usually the model thinking.

## Sources

WWDC26 339 "Bring an LLM provider to the Foundation Models framework" (protocol, executor
lifecycle, transcript-diff guidance, custom segments/metadata) and 241 "What's new in the
Foundation Models framework"; the `FoundationModels.swiftmodule` interface in the macOS 27
beta SDK (signatures above were read from it, not from docs); Apple's adapter source in
`coreai-models` `swift/Sources/CoreAILanguageModels/LanguageModel/CoreAILanguageModel.swift`.
