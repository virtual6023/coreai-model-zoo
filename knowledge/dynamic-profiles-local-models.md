# DynamicProfile with two local models: all-on-device model routing

> Verified 2026-06-13 (macOS 27 beta, M-series Mac). WWDC26 242 "Build agentic app experiences".
> Companion demo (six gates, `swift run`): `agent-demos/DualProfileChat`.

## What this is

WWDC26 session 242 introduces `DynamicProfile`: inside a single `LanguageModelSession` you
declare multiple **profiles** (each a model + instructions + tools + modifiers) and switch
between them as the conversation moves. Apple's example routes between `SystemLanguageModel`
(on-device) and `PrivateCloudComputeLanguageModel` (server).

Because `DynamicProfile.model(_:)` takes `some LanguageModel`, and any Core AI zoo bundle is a
`LanguageModel` via coreai-kit's `KitLanguageModel`, the same API routes between **two local
models** — a fast 0.6B for triage and a 4B (or larger) for hard questions — with **no server,
no PCC, in airplane mode**. This is the configuration Apple's demo does not show. It works:
the demo routes between `qwen3-0.6b` and `qwen3-4b` over one session, on-device, end to end.

## The API surface (verified against the macOS 27.0 SDK)

```swift
struct RoutingProfile: LanguageModelSession.DynamicProfile {
    let router: Router            // your state: which profile is active
    let fast: KitLanguageModel
    let smart: KitLanguageModel

    var body: some LanguageModelSession.DynamicProfile {
        if router.route == .smart {
            Profile { Instructions("You are the expert.") }
                .model(smart).maximumResponseTokens(384)
        } else {
            Profile { Instructions("You are fast triage.") }
                .model(fast)
        }
    }
}
let session = LanguageModelSession(profile: RoutingProfile(...))
```

Modifiers on a profile: `.model`, `.temperature`, `.samplingMode`, `.maximumResponseTokens`,
`.reasoningLevel`, `.toolCallingMode`, `.historyTransform`, `.transcriptErrorHandlingPolicy`,
lifecycle `.onActivate/.onDeactivate/.onPrompt/.onResponse/.onToolCall/.onToolOutput`, and
`.modifier(_:)`. Shared state across tools and profiles uses `@SessionPropertyEntry` (custom)
or the built-in `history` property.

## Behaviors you must design around (measured)

1. **The `body` is re-evaluated multiple times per turn** (7 evaluations for 3 turns). The
   framework reads it more than once to gather instructions and resolve the model. **Keep the
   body pure** — read your route variable there, never mutate state. Imperative work goes in
   lifecycle modifiers (`onResponse`, …), which fire once at their boundary.

2. **Lifecycle order on a switch** is `old.onDeactivate → new.onActivate → onPrompt →
   onResponse`. First entry into a profile fires `onActivate` before `onPrompt`.

3. **Switching models re-prefills the shared transcript on the newly active engine.** Each
   model has its own executor and KV cache; on a switch the receiving model re-prefills the
   conversation before answering. Measured (0.6B↔4B): switch-in first-delta **2.35 s**
   (re-prefill ~106 tok + the 4B's reasoning), switch-back **0.94 s**. Append-only KV reuse
   only helps across consecutive *same-model* turns.

4. **Two resident models cost two footprints.** Routing keeps both loaded so the switch is
   instant. qwen3-0.6b + qwen3-4b: ~102 MB with both bundles loaded but un-touched, rising to
   **~920 MB `phys_footprint` after the turns run**. Note `phys_footprint` is the jetsam-relevant
   *dirty* number and excludes clean read-only-mmapped weight pages — these are 4-bit bundles, so
   total mapped RSS is higher (~2.4 GB+ of weights). The 86→920 MB growth is runtime KV /
   activation / Metal buffers, not weights paging in. Report both numbers, labeled, if footprint
   matters for your jetsam budget.

## Routing decision: use guided generation, not a tool

242's baton-pass flips the route from inside a **tool** the model calls. On the kit's upstream
engine that path is unreliable: small/thinking models emit tool-call JSON the framework
rejects with `GenerationError.decodingFailure` ("failed to parse generated content"),
independent of the argument schema (verified with required, optional, and empty `@Generable`
arguments). The reliable "the model decides" channel is **guided generation**:

```swift
@Generable struct RouterDecision {
    @Guide(description: "true if the request needs the deep/expert model…")
    var needsExpert: Bool
}
// One persistent session on the sequential engine:
let session = LanguageModelSession(model: routerModel)           // engineVariant: .sequential
let decision = try await session.respond(to: "Classify: \(q)", generating: RouterDecision.self)
router.set(decision.content.needsExpert ? .smart : .fast)
```

Guided generation runs on the **sequential** engine (one logits step per token): the output
can't leak the model's `<think>` reasoning and can't be malformed, and that engine has no
over-generation pump (see #4 below), so it's also free of the consecutive-turn KV hazard.

## The two 242 patterns, fully local

- **Baton-pass** — collaboration over one shared transcript. The router classifies; the
  matching `DynamicProfile` branch answers; the transcript is visible to both. (Tool-flipped
  baton-pass is what the kit can't do reliably; guided-classification routing is the
  equivalent that works.)
- **Phone-a-friend** — consultation. The app opens a **short-lived child
  `LanguageModelSession`** on the big model with an isolated transcript, takes its answer, and
  the parent (fast model) writes the final reply. The child's transcript never merges into the
  parent's (verified: parent transcript = 1 prompt / 1 response). Tool-spawned consultation
  hits the same decodingFailure, so the consult is an app step.

## Hard-won rules (kit + upstream engine)

- **One engine, one session, for the engine's lifetime.** Two `LanguageModelSession`s over the
  same `KitLanguageModel` corrupt the KV state (the second resets the engine under the first).
  A per-turn fresh classifier session is the classic way to trip this — reuse one router
  session instead.
- **Consecutive same-model plain-respond turns can crash** (D1 over-generation leaves post-EOS
  tokens in the cache; the next same-model turn's KV fast-path meets garbage). Avoid by
  alternating models, changing the instructions each turn (e.g. inject a summary → the prefix
  changes → clean reset), or using guided gen.
- **A thinking model cut mid-`<think>` → decodingFailure.** If the token budget is spent in
  reasoning with no response text, the framework rejects the empty turn. Keep prompts bounded
  and maxTokens generous.
- **Prefer a single-profile `DynamicProfile` over `LanguageModelSession(model:instructions:)`**
  — the plain initializer's first respond can decodingFailure where the profile path is solid.
- **Model choice matters**: qwen3.5 is a hybrid (GDN, 4 KV states) the upstream engine won't
  load; VL *decode* bundles declare 4 per-token inputs and won't load either (true vision
  routing needs an own executor + the vision encoder). Use plain Qwen3 catalog bundles.

## Reproduce

```bash
cd agent-demos/DualProfileChat
swift run -c release DualProfileChat all      # g2–g6 (g1 needs Apple Intelligence)
# or a single gate, with your own bundles:
swift run -c release DualProfileChat g3 \
  --fast catalog:mlboydaisuke/qwen3-0.6b-CoreAI-official \
  --smart catalog:mlboydaisuke/qwen3-4b-CoreAI-official
```
