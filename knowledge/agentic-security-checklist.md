# Agentic security: a pre-ship checklist for local-LLM agent apps

Hard-won notes for shipping an **on-device LLM agent** (FM framework / App Intents) safely.
Distilled from WWDC26 **347 "Secure your app: mitigate risks to agentic features"** (verbatim:
`ondevice/_wwdc347_transcript.txt`) and the confirmation/ownership APIs in **343 "Explore advanced
App Intents features"** (`ondevice/_wwdc343_transcript.txt`). Written for the Vault app (Wave 2) but
applies to any zoo model put behind `LanguageModelSession` / Siri.

> ⚠️ API names below are as **spoken in the talks** — captions don't capture on-screen code, so confirm
> exact spelling/signatures (`.onToolCall`, `.historyTransform`, `authenticationPolicy`,
> `OwnershipProvidingEntity`, `IntentDonationManager`) against the developer docs before relying on them.
> The **concepts** are verbatim-confirmed; the **Swift symbols** are best-effort transcriptions.

---

## 1. The one sentence that matters

> Indirect prompt injection is an **unsolved research problem**. Apple's own framing (347): "our best
> approach at the moment is to understand how much your app is at risk, and aim to mitigate that risk."

So this is not a "turn on a flag and you're safe" page. It is a **threat-modeling exercise** plus a
catalog of **deterministic** mitigations you bolt onto specific points in the agent loop.

## 2. The risk model (347)

**Indirect prompt injection** = instructions embedded in *extra context* (not the user's prompt) that
redirect the agent's control flow. The context can arrive in the initial prompt **or in a tool result**.

Two distinct effects once an injection lands:
- **Data poisoning** — attacker influences the *parameters* of an action you were going to run anyway
  (user says "message Mom"; injection rewrites the recipient to the attacker).
- **Action poisoning** — attacker influences *which* action runs (user says "summarize this email";
  injection opens a URL with the email body appended → exfiltration).

**The Lethal Trifecta** (Simon Willison, cited verbatim in 347). Maximum danger when an agent
simultaneously has all three:
1. access to **private data**,
2. exposure to **untrusted content**,
3. the ability to **externally communicate** — generalize this to *any action with a side effect*.

> **Vault is born inside the Lethal Trifecta.** Its whole pitch is (1) your private notes/files/photos,
> and it will have (2) untrusted content (a note you saved from a web clip, a shared file, an email)
> and (3) side-effecting tools (send, post, delete, order). Treat the trifecta as Vault's **default
> state**, not an edge case. The design lever is to *break one leg per risky flow* (e.g. redact private
> data before it reaches a tool that can communicate; or gate the communicating tool behind confirmation).

Out of scope for 347 (and for this page): model *safety* (is the output itself harmful) and *guardrail
circumvention*. This is about an **external attacker** subverting your app.

## 3. The threat-modeling exercise (do this once per feature)

### Step A — Data-flow analysis of the prompt
List every source that feeds prompt construction. For Vault that's: system **instructions**, the
**user prompt**, and **extra context** — retrieved notes, file contents, calendar events, a "friend
feed"/shared items, **and every tool result**.

### Step B — Mark what is untrusted
Rule of thumb (347): **anything from an external entity is attack surface.** A calendar invite a
stranger sent you, a post on a feed, a shared document, the body of an inbound email, OCR text from an
image — all untrusted. Your own first-party UI input is the only thing that starts trusted.

### Step C — Enumerate actions and their side-effect risk
For each tool/intent, name the worst case:

| Side-effect class | Example (347) | Vault analogue |
|---|---|---|
| **Financial** | `OrderTeaTool` (lose money) | any purchase/booking tool |
| **Data exfiltration** | `PostAndFetchPublicFeedTool` (leak via public post) | share / send / export / "post" |
| **Data loss** | `DeletePhoto` (no undo) | delete note / file / event |
| **Stored / second-order** | `BrewingTimerIntent` *label* — injection writes instructions that a later "list timers" pulls back into context | any tool that **persists a model-controlled string** that is later re-read (tags, titles, notes, reminders) |

> The stored-injection row is the sneaky one. 347's `createTimer` example: a tool that looks harmless
> (no side effect) but takes an **optional `String` label** the *model* fills in. An injection sets the
> label to attacker text; a later "list timers" query reads it back → **context poisoning across turns**.
> **Vault must audit every place the model writes a string that is later read back into a prompt.**

## 4. Mitigations — deterministic first

347 is explicit: **prefer deterministic mitigations as the baseline** ("their security guarantees are
easier to audit and reason about"); use probabilistic ones as defense-in-depth, not as the only line.

### Prompt-level (cut the injection's fuel and flag it)
- **Redact PII / sensitive data before it reaches the LLM.** If it never enters the context, it can't be
  exfiltrated. (Deterministic.)
- **Spotlighting** — wrap untrusted spans in delimiter tags telling the model "this is untrusted data."
  **Probabilistic** (a crafted injection can negate it), but cheap and worth stacking; different models
  enforce it to different degrees.

### Action-level (gate the dangerous verbs)
- **User confirmation** before any side-effecting action.
- **Authentication / device-unlocked requirement** for high-risk actions — the agent may be reachable
  from the **lock screen** (Siri), so significant-risk actions must not run while locked.

## 5. The concrete APIs

### 5a. Foundation Models framework — lifecycle event modifiers (347)
Deterministic callbacks fired at fixed points in the session loop. Two that matter for security:

- **`.onToolCall`** — fires when the model emits a tool call, **before the executor runs it**. *If the
  callback throws, the tool never runs* and control returns to the loop. → **the single chokepoint for
  confirmations.** One `.onToolCall` that checks the tool name and calls your `confirmWithUser()` gives
  *full coverage of every tool call* from one place.
  ```
  // sketch from 347
  profile.onToolCall { call in
      guard call.toolName == "OrderTea" else { return }   // others run untouched
      guard await confirmWithUser(call) else { throw CancelledByUser() }  // throw == block
  }
  ```
- **`.historyTransform`** — fires *before the transcript is rendered to the model*, on every new user
  request **and every loop iteration**. Modifies the **tail** of the transcript. → the place to apply
  **spotlighting** (add delimiters to untrusted tool outputs) and **PII redaction** (swap sensitive spans
  for a placeholder). ⚠️ **Transforms are scoped to the current inference only** — not visible to the
  next call, so re-apply every iteration. For an expensive transform you want to persist, use the
  **`@SessionProperty`** annotation (stateful history transform).

The framework ships more modifiers and lets you package custom ones as reusable profile modifiers.

### 5b. App Intents — when the model is *Siri*, not your own loop (347 + 343)
When an App Intent adopts a schema it becomes a tool in Siri's toolbox, and **the model picks which
intent to call** — so injection can misuse it. The system gives you guardrails:

- **Risk-based, contextual confirmation** (347). A *Risk Evaluation* combines **static risk metadata** +
  **dynamic system state** → if high, Siri asks the user; decline blocks the intent.
  - **Risk metadata is auto-assigned when your intent adopts a schema** — you do nothing. The *schema*
    carries it (`deleteAssets` → destructive; exfiltrating/shared-content updates → risky). 343's nuance:
    Siri **assumes entities are private by default and may skip confirmation**; it confirms more for
    destructive actions and for content the user **shared/made public**.
  - **`OwnershipProvidingEntity`** (343, new) — conform shareable/publishable entities to it and keep the
    **ownership state** current (updated whenever the system requests the entity). This is how Siri knows
    "this event has attendees / this note is shared" and therefore *should* confirm. **Only add it to
    entities a user can actually share or make public.**
- **Lock-screen authentication** (347). Set **`authenticationPolicy = .requiresAuthentication`** on a
  custom intent so it won't run locked. A schema has its **own default policy** (by sensitivity), auto-
  assigned to your intent; you may **override only to make it stricter** — a weaker override is a *build
  error* that tells you the minimum.
- **Interaction-donation hygiene** (343). Donations (`IntentDonationManager`) teach Siri your app's usage
  — but **"if your app donates excessively, the system may ignore those donations."** Donate **real user
  UI actions only**, not synthetic/bulk events. (Donating from the *agent* loop would both pollute Siri's
  learning and risk laundering injected actions into "learned" behavior — don't.)
- **Build-time design hints** (240). Adopting `sendMessage` without `draftMessage` is a *build error* —
  Apple forces the draft/confirm path for messaging. Read these errors as security guidance, not noise.

## 6. Pre-ship checklist (tick every box before TestFlight)

**Threat model**
- [ ] Listed every prompt data source; marked each trusted/untrusted (external entity ⇒ untrusted).
- [ ] Classified every tool/intent by worst-case side effect (financial / exfil / data-loss / stored).
- [ ] Audited every tool that **persists a model-written string later re-read** (stored injection).

**Prompt-level (deterministic baseline)**
- [ ] PII/sensitive data **redacted before** entering the context (via `.historyTransform`/server-side).
- [ ] Untrusted spans **spotlighted** with delimiters (defense-in-depth, re-applied each iteration).

**Action-level**
- [ ] Every side-effecting tool routes through **one `.onToolCall` confirmation** chokepoint (FM loop) —
      *throw to block*.
- [ ] Destructive/exfil/financial actions: **confirmation required**, and **never run on a locked device**
      (`authenticationPolicy = .requiresAuthentication`, or stricter schema default left in place).
- [ ] Shareable entities conform to **`OwnershipProvidingEntity`** with live ownership state.
- [ ] No `git add -A`-style "donate everything" — donations reflect real UI actions only.

**Verification (close the loop with Evaluations — see `evaluations-framework.md`)**
- [ ] A **regression eval feeds poisoned context** and asserts (via a `disallowed` **TrajectoryExpectation**,
      299) that destructive tools are **not** called and parameters are **not** rewritten. This is the one
      deterministic test that turns "we mitigated injection" into a number you can hill-climb.

## 7. Vault design implications (concrete)

1. **Default-deny side effects.** Vault's read/ask path (Spotlight RAG, embeddings, VL) is low-risk and can
   run freely. Every *write/send/delete/order* goes through a single `.onToolCall` gate with confirmation.
   Build it as **one reusable profile modifier** so all current and future tools inherit it.
2. **A redaction `.historyTransform` on the retrieval path.** Vault reads the user's private corpus into
   context constantly — strip secrets (tokens, full card numbers, addresses) *before* the model sees them,
   so a "summarize and share" can't leak what was never present. Re-apply per iteration; cache with
   `@SessionProperty` if it's expensive.
3. **Spotlight untrusted corpus items.** Notes clipped from the web, shared files, inbound mail = untrusted.
   Tag them in `.historyTransform`; keep first-party typed input untagged.
4. **Lock-screen posture.** If Vault exposes Siri intents ("ask Vault…"), mark anything that exports/deletes
   `requiresAuthentication`. The "ask my notes a question" intent can stay lock-screen-friendly; "share this
   note" cannot.
5. **The PCC/cloud profile is a privacy boundary, made visible.** Sending the user's private corpus to PCC
   (or any server model) is *external communication* = trifecta leg #3. Keep it an **explicit opt-in profile**
   (DynamicProfile), show it in the UI, and prefer redaction before any off-device hop. (Aligns with the
   track's Value 4: "privacy by architecture.")
6. **Ship the injection regression eval with the app.** Per Value 1 ("measure before you claim"), Vault's
   Evaluations suite should include a poisoned-context trajectory test as a permanent gate, not a one-off.

## 8. Pointers
- Verbatim transcripts: `ondevice/_wwdc347_transcript.txt`, `ondevice/_wwdc343_transcript.txt`.
- Tool-call/trajectory verification & the poisoned-context test pattern: `evaluations-framework.md`.
- FM provider plumbing (where Vault's model sits): `fm-provider.md`.
- 347 explicitly defers model-safety/guardrail topics to a separate "model safety" talk.
