# Apple's Evaluations framework, mapped to the zoo's oracle/margin gates

How Apple's **Evaluations framework** (WWDC26 **298** Meet / **299** agentic / **335** hill-climbing;
verbatim in `ondevice/_wwdc{298,299,335}_transcript.txt`) lines up with the quality-gate discipline this
project already runs at the numeric layer (oracle references, the margin rule, flip budgets, device-
verified benches). The goal of this page: **express our methods in Apple's vocabulary** so the Vault app
(Wave 2) ships with an Evaluations suite that reads as native to the framework, while reusing everything we
already believe about measuring before claiming (Value 1).

> ⚠️ Symbols (`subject(from:)`, `ModelSample`, `Metric`, `Evaluator`, `ModelJudgeEvaluator`,
> `ScoreDimension`, `TrajectoryExpectation`, `ToolCallEvaluator`, `SampleGenerator`) are transcribed from
> talks — captions don't show code. Concepts are verbatim; confirm exact signatures in the docs.
> Framework is **new in Xcode 27**, runs in **Swift Testing**, supports **macOS/iOS/watchOS/visionOS**, and
> can run **on-device or against PCC**.

---

## 1. The shared premise (we and Apple independently arrived here)

298, verbatim: generative models "break a contract fundamental to software testing" — *the same input can
produce different outputs*, so "unit tests are insufficient." That is **exactly** why this project never
gated on raw string/tensor equality but on an **oracle compared within a margin**: at the fp16 ceiling the
reference itself flips on ~1e-4 noise, so an equality assert is meaningless. Apple's answer to the same
problem is **scoring + aggregate thresholds**; ours is the **margin rule + flip budget**. Same insight,
two altitudes. This page connects them.

## 2. Framework API in one screen

- **`subject(from:)`** — runs the thing under test for one sample, returns its output (the "subject").
- **`ModelSample`** — one input, plus an **expected** output (e.g. `expectedTags`). The dataset is `[ModelSample]`.
- **`Evaluator`** — a per-sample closure over the subject's output → emits a **`Metric`** (pass/fail *or* a
  numeric score). Quantitative.
- **`aggregateMetrics(using:)`** — roll per-sample metrics into trends (mean, ratio, custom — e.g. Cohen's κ).
- **Run it in a test** — `@Test(.evaluates(MyEval(), notes:))`; pull the results bundle; assert
  `#expect(results.aggregateValue(...) >= target)`. That **threshold is your optimization target**.
- **`ModelJudgeEvaluator`** — a *model* scores the output (qualitative). Same protocol/`Metric` as a
  quantitative evaluator, so **mix them freely** in one evaluation.
  - **`ScoreDimension`** — name + description + scale (use an **even** number of levels, e.g. 1–4, so the
    judge can't park on a neutral middle; "four levels = enough distinction without dilution").
  - **`ModelJudgePrompt`** — gives the judge app context + the expected value as reference.
  - Judge model should be **≥ as capable** as the model under test → 298 uses **PCC** to judge an on-device feature.
  - **Rationales are the product** — read them; they tell you *why* a score happened.
- **`SampleGenerator` / `makeSamples`** (299) — synthesize more `ModelSample`s. `sessionProvider` (which model
  drives generation; PCC for big context), `samplingStrategy` (`random` | `slidingWindow`), `validator`
  (accept/reject each → `samples` / `invalidSamples`). Coverage > count.
- **`TrajectoryExpectation` + `ToolCallEvaluator`** (299) — evaluate the agent's **path**, not just the
  answer: which tools, which arguments, in what order, and a **`disallowed`** set that must *not* appear.
  Matchers: `naturalLanguage` (intent, not string), `contains`, `oneOf`, `pattern`, `range`. `unordered`
  when timing doesn't matter. These are themselves `@Generable`, so synthetic data works for them too.
- **Xcode Evaluations report** — per-sample drill-down + a **Compare** button across two runs.

## 3. The correspondence table

| zoo gate (numeric/tensor layer) | Apple Evaluations equivalent (feature/behavior layer) | how to express ours as theirs |
|---|---|---|
| **Oracle** = fp32/bf16 reference output | `ModelSample.expected…` (closed answer) **or** a **Model Judge** (open answer) | our oracle is just an "expected value" the framework already models; for open-ended Vault answers, a PCC judge stands in for the oracle |
| **Numeric parity gate** (cos = 1.0, bit-exact, `engine ≡ python`, 24/24 tokens) | a **quantitative `Evaluator`** over a fixed dataset returning pass/fail | "if you can measure it in code → quantitative" (298). Our parity checks *are* quantitative evaluators; wrap each as one `Metric` |
| **Margin rule** (a flip counts only if logit margin ≥ ~0.1; ties ignored because the oracle itself flips) | tolerance inside a quantitative evaluator; `range`/`naturalLanguage` matcher; and the framework's whole reason to exist | our margin rule = a local tolerance band; in Apple terms it's "score within ε," not "assert equal." Encode the band in the `Evaluator`, not as an `==` |
| **Flip budget ≤ N** / "24 of 24" / "≤ 2 query flips on busy scenes" | **optimization target**: `#expect(aggregateValue >= rate)` | reframe "N of M must match" as a **pass-rate threshold** over the dataset — identical math, native shape |
| **Busy-scene tolerance** (torch itself flips on 1e-4) | **drift** + judge rationale; the dataset label is itself noisy | our "the oracle is unstable here" = Apple's "raters disagree / the judge drifts." Measure it, don't pretend it's zero |
| **Device-verified RELEASE bench** (tok/s on real HW) | run the eval target **on-device**; pair with **243 Instruments** (TTFT / tokens-per-sec / total latency) | Evaluations gives behavior on-device; the Foundation Models Instrument gives the perf numbers — and it **profiles any FM-framework model, including our ZooFMProvider models** |
| **Hill-climbing the quant scheme** (sym8 vs km8 vs km4 by flip-count) | **evaluation-driven development** + **Compare** view; one variable at a time (control vs experimental) | we already do this by hand; 335 gives it a name, a UI, and the discipline of isolating one change |
| *(no zoo equivalent yet)* | **`SampleGenerator`** synthetic data | NEW lever — grow numeric-gate prompt sets and Vault eval datasets without hand-writing thousands |
| *(no zoo equivalent yet)* | **Model judge + Cohen's κ alignment** (≥ 0.6) | NEW — score *answer quality* at scale, and **evaluate the evaluator** so it stays aligned as data grows |
| *(no zoo equivalent yet)* | **`TrajectoryExpectation` / `disallowed`** | NEW — verify the agent's *path*; doubles as the **prompt-injection regression** (see §5) |

## 4. The altitude insight (why these stack, not compete)

The two systems ask **different questions at different layers**, and a feature needs both:

- **zoo gate:** "Does the converted/quantized model emit the *same tokens* as the high-precision reference?"
  → **fidelity of the port** (tensor layer). A `sym8` bundle that matches fp16 within margin passes.
- **Evaluations:** "Does the *feature* behave as the user expects across a diverse dataset?"
  → **quality of the behavior** (feature layer). Does Vault answer correctly, cite the right note, refuse
  the poisoned action?

They are orthogonal: a model can pass zoo gates (bit-exact port) and still flunk Evaluations (the base model
is just bad at the task), or score well on a tiny Evaluations set yet be a broken port. **Run both.** zoo
gates protect "we shipped the model we think we shipped"; Evaluations protects "the feature is actually good."
299's punchline is the same — book-tagging eval checks *what* the model produces, tool eval checks *how* it
got there; "run both in the same suite for end-to-end confidence."

## 5. Two findings worth carrying into Vault

1. **The `disallowed` TrajectoryExpectation is a deterministic prompt-injection test.** 299's mechanism for
   "the model must *not* call `findSimilarBooks`" is exactly the gate `agentic-security-checklist.md` §6 asks
   for: feed **poisoned context**, then assert the destructive tool is **absent** from the trajectory and the
   parameters weren't rewritten (`naturalLanguage` matcher on the recipient/target). This converts "we
   mitigated injection" from a claim into a **number you can hill-climb** — the cleanest bridge between the
   security work and the eval work.
2. **Watch drift, target Cohen's κ ≥ 0.6.** If Vault uses a model judge for answer quality, the judge will
   diverge from human judgement as the dataset grows (335). Align it: add app context + a *few* worked
   examples (too many → overfit the alignment score), and gate on **κ ≥ 0.6** ("meaningful agreement").
   This is the *exact analogue* of our "is the oracle trustworthy here?" instinct, lifted to the judge.

## 6. A starter Vault Evaluations suite (sketch)

- **Quantitative** (`Evaluator`): retrieval hit-rate (did the cited note exist?), answer-contains-citation,
  latency budget (pair with 243 Instruments TTFT/total-latency), VL-route correctness (image query → VL model).
- **Qualitative** (`ModelJudgeEvaluator`, PCC judge, 1–4): groundedness (answer supported by retrieved
  context?), helpfulness. Split the moment you disagree with a score (298: a broad question = two questions).
- **Trajectory** (`ToolCallEvaluator`): ordered "search-before-answer"; **`disallowed` destructive tools under
  poisoned context** (the §5.1 injection gate); no unexpected tool calls.
- **Data**: start 20–30 hand-written samples (298 best practice), then `SampleGenerator` to ~hundreds; expect
  scores to **drop** when the dataset grows (299) — that's the small set having flattered you, not a regression.
- **Discipline**: evaluation-driven development — one change per loop, Compare against the prior run, keep the
  notes dict so runs are diffable. Ship the suite *with* the app (Value 1: measure, then claim).

## 7. Pointers
- Verbatim: `ondevice/_wwdc{298,299,335}_transcript.txt`. PCC judge/entitlement: `ondevice/_wwdc319_transcript.txt`.
- Perf side of the same loop: **243** Foundation Models Instrument (`ondevice/_wwdc243_transcript.txt`) —
  TTFT / tokens-per-sec / total latency, on any FM-framework model.
- Security gate that rides on §5.1: `agentic-security-checklist.md`.
- Where Vault's model plugs in: `fm-provider.md`.
