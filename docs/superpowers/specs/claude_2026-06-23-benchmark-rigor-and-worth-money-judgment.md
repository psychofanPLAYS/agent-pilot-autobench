# claude_2026-06-23 — Is pilotBENCHY a TRUE benchmark tester? Is it worth money?

> **Status:** Research-aid judgment from Claude (Opus 4.8). Codex owns code.
> Deliberately a *different altitude* than the order/contract specs: this is the
> harsh product+science critique and a roadmap of harder benchmark programs to
> build. Meant to inspire and push, not to block. Pairs with the verification and
> order-of-operations docs (`claude_2026-06-23-*`).

---

## The blunt verdict

**As-is, pilotBENCHY is a competent llama.cpp _measurement harness_. It is NOT
yet a "true benchmark tester," and it is NOT yet clearly worth money.**

It measures honestly (TPS/TTFT/prompt-eval, receipts, Karpathy-style ledger). The
separated-programs work in flight fixes the *scheduling* flaw. But three things
still stand between "useful tool" and "worth paying for":

1. **The intelligence signal is statistically too thin to trust.** A 10-question
   (here 5-question) fixed fixture, scored once, is a coin flip. `1/5` vs `2/5` is
   noise, not evidence. A real benchmark reports a number **with a confidence
   interval** and **run-to-run variance**. Right now one accuracy number is
   treated as truth. This is the single biggest scientific hole.

2. **The fixed public fixture is contamination-prone and tiny.** A pinned public
   SimpleBench snapshot is exactly what models memorize. "Intelligence" measured
   on memorizable items is not defensible. lm-eval-harness already does broad
   academic benchmarks better — competing there is a losing game.

3. **The output is a leaderboard, not a decision.** Buyers don't pay for receipts;
   they pay to stop doing hours of manual flag trial-and-error and be told:
   *"Use these flags. Expect 84 tok/s at 64k. Quality holds to ~96k then drops.
   18.9 GB VRAM, 5 GB headroom. Confidence: high."* The prescriptive,
   **Pareto-optimal recommendation with quantified confidence** is the product.
   It doesn't exist yet.

### What it IS uniquely good at (the real moat — lean into this)

Not "intelligence leaderboard." The defensible, monetizable question is:

> **"For MY specific GGUF on MY specific GPU, what exact llama.cpp flags + context
> give the best speed / quality / VRAM tradeoff — with evidence I can trust?"**

That is a genuine, expensive-to-answer pain (flag space is huge, hardware-specific,
and changes every llama.cpp build). Nobody does it well. Everything below pushes
toward owning that question.

---

## Roadmap: increasingly sophisticated benchmark programs

Tiered. Codex is building Tier 0 (separated programs). These are what make it
*worth money*. Each program lists **what it measures**, **why it's worth money**,
and **what's new to capture**.

### Tier 1 — Make the numbers trustworthy (foundation of "true")

**P1. Statistical intelligence (CI-backed).**
Repeat each question K times (default 5); report accuracy ± **Wilson score
interval**, per-question stability, and flag "noisy" items (answer flips across
repeats). Grow the item pool. *Worth money:* turns a coin flip into a measurement
a buyer can defend. *Capture:* per-item answer distribution, agreement rate,
CI width, "effective N."

**P2. Determinism & reproducibility audit.**
Run the same prompt N times at temp 0; hash outputs; report exact-match rate and,
when it diverges, where divergence starts. flash-attn + KV-quant + batching are
not bit-stable — *measure it instead of pretending.* *Worth money:* tells the user
whether their setup is reproducible at all. *Capture:* output-hash agreement,
first-divergence token index, divergence vs batch/parallel.

**P3. KV-quant quality sensitivity.**
Same items at `q8_0` vs `f16` KV (and `q4_0` if offered): Δaccuracy + Δperplexity.
*Worth money:* answers "is q8 KV actually safe for MY model?" — the exact tradeoff
the locked baseline assumes. *Capture:* accuracy delta with CI, perplexity delta,
VRAM saved.

### Tier 2 — Probe the failure modes a 30-second test misses (differentiation)

**P4. Sustained-load / thermal-throttle program.**
Generate continuously for N minutes; plot tok/s over wall-clock; detect the
throttle knee; sample GPU temp / power / clocks throughout. A 30 s probe reports
the *peak* the user will never sustain. *Worth money:* "can I actually run this
all day, or does it throttle at minute 5?" *Capture:* tok/s(t) curve, throttle
onset time, steady-state vs peak tok/s, temp/power/clock traces.

**P5. Long-generation stability.**
Force a long generation (4k–16k tokens); detect repetition loops, entropy
collapse, degeneration. Reasoning models think long — this is where they break.
*Worth money:* intelligence runs allow unlimited thinking; this proves the model
stays coherent that long. *Capture:* repetition rate, distinct-n-gram ratio,
entropy(t), loop-detection flag, tokens-to-degeneration.

**P6. Prompt-length sensitivity curve.**
Sweep prompt size 1k→max; plot prefill TPS and TTFT vs prompt tokens. Agents run
huge system prompts; a short-prompt TTFT is a lie about real latency. *Worth
money:* predicts real-world agent latency. *Capture:* prefill-TPS(prompt_len),
TTFT(prompt_len), the prompt size where TTFT crosses a usability threshold.

**P7. Concurrency / throughput-under-load.**
N parallel slots; measure aggregate throughput and p50/p95/**p99** latency,
KV-cache pressure, deferred/dropped requests. *Worth money:* "can this serve my
agent swarm / multi-user setup?" *Capture:* aggregate tok/s vs slots, tail
latency, KV pressure, request-drop onset.

### Tier 3 — The moat (worth real money)

**P8. Pareto-frontier recommender.**
Across the flag×context grid, compute the **non-dominated frontier** over
{gen TPS, prompt TPS, max quality-proven context, VRAM, accuracy-with-CI}. Output
THE recommendation + the dominated alternatives + *why*. *Worth money:* this is
the decision the buyer is paying for. *Capture:* frontier set, the chosen point,
sensitivity ("at +2 GB VRAM you gain X tok/s"), confidence label.

**P9. Procedural / uncontaminated task generators.**
Seeded generators that can't be memorized: needle-in-context (random
passcode+depth), multi-hop synthetic reasoning (randomized entities), constraint
instruction-following (verifiable), arithmetic chains, JSON-schema adherence.
Reproducible via seed, but novel every run. *Worth money:* a defensible
intelligence signal that survives "the fixture is contaminated." *Capture:* per
generator pass-rate with CI, difficulty-vs-pass curve, seed for replay.

**P10. llama.cpp build-regression tracking.**
Pin the build commit in every receipt; diff metrics build-over-build; flag
regressions ("new build: -12% prompt TPS, flag X now ignored"). *Worth money:*
longitudinal trust — the tool that warns you an upgrade broke your setup. *Capture:*
metric deltas keyed by build commit, flag-support diffs, regression alerts.

**P11. Self-consistency / calibration (advanced, no ground truth needed).**
Multi-sample agreement as a quality proxy where there's no answer key; does the
model "know when it's right"? *Worth money:* extends quality measurement to
open-ended tasks. *Capture:* sample-agreement entropy, confidence-vs-correctness
calibration on the items that DO have keys.

---

## Suggested build order for Codex (after Tier 0 lands)

1. **P1 (CI) + P9 (procedural)** first — together they fix the "thin, contaminated
   intelligence" hole that most undermines trust.
2. **P4 (sustained/thermal)** — highest "aha" value for a local-GPU buyer, low
   conceptual complexity, reuses the speed-probe loop.
3. **P8 (Pareto recommender)** — converts all the captured data into the decision
   that justifies payment. Needs P1's CIs to be honest.
4. P6, P7, P5, P3, P10, P11 as the program library matures.

Each is a clean, independently testable `ProgramSpec` + runner — the architecture
Codex is already building makes them additive, not rewrites.

## One-line answer to David's question

> *Is it a true benchmark tester / worth money / useful?* — Useful yes, as a
> measurement harness. "True benchmark" and "worth money": **not yet** — it needs
> confidence intervals (P1), uncontaminated tasks (P9), sustained-load truth (P4),
> and a Pareto recommendation (P8). Build those four and it's a product.
