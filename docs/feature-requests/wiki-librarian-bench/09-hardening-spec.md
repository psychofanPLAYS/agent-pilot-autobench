# 09 — Hardening spec for the whole librarian test

Status: draft — actionable. Items marked [VALIDITY] are integrity weaknesses in the
shipped v0 generators that should be addressed before any published score (verified
against the code 2026-06-24).

Goal: make every cell of the cube trustworthy. A benchmark that can be gamed by
position bias, contaminated by reuse, or silently broken by a chat template is worse
than no benchmark — it produces confident wrong recommendations into the SSOT.

## A. Preflight gates (fail-fast before scoring)

Run once per (model, quant, template, thinking) before any questions are asked. A
failed gate aborts the cell and records a `preflight_fail` class — never a quality 0
that looks like the model is dumb.

1. **Identity gate** — resolve the GGUF to a canonical HF slug + quant artifact; refuse
   to score an unidentified model (SSOT integrity). See [00-vision-ssot.md](00-vision-ssot.md).
2. **Single-BOS gate (Gemma)** — tokenize a known string; assert exactly one BOS. Two
   BOS => abort (double-BOS silently degrades every score).
3. **Template-load gate** — confirm the intended template is actually loaded (`--jinja`
   for froggeric v19); hash it so the knob value is reproducible.
4. **Thinking-sanity gate (Qwen)** — with thinking=on, assert the output contains a
   `<think>` block; with thinking=off, assert it does NOT. This is what catches the
   `chatml` template silently killing thinking. If on and off look identical, abort.
5. **Answer-channel gate** — one warmup question must yield a parseable `Final Answer:`
   / MC letter. If the model can't honor the contract at all, that's a format failure,
   recorded as such, not a content score.

## B. Determinism & stability protocol

Two distinct regimes, reported separately (never conflated):

- **Repro-stability** — the strictest reproducibility the model allows.
  - Gemma: `temp=0` greedy.
  - Qwen: greedy is BANNED (repetition) -> use `temp=0.1` + fixed seed, LABEL as
    `quasi_deterministic`. Do not pretend it is greedy.
- **Operational stability** — author-recommended sampling, fixed seed, **N=5 repeats**
  per cell. Report `exact_output_match_rate` and `score_stddev`.
- **MoE/batch caveat** — Qwen expert routing + server batching can jitter outputs at
  fixed seed. Record `parallel`/batch settings with each stability number and do not
  attribute batch jitter to the model. Prefer single-stream for the stability meter.
- Record the llama.cpp seed and build in every row.

## C. MC validity hardening

### [VALIDITY] Fixed letter encodes the class (dedupe, gate, contradiction)

Verified in code: `dedupe`, `gate`, and `contradiction` use a fixed `_CHOICES` tuple
and compute the gold letter directly from the class index (e.g. `gate._INJECT = "A"`;
`dedupe`: `letter = chr(ord("A") + gold_class)`). So the letter always encodes the
class, and the gold-letter frequency is set by class balance. A model with a letter
position bias, or one that infers the fixed label->letter convention, can be advantaged.
(`compress` and `rerank` already randomize the gold position — leave them as-is.)

Fix (apply to the three label-choice packs): randomize the label->letter assignment per
question with `make_rng(seed)` (shuffle which letter carries each class label), then set
`answer` accordingly. Keep a stable `subtype` tag for the semantic class so slicing
still works, but the LETTER must not deterministically encode the class. This brings the
three packs in line with the `compress`/`rerank` pattern.

Add tests:
- gold-letter distribution across a pack is approximately uniform (no letter > ~50%).
- report per-letter accuracy at score time to detect residual positional bias.

### Position rotation across seeds

For any planted-answer task (rerank, compress), rotate the correct option's position
across seeds so a fixed run never concentrates the answer in one slot.

## D. Adversarial / edge cases to add per job

Each is a new `subtype` with known gold. These convert "easy" packs into discriminating
ones.

- **gate**: `relevant-but-stale` (skip), `partially-relevant` (skip unless task needs
  it), `subtle-relevant` (inject — low keyword overlap), `secret-in-memory` (skip +
  flag; never inject credentials).
- **dedupe**: `paraphrase-with-negation` (NOT a duplicate — opposite meaning),
  `unit-change` (same number, different unit => related not duplicate),
  `subset/superset` (one strictly contains the other).
- **contradiction**: `temporal-supersession` (new replaces old), `numeric-drift`
  (value changed), `partial-contradiction` (one clause conflicts, rest agrees).
- **compress**: `faithful-but-over-budget` (correct facts, violates token budget =>
  fails budget meter), `faithful-but-reordered` (still correct), `plausible-hallucination`
  (adds a fact that "sounds right" but is absent).
- **triage**: `borderline-salience`, `secret/credential` (DROP + flag — security),
  `PII` (handle per policy), `duplicate-of-known` (drop).
- **write-entry**: slug edge cases (punctuation, unicode, digits, leading/trailing
  spaces), `ambiguous-type` (mark multiple acceptable via `accept`).
- **rerank**: `lost-in-the-middle` (rotate gold to deep positions), `hard-negative`
  (distractor shares the answer's entities but not the relation).

## E. Negative controls (the abstention tests)

The most valuable and most-skipped cases — they measure precision, not recall.

- **gate / no-relevant-memory**: candidate is clearly irrelevant; correct action is
  skip. Over-injection here is the failure mode that makes a memory layer harmful.
- **rerank / none-answer**: add an explicit `None of these` option and author cases
  where no snippet answers the query. Picking a snippet anyway = over-confidence.
- **contradiction / unrelated**: already present; ensure it's a real share of the pack.

## F. Anti-contamination

- Keep generation procedural + seed-driven (already true for v0).
- **Seed rotation**: never publish a score computed on a seed previously published; the
  SSOT records the seed so reuse is detectable.
- No fixture answer ever appears verbatim in a system prompt or filler.

## G. Cross-cutting meters wiring (make them gates, not vibes)

- **format-adherence** — a cell that can't produce a parseable answer is penalized as a
  format failure; it does not silently become a content 0.
- **grounding (compress)** — verify the chosen summary's fact clauses are a subset of
  the source (substring check); count hallucinated clauses.
- **calibration** — for jobs where we request a confidence, compute Brier/ECE.
- Every meter writes which knob values produced it (model, quant, thinking, template,
  sampling, seed) so the warehouse slices cleanly.

## H. Scoring robustness

- Maintain `accept` lists for EXACT near-forms (synonyms, number-word vs digit already
  handled by `normalize_exact`).
- Tag each question with a `difficulty` (`easy`/`medium`/`hard`/`adversarial`) so trend
  charts can show where a model breaks, not just an average.
- Balanced class distribution per pack, asserted in tests.

## I. Sweep-budget discipline (no silent truncation)

The Qwen grid is combinatorially large (2 models x 5 quant x 2 thinking x 3 template x
2 MTP x 7 jobs x 5 seeds). Apply the tiered strategy in
[04-knobs.md](04-knobs.md): preflight/determinism -> thinking x template 2-axis grid ->
quant+MTP ladder at the winner -> context ladder -> Optuna tail. Whenever the harness
caps coverage (top-N, sampling, no-retry), `log()` exactly what was dropped — a silent
cap reads as "we tested everything" when we didn't.

## J. Application checklist (for whoever implements this)

- [x] [VALIDITY] randomize label->letter in dedupe, gate, contradiction (DONE 2026-06-24 via `_common.shuffle_choices`; verified ~uniform across 60 seeds); compress/rerank already OK
- [ ] add per-letter accuracy reporting at score time
- [ ] add the adversarial subtypes in section D to each generator (+ gold-sanity tests)
- [ ] add negative-control cases (section E), incl. `None of these` in rerank
- [ ] implement the 5 preflight gates (section A) in the run path
- [ ] implement the two-regime stability protocol (section B)
- [ ] wire format-adherence + grounding as scored gates (section G)
- [ ] seed-rotation + seed recorded in every SSOT row (section F)
- [ ] difficulty tags + balanced-distribution asserts (section H)
