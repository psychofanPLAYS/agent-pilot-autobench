# claude_2026-06-23 — Order of operations & program design

> **Status:** Research-aid handoff from Claude (Opus 4.8). Codex owns all code.
> This is the agreed pipeline + capture taxonomy + reporting model from a
> brainstorm with David. One open question remains (Stage 1 probe task) — see end.
> Pairs with `claude_2026-06-23-karpathy-autoresearch-verification.md`.

## Decision taken

**Fit gates everything.** The fit climb is not a bare allocation check — at each
context tier it runs a reliably-gradeable task that forces the model to produce
output, so a single climb captures the **fit boundary**, the **speed-vs-context
falloff**, and a **quality-vs-context falloff** at once. SimpleBench (hard
reasoning) stays a separate stage at 64k.

David's words: *"have the fit gate run first … measure performance with some task
you can reliably grade that gets the LLM to produce output … so you could test the
context falloff speed this way … anything you can think of we need to capture."*

## The gated pipeline

```
Stage 0  SESSION SNAPSHOT   (once, no server)
Stage 1  FIT + CHARACTERIZE (the climb — gates everything below)
           for ctx in 16k, 32k, 64k, 128k, 256k  (stop at first non-serve):
             • fresh server @ ctx, q8_0 KV, standard flags
             • SERVED? no → record failure class, STOP climb
             • graded probe task → speed + quality at this tier
           → max_served_context, tps_vs_ctx curve, quality_vs_ctx curve
Stage 2  SPEED PROBE     @16k  (repeatable gen, N repeats, determinism + TPS stddev)
Stage 3  FLAG ABLATION   @16k  (all standard ON, then one-flag-OFF deltas)
Stage 4  INTELLIGENCE    @min(64k, max_served_context)
                         one SimpleBench question per FRESH server/window,
                         unlimited thinking (n_predict=-1, never truncated)
Stage 5  REPORT          keep / discard / partial / crash; surface partial data
```

Hard rules baked in:
- **Never start serious testing below 16k.** `MIN_SERIOUS_CONTEXT_SIZE = 16384`
  (already in `programs.py`). 4k allowed only for an explicit tiny smoke test.
- **Intelligence runs at 64k**, one question per fresh session/window, reasoning
  never truncated (`n_predict = -1`).
- **Intelligence context is clamped to `min(64k, max_served_context)`** so we
  never launch a 64k run that fit already proved will OOM. If 64k did not fit,
  Stage 4 is skipped or clamped with an honest note — never crashed.
- Each stage can run standalone; `deep` runs all in order.

## Standard baseline (forced, locked) — already correct in repo

`gpu_profiles.py` 4090 profile emits, and every program inherits:

```
--flash-attn on  --kv-unified  --cache-type-k q8_0  --cache-type-v q8_0
--jinja  --gpu-layers 99
```

- `cli._effective_forced_server_args()` concatenates standard + user-locked
  extras and validates; dedup happens in `flag_ladder.llama_server_args_for_settings`
  via `_option_names` (tests at `test_cli.py:1495` assert single occurrence).
- **Template lock-in:** if the user passes `--jinja` / `--chat-template-file`,
  it is additive and stays on for every profile (`test_cli.py:1477`). Template
  *testing mode* is deferred; for now the chosen template is part of the baseline.

## Capture taxonomy ("anything we can think of")

### Per session (Stage 0, once)
OS name+build · CPU model + cores/threads · RAM total/avail · GPU name + VRAM
total + driver + CUDA version · llama.cpp build/commit · GGUF arch/quant/param
count/file size · resolved standard flags · template file + sha256 · bench git
branch/commit · UTC timestamp. (Reuse `telemetry.sample_telemetry()` +
`gguf_metadata` helpers — do not reinvent.)

### Per context tier (Stage 1)
ctx size · served bool + failure class · KV type + KV mem (est/actual) · VRAM
at-load + peak · server_ready_ms · prompt-fill tokens · **prompt-eval TPS
(prefill)** · TTFT cold + warm · gen TPS · generated tokens · stop reason ·
**graded correctness** · TPS-falloff & TTFT-falloff vs the 16k baseline.

### Per speed attempt (Stage 2)
all of the above at fixed ctx + **determinism hash** (identical text each run?) +
gen-TPS stddev across repeats + completion status.

### Per intelligence question (Stage 4)
question_id · fresh cold session · ctx=64k · predicted vs expected · correct ·
generated + reasoning tokens · output chars · **hit-any-cap? (must be no)** ·
stop reason · TTFT · gen/prompt TPS · `/metrics` sample.

### Per ablation (Stage 3)
baseline (all standard ON) vs one-flag-OFF → ΔTPS · ΔTTFT · Δcorrectness ·
ΔVRAM · stability/warnings.

### llama.cpp `/metrics` between sessions (Prometheus endpoint)
`prompt_tokens_total` · `tokens_predicted_total` · KV-cache usage ratio ·
requests processing/deferred · `n_decode_total`. Sample at bounded intervals;
this is the `session_metrics.py` module in the Codex plan (Task 4).

## Reporting model (Stage 5) — restores Karpathy fidelity

Four honest outcomes (replaces `not ok → crash`):

| Outcome | When | Carries |
|---|---|---|
| `keep` | ok & score improved | full metrics |
| `discard` | ok & score ≤ best | full metrics |
| `partial` | budget expired **after ≥1 graded question** | completed_questions, attempted_questions, accuracy_over_completed, median_tps_over_completed, `reason: budget_exhausted` |
| `crash` | server died / OOM / model_load / 0 completed | failure class |

Implementation detail and code refs in the verification doc (item #4).

## How this maps onto Codex's existing plan

- `programs.py` (done) already encodes context floors + `asks_questions` +
  `one_question_per_window` + `unlimited_thinking` per program — the pipeline
  reads these to enforce the rules above.
- Tasks 3 (`speed_probe.py`), 4 (`session_metrics.py`), 5 (one-question/64k
  runner) slot directly into Stages 2, 0/4-metrics, and 4 respectively.
- **Gap not yet in Codex's plan:** the Stage-1 *graded* climb probe (fit currently
  only checks serve/allocate, see `context_limit.py`) and the Stage-5 `partial`
  outcome. Both are conflict-free additions.

## OPEN QUESTION (David accidentally dismissed — needs an answer)

Stage 1's graded probe task. Three candidates:

1. **Needle-in-context retrieval** — plant a known token (passcode) at varying
   depth in filler sized to the tier; ask the model to return it. Exact-match
   gradeable AND actually exercises long context (not just allocation). *Best at
   answering "does 64k context actually work."*
2. **Structured 500-word generation** — the repeatable poem ending with an exact
   `FINAL LINE`. Deterministic + clean speed numbers, but a short prompt at 64k
   does not prove the window is used. *Speed-falloff only.*
3. **Both** — needle (long-context quality + real prefill) + short generation
   (clean gen-TPS/TTFT) per tier. Richest curves, ~2× server time per tier.

Claude's recommendation: **(3) Both**, because it cleanly separates "can it read
long context" (needle) from "how fast does it generate" (short gen) — and David
asked to capture as much as possible. If time per tier is a concern, start with
(1) needle since it subsumes the long-context question the speed prompt cannot
answer.
