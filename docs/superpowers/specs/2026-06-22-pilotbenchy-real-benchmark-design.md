# PilotBENCHY Real-Benchmark Cockpit Design (Phases 1+2)

Date: 2026-06-22
Status: DESIGN — approved direction, pending spec review.
Author: Claude (with Dawid).
Relationship to prior work: This is the **successor** to
`2026-06-19-pilotbenchy-model-intelligence-design.md`. That design defines model
discovery, Hugging Face provenance, the catalog, campaign selection, and the
mandatory-SimpleBench + agentic-suite + best-balanced-winner *contract*, but
explicitly runs **no live benchmark**. This spec makes that contract actually
execute from the cockpit, adds difficulty-graded and agentic question packs aimed
at small local models, emphasizes reasoning, and makes the Karpathy self-learning
loop first-class. It does **not** re-scope discovery/provenance/catalog. The
in-progress `model_identity.py` work is left untouched.

---

## 1. The problem (verified, not assumed)

The cockpit has never actually asked a model any benchmark questions.

`_run_one_autoresearch` in `cli.py` selects the measurement engine like this:

```python
if flag_ladder:
    attempt_runner = LlamaServerSimpleBenchAttemptRunner(...)  # asks the 10 questions
else:
    attempt_runner = LlamaBenchAttemptRunner(...)              # synthetic llama-bench, no questions
```

`flag_ladder` defaults to `False`, and the TUI cockpit (`tui` command) and
`autoresearch-all` call `_run_one_autoresearch` **without** `flag_ladder=True`.
So every cockpit/default run uses synthetic `llama-bench` (`-pg 128,32`: load
model, generate 32 tokens off a 128-token fake prompt, print one speed number),
plus a workflow smoke and a TTFT probe. The 10 SimpleBench reasoning questions
with right-answer scoring run **only** behind the CLI-only `--flag-ladder` flag.
This is the root cause of "I don't think it's asking the LLMs questions at all."

Two facts confirmed during analysis:

- **Session-clear is already correct.** Each question is an independent
  `POST /v1/chat/completions` carrying only `[system, user]` — no prior Q&A is
  threaded, so every question runs in a clean session. No change needed.
- **The bundled data is correct.** `simple_bench_public.json` is byte-identical
  to the user's `simple_bench_public (1).json` (10 questions + answer keys).

A second, subtler problem: **SimpleBench is adversarially hard.** Frontier models
score ~40–60%; small local GGUFs (2B–13B) score near zero. Even once questions are
asked, a small model returns a flat zero, which still looks broken. The fix is a
**difficulty gradient** so weak models get a meaningful, discriminating score.

## 2. Goals / non-goals

### Goals
1. The default cockpit/`autoresearch` run **asks and scores real questions** through
   `llama-server`, finds best settings per model, and finds the best model overall.
2. A **difficulty-graded** question set (easy → hard) so every model class gets a
   non-flat score, with **reasoning weighted heavily**.
3. **Agentic task packs** representative of real small-model local work
   (knowledge-vault/wiki upkeep, transcript summarization, tool-use, multi-hop
   "deep research" over a local corpus).
4. **Cross-session self-learning**: persist champion settings + rung outcomes so the
   next session warm-starts and converges faster — the Karpathy wow-factor, made real
   for the default path.
5. **Cockpit UX**: champion verdict panel, live per-question progress, open-results
   key, working preset switch, benchmark/speed-scout toggle.

### Non-goals (this spec)
- The concurrency × context sweep (Phase 3 — separate spec; summarized in §10).
- Re-implementing discovery/provenance/catalog (owned by the 2026-06-19 design).
- Live web access as a *default* eval (offered only as an opt-in non-reproducible mode).
- Stopping or touching the user's running LLM service; large downloads; new GGUFs.

## 3. Two conceptual anchors

**Reasoning is the lever.** At inference time a model's underlying reasoning
*capacity* is fixed; RAG/databases add knowledge and grounding, not raw reasoning.
What moves *realized* reasoning quality is (a) **which model/quant you pick** — this
tool's purpose — and (b) **test-time compute** (chain-of-thought, self-consistency /
majority vote, larger reasoning budget). Therefore: measure reasoning per model so
the user can pick the best reasoner, and leave room to layer test-time-compute
scaffolding as a future scored dimension.

**Self-learning is the wow-factor, and it half-exists.** `learning.py` already runs a
**persistent per-model Optuna study** in SQLite (`load_if_exists=True`), so it
warm-starts across sessions — that is the Karpathy self-improvement loop. The catch:
the flag-ladder path sets `learning = False`. Promoting the ladder to default would
discard learning unless we add ladder-level memory. §9 closes that gap.

## 4. Architecture (bounded units)

Each unit is a small, independently testable module with one job. Build on existing
files; do not grow `cli.py` / `autoresearch.py` into larger multipurpose files.

### Unit A — Eval routing (the bug fix)
- Add `evaluation: Literal["benchmark", "speed_scout"]` to `RunConfig`
  (default `"benchmark"`).
- `_run_one_autoresearch` routes `"benchmark"` through the question-asking
  `llama-server` engine (the flag ladder over the question packs); `"speed_scout"`
  keeps `LlamaBenchAttemptRunner`. The TUI passes the active mode; a new
  `--speed-scout` CLI flag selects the fast path.
- `llama-server` becomes a hard prerequisite for the default path; `doctor` already
  reports it, and a missing server yields actionable output, never a false zero.
- Receipt/TSV contract is unchanged; `simple_bench_score` generalizes (see §5).

### Unit B — `BenchmarkPack` abstraction (extensibility spine)
- A pack = `{id, display_name, difficulty_tier, format, scorer, data_path, license_note}`.
  - `difficulty_tier ∈ {easy, medium, hard}`.
  - `format ∈ {multiple_choice, numeric_exact, generative_rubric}`
    (numeric_exact reserved for future GSM8K; not bundled now).
  - `scorer`: pure function `(question, model_output) -> QuestionScore`
    where `QuestionScore = {value: float in [0,1], correct: bool|None, detail: dict}`.
- Generalize `simple_bench.py`'s loader/scorer into the first concrete pack behind
  this interface. Loader accepts variable option counts (A–F and A–D).
- The stateless one-question-per-POST contract is preserved for every MC pack.

### Unit C — Bundled MC intelligence packs (the gradient)
Vendor tiny offline subsets (~20–40 items each) under `src/gguf_limit_bench/data/packs/`,
each with a `NOTICE.md` recording upstream source, revision, license, and checksum
(mirroring `SIMPLEBENCH_NOTICE.md`):
- **Easy:** ARC-Easy, OpenBookQA, CommonsenseQA.
- **Medium:** MMLU (a few subjects), ARC-Challenge.
- **Hard:** SimpleBench (already bundled).
Each model gets a per-pack accuracy; tiers make a 2B model's score meaningful instead
of flat zero. Licensing note: ARC/OpenBookQA/CommonsenseQA/MMLU are permissively
licensed; vendored subsets are small and attributed.

### Unit D — Agentic task packs (real local-agent work)
Extend `workflows.py`'s `WorkflowTask` with a pluggable `scorer` and optional input
`fixture`, then add representative **seed** tasks (≈3 items each — proof of interface,
not a full benchmark; expandable later via the same interface). Keep the existing 4
Hermes tool-choice/JSON tasks.
- **Knowledge-vault / wiki upkeep:** input = a markdown page + an edit instruction;
  output = edited markdown; deterministic scorer checks structure preserved
  (headings / front matter / existing links intact) **and** the specific fact updated.
- **Transcript summarization:** input = a transcript chunk; output = a summary;
  scorer checks required key-points present, length within bounds, and no fabricated
  sections (key-point checklist match, not LLM-graded).
- **Multi-hop "deep research":** input = a small fixed local corpus (a vendored mini
  "brain vault") + a question requiring 2–3 hops across documents; scorer checks the
  synthesized answer contains the chained facts. Reproducible and offline. A live-web
  variant is an explicit opt-in mode labeled non-reproducible.
These map onto the `hermes` / `openclaw` presets defined in the 2026-06-19 design and
serve as the repo-local deterministic tasks `AUTORESEARCH-PROGRAM.md` already calls for
where external suites are too heavy.

### Unit E — Composite scoring & the verdict
- Per attempt, compute component scores kept **visible, never blended into one hidden
  number**:
  - `reasoning_score` — weighted across MC packs, reasoning-heavy packs weighted up.
  - `agentic_score` — from the workflow/task packs.
  - existing speed/TTFT metrics retained as evidence.
- Selection stays **accuracy/quality-first, speed as a bounded tiebreaker** (reuse the
  existing `accuracy*1000 + bounded_speed_tiebreaker` rule, generalized over the
  combined quality signal so a fast-but-wrong profile never beats a slower-but-correct
  one).
- Per-model champion = best settings. **Best model of the bunch** = leaderboard
  champion across models. Reports answer best-for-quality, best-for-agentic,
  best-for-latency, and best-balanced separately (consistent with the 2026-06-19
  reporting contract).

### Unit F — Cockpit UX
- **Champion verdict panel:** best model + its best settings + per-pack score breakdown,
  shown live and final.
- **Live run detail:** `model 2/5 · rung L3 · Q4/10 ✓ · reasoning 50% · 3m12s` instead
  of a coarse per-model bar.
- **Open results** key → `results.html` / `leaderboard.md`.
- **Preset switch** key (fixes the currently inert preset menu — `run_config` is
  hardcoded to `normal` with no binding) and a **benchmark / speed-scout** toggle.

### Unit G — Verification
- Test-per-module, mocked `llama-server`/`llama-cli` (repo convention, TDD).
- **Live acceptance run** (the "make sure it actually works" gate): the
  `Gemma…Q8_K_P.gguf` + `cuda12\llama-server.exe`, confirming questions are asked +
  scored, packs produce a non-zero gradient, a champion is emitted, and a receipt is
  written. Runs on a free port; never touches an already-running served model.

### Unit H — Cross-session settings memory (§9)
Detailed below.

## 5. Scoring model (summary)

```
QuestionScore.value ∈ [0,1]            per question, from the pack scorer
pack_accuracy        = mean(value)      per pack
reasoning_score      = weighted mean over MC packs (reasoning packs up-weighted)
agentic_score        = mean over agentic task packs
quality              = w_r*reasoning_score + w_a*agentic_score   (weights configurable)
attempt_score        = quality*1000 + bounded_speed_tiebreaker(median_tps)
```

A failed mandatory phase (e.g., SimpleBench not asked) cannot be hidden by a high
component elsewhere; the report shows the failure and withholds a champion until every
selected model received every mandatory pack.

## 6. Self-learning / cross-session memory (Unit H)

Goal: "get further, quicker, next time." Two complementary mechanisms.

1. **Champion memory in `_db`.** Persist, keyed by `model + hardware_fingerprint +
   score_version`:
   - the best profile (settings) and its component scores;
   - per-rung outcomes (kept / discarded / crashed) with the comparable score.
   On the next session for the same key, the loop:
   - tries the **prior champion first** as the new baseline;
   - **deprioritizes or skips** rungs that previously crashed or clearly lost
     (configurable; always re-checkable with `--ignore-memory`);
   - seeds any later sweep (Phase 3) from the known champion.
2. **Persistent Optuna study** (existing) for fine settings refinement, now optimizing
   `attempt_score`. Bump `LEARNING_SCORE_VERSION` (e.g. `score-v4-intelligence-agentic`)
   so new scoring does not inherit stale-metric trials. The study remains per-model and
   warm-starts across sessions.

`hardware_fingerprint` = stable hash of GPU name + VRAM + llama.cpp build id, so memory
from a different box/build is not silently reused.

Receipts surface what memory did ("warm-started from champion X; skipped rung L5
(crashed last session)") so the self-improvement is visible, not magic.

## 7. Data, provenance, licensing

- Vendored pack subsets live under `data/packs/<pack-id>/` with a per-pack `NOTICE.md`
  (source URL, revision, license, checksum), mirroring `SIMPLEBENCH_NOTICE.md`.
- Subsets are intentionally small (offline, fast, deterministic); they are smoke
  signals, not full benchmark scores — reports must say so.
- No large dataset downloads at runtime; packs are vendored at authoring time.

## 8. Safety constraints (carried from prior design)

- Never stop/restart/kill the user's running served LLM. Benchmark server runs as a
  separate process on a free port (`_free_port`), bound to `127.0.0.1`, torn down by
  the PID we spawned only.
- Never reset/delete/stage unrelated files or the dirty worktree.
- OOM/crash: preserve receipt, mark the attempt, unload only the benchmark-owned
  process, continue per policy.
- Missing prerequisites (server, pack file) → actionable doctor output, never a false
  zero or silent skip.

## 9. Phase 3 (deferred to its own spec): concurrency × context sweep

After a model's ladder succeeds with non-zero quality, take its champion profile and
sweep `parallel ∈ {1,2,3,4} × ctx ∈ {4k,8k,16k,32k}`, firing N questions concurrently.
Record aggregate tok/s, slowest-stream tok/s, TTFT-under-load, quality-must-hold, VRAM
peak + cliff detection, with a VRAM-headroom guard that skips cells that won't fit.
Output `concurrency-profile.{md,tsv,json}`. Runs **last**, only when prior phases passed.

## 10. Testing strategy

### Unit (mocked, no model load)
- Pack loader: variable option counts, malformed rows, duplicate ids, empty set.
- Each scorer: MC correctness, summarization key-point match, wiki structure checks,
  multi-hop chained-fact check, numeric scorer (reserved) stubs.
- Eval routing: `benchmark` vs `speed_scout` selects the right runner.
- Composite scoring: quality-first ordering; failed mandatory phase withholds champion.
- Champion memory: persist/read round-trip; warm-start tries prior champion first;
  hardware-fingerprint mismatch does not reuse; `--ignore-memory` re-tests everything.
- Optuna score-version bump isolates old trials.

### Integration (no model load)
- Fake `llama-server` returning canned streamed completions → full attempt → receipt.
- TUI: champion panel renders; live detail updates; preset switch changes config;
  open-results key resolves paths.
- Dry-run still plans without starting a server.

### Hardware acceptance (Unit G)
- One live run on the Gemma model + cuda12 server: questions asked + scored, gradient
  non-zero, champion emitted, receipt complete, user's services untouched.

## 11. Acceptance criteria

1. A default cockpit run (no special flags) asks and scores real questions through
   `llama-server` and produces a per-model champion and an overall leaderboard champion.
2. `--speed-scout` still offers the fast synthetic path; it is clearly labeled
   speed-only and never declared agent-ready.
3. At least the easy + medium + hard MC packs load and score, giving a small model a
   non-flat gradient; reasoning packs are weighted up.
4. At least one wiki-upkeep, one summarization, and one multi-hop research task run and
   score deterministically through the agentic layer.
5. Component scores (reasoning, agentic, speed) are reported separately; a failed
   mandatory pack withholds the champion.
6. Champion settings + rung outcomes persist to `_db` and warm-start the next session
   for the same model+hardware+score-version; receipts show what memory did.
7. The cockpit shows the champion verdict, live per-question progress, a working preset
   switch, a benchmark/speed-scout toggle, and an open-results key.
8. Existing + new unit/integration tests, ruff format/check, mypy, compileall, and
   `uv build` all pass.
9. A documented live acceptance run on the Gemma model proves questions are asked and
   scored end-to-end, with the user's running services untouched.
10. The 2026-06-19 discovery/provenance/catalog work and the in-progress
    `model_identity.py` are not regressed or re-scoped.
