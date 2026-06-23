> **SUPERSEDED** by
> [`SESSION-2026-06-23-handoff-2-onboarding-phaseB.md`](SESSION-2026-06-23-handoff-2-onboarding-phaseB.md)
> (onboarding + Phase B + eval depth). This doc covers the earlier Phase-1
> cockpit work; keep it only for the deep autoresearch-loop mental model.

# Session Handoff — 2026-06-23 (pilotBENCHY real-benchmark + cockpit)

Audience: the next engineer/agent picking this up cold. Read this first, then
`docs/superpowers/specs/2026-06-22-pilotbenchy-real-benchmark-design.md` and
`docs/superpowers/specs/2026-06-22-pilotbenchy-modes-and-data-design.md`.

Branch: `codex/pilotbenchy-first-run-reports` · Draft PR: #9 ·
Repo: https://github.com/psychofanPLAYS/agent-pilot-autobench

## TL;DR

pilotBENCHY went from "looks like a benchmark but never actually asked the model
anything" to a working local autoresearch loop: it asks the SimpleBench questions
through `llama-server`, scores them accuracy-first, walks an ordered llama.cpp flag
ladder, and (on long runs) keeps searching with a persistent Optuna study. The TUI
cockpit is the front door (run modes via `M`). All changes are committed and pushed;
`pytest` (252 passing, 1 skipped), ruff format/check, mypy, and compileall are green.

## The one bug that mattered most

The cockpit/default run used the synthetic `llama-bench` probe (generate 32 tokens off
a fake prompt) and **never asked the model a question**. The real SimpleBench
question-asking engine existed but was gated behind a CLI-only `--flag-ladder` flag the
TUI never set. Fixed by making benchmark mode the default (`evaluation_mode.py`) and
routing the cockpit/CLI through the question engine; `--speed-scout` opts back into the
synthetic probe. Then a second bug: every answer hit the 1024-token cap so reasoning
models never reached their "Final Answer:" line (scored as None) — raised to 4096 and
hardened the extractor.

## What shipped (with commit refs)

Phase 1 — make it actually benchmark:
- `4adf7e3` design spec, `820753f` plan
- `8a3071b` `evaluation_mode.py` (BENCHMARK default | SPEED_SCOUT)
- `cd6b49f` benchmark mode is the default; `--speed-scout` opt-out
- `b63d9bb` cockpit mode toggle + champion verdict line
- `5afb5ee` capture answers reliably (4096 token cap, hardened `extract_final_answer`,
  stronger system prompt) — NOTE: this commit also accidentally folded in codex's
  in-progress `model_identity.py`/`discovery.py` WIP via `git add -A`.

Cockpit modes + methodology:
- `3972645` `Lmin-stripped` ladder baseline (fewer-flags-faster test)
- `e83035e` goal-shaped run modes in the cockpit (`modes.py`, `M` cycles)
- `54e4042` parallel tested last as capability (`Lpar-2/3`); Karpathy 5-minute rounds
  (`KARPATHY_ROUND_*`, loop `round_seconds` cap)

Depth + correctness:
- `a74c8b2` `forced_server_args` config + `PILOTBENCH_FORCED_SERVER_ARGS`; `tests/conftest.py`
  clears `PILOTBENCH_*` env so machine config can't break the suite
- `66b97be` real perf metrics on the question path: prefill tok/s (was hardcoded `0.0`),
  gen-tps stddev, TTFT p90/p99, server warmup before the scored batch
- `1f9b148` overnight convergence: ladder then persistent Optuna search, accuracy-first,
  objective-tagged studies, `LEARNING_SCORE_VERSION=score-v4-simplebench`
- `d80ba2b` show prefill tok/s in `flag-ladder-results`
- `2983042` README + dry-run example artifact updated to match the app

## How it works now (mental model)

1. `apb --start` (or `FIRST_RUN.bat`) → cockpit (`tui.py`, `BenchTui`). Arrow/Space to
   pick models, `M` to pick a mode (`modes.py` `RUN_MODES`), Enter to run.
2. A run calls `cli._run_one_autoresearch`. `evaluation=BENCHMARK` (default) →
   `flag_ladder` path → `LlamaServerSimpleBenchAttemptRunner` asks the 10 SimpleBench
   questions (stateless `POST /v1/chat/completions` per question) and scores them.
3. `AutoresearchLoop` (`autoresearch.py`) walks `candidate_sequence` (the ladder from
   `flag_ladder.build_core_flag_ladder`) first, each attempt capped at `round_seconds`
   (5 min). When `learner` is set and budget remains, it then drives via the Optuna
   learner (`learning.OptunaSettingsLearner`) until the budget — overnight convergence.
4. Objective is accuracy-first: `AttemptResult.score()` returns `simple_bench_score`
   (`accuracy*1000 + bounded_speed_tiebreaker`).
5. Champion per model + leaderboard across models via `reports.write_leaderboard`.
   Receipts under `_runs/<ts>-<model>/`; `flag-ladder-results.md/.tsv/.json` per run.

## Verified this session (evidence, not claims)

- Live run on `G:\AI\models\LM_Studio-gguf\Gemma-4-E2B-...Q8_K_P.gguf` + cuda12
  `llama-server.exe`: all 10 questions asked, answers captured (10/10 reached Final
  Answer after the token-cap fix; 3/10 correct — expected for a 2B model on hard
  SimpleBench), prefill ~1500 tok/s, gen-tps stddev ~2, TTFT p90/p99 captured, champion
  `L2-kv-unified` (same accuracy, ~19% faster than baseline).
- Documented `autoresearch --model … --dry-run` plans exactly
  `Lmin-stripped → L0 → L2-kv-unified → L3 → L4 → L5 → L6-q8-kv → T12/16/24/32 → Lpar-2 → Lpar-3`.
- Release gate green: `pytest` 252 passed / 1 skipped, ruff format/check, mypy, compileall.

## Environment / machine setup (already applied on this box)

- `apb` is on PATH (`apb setup` created `_bin/apb.bat`, added `_bin` to user PATH).
- Real paths set as persistent USER env vars (NOT in tracked `_CONFIG.toml`, which would
  break tests): `PILOTBENCH_MODEL_ROOTS=G:\AI\models`,
  `PILOTBENCH_LLAMA_{SERVER,BENCH,CLI,PERPLEXITY}=G:\AI\llama.cpp\cuda12\*.exe`.
- `tests/conftest.py` clears `PILOTBENCH_*` for tests, so these env vars don't break the
  suite.

## Known gaps / NOT done (next steps, roughly prioritized)

1. **Phase 2 question packs** (the agreed next build): difficulty-graded MC packs so small
   models get a non-zero gradient — easy tier (ARC-Easy, OpenBookQA, CommonsenseQA) +
   medium (MMLU, ARC-Challenge); plus agentic packs aimed at real local-agent work
   (wiki/knowledge-vault upkeep, YouTube-transcript summarization, multi-hop deep-research
   over a local corpus, Hermes/OpenClaw tool-use). Generalize `simple_bench.py` into a
   `BenchmarkPack` interface (see the real-benchmark design spec, Unit B/C/D).
2. **Expand the learner's search space** to include the flag dimensions (kv_unified,
   cache_type, threads, cont_batching) so overnight search explores flags, not just
   context/batch/parallel. Today the ladder handles flags and the learner handles the
   numeric dims (`learning.py` `OptunaSettingsLearner.suggest`).
3. **In-bench telemetry + graphs**: sample GPU util/VRAM/power/temp during each profile
   and correlate with tok/s; render charts into the HTML report (see modes-and-data spec).
4. **Cross-session champion memory surfaced in the TUI** ("last time best was …"); Optuna
   study already persists per model+objective.
5. **Phase 3 concurrency × context sweep** (deferred spec) — parallel {1,2,3,4} × ctx
   tiers with a VRAM-headroom guard, run last.
6. **Split commit `5afb5ee`** if you want clean history — it folded in codex's
   `model_identity.py`/`discovery.py` WIP.
7. **Custom-mode time picker** (15/30/45) — `custom` mode currently defaults to 15 min.

## Gotchas for the next session

- Do NOT put machine paths in the tracked `_CONFIG.toml` — it breaks tests that assume
  repo-relative defaults. Use `PILOTBENCH_*` env vars (the `conftest` isolates tests).
- Benchmark mode requires `llama-server`; the doctor reports missing paths.
- SimpleBench is intentionally hard; small models scoring low is correct, not a bug —
  that's why Phase 2 adds easier packs.
- `--parallel` is treated as capability, not single-stream speed; don't "fix" the ladder
  to rank it on tok/s.

## Key files

- `src/gguf_limit_bench/cli.py` — commands + `_run_one_autoresearch` orchestration
- `src/gguf_limit_bench/evaluation_mode.py` — benchmark vs speed-scout
- `src/gguf_limit_bench/modes.py` — cockpit run modes + Karpathy round constants
- `src/gguf_limit_bench/flag_ladder.py` — the ordered ladder (Lmin → … → Lpar)
- `src/gguf_limit_bench/simple_bench.py` / `simple_bench_runner.py` — question asking,
  scoring, answer extraction, perf metrics
- `src/gguf_limit_bench/autoresearch.py` — `AutoresearchLoop`, scoring, flag-ladder report
- `src/gguf_limit_bench/learning.py` — persistent Optuna learner
- `src/gguf_limit_bench/tui.py` — the cockpit
- `docs/superpowers/specs/2026-06-22-*` — the two design specs

## Resume / run

```powershell
# from a fresh terminal, apb is on PATH:
apb --start                 # cockpit: pick models, M for mode, Enter to run
apb doctor                  # confirm paths resolve
# dev gate:
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check . ; uv run --extra dev mypy src
```
