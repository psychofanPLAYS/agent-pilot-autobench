# Agent Pilot Autobench Implementation Plan

Goal: build a repeatable TUI benchmark cockpit for a local GGUF collection, starting with Qwen-family models and RTX 4090-style Windows workstations.

Architecture: keep this project in one folder, use ready-made tools for heavy work, and store searchable receipts for every run. The app wraps `llama-bench`, LM Studio `lms`, NVML telemetry, Optuna learning, and small benchmark prompts; it does not reinvent inference.

Tech stack: Python, Textual, Rich, Typer, pytest, psutil, nvidia-ml-py, Optuna, `llama-bench.exe`, LM Studio `lms`.

## 18 Step Plan

- [x] Establish project-local configuration and reusable development guidance.
- [x] Create the project folder and initialize git.
- [x] Create a small project plan, receipts folder, and test-first structure.
- [x] Add tests for GGUF discovery, family parsing, quant parsing, and mmproj vision detection.
- [ ] Implement discovery from configured model roots and optional LM Studio metadata caches.
- [x] Add tests for model selection state: space toggles rows, select-all selects all.
- [x] Implement a Textual model picker with select-all at the top.
- [x] Add tests for `llama-bench` command planning and safe profile expansion.
- [x] Implement quick, baseline, and limit-search profile planners.
- [x] Add tests for telemetry snapshots and OOM/failure classification.
- [x] Implement psutil plus NVML telemetry with `nvidia-smi` fallback.
- [x] Add tests for JSONL receipts, Markdown summaries, and recovery markers.
- [x] Implement run folders under `_runs/<timestamp>-<slug>`.
- [x] Wire a quick 5-minute per-model benchmark runner.
- [ ] Read LM Studio logs/settings as read-only profile hints.
- [x] Add a Karpathy-style loop: fixed budget, mutate one setting, measure, keep if score improves.
- [x] Add plain-English docs and a command board.
- [x] Add the required benchmark-suite ledgers and CLI wrapper before production-ready labeling.
- [x] Integrate a command-based general-purpose benchmark wrapper for EleutherAI `lm-evaluation-harness`.
- [x] Integrate a command-based agentic benchmark wrapper for Inspect AI tasks.
- [x] Add bundled local smoke, BFCL smoke, SWE-bench-style, tau2/tau3-bench-style, and repo-local deterministic task plans.
- [ ] Install and verify the heavier SWE-bench and tau2/tau3-bench harnesses before treating the heavy external plan as passing evidence.
- [x] Upgrade the autoresearch loop to Karpathy-style keep/discard/crash decisions over a comparable `agent_bench_score` backed by git history and TSV ledgers when `--benchmark-suite-plan` is provided.
- [ ] Run unit tests, smoke tests, and commit the project repo.

## Receipts

Each benchmark run writes:

- `events.jsonl`: every command, model, flag set, telemetry sample, warning, and result.
- `summary.md`: human-readable outcome with best settings and failure notes.
- `models.json`: discovered model inventory snapshot.
- `recovery.json`: last known step, crash/OOM classification, and resume instructions.

## First Supported Benchmark Profiles

- `quick`: about five minutes per model, meant to answer "is this viable?"
- `baseline`: a stronger comparison set for top candidates.
- `limit`: gradually pushes context, batch, flash attention, and GPU layers until failure or safety stop.

## Autoresearch Rule

The optimizer uses Karpathy's core pattern: fixed time budget, one metric, small setting changes, keep the change only when the recorded score improves. It now also persists Optuna trials under `_runs/learning/optuna.sqlite3`, so later runs can reuse prior results instead of starting from scratch.
This is the full Karpathy-style contract only when `--benchmark-suite-plan`
is provided. Without that plan, the loop is a system-viability scout. See
`docs\BENCHMARK-SUITE-PHASE.md`.
The context ladder starts at 4K, then climbs through 8K, 16K, 32K, and higher
contexts. Do not use an implicit/default context as the first production signal.

When learning is enabled, each new model study starts with the safe baseline settings, then Optuna proposes later trials from prior evidence. Failed trials are still useful because they teach the search to avoid unstable settings or models that fail to load.

For the current Qwen focus, use `--qwen-35b-only`; 27B and 9B models are available for smoke tests, but they are not the target queue.

For MTP-focused models, use `--mtp-only` plus `--workflow-eval`. The workflow evaluator uses `llama-cli` and adds a low-burn MTP draft probe (`--draft-max 16`) when the filename indicates MTP.

Current implemented score:


`score = generation_tokens_per_sec + prompt_tokens_per_sec / 100 + context_bonus + workflow_score + serving_tokens_per_sec / 10 - cold_serving_ttft_ms / 1000`

Failed attempts receive a large negative score so Optuna learns to avoid settings that crash or OOM.
When serving TTFT is missing, the score uses a 10-second penalty so old
speed-only receipts cannot look as strong as measured `llama-server` evidence.
The probe also records warm TTFT and warmup penalty so Qwen-style first-question
latency can be separated from warmed-up serving behavior.
The probe runs fixed questions in the same order every time: 1 at 4K, 2 at 8K,
3 at 16K, and 5 at 32K and above. Per-question rows append to
`_runs/serving-metrics.tsv` for charting.

Production-ready scoring still requires:

- `_runs\benchmark-suite.tsv` for general-purpose benchmark evidence.
- `_runs\agentic-suite.tsv` for agentic benchmark evidence.
- `_runs\agent-bench-score.tsv` for the combined benchmark-suite scalar.
- `--benchmark-suite-plan` on autoresearch, so keep/discard/crash decisions
  preserve winning settings and reject losing settings by `agent_bench_score`.
