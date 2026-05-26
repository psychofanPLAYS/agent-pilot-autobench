# Agent Pilot Autobench Implementation Plan

Goal: build a repeatable TUI benchmark cockpit for a local GGUF collection, starting with Qwen-family models and RTX 4090-style Windows workstations.

Architecture: keep this project in one folder, use ready-made tools for heavy work, and store searchable receipts for every run. The app wraps `llama-bench`, LM Studio `lms`, NVML telemetry, Optuna learning, and small benchmark prompts; it does not reinvent inference.

Tech stack: Python, Textual, Rich, Typer, pytest, psutil, nvidia-ml-py, Optuna, `llama-bench.exe`, LM Studio `lms`.

## 18 Step Plan

- [x] Back up the local Codex config and register selected skill folders by path.
- [x] Create the project folder and initialize git.
- [x] Create a small project plan, receipts folder, and test-first structure.
- [x] Add tests for GGUF discovery, family parsing, quant parsing, and mmproj vision detection.
- [ ] Implement discovery from `G:\AI\models` and LM Studio metadata caches.
- [x] Add tests for model selection state: space toggles rows, select-all selects all.
- [x] Implement a Textual model picker with select-all at the top.
- [x] Add tests for `llama-bench` command planning and safe profile expansion.
- [x] Implement quick, baseline, and limit-search profile planners.
- [x] Add tests for telemetry snapshots and OOM/failure classification.
- [x] Implement psutil plus NVML telemetry with `nvidia-smi` fallback.
- [x] Add tests for JSONL receipts, Markdown summaries, and recovery markers.
- [x] Implement run folders under `runs/<timestamp>-<slug>`.
- [x] Wire a quick 5-minute per-model benchmark runner.
- [ ] Read LM Studio logs/settings as read-only profile hints.
- [x] Add a Karpathy-style loop: fixed budget, mutate one setting, measure, keep if score improves.
- [x] Add plain-English docs and a command board.
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

The optimizer uses Karpathy's core pattern: fixed time budget, one metric, small setting changes, keep the change only when the recorded score improves. It now also persists Optuna trials under `runs/learning/optuna.sqlite3`, so later runs can reuse prior results instead of starting from scratch.

When learning is enabled, each new model study starts with the safe baseline settings, then Optuna proposes later trials from prior evidence. Failed trials are still useful because they teach the search to avoid unstable settings or models that fail to load.

For the current Qwen focus, use `--qwen-35b-only`; 27B and 9B models are available for smoke tests, but they are not the target queue.

For MTP-focused models, use `--mtp-only` plus `--workflow-eval`. The workflow evaluator uses `llama-cli` and adds a low-burn MTP draft probe (`--draft-max 16`) when the filename indicates MTP.

Current implemented score:


`score = generation_tokens_per_sec + prompt_tokens_per_sec / 100 + context_bonus - ttft_penalty`

Failed attempts receive a large negative score so Optuna learns to avoid settings that crash or OOM.
