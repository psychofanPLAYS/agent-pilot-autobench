# 10 - Runnable presets and first live comparison

Status: runnable plan artifacts, 2026-06-27

The Gemma and Qwen model plans now have bundled benchmark-suite plans:

- `benchmarks/plans/wiki-librarian-gemma3-27b-direct.plan.json`
- `benchmarks/plans/wiki-librarian-qwen3-moe-thinking.plan.json`

Both plans run the seven v0 librarian packs through `gguf_limit_bench.librarian_suite`
against a local llama.cpp OpenAI-compatible endpoint on `http://127.0.0.1:8080`.
They split the packs into two suite phases so the existing ledgers work:

- general: `librarian-write-entry`, `librarian-triage`, `librarian-dedupe`
- agentic: `librarian-gate`, `librarian-rerank`, `librarian-compress`,
  `librarian-contradiction`

Before any pack question is scored, `librarian_suite` now runs the five hardening
preflight gates. If model identity, template loading, Gemma BOS, Qwen thinking, or
answer-channel checks fail, the task emits `preflight_fail` receipts with `asked: 0`
instead of a bad quality score.

## First live Gemma-vs-Qwen comparison

Run one model at a time through the same path. First serve the Gemma GGUF on
`127.0.0.1:8080`, then run:

```powershell
uv run --extra dev agent-autobench benchmark-suite --plan benchmarks\plans\wiki-librarian-gemma3-27b-direct.plan.json --runs-root _runs\wiki-librarian-first-live
```

Then stop Gemma, serve the Qwen GGUF on the same endpoint, preferably with the
froggeric-v19 template and `--jinja`, then run:

```powershell
uv run --extra dev agent-autobench benchmark-suite --plan benchmarks\plans\wiki-librarian-qwen3-moe-thinking.plan.json --runs-root _runs\wiki-librarian-first-live
```

The browser cockpit can launch the same plan path while running a selected model:

```powershell
uv run --extra dev agent-autobench start --benchmark-suite-plan benchmarks\plans\wiki-librarian-gemma3-27b-direct.plan.json
uv run --extra dev agent-autobench start --benchmark-suite-plan benchmarks\plans\wiki-librarian-qwen3-moe-thinking.plan.json
```

Use the Gemma plan for the Gemma model selection and the Qwen plan for the Qwen
model selection.

## Artifact contract

Each `benchmark-suite` command emits a timestamped suite directory under the chosen
runs root, for example:

- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/suite-plan.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/suite-summary.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma_librarian_core/score.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma_librarian_core/librarian-suite-summary.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma_librarian_core/librarian-suite.tsv`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma_librarian_core/librarian-suite.md`
- one per-pack JSON file in each task directory, such as `librarian-gate.json`

The runs root also receives the cross-run ledgers:

- `_runs/wiki-librarian-first-live/benchmark-suite.tsv`
- `_runs/wiki-librarian-first-live/agentic-suite.tsv`
- `_runs/wiki-librarian-first-live/agent-bench-score.tsv`

Compare the Gemma and Qwen rows in `agent-bench-score.tsv` first. Then inspect each
run's `librarian-suite.tsv` files to see which librarian job family caused the win
or loss.

## Current limits

These are first live presets, not the full cube. They encode the model-plan metadata
and first recommended settings, but they do not yet automate every quant, context,
template negative-control, MTP, or seed-repeat rung from docs 07 and 08. The preset
plans still need real live serving flags and model identity evidence to pass preflight.
