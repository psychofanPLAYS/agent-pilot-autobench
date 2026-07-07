# 10 - Runnable presets and first live comparison

Status: runnable plan artifacts, 2026-06-27

The current example model plans have bundled benchmark-suite plans for Gemma 4
and Qwen3.6.

- `benchmarks/plans/wiki-librarian-gemma4-26b-a4b-thinking.plan.json`
- `benchmarks/plans/wiki-librarian-qwen3-moe-thinking.plan.json`

Both plans run the eight v0 librarian packs through `gguf_limit_bench.librarian_suite`
against a local llama.cpp OpenAI-compatible endpoint on `http://127.0.0.1:8080`.
They split the packs into two suite phases so the existing ledgers work:

- general: `librarian-write-entry`, `librarian-triage`, `librarian-dedupe`
- agentic: `librarian-gate`, `librarian-rerank`, `librarian-query`,
  `librarian-compress`, `librarian-contradiction`

Before any pack question is scored, `librarian_suite` now runs the five hardening
preflight gates. If model identity, template loading, Gemma BOS, Qwen thinking, or
answer-channel checks fail, the task emits `preflight_fail` receipts with `asked: 0`
instead of a bad quality score.

## First live model-vs-model comparison (Gemma and Qwen as examples)

Run one model at a time through the same path. First serve the Gemma 4 GGUF on
`127.0.0.1:8080`, then run:

```powershell
uv run --extra dev agent-autobench benchmark-suite --plan benchmarks\plans\wiki-librarian-gemma4-26b-a4b-thinking.plan.json --runs-root _runs\wiki-librarian-first-live
```

Then stop Gemma, serve the Qwen GGUF on the same endpoint with the Froggeric
v21.3 template, `--jinja`, `enable_thinking=true`, `preserve_thinking=true`,
`--reasoning on`, and `--reasoning-format deepseek`, then run:

```powershell
uv run --extra dev agent-autobench benchmark-suite --plan benchmarks\plans\wiki-librarian-qwen3-moe-thinking.plan.json --runs-root _runs\wiki-librarian-first-live
```

The browser cockpit can launch the same plan path while running a selected model:

```powershell
uv run --extra dev agent-autobench start --benchmark-suite-plan benchmarks\plans\wiki-librarian-gemma4-26b-a4b-thinking.plan.json
uv run --extra dev agent-autobench start --benchmark-suite-plan benchmarks\plans\wiki-librarian-qwen3-moe-thinking.plan.json
```

Use the Gemma 4 plan for the current Google-family challenger and the Qwen plan
for current Qwen selection work.

## Artifact contract

Each `benchmark-suite` command emits a timestamped suite directory under the chosen
runs root, for example:

- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/suite-plan.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/suite-summary.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma4_librarian_core/score.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma4_librarian_core/librarian-suite-summary.json`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma4_librarian_core/librarian-suite.tsv`
- `_runs/wiki-librarian-first-live/<timestamp>-benchmark-suite/gemma4_librarian_core/librarian-suite.md`
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
