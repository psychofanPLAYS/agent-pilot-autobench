# Agent Pilot Autobench Command Board

This is the compact command map. The public command is `agent-autobench`; `apb` is the short alias; `pilotbench` remains as an older compatibility command.

## Start

For Windows users:

```text
Double-click FIRST_RUN.bat once
```

For terminal users:

```powershell
uv run --extra dev --extra bench agent-autobench --first-run
```

After setup adds `_bin` to PATH:

```powershell
apb --start
agent-autobench --start
```

## Checks

```powershell
agent-autobench doctor
agent-autobench doctor --strict
agent-autobench doctor --json-out
```

## Model Discovery

```powershell
agent-autobench survey
agent-autobench survey --qwen-only
agent-autobench survey --qwen-35b-only
agent-autobench survey --qwen-35b-only --mtp-only
```

Default paths are repo-relative and can be overridden in `_CONFIG.toml`:

- `_models`
- `_llama`
- `_runs`
- `_db`
- `_bin`

## Results

```powershell
agent-autobench results
```

Important files:

- `_runs\leaderboard.md`
- `_runs\champion.json`
- `_runs\results.html`
- `_runs\autoresearch-results.tsv`
- `_runs\autoresearch-attempts.tsv`
- `_runs\serving-metrics.tsv`
- `_runs\benchmark-suite.tsv`
- `_runs\agentic-suite.tsv`
- `_runs\agent-bench-score.tsv`

## One Model

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --budget-minutes 5 --parallel-max 4
```

Learning is on by default. To avoid updating the local Optuna memory:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --no-learning
```

## Queue

```powershell
agent-autobench autoresearch-all --qwen-only --budget-minutes 5 --parallel-max 4
agent-autobench autoresearch-all --qwen-35b-only --total-budget-minutes 30 --budget-minutes 5 --parallel-max 4 --workflow-eval
agent-autobench autoresearch-all --qwen-35b-only --mtp-only --total-budget-minutes 20 --budget-minutes 5 --workflow-eval
```

## Serving Probe

```powershell
agent-autobench serve-probe --model "path\to\model.gguf" --context-size 4096
```

The probe records server ready time, cold TTFT, warm TTFT, warmup penalty, and serving tokens/sec. Context tiers ask the same ordered question suite each time so `_runs\serving-metrics.tsv` stays comparable.

## Benchmark Suite

Install harness extras:

```powershell
uv sync --extra dev --extra bench
```

List bundled plans:

```powershell
agent-autobench benchmark-suite-plans
```

Run a smoke plan:

```powershell
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
```

Use the suite as the optimizer target:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

Profile context falloff:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384 --context-ladder 32768
```

Profile perplexity falloff:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --perplexity-corpus "path\to\corpus.txt" --perplexity-context 4096 --perplexity-context 8192 --perplexity-context 16384
```

## Receipts

Every run writes a folder under `_runs\<timestamp>-<model>`.

Receipt files:

- `events.jsonl`: attempt events, settings, telemetry, and failure class
- `summary.md`: plain-English result
- `itemized-report.md`: attempt-by-attempt report
- `report.html`: browser report for the run
- `report.json`: machine-readable itemized report with metric coverage
- `context-profile.md`: context ladder report when `--context-ladder` is used
- `context-profile.tsv`: grep/chart-friendly context ladder rows
- `context-profile.json`: machine-readable context ladder rows
- `perplexity-profile.md`: quality falloff report when `--perplexity-corpus` is used
- `perplexity-profile.tsv`: grep/chart-friendly perplexity ladder rows
- `perplexity-profile.json`: machine-readable perplexity ladder rows
- `best-settings.json`: best settings and score
- `learning.json`: best Optuna settings when learning is enabled
- `recovery.json`: latest recovery status

Folder-level reports:

- `_runs\leaderboard.md`: run-level ranking
- `_runs\model-comparison.md`: best-known result grouped by model
- `_runs\results.html`: browser-friendly results dashboard

Open the browser report:

```powershell
agent-autobench results --open-browser
```

Serve the report folder on localhost:

```powershell
agent-autobench results --serve
```

Receipts should stay small, deterministic, and inspectable. They should record the settings, score, failure class, and short command evidence needed to understand a result later.
