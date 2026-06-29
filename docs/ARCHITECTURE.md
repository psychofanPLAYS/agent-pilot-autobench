# Architecture and Code Map

Agent Pilot Autobench is a local-first Python CLI plus the pilotBENCHY web UI around existing llama.cpp tools.
It does not replace `llama-bench`, `llama-cli`, or `llama-server`; it makes experiments
repeatable, records evidence, and compares candidates under explicit budgets.

## Data Flow

```text
CLI, WebSocket pilotBENCHY web UI, or terminal TUI fallback
  -> resolved config and discovered GGUF models
  -> benchmark/autoresearch runner
  -> llama.cpp subprocess or benchmark harness
  -> telemetry, score, and failure classification
  -> append-only ledgers and per-run receipt
  -> Markdown, JSON, TSV, and HTML reports
```

Subprocesses receive argument lists directly (`shell=False`). Benchmark-controlled
servers bind to `127.0.0.1` by default. Generated models, databases, environments, and
receipts stay in ignored project-local directories.

## Feature Map

| Feature | Command | Main implementation | Outputs | Primary tests |
| --- | --- | --- | --- | --- |
| First-run setup | `agent-autobench --first-run` | `cli.py`, `installer.py`, `doctor.py` | `_bin/`, `_db/`, `_runs/` | `test_cli.py`, `test_installer.py`, `test_doctor.py` |
| pilotBENCHY web UI | `apb` / `agent-autobench start` | `webui.py`, `cli.py`, `discovery.py`, FastAPI/WebSocket service | selected model paths, WebSocket run events, telemetry, receipts | `test_webui.py`, `test_cli.py` |
| Model discovery | `agent-autobench survey` | `discovery.py`, `selection.py`, `tui.py` | selected model paths | `test_discovery.py`, `test_selection.py`, `test_tui.py` |
| Raw speed probe | `agent-autobench quick` | `runner.py`, `autoresearch.py` | receipt and speed metrics | `test_autoresearch.py` |
| Adaptive autoresearch | `agent-autobench autoresearch` | `autoresearch.py`, `learning.py` | best settings, attempts/results TSV | `test_autoresearch.py`, `test_learning.py` |
| Flag ablations | `autoresearch --flag-ladder` | `flag_ladder.py`, `simple_bench.py`, `simple_bench_runner.py` | transcripts, warnings, comparison table | `test_simple_bench.py`, `test_cli.py` |
| Serving TTFT | `agent-autobench serve-probe` | `server_probe.py` | cold/warm TTFT and TPS samples | `test_server_probe.py` |
| Benchmark suites | `agent-autobench benchmark-suite` | `bench_plan.py`, `benchmark_suite.py` | suite receipts and score ledgers | `test_benchmark_suite.py`, `test_cli.py` |
| Reports | `agent-autobench results` | `reports.py`, `run_report.py`, `run_history.py` | leaderboard, model comparison, HTML | `test_reports.py`, `test_run_history.py` |
| Profile export | `agent-autobench export-profile` | `deployment.py` | YAML and PowerShell launch profile | `test_deployment.py` |

All source modules live under `src/gguf_limit_bench/`. The Typer application in
`cli.py` is the public entry point for `agent-autobench`, `apb`, and the compatibility
alias `pilotbench`.

## Evidence Model

`RunReceipt` creates a unique timestamped directory and writes JSONL events plus a
recovery file during execution. Successful and failed attempts are both retained.
Top-level TSV ledgers support comparisons across runs; per-run reports retain exact
settings and bounded stdout/stderr tails. Server launch argv is stored as JSON data,
not as a double-clickable command script. The report layer does not upgrade missing
quality measurements into inferred claims.

## Boundaries

- `config.py` resolves file, CLI, and environment configuration.
- `autoresearch.py` owns candidate selection, budgets, scoring, and receipts.
- `flag_ladder.py` owns independent llama-server argument profiles.
- `simple_bench.py` owns the fixed public dataset contract and answer scoring.
- `simple_bench_runner.py` owns one benchmark server lifecycle and question batch.
- `benchmark_suite.py` executes external harness commands with timeouts and redacted
  environment receipts.
- `reports.py` and `run_report.py` turn recorded evidence into public artifacts.

## Generated and Ignored State

| Path | Purpose |
| --- | --- |
| `.venv/` | uv-managed development environment |
| `_models/` | optional local GGUF model directory |
| `_llama/` | optional local llama.cpp executables |
| `_db/` | SQLite app and learning state |
| `_runs/` | current receipts, ledgers, and reports |
| `runs/`, `db/` | legacy-compatible local state |

These paths must not be committed. Small source-controlled benchmark plans live under
`benchmarks/`; bundled public SimpleBench data and its license notice live under
`src/gguf_limit_bench/data/`.

## Quality Gates

The repository enforces formatting, linting, tests on Python 3.11-3.13, MyPy,
compilation, package build, isolated wheel installation, bundled-data loading, and CLI
startup. The exact local commands are in [CONTRIBUTING.md](../CONTRIBUTING.md).
