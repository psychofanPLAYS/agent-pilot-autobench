# Architecture and Code Map

pilotBENCHY is a local-first benchmarking app around existing llama.cpp tools. The
**web cockpit is the primary surface** a user interacts with; a CLI and TUI exist for
power users and automation. It does not replace `llama-bench`, `llama-cli`, or
`llama-server`; it makes experiments repeatable, records evidence, and compares
candidates under explicit budgets.

---

## ⚠️ DESIGN DIRECTION — SSOT, NON-NEGOTIABLE (read before touching web/engine)

This is the single source of truth for how the run-flow is built. Any change that
violates these must update this section first (with the owner's sign-off) — do **not**
silently build a different architecture. (A prior agent diverged by re-adding
in-process web evaluation; that is exactly what these rules forbid.)

1. **The engine is the only thing that evaluates.** Evaluation runs in a **detached
   engine subprocess** (its own process group), launched by `cli.py`'s `engine`
   command. It runs the models, scores them, and writes the canonical record.
2. **The web UI is a THIN CLIENT and the PRIMARY user surface.** Two jobs only:
   (a) pass instructions to the engine, (b) render what the engine wrote. It **never**
   computes a score, queries a model, or holds evaluation state. It must ALSO be a
   *work of art* — beautiful and genuinely functional, since it is what users see.
   `serve_webui` / `WebUiState` must **not** accept a `run_model` callback or run eval
   in-process. The web server may die and the run continues.
3. **The run directory on disk is the ONLY seam** between web and engine
   (`run-spec.json`, `control.json`, `status.json`, `live.jsonl`, `telemetry.jsonl`,
   receipts). No other coupling.
4. **Impeccable process & window hygiene** — detached process groups, kill-tree on
   stop/abort (no orphaned engine or llama-server), a status heartbeat, and reattach
   on browser refresh (a refresh must never kill or orphan an in-flight run).
5. **Sequential by default.** One model fully evaluated before the next; the only
   parallel case is a deliberate concurrency "pressure test" against a single model.

Deep reference: `docs/superpowers/specs/2026-06-30-inflight-cockpit-design.md`.
Modules that implement the seam: `engine.py`, `run_dir.py`, `events.py`,
`server_probe.py` (kill-tree), and the thin `webui.py`.

## Data Flow (detached-engine architecture)

```text
web cockpit (thin)          run directory (the ONLY seam)         engine (detached)
  browser renders    <— WS —>  control.json / status.json  <——   sequential runner
  web server               live.jsonl / telemetry.jsonl / metrics —>  llama.cpp
  passes orders  —— r/w ——>  run-spec.json  ————————————————————>  (own proc group)
                                     |
                              receipts + Markdown/JSON/TSV/HTML reports
```

The web server **launches the engine detached and returns immediately** — it never
evaluates in-process. The engine appends `live.jsonl` (per-question thinking/answer/
score + lifecycle) and rewrites `status.json` (heartbeat) every ~2s; the UI tails from
a cursor and re-reads from offset 0 on reconnect. Subprocesses receive argument lists
directly (`shell=False`); benchmark-controlled servers bind to `127.0.0.1` by default.
Generated models, databases, environments, and receipts stay in ignored project-local
directories.

## Feature Map

| Feature | Command | Main implementation | Outputs | Primary tests |
| --- | --- | --- | --- | --- |
| First-run setup | `agent-autobench --first-run` | `cli.py`, `installer.py`, `doctor.py` | `_bin/`, `_db/`, `_runs/` | `test_cli.py`, `test_installer.py`, `test_doctor.py` |
| Browser cockpit (thin client, PRIMARY surface) | `pilotbench start` | `webui.py` (renders only; spawns detached engine, tails run dir) | run-spec.json + rendered live cockpit | `test_webui.py`, `test_cli.py` |
| Detached engine (does ALL eval) | `pilotbench engine --run-dir` | `engine.py`, `run_dir.py`, `events.py` | live.jsonl, status.json, receipts | `test_engine.py`, `test_run_dir.py`, `test_events.py` |
| GPU-free cockpit replay | `pilotbench engine-replay` | `engine_replay.py` | re-streamed live.jsonl | `test_engine_replay.py` |
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
