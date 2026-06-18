# Agent Pilot Autobench

[![CI](https://github.com/psychofanPLAYS/agent-pilot-autobench/actions/workflows/ci.yml/badge.svg)](https://github.com/psychofanPLAYS/agent-pilot-autobench/actions/workflows/ci.yml)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Local-first GGUF and llama.cpp benchmarking for agent workloads, with reproducible
receipts instead of one-off speed claims.

Agent Pilot Autobench answers a practical question with evidence instead of guesswork:

> _**Which local model and runtime settings are actually useful for agent work?**_
> _**Which GGUF should I use to power my Assistant's Harness? (ie. Hermes Agent, OpenClaw)**_

> **You went on Hugging Face and pulled 10+ models. Let's figure out which one will serve you best.**

It wraps existing tools such as `llama-bench`, `llama-cli`, `llama-server`, Optuna, Textual, Rich, and pytest. The project records receipts, scores, failures, and champion settings so results are based on repeatable evidence instead of vibes.

**Fastest first run on Windows:** double-click `FIRST_RUN.bat`.

**What proves it works:** every experiment records its command, settings, telemetry,
score, failure class, and generated reports under `_runs/`; CI independently tests
Python 3.11, 3.12, and 3.13 and smoke-installs the built wheel.

Start with the [architecture and code map](docs/ARCHITECTURE.md), or jump directly
to the [command board](docs/COMMAND-BOARD.md).

## Start

On Windows, first run:

```text
FIRST_RUN.bat
```

That installs the local command, adds `apb` to PATH, checks the machine, and opens the model picker when the required paths are ready.

Terminal users can run the same first-run flow with:

```powershell
uv run --extra dev --extra bench agent-autobench --first-run
```

After first run adds the repo-local `_bin` folder to your user PATH, new terminals can use:

```powershell
apb --start
```

## What Setup Creates

The setup command prepares only local project state:

- `.venv`: Python environment managed by `uv`
- `_bin`: repo-local command shims for `agent-autobench` and `apb`
- `_db`: local SQLite app state
- `_runs`: benchmark receipts, ledgers, leaderboards, and HTML results

It also runs the doctor checks for model folders and llama.cpp executables. If a path is missing, setup prints the exact check that failed.

## What The App Measures

- load success or failure
- prompt and generation throughput
- context size and runtime settings
- cold and warm serving TTFT when `llama-server` probing is enabled
- workflow smoke scores when workflow evaluation is enabled
- benchmark-suite scores when a suite plan is provided
- failure class such as timeout, model load failure, crash, GPU OOM, or memory allocation failure

The optimizer follows the simple Karpathy-style loop: fixed budget, one measured score, small setting changes, keep the change only when the recorded score improves.

## Results

Each run writes a receipt folder under `_runs/<timestamp>-<model-name>/`.

Important outputs:

- `_runs/leaderboard.md`: compact ranked summary
- `_runs/model-comparison.md`: best-known result grouped by model
- `_runs/model-comparison.json`: machine-readable per-model comparison
- `_runs/results.html`: browser-friendly results report
- `_runs/champion.json`: current machine-readable champion
- `_runs/autoresearch-results.tsv`: one row per completed run
- `_runs/autoresearch-attempts.tsv`: one row per attempted setting with keep, discard, or crash
- `_runs/serving-metrics.tsv`: per-question serving metrics
- `_runs/benchmark-suite.tsv`, `_runs/agentic-suite.tsv`, `_runs/agent-bench-score.tsv`: standardized benchmark-suite evidence

Each receipt also includes:

- `itemized-report.md`: readable attempt-by-attempt report
- `report.html`: browser report for the run
- `report.json`: machine-readable itemized report, including metric coverage
- `context-profile.md`, `context-profile.tsv`, `context-profile.json`: fixed context ladder profile when enabled

The TUI remembers the last selected models and shows truncated previous-run summaries so the current session can be compared against earlier evidence without opening every receipt.

For a safe preview that starts no model server, see the
[sanitized flag-ladder dry-run artifact](docs/examples/flag-ladder-dry-run.md). It shows
the exact profiles and commands the live run would evaluate without claiming benchmark
performance that has not been measured.

## Common Commands

```powershell
apb --first-run
apb --start
agent-autobench doctor
agent-autobench results
agent-autobench results --open-browser
agent-autobench results --serve
agent-autobench benchmark-suite-plans
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
agent-autobench autoresearch --model "path\to\model.gguf" --budget-minutes 5
agent-autobench autoresearch --model "path\to\model.gguf" --flag-ladder --dry-run
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --flag-ladder --budget-minutes 20 --parallel-max 6
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384
agent-autobench autoresearch --model "path\to\model.gguf" --perplexity-corpus "path\to\corpus.txt" --perplexity-context 4096 --perplexity-context 8192
```

The short alias `apb` is created by setup for people who prefer a smaller command. The older `pilotbench` command remains available for compatibility.

## Configuration

Defaults are repo-relative so the project can be unpacked anywhere:

- models: `_models`
- llama.cpp executables: `_llama`
- results: `_runs`
- app state: `_db`
- command shims: `_bin`

Override paths and defaults in the single repo-root `_CONFIG.toml`, with environment variables, or with CLI options.

Important defaults in `_CONFIG.toml`:

- `default_preset = "deep"` for long, evidence-heavy runs by default
- `learning = true`
- `workflow_eval = true`
- `ttft_probe = true`
- `perplexity_corpus = ""` until you choose a local quality-test corpus

Useful environment variables:

```text
PILOTBENCH_MODEL_ROOTS
PILOTBENCH_LLAMA_BENCH
PILOTBENCH_LLAMA_CLI
PILOTBENCH_LLAMA_SERVER
PILOTBENCH_LLAMA_PERPLEXITY
PILOTBENCH_RUNS_ROOT
PILOTBENCH_DEFAULT_PRESET
PILOTBENCH_PARALLEL_MAX
```

To inspect resolved paths:

```powershell
agent-autobench doctor --json-out
```

## Benchmark Suite

Speed-only tests are useful for scouting, but they are not enough to prove agent usefulness.

The metric contract is intentionally practical:

- TTFT: cold and warm first-token latency from `llama-server`
- TPS: generation speed from `llama-bench` and reliable serving samples
- context growth: 4K upward through larger context tiers
- falloff: token/sec retention as context grows
- perplexity falloff: optional `llama-perplexity` profile over a fixed corpus
- metric coverage: every report says which metrics were measured, estimated, or still missing
- stability: load failures, OOM, timeout, crash, and memory allocation failures
- usefulness: benchmark-suite and workflow evidence when configured

Perplexity falloff stays `not_measured` unless you provide a real text corpus
and context tiers. That is better than pretending speed proves quality.

For stronger evidence, run a benchmark-suite plan:

```powershell
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
```

Autoresearch can optimize against the same suite score:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

For an immediately useful local comparison, run the llama.cpp flag ladder
against the 10-question SimpleBench public set:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --flag-ladder --dry-run
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --flag-ladder --budget-minutes 20 --parallel-max 6
```

`--dry-run` writes the launch plan without starting a server. The live run
starts one benchmark-owned `llama-server` process per flag profile, asks the
same 10 questions, records `transcript.jsonl` and `summary.json`, and picks the
best settings by accuracy first and speed second. Extra llama.cpp flags can be
tested with repeatable `--llama-server-extra-arg=...`.

The generated `flag-ladder-results.md` shows each independent flag ablation's
TPS slowdown versus baseline, TTFT, strict answer accuracy, warnings, and exact
receipt. MTP-named models automatically add native `--draft-max 8/16/32`
profiles. Full server logs are retained, with short `warnings.log` and
`server-tail.log` files for quick review.

For context scaling evidence, add a fixed ladder:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384 --context-ladder 32768
```

Normal, deep, and overnight presets also carry context ladder targets. Quick
Scout stays small so first runs do not unexpectedly become long tests.

For quality falloff evidence, add a fixed corpus and perplexity ladder:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --perplexity-corpus "path\to\corpus.txt" --perplexity-context 4096 --perplexity-context 8192 --perplexity-context 16384
```

See [docs/BENCHMARK-SUITE-PHASE.md](docs/BENCHMARK-SUITE-PHASE.md) for the scoring contract and [docs/PRODUCT-DESIGN.md](docs/PRODUCT-DESIGN.md) for the product direction.

## Safety And Privacy

- Runs stay local by default.
- The app does not upload models, prompts, or receipts to a cloud service.
- Exported server profiles bind to `127.0.0.1` unless you intentionally change them.
- Heavy artifacts should live in ignored folders such as `_runs`, `_models`, and `_llama`.

## Development

Install dependencies and run tests:

```powershell
uv sync --extra dev --extra bench
uv run --extra dev pytest -q
```

Narrow checks:

```powershell
uv run --extra dev python -m compileall src tests
uv run --extra dev agent-autobench doctor
uv run --extra dev agent-autobench results
```

## Verification

The release gate is intentionally copy-pasteable:

```powershell
uv sync --extra dev --locked
uv run --extra dev python -m pytest -q
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev python -m compileall -q src tests
uv build
```

GitHub Actions runs tests on Python 3.11-3.13, exercises the Windows launchers and CLI,
then builds and installs the wheel in an isolated environment. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the complete contributor workflow and
[CHANGELOG.md](CHANGELOG.md) for release-facing changes.

## Project Status

Agent Pilot Autobench is an **alpha release candidate**. Its offline release gates,
packaging, dry-run orchestration, and mocked server lifecycle are automated. A live
flag-ladder sweep on representative GGUF models and current llama.cpp binaries is the
remaining hardware acceptance gate before a stable release is considered.

## Limitations

- Benchmark results are machine-, model-, quantization-, and llama.cpp-build-specific.
- The bundled SimpleBench public snapshot contains only ten multiple-choice questions;
  it is a deterministic smoke signal, not a general intelligence score.
- A faster profile is not automatically a better agent. Use workflow or benchmark-suite
  evidence when task quality matters.
- The project currently targets Windows for the guided first-run experience. Core Python
  commands and CI also run on Linux, but the `.bat` launchers are Windows-only.
- Model downloads and llama.cpp binaries are intentionally not bundled.

## Project Documentation

- [Architecture and code map](docs/ARCHITECTURE.md)
- [Autoresearch contract](docs/AUTORESEARCH-PROGRAM.md)
- [Command board](docs/COMMAND-BOARD.md)
- [Product design](docs/PRODUCT-DESIGN.md)
- [Security policy](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).

The bundled SimpleBench public snapshot is also MIT-licensed by its upstream authors;
see the included [dataset notice](src/gguf_limit_bench/data/SIMPLEBENCH_NOTICE.md) for
the pinned source revision and checksum.

## Tiny Glossary

- `apb`: short for Agent Pilot Autobench, the quick command alias.
- `pilotBENCHY`: the friendly TUI/workflow name for the benchmark cockpit.
