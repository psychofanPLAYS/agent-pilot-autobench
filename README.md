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

**Fastest start on Windows:** double-click `INSTALL.bat` once, then type `apb`.

**What proves it works:** every experiment records its command, settings, telemetry,
score, failure class, and generated reports under `_runs/`; CI independently tests
Python 3.11, 3.12, and 3.13 and smoke-installs the built wheel.

Start with the [architecture and code map](docs/ARCHITECTURE.md), or jump directly
to the [command board](docs/COMMAND-BOARD.md).

## Start

Two steps, total. **Install once:**

```text
INSTALL.bat
```

(or from a terminal: `powershell -ExecutionPolicy Bypass -File install.ps1`)

That one command auto-installs `uv` if needed, builds the local environment,
pulls dependencies, adds `apb` to your PATH, and checks the machine — no flags,
nothing else to run.

**Then, from any terminal, just type:**

```powershell
apb
```

That's it. Plain `apb` opens the browser cockpit. The very first time you run it on
a new machine it sets itself up automatically before opening; every run after
that goes straight to the app. Power-user subcommands still exist — run
`apb --help` to see them.

## Cockpit Modes

Plain `apb` (or double-clicking `START.bat`) opens the local browser cockpit.
The browser is the primary workflow for model selection, benchmark-suite plan
selection, live WebSocket run progress, telemetry, and receipt links. Pick
model(s) with checkboxes, choose a mode from the menu, review the
recommended/forced llama.cpp flags, then press **Start benchmark**. While a run is
active, the cockpit shows the live activity feed, telemetry, current winner, and
receipt/report links so you do not have to dig through `_runs` by hand. The older
terminal cockpit is still available with `apb tui` as a fallback.

- **Quick check** — does it load, and how fast? No questions asked (fast scout).
- **Find best settings** — walk the flag ladder, ask the questions, crown the best settings.
- **Librarian bot test** — score any local model(s) as memory/RAG agent workers, head-to-head.
- **How flags affect speed** — see how each llama.cpp flag changes tok/s and TTFT.
- **Context limits** — how much context fits, and how long context affects tok/s.
- **Deep / overnight** — everything, big budget; keeps searching and converging.
- **Custom** — you choose the time.

Each mode maps to a time budget (measured in Andrej Karpathy's fixed 5-minute
rounds), whether the SimpleBench questions are asked, and which ladders run. The
cockpit shows the active mode, live status/telemetry, and the champion
(best model and its best settings) when a run finishes.

## What Setup Creates

The setup command prepares only local project state:

- `.venv`: Python environment managed by `uv`
- `_bin`: repo-local command shims for `agent-autobench` and `apb`
- `_db`: local SQLite app state
- `_runs`: benchmark receipts, ledgers, leaderboards, and HTML results

It also runs the doctor checks for model folders and llama.cpp executables. If a path is missing, setup prints the exact check that failed.

## What The App Measures

By default a run **asks the model the bundled 10-question SimpleBench set through
`llama-server`, one question at a time with a fresh session each, and scores the
answers against the key** — so the result reflects reasoning, not just speed. For
each runtime profile it records:

- answer accuracy (accuracy-first; speed is only a tiebreaker)
- generation throughput (tok/s) and **prefill / prompt-processing throughput** (tok/s)
- generation-speed stability (stddev) and TTFT, including p90/p99 tail latency
- context size and the exact llama.cpp flags used
- load success or failure, and failure class (timeout, model-load failure, crash,
  GPU OOM, memory allocation failure)
- workflow smoke scores and benchmark-suite scores when those are enabled

The loop follows Karpathy's autoresearch pattern — a fixed time budget per round,
one comparable score, keep a change only when the score improves, durable receipts
— and walks an ordered llama.cpp flag ladder first. When learning is on and the
budget allows (a long or overnight run), it then keeps searching with a
**persistent per-model Optuna study** that warm-starts from the previous session,
so each run starts smarter and converges toward the best settings.

For a fast "does it load and roughly how fast" check that asks no questions, use
`--speed-scout` (or the **Quick check** cockpit mode).

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

The fallback TUI remembers the last selected models and shows truncated previous-run summaries so the current session can be compared against earlier evidence without opening every receipt.

For a safe preview that starts no model server, see the
[sanitized flag-ladder dry-run artifact](docs/examples/flag-ladder-dry-run.md). It shows
the exact profiles and commands the live run would evaluate without claiming benchmark
performance that has not been measured.

## Common Commands

```powershell
apb
agent-autobench doctor
agent-autobench results
agent-autobench results --open-browser
agent-autobench results --serve
agent-autobench benchmark-suite-plans
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --budget-minutes 20
agent-autobench autoresearch --model "path\to\model.gguf" --speed-scout --budget-minutes 5
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --dry-run
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --parallel-max 6
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384
agent-autobench autoresearch --model "path\to\model.gguf" --perplexity-corpus "path\to\corpus.txt" --perplexity-context 4096 --perplexity-context 8192
```

By default `autoresearch` runs in **benchmark mode**: it asks the SimpleBench
questions through `llama-server` and walks the flag ladder, so it needs
`--llama-server`. Add `--speed-scout` for the fast synthetic `llama-bench` probe
that asks no questions. `--dry-run` writes the launch plan without starting a
server. The older `--flag-ladder` flag is kept as a compatibility alias for the
benchmark default.

The short alias `apb` is created by setup for people who prefer a smaller command. The older `pilotbench` command remains available for compatibility.

## Configuration

Defaults are repo-relative so the project can be unpacked anywhere:

- models: `_models`
- llama.cpp executables: `_llama`
- results: `_runs`
- app state: `_db`
- command shims: `_bin`

Override paths and defaults in the single repo-root `_CONFIG.toml`, with environment variables, or with CLI options.

Important settings in `_CONFIG.toml`:

- `learning = true` so long runs keep searching and remember across sessions
- `workflow_eval = true`
- `ttft_probe = true`
- `parallel_max = 4` caps the parallel slots the capability test will try
- `forced_server_args = []` — raw llama-server flags forced ON for every benchmark
  profile (e.g. `["--no-mmap"]`); managed flags like `--host`/`--port`/`--model`
  are rejected
- `perplexity_corpus = ""` until you choose a local quality-test corpus

The cockpit now drives runs by **mode** (see Cockpit Modes above). The older
`default_preset` still applies to non-cockpit code paths but the mode you pick in
the browser cockpit or TUI takes precedence there.

Keep your machine-specific paths out of the tracked `_CONFIG.toml` (it ships with
repo-relative defaults). Set them with environment variables instead, which the
`apb setup` flow can persist for you:

```text
PILOTBENCH_MODEL_ROOTS
PILOTBENCH_LLAMA_BENCH
PILOTBENCH_LLAMA_CLI
PILOTBENCH_LLAMA_SERVER
PILOTBENCH_LLAMA_PERPLEXITY
PILOTBENCH_RUNS_ROOT
PILOTBENCH_DEFAULT_PRESET
PILOTBENCH_PARALLEL_MAX
PILOTBENCH_FORCED_SERVER_ARGS
```

To inspect resolved paths:

```powershell
agent-autobench doctor --json-out
```

## Benchmark Suite

Speed-only tests are useful for scouting, but they are not enough to prove agent usefulness.

The metric contract is intentionally practical:

- TTFT: first-token latency from `llama-server`, with p90/p99 tail latency
- TPS: generation speed plus prompt/prefill throughput from `llama-server`, with
  generation-speed stability (stddev) across the question batch
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

The flag ladder is what a benchmark-mode run does by default: it starts one
benchmark-owned `llama-server` per flag profile and asks the same 10 SimpleBench
questions in the same order, recording `transcript.jsonl` and `summary.json` and
picking the best settings by accuracy first, speed second.

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --dry-run
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --budget-minutes 20 --parallel-max 6
```

`--dry-run` writes the launch plan without starting a server. The ladder walks
rungs in order: a stripped bare-minimum baseline (`Lmin-stripped`), then the
standard flags one at a time (`--kv-unified`, RAM cache, cache reuse, context
checkpoints, q8 KV, a thread sweep) — all at single stream so the tok/s
comparison is clean. Concurrency (`--parallel`) is a capability axis rather than
a single-stream speed axis, so it is measured **last** (`Lpar-2`, `Lpar-3`). When
the model filename signals preserved MTP heads and the local llama.cpp build
supports them, the ladder adds one native speculative profile using
`--spec-type draft-mtp --spec-draft-n-max` (not the removed `--draft-max`). Extra
flags can be tested with repeatable `--llama-server-extra-arg=...`, and
`forced_server_args` in `_CONFIG.toml` forces flags on for every profile.

The generated `flag-ladder-results.md` shows each profile's answer accuracy,
generation tok/s, prefill tok/s, slowdown versus the `L0-baseline`, TTFT,
warnings, and exact receipt. Full server logs are retained, with short
`warnings.log` and `server-tail.log` files for quick review.

On a long or overnight run, once the ladder is walked the loop keeps searching
with a persistent per-model Optuna study, so it converges toward the best settings
and starts smarter next session.

For context scaling evidence, add a fixed ladder:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384 --context-ladder 32768
```

The Context limits and Deep / overnight cockpit modes carry context-ladder
targets; Quick check stays small so a first run does not unexpectedly become a
long test.

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
packaging, dry-run orchestration, and mocked server lifecycle are automated, and a
live benchmark run — asking and scoring the SimpleBench set through `llama-server`
across the flag ladder — has been exercised on local GGUF models and current
llama.cpp binaries. Broader live sweeps across more models and hardware remain the
path to a stable release.

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
- Agent Pilot cockpit: the browser-first workflow for running benchmarks and comparing models.
