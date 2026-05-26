# Agent Pilot Autobench Command Board

This project is a local-first pilot tester for finding practical Hermes-agent GGUF settings without loading models one by one in LM Studio.

Hero command after install: `agent-autobench`

Tiny shortcut after install: `apb`

Beginner command before install:

```powershell
uv run --extra dev agent-autobench first-run
```

Beginner command after install:

```powershell
agent-autobench first-run
```

Tiny shortcut:

```powershell
apb first-run
```

`first-run` is the friendly installer. It syncs the `.venv`, creates the Windows
command shims, checks the machine, prepares local state, explains missing tools
or folders, and tells the user what to run next.

## Safe First Commands

For complete beginners on Windows:

```text
Double-click START-HERE.bat
```

To make `agent-autobench` work from any terminal folder:

```text
Double-click INSTALL-COMMAND.bat
```

Or run the installer with the PATH option:

```powershell
agent-autobench first-run --add-to-path
```

Prepare the app for the first time before installing the command:

```powershell
uv run --extra dev agent-autobench first-run
```

Prepare the app for the first time after installing the command:

```powershell
agent-autobench first-run
```

Open the easy model picker from a terminal:

```powershell
agent-autobench --start
```

Check only, without opening the picker:

```powershell
agent-autobench --start --check-only
```

Check whether the local folders and llama.cpp tools are ready:

```powershell
uv run --extra dev pilotbench doctor
```

Fail fast in scripts when a required path is missing:

```powershell
uv run --extra dev pilotbench doctor --strict
```

List Qwen-family GGUF models:

```powershell
uv run --extra dev pilotbench survey --qwen-only
```

List only Qwen 35B models:

```powershell
uv run --extra dev pilotbench survey --qwen-35b-only
```

List only Qwen 35B MTP models:

```powershell
uv run --extra dev pilotbench survey --qwen-35b-only --mtp-only
```

Default discovery should cover common G-drive model roots:

- `G:\AI\models`
- `G:\AI\models\LM_Studio-gguf`

Open the model picker TUI:

```powershell
agent-autobench --start
```

Open the picker for suite-backed production-readiness runs:

```powershell
agent-autobench start --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

Show the latest champion and write `runs\leaderboard.md` plus `runs\champion.json`:

```powershell
uv run --extra dev pilotbench results
```

Measure real local serving TTFT, warmed-up TTFT, and serving token/sec for one model:

```powershell
uv run --extra dev pilotbench serve-probe --model "G:\AI\models\path\to\model.gguf" --context-size 4096
```

By default this sends three requests to one live `llama-server` process with
`cache_prompt` enabled. That exposes the first-question tax seen during use:
`server ready`, `server start to first token`, `cold TTFT`, `warm TTFT`, and
`warmup penalty`.

The probe uses the fixed agent question suite in the same order every time:
4K asks question 1, 8K asks questions 1-2, 16K asks questions 1-3, and 32K+
asks all 5. That makes `runs\serving-metrics.tsv` chartable over time.

Create the local SQLite experiment memory:

```powershell
uv run --extra dev pilotbench init-db
```

List benchmark packs:

```powershell
uv run --extra dev pilotbench packs
```

Export the latest champion as deployment files:

```powershell
uv run --extra dev pilotbench export-profile
```

Exported server profiles bind to `127.0.0.1` by default. Change that only when you intentionally want LAN or Tailscale access.

## Install The Windows Command

Use this when a beginner wants to type `agent-autobench` from any terminal
folder:

```text
Double-click INSTALL-COMMAND.bat
```

What it does:

- Creates `G:\_codex_global\bin\agent-autobench.bat`.
- Creates `G:\_codex_global\bin\apb.bat`.
- Points that shim back to this repo.
- Asks before adding `G:\_codex_global\bin` to the user PATH.
- Does not touch the system PATH.
- Does not need admin rights.

The `first-run` command also creates those shims. Use `INSTALL-COMMAND.bat` when
you want the double-click PATH prompt.

After adding PATH, close and reopen the terminal, then run:

```powershell
agent-autobench first-run
apb first-run
```

Run the basic autoresearch loop for one model:

```powershell
uv run --extra dev pilotbench autoresearch --model "G:\AI\models\path\to\model.gguf" --budget-minutes 5 --parallel-max 4
```

By default, autoresearch also starts `llama-server` for the winning attempt and
records cold streaming TTFT, warm streaming TTFT, warmup penalty, and serving
token/sec. It also records `llama-server` launch-to-ready time. Use
`--no-ttft-probe` only when you deliberately want a quick `llama-bench`
speed-only run.

Autoresearch starts context at 4K. As successful settings climb the context
ladder, the serving probe asks progressively more ordered questions: 1 at 4K,
2 at 8K, 3 at 16K, and 5 at 32K and above.

Run the real Karpathy-style loop with the benchmark suite as the score:

```powershell
uv run --extra dev --extra bench pilotbench autoresearch `
  --model "G:\AI\models\path\to\model.gguf" `
  --budget-minutes 20 `
  --parallel-max 4 `
  --benchmark-suite-plan benchmark-suite.plan.json
```

When `--benchmark-suite-plan` is present, each successful attempt runs the
general and agentic suite and the loop uses `agent_bench_score` for
`keep`/`discard`/`crash`. This is the production-readiness path; raw speed-only
runs are still useful scouting, but they are not production-ready evidence.

Learning is on by default. To do a one-off run without updating the learning database:

```powershell
uv run --extra dev pilotbench autoresearch --model "G:\AI\models\path\to\model.gguf" --no-learning
```

Run the basic autoresearch loop across discovered Qwen models:

```powershell
uv run --extra dev pilotbench autoresearch-all --qwen-only --budget-minutes 5 --parallel-max 4
```

Run the recommended Qwen 35B queue:

```powershell
uv run --extra dev pilotbench autoresearch-all --qwen-35b-only --total-budget-minutes 30 --budget-minutes 5 --parallel-max 4 --workflow-eval
```

Run only Qwen 35B MTP candidates:

```powershell
uv run --extra dev pilotbench autoresearch-all --qwen-35b-only --mtp-only --total-budget-minutes 20 --budget-minutes 5 --workflow-eval
```

For a Codex-friendly bulk loop, the target behavior is:

- Accept a total wall-clock budget for the whole queue, not only a per-model budget.
- Support a finish-early control so one strong-enough result can stop the queue before spending the full budget.
- Keep every run resumable from its receipt folder, so Codex can inspect what happened without rerunning expensive benchmarks.

## Receipts

Each autoresearch run writes a folder under `runs/<timestamp>-<model>`:

- `events.jsonl`: every attempt, settings, result, telemetry, and failure class.
- `summary.md`: plain-English best result.
- `best-settings.json`: machine-readable best settings and score.
- `learning.json`: best learned settings when learning is enabled.
- `recovery.json`: latest recovery status.
- `leaderboard.md`: ranked cross-run summary written under `runs\`.
- `champion.json`: latest champion written under `runs\`.

The cross-run chart ledgers are:

- `runs\autoresearch-results.tsv`: one compact row per run.
- `runs\autoresearch-attempts.tsv`: one row per attempted setting with
  `keep`, `discard`, or `crash`, plus current git branch/commit metadata.
- `runs\serving-metrics.tsv`: one row per serving question, stable question order.
- `runs\benchmark-suite.tsv`: one row per general-purpose benchmark task.
- `runs\agentic-suite.tsv`: one row per agentic benchmark task.
- `runs\agent-bench-score.tsv`: the combined `agent_bench_score` gate.

Receipts should stay small and deterministic. They should contain the settings, scores, failure class, and short command output needed for debugging, but not huge raw logs or machine-specific noise. This makes them friendly for Codex, Git diffs, and quick human review.

## Learning Brain

The app uses Optuna with a local SQLite database at `runs/learning/optuna.sqlite3`.

In plain English:

- Before an attempt, Optuna suggests settings from the search space.
- The app runs `llama-bench`.
- The app scores the result.
- The score is written back to Optuna.
- Future runs for the same model reuse those past trials instead of starting blind.

This is real persistent optimization, but it is still local and small. It does not train a huge neural network or send benchmark data to a cloud service.

## Current Autoresearch Shape

The loop follows the Karpathy-style autoresearch shape:

- Fixed wall-clock budget.
- One main score.
- Baseline first.
- Mutate one setting at a time when learning is off.
- Use Optuna's persistent suggestions when learning is on, while still accepting only measured score improvements.
- Measure both `llama-bench` speed and `llama-server` cold/warm streaming TTFT plus serving token/sec when the TTFT probe is enabled.
- Record failures such as GPU OOM, model-load failure, crash, and timeout, then keep going.
- Keep unified KV cache marked as mandatory target metadata for Hermes deployment.

Required production-ready gate:

- Use the benchmark-suite phase from `docs\BENCHMARK-SUITE-PHASE.md`.
- General-purpose benchmarks should use an existing harness first, starting with
  EleutherAI `lm-evaluation-harness`, and append rows to
  `runs\benchmark-suite.tsv`.
- Agentic benchmarks should cover BFCL-style tool/function calling,
  SWE-bench-style coding-agent tasks, tau2-bench/tau3-bench-style task
  completion, and repo-local deterministic tasks for JSON repair, command
  safety, receipt inspection, and planning. Append rows to
  `runs\agentic-suite.tsv`.
- Pass `--benchmark-suite-plan` to make autoresearch use Karpathy-style
  keep/discard/crash decisions over a real `agent_bench_score`.

## Benchmark Suite Phase

Install and sync all benchmark harness wrappers:

```powershell
uv sync --extra dev --extra bench
```

The installed base harness commands are:

```powershell
uv run --extra bench lm-eval --help
uv run --extra bench inspect --help
.venv-bfcl\Scripts\bfcl.exe --help
```

Write an editable external-harness plan:

```powershell
agent-autobench benchmark-suite-template --output benchmark-suite.plan.json --model "qwen-local"
```

List bundled plans:

```powershell
agent-autobench benchmark-suite-plans
```

Useful bundled plans:

- `benchmarks\plans\local-openai-smoke.plan.json`: `lm-eval` GSM8K CoT
  zero-shot smoke plus repo-local Inspect JSON repair, for a local
  OpenAI-compatible endpoint.
- `benchmarks\plans\local-bfcl-smoke.plan.json`: GSM8K CoT zero-shot plus BFCL
  `simple_python`, using `.venv-bfcl`.
- `benchmarks\plans\external-agentic-heavy.plan.json`: SWE-bench and tau2
  integration plan. It is expected to fail honestly until those external
  harnesses are installed and configured.

Run the plan:

```powershell
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
```

Run autoresearch with the plan as the actual optimizer target:

```powershell
agent-autobench autoresearch --model "G:\AI\models\path\to\model.gguf" --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

The easy picker also accepts the same plan:

```powershell
agent-autobench start --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

The plan format is deliberately command-based. Each task calls a real external
harness such as `lm-eval run` or `inspect eval`, then PilotBENCHY captures the
receipt, extracts a numeric score from JSON output, and appends the phase TSVs.
If a harness is missing, crashes, or does not produce a score, the suite writes
failed evidence instead of pretending the run is useful.

Tasks may use either one `command` or a sequence of `commands`. The sequence form
is for harnesses such as BFCL that generate model responses first and evaluate
them second.

BFCL is installed in an isolated Python 3.11 venv because Python 3.13 tries to
build `tree-sitter` locally. Use `.venv-bfcl\Scripts\bfcl.exe generate ...`
and `.venv-bfcl\Scripts\bfcl.exe evaluate ...` inside a suite task's `commands`
when adding the BFCL task group.

## Beginner Presets

The TUI presents simple presets first:

- `Quick Scout`: load and speed sanity check.
- `Normal`: good default run.
- `Deep Pilot`: serious agent-pilot run, up to 20 minutes per model.
- `Overnight`: longer campaign controlled by a total time cap.

Advanced settings should expose the real llama.cpp knobs in plain groups: context, KV cache, MTP, batch, ubatch, threads, parallel slots, flash attention, and deployment targets.

## MTP Notes

The app detects MTP candidates from the model filename and exposes `--mtp-only` for focused queues. For real workflow evals, MTP candidates get a llama-cli draft probe using `--draft-max 16`.

Current local evidence:

- `Qwen3.6-35B-A3B-MTP-TQ3_4S.gguf` failed to load in this llama.cpp build.
- `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf` also failed to load in this llama.cpp build.
- Those failures are classified as `model_load` and written to receipts, so Codex can inspect them without rerunning.

Important note: `llama-bench.exe` does not expose a `--parallel` flag in the local build checked on this machine. The loop still explores `parallel` as target metadata for the later `llama-server` adapter, while the current executable checks focus on speed, batch, ubatch, GPU layers, flash attention, and context depth.
