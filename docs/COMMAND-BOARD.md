# Agent Pilot Autobench Command Board

This project is a local-first pilot tester for finding practical Hermes-agent GGUF settings without loading models one by one in LM Studio.

Hero command after install: `agent-autobench`

Beginner command before install:

```powershell
uv run --extra dev agent-autobench first-run
```

Beginner command after install:

```powershell
agent-autobench first-run
```

`first-run` is the friendly startup check. It checks the machine, prepares local
state, explains missing tools or folders, and tells the user what to run next.

## Safe First Commands

For complete beginners on Windows:

```text
Double-click START-HERE.bat
```

To make `agent-autobench` work from any terminal folder:

```text
Double-click INSTALL-COMMAND.bat
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

Show the latest champion and write `runs\leaderboard.md` plus `runs\champion.json`:

```powershell
uv run --extra dev pilotbench results
```

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
- Points that shim back to this repo.
- Asks before adding `G:\_codex_global\bin` to the user PATH.
- Does not touch the system PATH.
- Does not need admin rights.

After adding PATH, close and reopen the terminal, then run:

```powershell
agent-autobench first-run
```

Run the basic autoresearch loop for one model:

```powershell
uv run --extra dev pilotbench autoresearch --model "G:\AI\models\path\to\model.gguf" --budget-minutes 5 --parallel-max 4
```

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
- Record failures such as GPU OOM, model-load failure, crash, and timeout, then keep going.
- Keep unified KV cache marked as mandatory target metadata for Hermes deployment.

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
