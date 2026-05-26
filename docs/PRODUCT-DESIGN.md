# Agent Pilot Autobench Product Design

Agent Pilot Autobench should feel like a practical local model lab: it helps a
developer choose a local model/runtime configuration with receipts, not guesswork.
The product should stay small, local-first, and evidence-driven.

## Product Promise

Given a local GGUF model collection and a llama.cpp runtime, Agent Pilot
Autobench answers:

1. Which candidates load reliably on this machine?
2. Which settings are fast, stable, and useful for agent-style work?
3. Which measured winner should be promoted into a local serving profile?
4. Where is the proof for every pass, failure, and recommendation?

## Design Principles

- **Beginner-safe first run**: the first command should explain missing tools,
  write local state, and give one clear next command.
- **Evidence over impressions**: every recommendation needs a JSON and Markdown
  receipt that can be reviewed without rerunning a benchmark.
- **No silent fallback**: failed loads, CPU fallback, OOM, timeout, and bad JSON
  behavior should be recorded as useful evidence.
- **Use proven tools**: llama.cpp, Optuna, Typer, Rich, Textual, pytest, NVML,
  and local SQLite should carry the core workload.
- **Local by default**: benchmarks, receipts, learning state, and deployment
  exports should stay on the operator's machine unless explicitly shared.
- **Deployment-aware scoring**: the winner is not only the fastest model; it must
  be stable enough to serve real agent tasks.

## Recommended Feature Tracks

### 1. Readiness Wizard

The first-run flow should become a proper readiness wizard that ends in either a
green start state or a clear fix list.

Expected behavior:

- Check Python, `uv`, model roots, `llama-bench`, `llama-cli`, optional
  `llama-server`, writable receipt folders, and GPU telemetry access.
- Explain each failure in plain language.
- Offer copy-paste commands for the next step.
- Write `runs/readiness.json` so future agents can inspect setup state.

Success measure:

- A new user can run one command and know whether the machine is ready.

### 2. Campaign Builder

The current presets should become named benchmark campaigns with explicit time
budgets, target packs, and stop rules.

Expected behavior:

- Keep `Quick Scout`, `Normal`, `Deep Pilot`, and `Overnight`.
- Show the exact packs and time budget before a run starts.
- Support finish-early rules such as "stop when a candidate beats the current
  champion by enough margin."
- Resume interrupted campaigns from receipt state.

Success measure:

- The operator can start a campaign, stop it, and resume without losing evidence.

### 3. Champion Promotion

The results flow should distinguish a measured winner from a promoted champion.

Expected behavior:

- `results` shows the current measured winner.
- `promote-champion` records the operator's decision to approve a winner for local
  serving.
- Promotion writes a receipt with model path, settings, score, risks, and
  operator note.
- `export-profile` uses the promoted champion by default.

Success measure:

- Deployment files only come from an intentional champion decision.

### 4. Endpoint Validation

The product should verify the exported profile against an OpenAI-compatible
localhost endpoint before claiming it is ready for agent work.

Expected behavior:

- Start or target a `llama-server` endpoint.
- Check `/v1/models` and one minimal generation request.
- Run a tiny JSON/tool-discipline probe.
- Write `endpoint-validation.json` and `endpoint-validation.md`.

Success measure:

- A profile is not called deployment-ready until a real endpoint responds.

### 5. Evidence Report

The HTML report should become the main inspection surface for humans and agents.

Expected behavior:

- Show current measured winner, promoted champion, last readiness state, and
  recent failures.
- Link each row to its receipt folder.
- Explain common failure classes.
- Keep the report static and local so it can be opened without a server.

Success measure:

- A reviewer can understand the run history from `runs/results.html` alone.

### 6. Benchmark Pack Contract

Benchmark packs should become the extension point for real agent tasks without
hardcoding every test into the core loop.

Expected behavior:

- Keep built-in packs for load smoke, speed, JSON discipline, tool calling,
  coding smoke, context limit, and deployment validation.
- Allow local private packs from a plugin folder.
- Version pack manifests and receipt schemas.
- Make pack output small, deterministic, and reviewable.

Success measure:

- New task categories can be added without changing the benchmark runner core.

## Suggested Roadmap

1. **Stabilize first-run and reporting**
   - Readiness wizard receipt.
   - Clear setup errors.
   - Static HTML report improvements.

2. **Make champions intentional**
   - Measured winner versus promoted champion.
   - Promotion receipt.
   - Safer deployment export defaults.

3. **Validate local serving**
   - `llama-server` endpoint probe.
   - OpenAI-compatible generation smoke.
   - JSON/tool behavior receipt.

4. **Deepen benchmark packs**
   - BFCL-style tool-call pack.
   - Long-context pack.
   - MTP efficiency receipt.

## Non-Goals For Now

- Cloud-hosted benchmark telemetry.
- Training or fine-tuning models.
- Replacing llama.cpp with a custom runtime.
- Full benchmark leaderboard claims across other machines.
- Silent changes to system PATH, services, or network exposure.

## Public Language

Use **Agent Pilot Autobench** as the public product name. Keep copy specific:

- "first-run command" for the beginner entry point
- "measured winner" for the best result from receipts
- "promoted champion" for an operator-approved deployment candidate
- "receipt" for the evidence folder that proves what happened
- "local-first" only when the feature truly stays local by default
