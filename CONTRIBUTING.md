# Contributing

Thanks for improving Agent Pilot Autobench. Keep changes local-first, evidence-backed,
and safe to run on a workstation containing large models and existing benchmark data.

## Development Setup

Requirements: Git, Python 3.11-3.13, and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/psychofanPLAYS/agent-pilot-autobench.git
cd agent-pilot-autobench
uv sync --extra dev --locked
```

Optional benchmark harness dependencies are installed with `--extra bench`. Models,
llama.cpp binaries, environments, databases, and run output belong in the ignored
local directories described in `docs/ARCHITECTURE.md`.

## Required Checks

Run these before opening a pull request:

```powershell
uv run --extra dev python -m pytest -q
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev python -m compileall -q src tests
uv build
git diff --check
```

Use test-first development for behavior changes. Tests must cover failures and receipts,
not only successful output. Avoid assertions that depend on a particular local GPU,
model inventory, absolute path, or live network service.

## Pull Requests

- Keep commits focused and explain the user-visible impact.
- Include exact verification commands and results.
- Do not commit `_runs/`, `runs/`, `_models/`, `_llama/`, `_db/`, virtual environments,
  caches, tokens, model files, or machine-specific configuration.
- Document new commands and artifact formats.
- Preserve localhost defaults and direct argument-list subprocess execution.

## Benchmark Claims

State the model, quantization, llama.cpp build, hardware context, settings, sample size,
and receipt path. Do not generalize a ten-question SimpleBench score or one machine's
throughput into a universal model ranking.
