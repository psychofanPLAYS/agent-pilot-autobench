# Public Autoresearch Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing GGUF benchmark cockpit easier to run, safer to publish, and clearer as the start of Agent Pilot Autobench.

**Architecture:** Keep the existing package name and benchmark internals stable. Add a small readiness-check module, public-facing README, and CI while preserving configurable local Windows defaults.

**Tech Stack:** Python, Typer, Rich, pytest, uv, GitHub Actions.

---

### Task 1: Add Readiness Checks

**Files:**
- Create: `src/gguf_limit_bench/doctor.py`
- Modify: `src/gguf_limit_bench/cli.py`
- Test: `tests/test_doctor.py`

- [x] **Step 1: Write failing tests**

```powershell
uv run --extra dev pytest tests\test_doctor.py -q
```

Expected before implementation: import failure for `gguf_limit_bench.doctor`.

- [x] **Step 2: Implement `build_doctor_report`**

Add dataclasses for checks and reports. Check model roots, `llama-bench`, `llama-cli`, and create the receipts root if needed.

- [x] **Step 3: Wire `doctor` into Typer**

Expose `pilotbench doctor` with `--root`, `--llama-bench`, `--llama-cli`, `--runs-root`, `--strict`, and `--json-out`.

- [x] **Step 4: Verify focused tests**

```powershell
uv run --extra dev pytest tests\test_doctor.py -q
```

Expected: five doctor tests pass.

### Task 2: Add Public GitHub Front Door

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `docs/COMMAND-BOARD.md`

- [x] **Step 1: Add README**

Describe the mission, install path, first run, common commands, receipts, project status, references, and license status in plain English.

- [x] **Step 2: Add CI**

Run `uv sync --extra dev --locked` and `uv run pytest -q` on push and pull request.

- [x] **Step 3: Update package metadata**

Set `readme = "README.md"` and use a public-friendly description.

- [x] **Step 4: Update command board**

Add `doctor` and `doctor --strict` as the first safe commands.

### Task 3: Verification

**Files:**
- No new files.

- [x] **Step 1: Run full tests**

```powershell
uv run --extra dev pytest
```

Expected: all tests pass.

- [x] **Step 2: Run CLI smoke checks**

```powershell
uv run --extra dev pilotbench --help
uv run --extra dev pilotbench doctor --json-out
```

Expected: help renders and doctor returns JSON.
