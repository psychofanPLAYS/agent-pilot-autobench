# PilotBENCHY Cockpit Asks Questions — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default cockpit (TUI), `autoresearch`, and `autoresearch-all` runs actually ask + score the 10 SimpleBench questions through `llama-server` and report a champion — instead of synthetic `llama-bench`.

**Architecture:** The SimpleBench question-asking engine (`LlamaServerSimpleBenchAttemptRunner`) and the flag ladder already exist and are tested; today they only run behind the opt-in `--flag-ladder` flag. This phase introduces an explicit `evaluation` mode (`benchmark` default | `speed_scout`) and routes the default paths through the existing engine, with `--speed-scout` as the opt-out for the old synthetic fast path. Minimal new machinery; mostly routing + tests.

**Tech Stack:** Python 3.11–3.13, Typer CLI, Textual TUI, pytest, `uv`. Repo convention: test-per-module, TDD, frequent commits, ruff/mypy clean.

**Verification gate after every task:** `uv run --extra dev python -m pytest -q` stays green.

---

### Task 1: `evaluation` mode constant + resolver

**Files:**
- Create: `src/gguf_limit_bench/evaluation_mode.py`
- Test: `tests/test_evaluation_mode.py`

- [ ] **Step 1: Write the failing test**

```python
from gguf_limit_bench.evaluation_mode import (
    EvaluationMode,
    resolve_evaluation_mode,
    asks_questions,
)


def test_default_is_benchmark():
    assert resolve_evaluation_mode(speed_scout=False, flag_ladder=False) is EvaluationMode.BENCHMARK


def test_speed_scout_opt_out():
    assert resolve_evaluation_mode(speed_scout=True, flag_ladder=False) is EvaluationMode.SPEED_SCOUT


def test_flag_ladder_forces_benchmark():
    assert resolve_evaluation_mode(speed_scout=True, flag_ladder=True) is EvaluationMode.BENCHMARK


def test_benchmark_asks_questions():
    assert asks_questions(EvaluationMode.BENCHMARK) is True
    assert asks_questions(EvaluationMode.SPEED_SCOUT) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_evaluation_mode.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from enum import StrEnum


class EvaluationMode(StrEnum):
    BENCHMARK = "benchmark"
    SPEED_SCOUT = "speed_scout"


def resolve_evaluation_mode(*, speed_scout: bool, flag_ladder: bool) -> EvaluationMode:
    """Benchmark (asks questions) is the default. --speed-scout opts out.

    An explicit --flag-ladder always means benchmark, even if speed_scout is set,
    so the legacy flag keeps working.
    """
    if flag_ladder:
        return EvaluationMode.BENCHMARK
    return EvaluationMode.SPEED_SCOUT if speed_scout else EvaluationMode.BENCHMARK


def asks_questions(mode: EvaluationMode) -> bool:
    return mode is EvaluationMode.BENCHMARK
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev python -m pytest tests/test_evaluation_mode.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/evaluation_mode.py tests/test_evaluation_mode.py
git commit -m "feat: add evaluation mode resolver (benchmark default, speed-scout opt-out)"
```

---

### Task 2: Route `_run_one_autoresearch` benchmark mode through the SimpleBench engine

**Context:** `_run_one_autoresearch` in `src/gguf_limit_bench/cli.py` already routes through `LlamaServerSimpleBenchAttemptRunner` when `flag_ladder=True`, else `LlamaBenchAttemptRunner`. We make `flag_ladder` default-true via the new mode so callers that don't pass `flag_ladder=False/speed_scout` ask questions. Implementation: add an `evaluation: EvaluationMode = EvaluationMode.BENCHMARK` parameter; compute the effective ladder boolean as `flag_ladder or asks_questions(evaluation)`.

**Files:**
- Modify: `src/gguf_limit_bench/cli.py` (`_run_one_autoresearch`, around line 1280–1346)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_cli.py`)

```python
def test_run_one_autoresearch_benchmark_mode_uses_simplebench_runner(monkeypatch, tmp_path):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    captured = {}

    class FakeLoop:
        def __init__(self, **kwargs):
            captured["candidate_sequence"] = kwargs.get("candidate_sequence")
        def run(self):
            class R:
                path = tmp_path
            return R()

    monkeypatch.setattr(cli, "AutoresearchLoop", FakeLoop)
    # benchmark mode should build a flag-ladder candidate sequence (questions path)
    cli._run_one_autoresearch(
        model=tmp_path / "m.gguf",
        llama_bench=tmp_path / "llama-bench",
        llama_cli=tmp_path / "llama-cli",
        llama_server=tmp_path / "llama-server",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=1,
        max_attempts=1,
        learning=True,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.BENCHMARK,
    )
    assert captured["candidate_sequence"] is not None  # benchmark => ladder/questions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_cli.py::test_run_one_autoresearch_benchmark_mode_uses_simplebench_runner -v`
Expected: FAIL (`_run_one_autoresearch` has no `evaluation` parameter).

- [ ] **Step 3: Implement**

In `cli.py`, import at top: `from gguf_limit_bench.evaluation_mode import EvaluationMode, asks_questions`.

Add parameter to `_run_one_autoresearch` signature (near `flag_ladder: bool = False`):
```python
    evaluation: EvaluationMode = EvaluationMode.BENCHMARK,
```
At the very start of the body, before `if flag_ladder:`, compute the effective flag:
```python
    flag_ladder = flag_ladder or asks_questions(evaluation)
```
Leave the rest of the function unchanged (the existing `if flag_ladder:` block builds the SimpleBench runner + candidate sequence).

- [ ] **Step 4: Run tests**

Run: `uv run --extra dev python -m pytest tests/test_cli.py -q`
Expected: PASS (new test + existing).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/cli.py tests/test_cli.py
git commit -m "feat: route benchmark-mode autoresearch through the SimpleBench question engine"
```

---

### Task 3: TUI cockpit runs benchmark mode (asks questions) with a benchmark/speed-scout toggle

**Context:** The TUI `run_model` callback (cli.py ~1179) calls `_run_one_autoresearch` with no benchmark routing → synthetic. We pass `evaluation` and add a `BenchTui` toggle (default benchmark). The toggle lives in `tui.py`; the `tui` command reads it via the callback.

**Files:**
- Modify: `src/gguf_limit_bench/tui.py` (add `evaluation_mode` state + `m` binding + status text)
- Modify: `src/gguf_limit_bench/cli.py` (`tui` command callback passes `picker.evaluation_mode`)
- Test: `tests/test_tui.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from gguf_limit_bench.tui import BenchTui
from gguf_limit_bench.evaluation_mode import EvaluationMode


def test_tui_defaults_to_benchmark_mode(tmp_path):
    app = BenchTui(root=tmp_path, runs_root=tmp_path)
    assert app.evaluation_mode is EvaluationMode.BENCHMARK


def test_tui_toggle_switches_to_speed_scout(tmp_path):
    app = BenchTui(root=tmp_path, runs_root=tmp_path)
    app.action_toggle_evaluation()
    assert app.evaluation_mode is EvaluationMode.SPEED_SCOUT
    app.action_toggle_evaluation()
    assert app.evaluation_mode is EvaluationMode.BENCHMARK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_tui.py -v`
Expected: FAIL (`evaluation_mode` / `action_toggle_evaluation` missing).

- [ ] **Step 3: Implement** in `tui.py`

Add import: `from gguf_limit_bench.evaluation_mode import EvaluationMode`.
In `BenchTui.__init__`, add: `self.evaluation_mode = EvaluationMode.BENCHMARK`.
Add binding to `BINDINGS`: `("m", "toggle_evaluation", "Mode")`.
Add method:
```python
    def action_toggle_evaluation(self) -> None:
        self.evaluation_mode = (
            EvaluationMode.SPEED_SCOUT
            if self.evaluation_mode is EvaluationMode.BENCHMARK
            else EvaluationMode.BENCHMARK
        )
        if self.is_running:
            self._refresh_table()
```
Guard `is_running`: Textual sets `self.is_running`; in unit tests (no event loop) it is False, so the guard avoids touching widgets. In `_refresh_table`'s status string, append: ` Mode: {self.evaluation_mode.value}.`

- [ ] **Step 4: Implement** in `cli.py` `tui` command

In the `run_model` lambda, pass `evaluation=picker.evaluation_mode` to `_run_one_autoresearch`. Because the lambda is created before `picker.run()`, capture the picker: define `picker` first, then set `picker.run_model = lambda model: _run_one_autoresearch(..., evaluation=picker.evaluation_mode).path`. (Move the lambda assignment to after `picker = BenchTui(...)`.)

- [ ] **Step 5: Run tests**

Run: `uv run --extra dev python -m pytest tests/test_tui.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gguf_limit_bench/tui.py src/gguf_limit_bench/cli.py tests/test_tui.py
git commit -m "feat: cockpit runs benchmark mode by default with m-key speed-scout toggle"
```

---

### Task 4: `--speed-scout` CLI flag on `autoresearch` and `autoresearch-all`

**Files:**
- Modify: `src/gguf_limit_bench/cli.py` (`autoresearch` ~934, `autoresearch_all` ~1056)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
def test_autoresearch_speed_scout_resolves_to_speed_mode():
    from gguf_limit_bench.evaluation_mode import resolve_evaluation_mode, EvaluationMode
    assert resolve_evaluation_mode(speed_scout=True, flag_ladder=False) is EvaluationMode.SPEED_SCOUT
    assert resolve_evaluation_mode(speed_scout=False, flag_ladder=False) is EvaluationMode.BENCHMARK
```

(Routing itself is covered by Task 1/2; this guards the default contract the CLI relies on.)

- [ ] **Step 2: Run to verify it passes already** (it depends only on Task 1)

Run: `uv run --extra dev python -m pytest tests/test_cli.py::test_autoresearch_speed_scout_resolves_to_speed_mode -v`
Expected: PASS.

- [ ] **Step 3: Implement** — add to both commands a Typer option:
```python
    speed_scout: bool = typer.Option(
        False,
        "--speed-scout",
        help="Fast synthetic llama-bench scout (does NOT ask the benchmark questions).",
    ),
```
In each command body, compute:
```python
    from gguf_limit_bench.evaluation_mode import resolve_evaluation_mode
    evaluation = resolve_evaluation_mode(speed_scout=speed_scout, flag_ladder=flag_ladder)
```
`autoresearch` passes `evaluation=evaluation` to `_run_one_autoresearch` (and may drop the explicit `flag_ladder=flag_ladder` since it is folded in, but keep it for `--dry-run`). `autoresearch_all` (no `flag_ladder` param today) passes `evaluation=resolve_evaluation_mode(speed_scout=speed_scout, flag_ladder=False)`.
Update the `--dry-run` guard in `autoresearch` to: `if dry_run and evaluation is EvaluationMode.SPEED_SCOUT: raise typer.BadParameter("--dry-run needs benchmark mode (remove --speed-scout)")`.

- [ ] **Step 4: Run full suite**

Run: `uv run --extra dev python -m pytest -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/cli.py tests/test_cli.py
git commit -m "feat: --speed-scout opt-out; benchmark mode is the default for autoresearch + all"
```

---

### Task 5: Champion verdict line in the cockpit

**Context:** `tui.py` `_dashboard_text` shows run status; after a run, surface the leaderboard champion (best model of the bunch) in the dashboard. `write_leaderboard(runs_root)` returns an object with `.entries` and `.champion.model_name` / `.champion.score` (used already in cli.py:1201).

**Files:**
- Modify: `src/gguf_limit_bench/tui.py` (`_run_models_inside_tui` end → set champion text)
- Test: `tests/test_tui.py`

- [ ] **Step 1: Write the failing test**

```python
def test_champion_line_formats(tmp_path):
    from gguf_limit_bench.tui import format_champion_line
    assert format_champion_line("QwenX", 950.0) == "Champion: QwenX (950.00)"
    assert format_champion_line(None, None) == "Champion: not decided yet"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_tui.py::test_champion_line_formats -v`
Expected: FAIL (`format_champion_line` missing).

- [ ] **Step 3: Implement** in `tui.py` (module-level function):
```python
def format_champion_line(model_name: str | None, score: float | None) -> str:
    if model_name is None or score is None:
        return "Champion: not decided yet"
    return f"Champion: {model_name} ({score:.2f})"
```
In `_run_models_inside_tui`, after the loop (before `self.phase = "finished"`), compute and display via `call_from_thread`:
```python
            from gguf_limit_bench.reports import write_leaderboard
            board = write_leaderboard(self.runs_root)
            name = board.champion.model_name if board.entries else None
            score = board.champion.score if board.entries else None
            self.call_from_thread(
                self.query_one("#dashboard", Static).update,
                self._dashboard_text(format_champion_line(name, score)),
            )
```
(Confirm `write_leaderboard` import path with `grep -n "def write_leaderboard" src/gguf_limit_bench/reports.py`; adjust import if it lives elsewhere.)

- [ ] **Step 4: Run tests**

Run: `uv run --extra dev python -m pytest tests/test_tui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/tui.py tests/test_tui.py
git commit -m "feat: show champion verdict in the cockpit after a run"
```

---

### Task 6: Full release-gate sweep

- [ ] **Step 1: Run the gate**

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev python -m compileall -q src tests
```
Expected: all green. Fix anything red before continuing.

- [ ] **Step 2: Commit any formatting/type fixes**

```bash
git add -A
git commit -m "chore: release-gate fixes for benchmark-default cockpit"
```

---

### Task 7: Live acceptance run (the "make sure it actually works" gate)

**This is hardware-dependent and run by the operator, not a subagent. Uses a free port; never touches an already-running served model.**

- [ ] **Step 1: Run a real benchmark cockpit-equivalent on a small model**

```bash
PILOTBENCH_RUNS_ROOT=/tmp/apb-live uv run --extra dev agent-autobench autoresearch \
  --model "G:/AI/models/LM_Studio-gguf/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf" \
  --llama-server "G:/AI/llama.cpp/cuda12/llama-server.exe" \
  --budget-minutes 8
```
Expected: a `simplebench-L0-baseline/transcript.jsonl` with 10 question rows, a `summary.json` with accuracy, and a champion in the leaderboard. Confirm questions were actually asked (transcript has model responses + predicted vs expected answers).

- [ ] **Step 2: Record evidence**

Confirm in the receipt: per-question responses present, `simple_bench_accuracy` populated, `flag-ladder-results.md` (or equivalent) names a champion profile. Note the accuracy (a 2B model on hard SimpleBench will likely score low — that is expected and is exactly why Phase 2 adds easier packs).

---

## Self-Review

- **Spec coverage (Phase 1 portion of the design):** Unit A (eval routing) → Tasks 1,2,4. Cockpit asks questions (the core bug) → Tasks 2,3. Unit F champion verdict (minimal) + preset/mode toggle → Tasks 3,5. Unit G live run → Task 7. Phase-2 items (packs C/D, composite scoring E, cross-session memory H) are intentionally **out of this plan** and get their own plan next.
- **Placeholder scan:** none — every code step has concrete code; the one runtime-path uncertainty (`write_leaderboard` import) has an explicit verify-and-adjust instruction.
- **Type consistency:** `EvaluationMode` enum, `resolve_evaluation_mode(speed_scout, flag_ladder)`, `asks_questions(mode)`, `format_champion_line(model_name, score)`, `BenchTui.evaluation_mode` / `action_toggle_evaluation` are used consistently across tasks.
- **Known follow-up:** benchmark mode currently disables `learning`/`workflow_eval`/`ttft_probe` inside `_run_one_autoresearch` (existing flag-ladder behavior). Restoring learning under benchmark mode is **Phase 2 / Unit H** and is noted, not silently dropped.
