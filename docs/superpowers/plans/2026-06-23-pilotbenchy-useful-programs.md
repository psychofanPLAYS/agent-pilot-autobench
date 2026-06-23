# pilotBENCHY Useful Benchmark Programs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-size-fits-all 4k flag ladder with useful benchmark programs: fit-first context discovery, 32k-start fit search with 16k/8k refinement, 16k+ speed probes, 64k one-question intelligence windows, standard/on-off flag ablations, locked template support, and per-session machine/llama metrics snapshots.

**Architecture:** Add a small program layer that maps user intent to an execution contract before any llama-server launch. Speed programs use repeatable generation prompts at 16k+ and measure throughput/TTFT/metrics. Intelligence programs run question packs in fresh 64k windows after context fit is known. Telemetry and `/metrics` are captured as first-class evidence for every session and attempt.

**Tech Stack:** Python 3.11+, Typer CLI, Textual TUI, llama.cpp `llama-server`, existing `AutoresearchLoop`, `AutoresearchSettings`, `RunReceipt`, `TelemetrySnapshot`, JSONL/Markdown/TSV artifacts.

---

## Current Evidence Driving This Plan

Live run under `_runs\20260623-161636-Qwen3-5-9B-Claude-4-6-AWARE_UNCENSORED-Q8_0` proved the current design flaw:

- It launched SimpleBench at `--ctx-size 4096`, which is below the usable floor for David's target workflows.
- Profiles generated real output, but repeatedly hit 4095-token truncation and then failed as `SimpleBench attempt budget exhausted`.
- It consumed more than 20 minutes while producing partial transcripts, not a clean answer to "which settings should I use?"
- The effective standard flags were correct after this session's patch: `--flash-attn on --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --jinja --gpu-layers 99`.

The product correction is not "make the timeout longer." The correction is different programs with different evidence contracts.

## File Structure

- Create `src/gguf_limit_bench/programs.py`
  - Owns named benchmark programs, context floors, fit-search step sizes, default prompts, and program-to-settings decisions.
- Create `src/gguf_limit_bench/speed_probe.py`
  - Runs repeatable speed prompts and records generation/TTFT/prompt speed without SimpleBench scoring.
- Create `src/gguf_limit_bench/session_metrics.py`
  - Writes OS/GPU/process snapshots and llama.cpp `/metrics` samples at bounded polling intervals.
- Modify `src/gguf_limit_bench/flag_ladder.py`
  - Builds standard-on plus one-at-a-time-off ablations instead of pretending 4k stripped profiles are useful.
- Modify `src/gguf_limit_bench/simple_bench_runner.py`
  - Supports one-question-per-fresh-window mode at 64k+.
- Modify `src/gguf_limit_bench/cli.py`
  - Adds program selection and enforces minimum contexts before running.
- Modify `src/gguf_limit_bench/modes.py`
  - Replaces vague modes with named programs: speed, fit, intelligence, ablation, long-context-dropoff, deep.
- Modify `src/gguf_limit_bench/tui.py`
  - Shows program choice and prevents accidental 4k question runs.
- Add/modify tests under `tests/` matching each unit.
- Update `docs/COMMAND-BOARD.md` and `docs/superpowers/SESSION-2026-06-23-handoff-2-onboarding-phaseB.md`.

---

### Task 1: Add Program Contracts

**Files:**
- Create: `src/gguf_limit_bench/programs.py`
- Test: `tests/test_programs.py`

- [ ] **Step 1: Write failing tests**

```python
from gguf_limit_bench.programs import (
    INTELLIGENCE_CONTEXT_SIZE,
    MIN_SPEED_CONTEXT_SIZE,
    ProgramId,
    program_by_id,
    speed_probe_prompt,
)


def test_speed_program_never_uses_less_than_16k():
    program = program_by_id(ProgramId.SPEED)
    assert program.min_context_size == MIN_SPEED_CONTEXT_SIZE
    assert program.min_context_size == 16_384


def test_intelligence_program_uses_64k_windows():
    program = program_by_id(ProgramId.INTELLIGENCE)
    assert program.default_context_size == INTELLIGENCE_CONTEXT_SIZE
    assert program.default_context_size == 65_536
    assert program.one_question_per_window is True


def test_speed_prompt_is_repeatable_and_long_enough():
    prompt = speed_probe_prompt()
    assert "500 word" in prompt.lower()
    assert "poem" in prompt.lower()
    assert "same text every run" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_programs.py -q
```

Expected: FAIL because `gguf_limit_bench.programs` does not exist.

- [ ] **Step 3: Add program contract implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


MIN_SPEED_CONTEXT_SIZE = 16_384
FIT_START_CONTEXT_SIZE = 32_768
FIT_ASCENT_STEP = 32_768
FIT_BACKOFF_STEP = 16_384
FIT_REFINE_STEP = 8_192
INTELLIGENCE_CONTEXT_SIZE = 65_536
LONG_CONTEXT_TIERS = (16_384, 65_536, 131_072, 262_144)


class ProgramId(StrEnum):
    SPEED = "speed"
    FIT = "fit"
    INTELLIGENCE = "intelligence"
    FLAG_ABLATION = "flag-ablation"
    LONG_CONTEXT_DROPOFF = "long-context-dropoff"
    DEEP = "deep"


@dataclass(frozen=True)
class ProgramSpec:
    id: ProgramId
    label: str
    description: str
    min_context_size: int
    default_context_size: int
    asks_questions: bool
    one_question_per_window: bool = False


PROGRAMS = {
    ProgramId.SPEED: ProgramSpec(
        id=ProgramId.SPEED,
        label="Speed probe",
        description="Generate the same long text at 16k+ and measure throughput, TTFT, prompt speed, metrics, and stability.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=MIN_SPEED_CONTEXT_SIZE,
        asks_questions=False,
    ),
    ProgramId.FIT: ProgramSpec(
        id=ProgramId.FIT,
        label="Find fit",
        description="Climb context safely from 16k upward with q8_0 KV and remember the maximum context that actually serves.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=MIN_SPEED_CONTEXT_SIZE,
        asks_questions=False,
    ),
    ProgramId.INTELLIGENCE: ProgramSpec(
        id=ProgramId.INTELLIGENCE,
        label="Intelligence",
        description="Ask benchmark questions one per fresh 64k window after fit is known.",
        min_context_size=INTELLIGENCE_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        asks_questions=True,
        one_question_per_window=True,
    ),
    ProgramId.FLAG_ABLATION: ProgramSpec(
        id=ProgramId.FLAG_ABLATION,
        label="Flag ablation",
        description="Keep 2026 standard flags on, then test one off/change at a time with speed probes and selected intelligence checks.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=MIN_SPEED_CONTEXT_SIZE,
        asks_questions=False,
    ),
    ProgramId.LONG_CONTEXT_DROPOFF: ProgramSpec(
        id=ProgramId.LONG_CONTEXT_DROPOFF,
        label="Long-context dropoff",
        description="Run the same question packs across 16k/64k/128k/256k to measure accuracy and speed degradation.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        asks_questions=True,
        one_question_per_window=True,
    ),
    ProgramId.DEEP: ProgramSpec(
        id=ProgramId.DEEP,
        label="Deep",
        description="Fit, speed, ablation, intelligence, and long-context dropoff in a resumable campaign.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        asks_questions=True,
        one_question_per_window=True,
    ),
}


def program_by_id(program_id: ProgramId | str) -> ProgramSpec:
    return PROGRAMS[ProgramId(program_id)]


def speed_probe_prompt() -> str:
    return (
        "Generate the same text every run: write a 500 word poem about a local AI "
        "workbench tuning a model through measured experiments. Use complete sentences, "
        "no markdown table, and do not stop early."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_programs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gguf_limit_bench/programs.py tests/test_programs.py
git commit -m "feat: define useful benchmark programs"
```

---

### Task 2: Enforce 16k Minimum for Speed and Flag Programs

**Files:**
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `src/gguf_limit_bench/flag_ladder.py`
- Test: `tests/test_cli.py`, `tests/test_simple_bench.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_simple_bench.py`:

```python
from gguf_limit_bench.programs import MIN_SPEED_CONTEXT_SIZE


def test_core_flag_ladder_defaults_to_16k_context():
    ladder = build_core_flag_ladder()
    assert all(settings.context_size >= MIN_SPEED_CONTEXT_SIZE for settings in ladder)
```

Add to `tests/test_cli.py`:

```python
def test_autoresearch_dry_run_uses_16k_default_flag_context(tmp_path, monkeypatch):
    import json
    import gguf_limit_bench.cli as cli

    monkeypatch.setattr(cli, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"fake")
    receipt = cli._run_one_autoresearch(
        model=model,
        llama_bench=tmp_path / "llama-bench.exe",
        llama_cli=tmp_path / "llama-cli.exe",
        llama_server=tmp_path / "llama-server.exe",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=2,
        max_attempts=None,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        flag_ladder=True,
        dry_run=True,
        forced_server_args=cli._effective_forced_server_args(),
    )
    plan = json.loads((receipt.path / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    assert plan["context_size"] == 16_384
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_simple_bench.py::test_core_flag_ladder_defaults_to_16k_context tests/test_cli.py::test_autoresearch_dry_run_uses_16k_default_flag_context -q
```

Expected: FAIL because defaults still use 4096.

- [ ] **Step 3: Implement minimum defaults**

In `src/gguf_limit_bench/flag_ladder.py`, import:

```python
from gguf_limit_bench.programs import MIN_SPEED_CONTEXT_SIZE
```

Change `build_core_flag_ladder` default:

```python
def build_core_flag_ladder(
    *,
    context_size: int = MIN_SPEED_CONTEXT_SIZE,
    parallel_max: int = 6,
    extra_server_args: tuple[str, ...] = (),
    enable_mtp: bool = False,
) -> tuple[AutoresearchSettings, ...]:
```

In `src/gguf_limit_bench/cli.py`, import:

```python
from gguf_limit_bench.programs import MIN_SPEED_CONTEXT_SIZE
```

Change the `autoresearch` option default:

```python
flag_context_size: int = typer.Option(
    MIN_SPEED_CONTEXT_SIZE,
    min=MIN_SPEED_CONTEXT_SIZE,
    help="Context size for speed/flag programs. Question programs use 64k windows.",
)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_simple_bench.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gguf_limit_bench/cli.py src/gguf_limit_bench/flag_ladder.py tests/test_cli.py tests/test_simple_bench.py
git commit -m "fix: start benchmark programs at 16k context"
```

---

### Task 3: Add Repeatable Speed Probe Runner

**Files:**
- Create: `src/gguf_limit_bench/speed_probe.py`
- Test: `tests/test_speed_probe.py`

- [ ] **Step 1: Write failing tests**

```python
from gguf_limit_bench.programs import MIN_SPEED_CONTEXT_SIZE
from gguf_limit_bench.speed_probe import SpeedProbeResult, build_speed_probe_payload


def test_speed_payload_uses_repeatable_prompt_and_token_cap():
    payload = build_speed_probe_payload(max_tokens=768)
    assert payload["stream"] is True
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 768
    assert "500 word poem" in payload["messages"][1]["content"].lower()


def test_speed_probe_result_records_metrics():
    result = SpeedProbeResult(
        ok=True,
        context_size=MIN_SPEED_CONTEXT_SIZE,
        generated_tokens=700,
        output_chars=2800,
        ttft_ms=120.0,
        tokens_per_second=90.0,
        prompt_tokens_per_second=2500.0,
        failure="none",
    )
    assert result.tokens_per_second == 90.0
    assert result.context_size == 16_384
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_speed_probe.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Add minimal speed probe module**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

from gguf_limit_bench.programs import MIN_SPEED_CONTEXT_SIZE, speed_probe_prompt


@dataclass(frozen=True)
class SpeedProbeResult:
    ok: bool
    context_size: int
    generated_tokens: int
    output_chars: int
    ttft_ms: float | None
    tokens_per_second: float
    prompt_tokens_per_second: float
    failure: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def build_speed_probe_payload(max_tokens: int = 768) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a deterministic benchmark generator."},
            {"role": "user", "content": speed_probe_prompt()},
        ],
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def validate_speed_context(context_size: int) -> int:
    if context_size < MIN_SPEED_CONTEXT_SIZE:
        raise ValueError("speed probes must use at least 16k context")
    return context_size
```

- [ ] **Step 4: Run tests**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_speed_probe.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gguf_limit_bench/speed_probe.py tests/test_speed_probe.py
git commit -m "feat: add repeatable speed probe contract"
```

---

### Task 4: Add Session Metrics Snapshots

**Files:**
- Create: `src/gguf_limit_bench/session_metrics.py`
- Test: `tests/test_session_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
import json

from gguf_limit_bench.session_metrics import MachineSnapshot, append_metrics_sample


def test_machine_snapshot_serializes_static_context():
    snap = MachineSnapshot(
        os_name="Windows",
        cpu_name="test cpu",
        ram_total_mb=65536,
        gpu_name="RTX 4090",
        gpu_total_mb=24564,
        llama_build="b9596",
    )
    payload = snap.to_dict()
    assert payload["gpu_name"] == "RTX 4090"
    assert payload["ram_total_mb"] == 65536


def test_append_metrics_sample_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metrics_sample(path, {"gpu_util_percent": 50}, {"llamacpp:tokens_predicted_total": 123})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["telemetry"]["gpu_util_percent"] == 50
    assert row["llama_metrics"]["llamacpp:tokens_predicted_total"] == 123
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_session_metrics.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Add session metrics module**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json


@dataclass(frozen=True)
class MachineSnapshot:
    os_name: str
    cpu_name: str
    ram_total_mb: int
    gpu_name: str
    gpu_total_mb: int | None
    llama_build: str

    def to_dict(self) -> dict:
        return asdict(self)


def append_metrics_sample(
    path: Path,
    telemetry: dict,
    llama_metrics: dict[str, int | float | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "telemetry": telemetry,
        "llama_metrics": llama_metrics,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
```

- [ ] **Step 4: Run tests**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_session_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gguf_limit_bench/session_metrics.py tests/test_session_metrics.py
git commit -m "feat: record session metrics snapshots"
```

---

### Task 5: One Question Per 64k Intelligence Window

**Files:**
- Modify: `src/gguf_limit_bench/simple_bench_runner.py`
- Test: `tests/test_simple_bench.py`

- [ ] **Step 1: Write failing tests**

```python
from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.programs import INTELLIGENCE_CONTEXT_SIZE


def test_intelligence_settings_use_64k_context():
    settings = AutoresearchSettings(context_size=INTELLIGENCE_CONTEXT_SIZE)
    assert settings.context_size == 65_536
```

If adding an explicit runner flag:

```python
def test_simple_bench_runner_can_be_configured_one_question_per_window(tmp_path):
    runner = LlamaServerSimpleBenchAttemptRunner(
        llama_server=tmp_path / "llama-server.exe",
        model=tmp_path / "m.gguf",
        one_question_per_window=True,
    )
    assert runner.one_question_per_window is True
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_simple_bench.py::test_simple_bench_runner_can_be_configured_one_question_per_window -q
```

Expected: FAIL until constructor accepts the flag.

- [ ] **Step 3: Add runner constructor option**

In `LlamaServerSimpleBenchAttemptRunner.__init__`, add:

```python
one_question_per_window: bool = False,
```

and assign:

```python
self.one_question_per_window = one_question_per_window
```

Do not change the launch loop in this task. This task only adds the interface safely. The actual per-question server-session refactor should be a separate task because it changes runtime behavior.

- [ ] **Step 4: Run tests**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_simple_bench.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gguf_limit_bench/simple_bench_runner.py tests/test_simple_bench.py
git commit -m "feat: prepare one-question intelligence windows"
```

---

### Task 6: Update TUI/CLI Modes to Match Programs

**Files:**
- Modify: `src/gguf_limit_bench/modes.py`
- Modify: `src/gguf_limit_bench/tui.py`
- Test: `tests/test_tui.py`

- [ ] **Step 1: Write failing tests**

```python
from gguf_limit_bench.modes import RUN_MODES


def test_modes_include_fit_speed_intelligence_and_ablation():
    ids = {mode.id for mode in RUN_MODES}
    assert "speed" in ids
    assert "fit" in ids
    assert "intelligence" in ids
    assert "flag-ablation" in ids
```

- [ ] **Step 2: Run test to verify failure**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_tui.py::test_modes_include_fit_speed_intelligence_and_ablation -q
```

Expected: FAIL because old modes use `quick`, `best_settings`, etc.

- [ ] **Step 3: Replace mode list with program-shaped labels**

Use:

```python
RUN_MODES: tuple[RunMode, ...] = (
    RunMode(
        id="fit",
        label="Find fit",
        description="Start at 16k q8 KV, climb safely, and remember the max context.",
        budget_minutes=20,
        evaluation=EvaluationMode.SPEED_SCOUT,
    ),
    RunMode(
        id="speed",
        label="Speed probe",
        description="Generate repeatable long text at 16k+ and measure throughput/TTFT/metrics.",
        budget_minutes=10,
        evaluation=EvaluationMode.SPEED_SCOUT,
    ),
    RunMode(
        id="intelligence",
        label="Intelligence",
        description="Ask benchmark questions one per fresh 64k window.",
        budget_minutes=30,
        evaluation=EvaluationMode.BENCHMARK,
    ),
    RunMode(
        id="flag-ablation",
        label="Flag ablation",
        description="Keep standard flags on, then test one meaningful change at a time.",
        budget_minutes=30,
        evaluation=EvaluationMode.SPEED_SCOUT,
    ),
    RunMode(
        id="long-context-dropoff",
        label="Long-context dropoff",
        description="Measure speed and accuracy across 16k/64k/128k/256k.",
        budget_minutes=60,
        evaluation=EvaluationMode.BENCHMARK,
        context_ladder=(16_384, 65_536, 131_072, 262_144),
    ),
)
DEFAULT_RUN_MODE = RUN_MODES[0]
```

- [ ] **Step 4: Update affected tests**

Where tests expect `best_settings`, update them to expect `fit` or the new default. Keep backward compatibility only if existing CLI commands depend on the old names.

- [ ] **Step 5: Run tests**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_tui.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/gguf_limit_bench/modes.py src/gguf_limit_bench/tui.py tests/test_tui.py tests/test_cli.py
git commit -m "feat: expose useful benchmark programs in cockpit"
```

---

### Task 7: Real Acceptance Run

**Files:**
- No code required if prior tasks pass.
- Evidence: `_runs\<timestamp>-<model>\`

- [ ] **Step 1: Run fit first**

```powershell
$model = 'G:\AI\models\LM_Studio-gguf\DavidAU\Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED\Qwen3.5-9B-Claude-4.6-AWARE_UNCENSORED-Q8_0.gguf'
& '.venv\Scripts\apb.exe' context-limit --model $model --kv-cache-type q8_0 --min-context 16384 --max-context 262144
```

Expected:
- First tier is 16k, not 4k.
- Largest served context is printed.
- Context limit is recorded in state DB.

- [ ] **Step 2: Run speed probe**

```powershell
& '.venv\Scripts\apb.exe' autoresearch --model $model --program speed --budget-minutes 10 --flag-context-size 16384
```

Expected:
- Uses repeatable 500-word prompt.
- Writes speed result with prompt TPS, generation TPS, TTFT, `/metrics`, and machine snapshot.

- [ ] **Step 3: Run intelligence probe**

```powershell
& '.venv\Scripts\apb.exe' autoresearch --model $model --program intelligence --budget-minutes 30 --sample-size 5 --selection sequential
```

Expected:
- Uses 64k context by default.
- One question per fresh window.
- Transcript has one row per asked question.
- No 4095-token truncation for ordinary SimpleBench answers.

- [ ] **Step 4: Review result as a buyer**

Open:

```powershell
Get-Content _runs\leaderboard.md -TotalCount 120
Get-ChildItem _runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name
```

Answer:
- Did the app save time versus manual flag trial-and-error?
- Did it produce a defensible recommendation?
- Did it show enough raw evidence for a llama.cpp person to trust it?
- Did it tell the user what to do next?

- [ ] **Step 5: Commit docs/evidence summary**

```powershell
git add docs/COMMAND-BOARD.md docs/superpowers/SESSION-2026-06-23-handoff-2-onboarding-phaseB.md
git commit -m "docs: record useful benchmark program workflow"
```

---

## Self-Review

- Spec coverage: Covers 16k minimum, 64k SimpleBench windows, standard q8/flash/KV/Jinja baseline, template lock-in, machine snapshots, `/metrics`, on/off ablations, and named programs.
- Placeholder scan: No TBD/TODO placeholders are present.
- Type consistency: Uses existing `AutoresearchSettings`, `RunMode`, `EvaluationMode`, `LlamaServerSimpleBenchAttemptRunner`, and test command style from the repo.
- Known gap: Task 5 intentionally introduces the one-question-per-window interface before the full runtime refactor. The runtime refactor should be implemented immediately after Task 5, but separately, because it changes server lifecycle and can affect long-running benchmarks.

Plan complete and saved to `docs/superpowers/plans/2026-06-23-pilotbenchy-useful-programs.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
