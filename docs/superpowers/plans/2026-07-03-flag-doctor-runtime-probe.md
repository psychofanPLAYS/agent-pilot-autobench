# Flag Doctor Runtime Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the "Flag Doctor" workflow that turns today's manual Qwen/Froggeric/llama.cpp investigation into repeatable pilotBENCHY and S2 runtime checks before expensive current-model wiki-librarian benchmarks run.

**Architecture:** Add a small runtime-doctor layer that inspects the installed llama.cpp binary, recommended model/template flags, live `/props` state, and tiny chat probes, then writes receipt artifacts. pilotBENCHY uses it as a preflight/Flight Plan gate; S2 later consumes the same concepts for "Check this profile" without auto-reloading models.

**Tech Stack:** Python 3.13, pytest, llama.cpp OpenAI-compatible HTTP endpoints, existing `RunReceipt`/`AutoresearchSettings`, S2 `llama_supervisor.py` and `llama_meter.py`.

---

## File Structure

- Modify `src/gguf_limit_bench/template_recommend.py`: emit full Qwen Froggeric v21.3 thinking defaults, not only `--jinja` and `--chat-template-file`.
- Modify `tests/test_template_recommend.py`: lock the new recommended Qwen flags and keep Gemma behavior unchanged.
- Create `src/gguf_limit_bench/runtime_doctor.py`: pure-ish probe helpers for binary flag support, template repo/version checks, live `/props` comparison, and thinking extraction sanity.
- Create `tests/test_runtime_doctor.py`: deterministic tests for supported flags, stale template detection, and reasoning-output classification.
- Modify `src/gguf_limit_bench/librarian/preflight.py`: replace the brittle `<think>`-only Qwen gate with runtime-doctor thinking evidence that accepts `reasoning_content`.
- Modify `tests/test_librarian_preflight.py`: add Qwen cases for `reasoning_content`, stale template detail, and `enable_thinking` kwargs.
- Modify `src/gguf_limit_bench/receipts.py` only if the existing receipt API lacks a convenient `runtime-doctor.json` writer; otherwise keep writes inside existing run directories.
- Modify `src/gguf_limit_bench/cli.py`: add `agent-autobench doctor-runtime` or extend the existing doctor command with `--llama-server`, `--model`, `--base-url`, and `--json-out`.
- Modify `src/gguf_limit_bench/webui.py`: surface the latest runtime-doctor receipt in the cockpit proof chain before a Flight Plan run starts.
- Modify `docs/feature-requests/wiki-librarian-bench/10-runnable-presets.md`: replace the old v19 serving recommendation with v21.3, `enable_thinking`, `preserve_thinking`, `--reasoning on`, and `--reasoning-format deepseek`.
- Modify `docs/AUTORESEARCH-PROGRAM.md`: document the flag-doctor loop as the safe first phase before flag ladders.
- Create `docs/feature-requests/wiki-librarian-bench/11-flag-doctor-doctrine.md`: explain the operator loop and current-model proxy evidence requirements.
- In S2 later: modify `G:\_repos\switch2\backend\llama_meter.py` to include reasoning/template/version summary in `_trim_props`, and add a non-mutating `profile_doctor` helper near `llama_supervisor.build_cmdline()`.

---

### Task 1: Promote Qwen Froggeric v21.3 Defaults

**Files:**
- Modify: `src/gguf_limit_bench/template_recommend.py`
- Modify: `tests/test_template_recommend.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_template_recommend.py`:

```python
def test_qwen_recommends_froggeric_v21_reasoning_defaults_when_template_present(tmp_path):
    template = _make_template(tmp_path)
    model = tmp_path / "models" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.touch()

    flags = recommended_model_flags(model, search_roots=(tmp_path,))

    assert flags == (
        "--jinja",
        "--chat-template-file",
        str(template),
        "--chat-template-kwargs",
        '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_template_recommend.py::test_qwen_recommends_froggeric_v21_reasoning_defaults_when_template_present -q
```

Expected: FAIL because `recommended_model_flags()` currently returns only `("--jinja", "--chat-template-file", path)`.

- [ ] **Step 3: Implement the minimal code**

Update the Qwen branch in `src/gguf_limit_bench/template_recommend.py`:

```python
QWEN_THINKING_KWARGS = '{"enable_thinking":true,"preserve_thinking":true}'


def _qwen_reasoning_flags() -> tuple[str, ...]:
    return (
        "--chat-template-kwargs",
        QWEN_THINKING_KWARGS,
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )
```

Then change the Qwen return path:

```python
    if family == "qwen":
        template = template_override or discover_chat_template(family, search_roots)
        if template is not None:
            return ("--jinja", "--chat-template-file", str(template), *_qwen_reasoning_flags())
        return ("--jinja", *_qwen_reasoning_flags())
```

- [x] **Step 4: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_template_recommend.py -q
```

Expected: all tests pass after updating the old expected Qwen tuples to include the new reasoning flags.

- [ ] **Step 5: Commit**

```powershell
git add src\gguf_limit_bench\template_recommend.py tests\test_template_recommend.py
git commit -m "feat: recommend qwen froggeric reasoning defaults"
```

---

### Task 2: Add Runtime Doctor Receipt Model

**Files:**
- Create: `src/gguf_limit_bench/runtime_doctor.py`
- Create: `tests/test_runtime_doctor.py`

- [ ] **Step 1: Write tests for binary flag support**

Create `tests/test_runtime_doctor.py` with:

```python
from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.runtime_doctor import (
    RuntimeDoctorReceipt,
    detect_template_version,
    flag_supported,
    live_template_status,
    reasoning_status_from_message,
)


HELP = """
--chat-template-kwargs STRING
--reasoning [on|off|auto]
--reasoning-format FORMAT
--chat-template-file JINJA_TEMPLATE_FILE
"""


def test_flag_supported_finds_long_options():
    assert flag_supported(HELP, "--reasoning") is True
    assert flag_supported(HELP, "--reasoning-format") is True
    assert flag_supported(HELP, "--missing") is False
```

- [ ] **Step 2: Write tests for template version detection**

Append:

```python
def test_detect_template_version_reads_froggeric_constant(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text('{%- set template_version = "qwen3.6-froggeric-v21.3" %}\n', encoding="utf-8")

    assert detect_template_version(template) == "qwen3.6-froggeric-v21.3"
```

- [ ] **Step 3: Write tests for stale live template evidence**

Append:

```python
def test_live_template_status_detects_stale_loaded_template(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text('{%- set template_version = "qwen3.6-froggeric-v21.3" %}', encoding="utf-8")
    props = {"chat_template": '{%- set template_version = "qwen3.6-froggeric-v19" %}'}

    status = live_template_status(template, props)

    assert status["disk_version"] == "qwen3.6-froggeric-v21.3"
    assert status["live_version"] == "qwen3.6-froggeric-v19"
    assert status["matches_disk"] is False
```

- [ ] **Step 4: Write tests for reasoning content**

Append:

```python
def test_reasoning_status_accepts_reasoning_content():
    message = {"content": "Final Answer: A", "reasoning_content": "I should inspect the evidence."}

    assert reasoning_status_from_message(message) == {
        "has_reasoning": True,
        "source": "reasoning_content",
        "content_has_think_tags": False,
    }
```

- [ ] **Step 5: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_doctor.py -q
```

Expected: import failure because `runtime_doctor.py` does not exist.

- [ ] **Step 6: Implement `runtime_doctor.py`**

Create `src/gguf_limit_bench/runtime_doctor.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Any


_VERSION_RE = re.compile(r'template_version\s*=\s*"([^"]+)"')


@dataclass(frozen=True)
class RuntimeDoctorReceipt:
    ok: bool
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flag_supported(help_text: str, option: str) -> bool:
    return any(line.lstrip().startswith(option) or f", {option}" in line for line in help_text.splitlines())


def detect_template_version(template_path: Path) -> str | None:
    try:
        text = template_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _template_version_from_text(text: str) -> str | None:
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else None


def live_template_status(template_path: Path, props: dict[str, Any]) -> dict[str, Any]:
    disk_version = detect_template_version(template_path)
    live_version = _template_version_from_text(str(props.get("chat_template") or ""))
    return {
        "template_file": str(template_path),
        "disk_version": disk_version,
        "live_version": live_version,
        "live_template_set": bool(props.get("chat_template")),
        "matches_disk": bool(disk_version and live_version and disk_version == live_version),
    }


def reasoning_status_from_message(message: dict[str, Any]) -> dict[str, Any]:
    reasoning = message.get("reasoning_content")
    content = str(message.get("content") or "")
    has_tags = "<think>" in content.lower()
    if reasoning:
        return {"has_reasoning": True, "source": "reasoning_content", "content_has_think_tags": has_tags}
    return {"has_reasoning": has_tags, "source": "content" if has_tags else "none", "content_has_think_tags": has_tags}
```

- [x] **Step 7: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_doctor.py -q
```

Expected: all runtime doctor tests pass.

- [ ] **Step 8: Commit**

```powershell
git add src\gguf_limit_bench\runtime_doctor.py tests\test_runtime_doctor.py
git commit -m "feat: add llama runtime doctor probes"
```

---

### Task 3: Upgrade Librarian Qwen Thinking Preflight

**Files:**
- Modify: `src/gguf_limit_bench/librarian/preflight.py`
- Modify: `tests/test_librarian_preflight.py`

- [ ] **Step 1: Add a failing test for `reasoning_content`**

Add to `tests/test_librarian_preflight.py`:

```python
def test_librarian_preflight_qwen_thinking_accepts_reasoning_content(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Qwen3.6-35B-A3B-Q4_K_M.gguf")

    def fake_chat(**kwargs):
        if "thinking check" in kwargs["user_content"].lower():
            return ("Final Answer: A", 1.0, 1.0, 1.0, 1, {"reasoning_content": "I should reason."})
        return ("Final Answer: A", 1.0, 1.0, 1.0, 1, {"content": "Final Answer: A"})

    monkeypatch.setattr("gguf_limit_bench.librarian.preflight._chat", fake_chat)

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(
            extra_server_args=(
                "--jinja",
                "--chat-template-kwargs",
                '{"enable_thinking":true,"preserve_thinking":true}',
                "--reasoning",
                "on",
                "--reasoning-format",
                "deepseek",
            )
        ),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    gates = {gate.name: gate for gate in receipt.gates}
    assert gates["thinking_sanity"].status == "pass"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_librarian_preflight.py::test_librarian_preflight_qwen_thinking_accepts_reasoning_content -q
```

Expected: FAIL because `_thinking_sanity_gate()` currently only checks for `<think>` in returned text.

- [ ] **Step 3: Adapt `_chat` result handling**

In `src/gguf_limit_bench/librarian/preflight.py`, import:

```python
from gguf_limit_bench.runtime_doctor import reasoning_status_from_message
```

Then add a helper near `_thinking_sanity_gate`:

```python
def _chat_text_and_message(result: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    text = str(result[0] if result else "")
    message = result[5] if len(result) > 5 and isinstance(result[5], dict) else {"content": text}
    return text, message
```

Update the thinking probe:

```python
    result = _chat(
        base_url=base_url,
        system_prompt="You are a benchmark preflight assistant.",
        user_content=_THINKING_PROMPT,
        max_tokens=256,
        timeout_seconds=timeout_seconds,
    )
    text, message = _chat_text_and_message(result)
    reasoning = reasoning_status_from_message(message)
    has_think = reasoning["has_reasoning"]
    evidence = {"thinking_mode": mode, "reasoning": reasoning, "response_chars": len(text)}
```

- [ ] **Step 4: Preserve backwards compatibility**

Existing tests monkeypatch `_chat` to return a 5-tuple. Do not change those tests except the new one. `_chat_text_and_message()` must treat a 5-tuple as `{"content": text}`.

- [x] **Step 5: Run focused preflight tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_librarian_preflight.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src\gguf_limit_bench\librarian\preflight.py tests\test_librarian_preflight.py
git commit -m "fix: accept qwen reasoning_content in librarian preflight"
```

---

### Task 4: Add Operator-Facing Runtime Doctor Command

**Files:**
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/AUTORESEARCH-PROGRAM.md`

- [ ] **Step 1: Add CLI test for dry receipt output**

Add a focused test near existing CLI/doctor tests:

```python
def test_doctor_runtime_writes_json_receipt(tmp_path, cli_runner):
    output = tmp_path / "runtime-doctor.json"
    result = cli_runner.invoke(
        app,
        [
            "doctor-runtime",
            "--llama-server",
            "llama-server.exe",
            "--model",
            str(tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            "--base-url",
            "http://127.0.0.1:8080",
            "--json-out",
            str(output),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["mode"] == "dry-run"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_doctor_runtime_writes_json_receipt -q
```

Expected: FAIL because `doctor-runtime` does not exist.

- [ ] **Step 3: Implement minimal dry-run command**

Add a Typer command that writes:

```json
{
  "ok": true,
  "mode": "dry-run",
  "llama_server": "llama-server.exe",
  "model": "...",
  "base_url": "http://127.0.0.1:8080",
  "checks": []
}
```

Do not start or stop servers in this command. Live HTTP probes can be a later flag, but dry-run must be safe and deterministic.

- [ ] **Step 4: Document the command**

In `docs/AUTORESEARCH-PROGRAM.md`, add a short section before flag ladders:

```markdown
### Runtime Doctor Before Flag Ladders

Run `agent-autobench doctor-runtime --dry-run --json-out runtime-doctor.json` before expensive ladders. The doctor checks the installed llama.cpp flag surface, model-aware template defaults, and later live `/props` evidence. A failed doctor check blocks scoring and writes a receipt instead of producing a misleading low score.
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_doctor_runtime_writes_json_receipt -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\gguf_limit_bench\cli.py tests\test_cli.py docs\AUTORESEARCH-PROGRAM.md
git commit -m "feat: add runtime doctor command shell"
```

---

### Task 5: Update Wiki-Librarian Doctrine Docs

**Files:**
- Create: `docs/feature-requests/wiki-librarian-bench/11-flag-doctor-doctrine.md`
- Modify: `docs/feature-requests/wiki-librarian-bench/10-runnable-presets.md`

- [ ] **Step 1: Add the doctrine doc**

Create `docs/feature-requests/wiki-librarian-bench/11-flag-doctor-doctrine.md`:

```markdown
# 11 - Flag Doctor Doctrine

Status: implementation target, 2026-07-03

pilotBENCHY should discover good llama.cpp settings the way an expert operator does:

1. Inspect the local command/profile.
2. Check the installed `llama-server --help`.
3. Verify model path, template path, template repo version, and live `/props`.
4. Start from the known-good baseline.
5. Change one behavior at a time.
6. Run tiny probes before expensive benchmark packs.
7. Write receipts for every probe.
8. Treat invalid setups as `preflight_fail`, not bad model scores.

For Qwen/Froggeric v21.3, the default thinking block is:

```powershell
--jinja `
--chat-template-file 'G:\AI\models\Qwen-Fixed-Chat-Templates\chat_template.jinja' `
--chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' `
--reasoning on `
--reasoning-format deepseek
```

For Gemma, the first negative-control checks are single-BOS correctness and answer-channel discipline.

The cockpit should show the operator these facts before a run:

- installed llama.cpp build and supported flags
- disk template version
- live template version from `/props`
- whether live template matches disk
- Qwen thinking evidence source (`reasoning_content` or content tags)
- Gemma BOS result
- final answer parse result
```

- [ ] **Step 2: Update v19 wording in runnable presets**

Change the Qwen serving sentence in `10-runnable-presets.md` from:

```markdown
Froggeric v21.3 template, `--jinja`, `enable_thinking=true`,
`preserve_thinking=true`, `--reasoning on`, and `--reasoning-format deepseek`
```

to:

```markdown
Froggeric v21.3 template, `--jinja`, `enable_thinking=true`, `preserve_thinking=true`, `--reasoning on`, and `--reasoning-format deepseek`
```

- [x] **Step 3: Run docs grep**

Run:

```powershell
rg -n "froggeric-v19|v19 template" docs\feature-requests\wiki-librarian-bench
```

Expected: no stale v19 serving recommendation remains outside historical changelog/context.

- [ ] **Step 4: Commit**

```powershell
git add docs\feature-requests\wiki-librarian-bench\10-runnable-presets.md docs\feature-requests\wiki-librarian-bench\11-flag-doctor-doctrine.md
git commit -m "docs: add wiki librarian flag doctor doctrine"
```

---

### Task 6: Add S2 Non-Mutating Profile Doctor Plan Slice

**Files:**
- Create later in S2 repo: `G:\_repos\switch2\backend\profile_doctor.py`
- Modify later in S2 repo: `G:\_repos\switch2\backend\llama_meter.py`
- Test later in S2 repo: `G:\_repos\switch2\switch2\tests\test_profile_doctor.py`

- [ ] **Step 1: Create S2 tests first**

In S2, create `switch2/tests/test_profile_doctor.py`:

```python
from profile_doctor import compare_template_versions, profile_reasoning_flags


def test_compare_template_versions_detects_stale_live_template(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text('{%- set template_version = "qwen3.6-froggeric-v21.3" %}', encoding="utf-8")
    props = {"chat_template": '{%- set template_version = "qwen3.6-froggeric-v19" %}'}

    result = compare_template_versions(template, props)

    assert result["disk_version"] == "qwen3.6-froggeric-v21.3"
    assert result["live_version"] == "qwen3.6-froggeric-v19"
    assert result["matches"] is False


def test_profile_reasoning_flags_summarizes_qwen_thinking_defaults():
    flags = {
        "--chat-template-kwargs": '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning": "on",
        "--reasoning-format": "deepseek",
    }

    assert profile_reasoning_flags(flags) == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning": "on",
        "reasoning_format": "deepseek",
    }
```

- [ ] **Step 2: Implement S2 pure helper**

Create `G:\_repos\switch2\backend\profile_doctor.py` with pure functions only. Do not start, stop, arm, disarm, or reload llama-server.

```python
from __future__ import annotations

import json
import re
from pathlib import Path


_VERSION_RE = re.compile(r'template_version\s*=\s*"([^"]+)"')


def _version(text: str) -> str | None:
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else None


def compare_template_versions(template: Path, props: dict) -> dict:
    try:
        disk_text = template.read_text(encoding="utf-8", errors="replace")
    except OSError:
        disk_text = ""
    disk = _version(disk_text)
    live = _version(str((props or {}).get("chat_template") or ""))
    return {"template_file": str(template), "disk_version": disk, "live_version": live, "matches": bool(disk and live and disk == live)}


def profile_reasoning_flags(flags: dict) -> dict:
    raw = str((flags or {}).get("--chat-template-kwargs") or "{}")
    try:
        kwargs = json.loads(raw)
    except json.JSONDecodeError:
        kwargs = {}
    return {
        "enable_thinking": kwargs.get("enable_thinking"),
        "preserve_thinking": kwargs.get("preserve_thinking"),
        "reasoning": (flags or {}).get("--reasoning"),
        "reasoning_format": (flags or {}).get("--reasoning-format"),
    }
```

- [ ] **Step 3: Extend S2 snapshot, not control path**

Modify `_trim_props()` in `backend/llama_meter.py` to include:

```python
"reasoning_format": params.get("reasoning_format"),
"reasoning_in_content": params.get("reasoning_in_content"),
```

Only add read-only fields. Do not add service-control behavior.

- [ ] **Step 4: Run S2 focused tests**

Run:

```powershell
python -m pytest switch2\tests\test_profile_doctor.py switch2\tests\test_flag_catalog.py -q
```

Expected: PASS. If S2 source tests need local import path setup, use the repo's existing test invocation pattern.

- [ ] **Step 5: Commit in S2 only after source and live-copy strategy is confirmed**

Do not copy to `G:\AI\switch2` until source tests pass and David explicitly wants live deployment.

---

### Task 7: Final Integration Verification

**Files:**
- No new files unless tests reveal gaps.

- [ ] **Step 1: Run pilotBENCHY focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_template_recommend.py tests\test_runtime_doctor.py tests\test_librarian_preflight.py -q
```

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall src tests
```

Expected: no syntax failures.

- [ ] **Step 3: Run plan-load smoke**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m gguf_limit_bench.cli flight-plans --json-out
```

Expected: JSON includes the librarian benchmark Flight Plan and both Gemma/Qwen plan IDs remain loadable.

- [ ] **Step 4: Review receipts**

For any live probe, verify receipts include:

- `runtime-doctor.json`
- `preflight.json`
- `resolved-plan.json`
- `command.txt`
- `status.json`

- [ ] **Step 5: Commit final docs if separate**

```powershell
git status --short
git log --oneline -5
```

Expected: only intentional commits from this plan.

---

## Self-Review

Spec coverage:

- The plan turns today's manual flag/template/reasoning investigation into pilotBENCHY code paths: template recommendation, runtime doctor, preflight, CLI, docs, and receipts.
- The plan covers S2 as a non-mutating profile/snapshot doctor so S2 benefits without surprise reloads or GPU disruption.
- The plan preserves legacy Gemma artifacts as explicit baselines only, while making the current wiki-librarian decision path Qwen3.6-first with current challengers such as Gemma 4 only when deliberately selected.

Placeholder scan:

- No unfinished markers or undefined "add handling later" steps remain.
- Code snippets define the functions they reference.

Type consistency:

- `runtime_doctor.reasoning_status_from_message()` returns the same keys used by `librarian/preflight.py`.
- Qwen recommendation flags match today's verified launch block.
- S2 profile-doctor functions are pure and do not overlap with `llama_supervisor.start()`/`restart()`.
