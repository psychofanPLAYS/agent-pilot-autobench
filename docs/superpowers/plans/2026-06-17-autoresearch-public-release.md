# Autoresearch Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing branch and PR #9 into a verified, public, portfolio-grade release candidate without creating a duplicate branch or PR.

**Architecture:** Preserve the current package boundaries and harden the release seams: typed configuration and metrics, validated SimpleBench inputs, package-safe data, deterministic failure receipts, and CI-enforced build quality. Public documentation will map the CLI to source, evidence artifacts, and tests so an evaluator can verify claims quickly.

**Tech Stack:** Python 3.11-3.13, Typer, llama.cpp executables, pytest, Ruff, MyPy, Hatchling, uv, GitHub Actions.

---

### Task 1: Make the existing codebase type-clean

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/gguf_limit_bench/telemetry.py`
- Modify: `src/gguf_limit_bench/config.py`
- Modify: `src/gguf_limit_bench/score_extract.py`
- Modify: `src/gguf_limit_bench/packs.py`
- Modify: `src/gguf_limit_bench/benchmark_suite.py`
- Modify: `src/gguf_limit_bench/server_probe.py`

- [ ] **Step 1: Capture the current type failures**

Run: `uv run --extra dev mypy src`

Expected: 12 errors across the seven listed concerns, including the missing `psutil` stubs.

- [ ] **Step 2: Add the maintained psutil stub package**

Add `types-psutil>=7.0.0.20250601` to the `dev` dependency group and refresh `uv.lock` with:

```powershell
uv add --optional dev "types-psutil>=7.0.0.20250601"
```

- [ ] **Step 3: Narrow dynamic values before conversion**

Use explicit local variables and collection checks so MyPy can prove the conversions:

```python
min_score_value = payload.get("min_score")
min_score = None if min_score_value is None else float(min_score_value)
```

```python
if not isinstance(value, list | tuple):
    raise TypeError("Expected a list or tuple")
```

Use `TypeGuard` for numeric JSON values and type the built-in pack defaults explicitly rather than relying on `**dict` inference.

- [ ] **Step 4: Generalize the mean helper input**

Change `_mean` to accept `Iterable[float | None]`, preserving its filtering and average behavior for both optional and concrete float lists.

- [ ] **Step 5: Verify type health**

Run: `uv run --extra dev mypy src`

Expected: `Success: no issues found in 33 source files`.

- [ ] **Step 6: Commit the type-health slice**

```powershell
git add pyproject.toml uv.lock src/gguf_limit_bench/telemetry.py src/gguf_limit_bench/config.py src/gguf_limit_bench/score_extract.py src/gguf_limit_bench/packs.py src/gguf_limit_bench/benchmark_suite.py src/gguf_limit_bench/server_probe.py
git commit -m "chore: enforce autoresearch type health"
```

### Task 2: Validate SimpleBench inputs before starting a server

**Files:**
- Modify: `src/gguf_limit_bench/simple_bench.py`
- Modify: `src/gguf_limit_bench/simple_bench_runner.py`
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `tests/test_simple_bench.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing dataset-validation tests**

Add tests proving that the loader rejects a non-list `eval_data`, duplicate question IDs,
empty prompts, and answers outside `A` through `F`:

```python
@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"question_id": 1, "prompt": "", "answer": "A"}, "non-empty prompt"),
        ({"question_id": 1, "prompt": "Question", "answer": "Z"}, "answer A-F"),
    ],
)
def test_load_simple_bench_rejects_invalid_rows(tmp_path, row, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"eval_data": [row]}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_simple_bench_questions(path)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run --extra dev python -m pytest tests/test_simple_bench.py -q`

Expected: the new validation cases fail because rows are currently accepted or raise unhelpful exceptions.

- [ ] **Step 3: Implement explicit validation**

Parse the top-level shape and each row with clear `ValueError` messages before constructing `SimpleBenchQuestion`. Track question IDs in a set and reject duplicates.

- [ ] **Step 4: Add CLI numeric boundary tests**

Use `CliRunner` to prove `--budget-minutes 0`, `--flag-context-size 0`,
`--simple-bench-max-tokens 0`, and negative `--max-attempts` fail before any runner starts.

- [ ] **Step 5: Implement Typer range constraints**

Use `typer.Option(min=1)` for positive numeric options and preserve current defaults.

- [ ] **Step 6: Verify and commit the validation slice**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_simple_bench.py tests/test_cli.py -q
uv run --extra dev ruff check src tests
```

Then commit:

```powershell
git add src/gguf_limit_bench/simple_bench.py src/gguf_limit_bench/simple_bench_runner.py src/gguf_limit_bench/cli.py tests/test_simple_bench.py tests/test_cli.py
git commit -m "feat: validate autoresearch benchmark inputs"
```

### Task 3: Enforce release packaging and CI gates

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `tests/test_package_metadata.py`

- [ ] **Step 1: Write failing package-metadata tests**

Assert the project metadata has the correct description spelling, MIT license expression,
repository URLs, keywords, classifiers, and bundled SimpleBench assets:

```python
def test_public_package_metadata_is_release_ready():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert "LLLM" not in project["description"]
    assert project["license"] == "MIT"
    assert project["urls"]["Repository"].endswith("agent-pilot-autobench")
```

- [ ] **Step 2: Run the metadata test and confirm RED**

Run: `uv run --extra dev python -m pytest tests/test_package_metadata.py -q`

Expected: metadata fields are missing and the description contains `LLLM`.

- [ ] **Step 3: Add accurate package metadata**

Set the description to `Local-first GGUF and llama.cpp benchmarking for agent workloads.`
Add MIT license, repository/issues URLs, local-AI keywords, and supported Python/OS classifiers.

- [ ] **Step 4: Add CI release gates**

Keep the Python 3.11-3.13 test matrix and add one Python 3.13 quality job that runs:

```yaml
- run: uv run --extra dev mypy src
- run: uv run --extra dev python -m compileall -q src tests
- run: uv build
```

- [ ] **Step 5: Prove wheel installation in isolation**

Build the wheel into an ignored temporary directory, install it into a temporary `G:` venv,
and confirm the bundled dataset loads ten questions and `agent-autobench --help` exits zero.

- [ ] **Step 6: Commit the packaging slice**

```powershell
git add pyproject.toml uv.lock .github/workflows/ci.yml tests/test_package_metadata.py
git commit -m "build: add public release gates"
```

### Task 4: Create an evaluator-friendly public front door

**Files:**
- Modify: `README.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `CHANGELOG.md`
- Modify: `docs/AUTORESEARCH-PROGRAM.md`
- Modify: `docs/COMMAND-BOARD.md`
- Modify: `tests/test_public_identity.py`

- [ ] **Step 1: Add failing public-surface assertions**

Assert the README contains a first-run command, proof/verification section, limitations,
architecture link, security link, and no mandatory workstation-specific paths.

- [ ] **Step 2: Run public identity tests and confirm RED**

Run: `uv run --extra dev python -m pytest tests/test_public_identity.py -q`

Expected: architecture, security, and release-evidence requirements are absent.

- [ ] **Step 3: Rewrite the README first screen**

Lead with one value proposition, the Windows first-run command, a concise evidence list,
and links to architecture and example artifacts. Move long command/reference material lower.

- [ ] **Step 4: Add durable public documentation**

`docs/ARCHITECTURE.md` must map feature -> command -> source -> output -> tests.
`CONTRIBUTING.md` must document setup and all local gates.
`SECURITY.md` must describe private vulnerability reporting and the localhost boundary.
`CHANGELOG.md` must describe the unreleased autoresearch flag-ladder work without claiming a published release.

- [ ] **Step 5: Replace private path requirements with portable examples**

Use quoted placeholders such as `G:\models\model.gguf` only as optional Windows examples;
remove internal upstream checkout paths and personal workstation setup tasks from public docs.

- [ ] **Step 6: Verify and commit public documentation**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_public_identity.py -q
git grep -n -E 'C:\\Users\\|G:\\AI\\_codex_projects|clamshell|xtreme|semloc' -- README.md docs CONTRIBUTING.md SECURITY.md CHANGELOG.md
git diff --check
```

Expected: tests pass and the safety scan returns no private-machine requirements.

Commit the reviewed files with `docs: polish the public autoresearch story`.

### Task 5: Run the complete release-candidate audit

**Files:**
- Inspect: all tracked and intended untracked files
- Inspect: ignored runtime folders without staging them

- [ ] **Step 1: Run every local gate**

```powershell
uv run --extra dev python -m pytest -q
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev python -m compileall -q src tests
uv build
git diff --check
```

- [ ] **Step 2: Run CLI and wheel smoke checks**

Verify `agent-autobench --help`, `doctor --json-out`, `results`, and a flag-ladder
`--dry-run` using a dummy model path. Confirm no heavyweight server is started.

- [ ] **Step 3: Audit publication scope**

Inspect `git status`, staged diff, file sizes, secret-like patterns, private paths, and ignored
runtime folders. Do not stage models, databases, environments, caches, logs, or receipts.

- [ ] **Step 4: Simplify recently changed code**

Review only the autoresearch, flag ladder, SimpleBench, packaging, and new validation changes.
Remove duplication and improve names without changing behavior, then rerun all gates.

### Task 6: Publish through existing PR #9

**Files:**
- Update: GitHub PR #9 title and body

- [ ] **Step 1: Stage and commit any final reviewed files explicitly**

Use path-specific `git add`; inspect `git diff --cached --stat` and
`git diff --cached --check` before committing.

- [ ] **Step 2: Push the existing branch**

```powershell
git push origin codex/pilotbenchy-first-run-reports
```

- [ ] **Step 3: Rewrite PR #9 for the final diff**

Title: `[codex] Ship portfolio-grade local autoresearch benchmarking`

The body must explain the problem, architecture, safety model, benchmark evidence,
verification commands, limitations, and intentional exclusions.

- [ ] **Step 4: Verify remote checks**

Watch PR #9 until CI, dependency review, and CodeQL complete. Inspect logs for every failure,
fix the root cause, push again, and repeat until all required checks are green.

- [ ] **Step 5: Mark PR #9 ready for review**

Only convert the draft after the local audit and GitHub check rollup both pass.
