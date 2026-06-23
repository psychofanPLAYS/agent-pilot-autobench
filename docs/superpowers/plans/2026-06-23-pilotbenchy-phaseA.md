# pilotBENCHY Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make single-model LLM evaluation good and visible — multiple question packs, "let the model think", 5-per-set random/sequential selection, a 4090-tuned recommended-flags profile, and per-run + lifetime results display.

**Architecture:** Generalize the existing SimpleBench dataclasses/runner into a `QuestionPack` abstraction with a registry; add pure helpers for question selection and a richer outcome taxonomy (correct/wrong/incomplete) with a forced-final follow-up; render per-run `results.md`/`results.json`; accumulate per-model lifetime stats in `state_db`; surface a scoreboard + lifetime line in the TUI. Keep the `simple-bench` path working throughout.

**Tech Stack:** Python 3.13, dataclasses, `uv`, pytest, ruff, mypy, Typer (CLI), Textual (TUI), sqlite3 (`state_db`).

**Reference:** spec `docs/superpowers/specs/2026-06-23-pilotbenchy-phaseA-llm-eval-design.md`.

**Pre-built & audited (do NOT regenerate):** `src/gguf_limit_bench/data/packs/{easy-gotcha.json, easy-mc.json, system_prompt_exact.txt, LICENSES.md}`. `easy-mc` answers are authoritative dataset labels (verified). Apply the audit fixes in Task 8.

---

## File Structure

- Create `src/gguf_limit_bench/packs.py` — `AnswerType`, `PackQuestion`, `QuestionPack`, JSON loader, registry, bundled-pack loading. answer_source optional+derived.
- Create `src/gguf_limit_bench/answer_scoring.py` — extraction + normalization + scoring for both answer types (generalizes `scorers.extract_answer`).
- Create `src/gguf_limit_bench/question_selection.py` — pure `select_questions(pack, size, mode, seed, cursor)`.
- Create `src/gguf_limit_bench/gpu_profiles.py` — GPU name → recommended-always-on flags.
- Create `src/gguf_limit_bench/results_report.py` — render `results.md`/`results.json`.
- Modify `src/gguf_limit_bench/simple_bench.py` — add `outcome` to question result + `incomplete` to batch result (back-compatible defaults).
- Modify `src/gguf_limit_bench/simple_bench_runner.py` — run a pack, forced-final follow-up, outcome taxonomy.
- Modify `src/gguf_limit_bench/state_db.py` — `question_attempts` table, lifetime aggregates, sequential cursor get/set.
- Modify `src/gguf_limit_bench/config.py` — `question_sample_size`, `question_selection`.
- Modify `src/gguf_limit_bench/cli.py` — `--sample-size`, `--selection`, results rendering, blank-help fix.
- Modify `src/gguf_limit_bench/autoresearch.py` — thread packs/selection/outcomes + write results.
- Modify `src/gguf_limit_bench/tui.py` — scoreboard + lifetime line.

Each task is TDD: failing test → verify red → minimal impl → verify green → commit. Run tests with `uv run --extra dev python -m pytest <path> -q`. Keep the full gate green at the end of each task: `uv run --extra dev ruff check . ; uv run --extra dev mypy src`.

---

### Task 1: QuestionPack core (`packs.py`)

**Files:** Create `src/gguf_limit_bench/packs.py`; Test `tests/test_packs.py`.

- [ ] **Step 1 — failing test.** Cover: `AnswerType` enum; `load_pack("simple-bench")` returns a `QuestionPack` whose `answer_type is AnswerType.MULTIPLE_CHOICE` and `len(questions) == 10`; `load_pack("easy-gotcha")` has `answer_type EXACT` and `>= 20` questions; `load_pack("easy-mc")` is MC with `>= 20`; every `PackQuestion` has non-empty `prompt` and `answer`; `answer_source` is derived (`"curated_fact"` for easy-gotcha, starts with `"dataset_label:"` for easy-mc); `available_packs()` includes the three ids; unknown id raises `KeyError`.

```python
# tests/test_packs.py
from gguf_limit_bench.packs import AnswerType, available_packs, load_pack

def test_simple_bench_pack_loads_as_multiple_choice():
    pack = load_pack("simple-bench")
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert len(pack.questions) == 10
    assert all(q.prompt and q.answer for q in pack.questions)

def test_easy_gotcha_is_exact_with_curated_source():
    pack = load_pack("easy-gotcha")
    assert pack.answer_type is AnswerType.EXACT
    assert len(pack.questions) >= 20
    assert all(q.answer_source == "curated_fact" for q in pack.questions)

def test_easy_mc_carries_dataset_label_provenance():
    pack = load_pack("easy-mc")
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert len(pack.questions) >= 20
    assert all(q.answer_source.startswith("dataset_label:") for q in pack.questions)

def test_available_packs_and_unknown():
    ids = available_packs()
    assert {"simple-bench", "easy-gotcha", "easy-mc"} <= set(ids)
    import pytest
    with pytest.raises(KeyError):
        load_pack("does-not-exist")
```

- [ ] **Step 2 — verify red:** `uv run --extra dev python -m pytest tests/test_packs.py -q` → FAIL (module missing).

- [ ] **Step 3 — implement `packs.py`.** Define `AnswerType(StrEnum)` (`MULTIPLE_CHOICE="multiple_choice"`, `EXACT="exact"`). `PackQuestion` frozen dataclass: `question_id, prompt, answer, answer_source, choices=None, tags=()` plus optional `accept: tuple[str,...]=()` (acceptable answer variants, used by Task 2). `QuestionPack` frozen dataclass: `pack_id, title, tier, answer_type, system_prompt, questions`. Loader reads `data/packs/<id>.json`; the existing `simple-bench` lives at `data/simple_bench_public.json` (eval_data rows → MC questions, answer field) with `data/system_prompt.txt` — special-case it so it maps onto the pack interface (pack_id "simple-bench", tier "hard"). For pack JSON files, resolve `system_prompt_ref` against `data/packs/`. Derive `answer_source` when absent: `curated_fact` if `answer_type==EXACT` else `dataset_label:<prefix>` where prefix is the part of `question_id` before the first digit run (e.g. `arc-easy-0001`→`dataset_label:arc-easy`, `csqa-0001`→`dataset_label:csqa`). Registry: `_PACK_IDS = ("simple-bench","easy-gotcha","easy-mc")`; `available_packs()` returns that tuple; `DEFAULT_PACKS` same. `load_pack` raises `KeyError` for unknown ids. Validate on load: MC questions must have `choices` and a single-letter `answer` indexing them; raise `ValueError` otherwise.

- [ ] **Step 4 — verify green:** rerun the test file → PASS.

- [ ] **Step 5 — commit:** `git add src/gguf_limit_bench/packs.py tests/test_packs.py && git commit -m "feat: QuestionPack interface + registry (simple-bench, easy-gotcha, easy-mc)"`

---

### Task 2: Answer extraction + scoring (`answer_scoring.py`)

**Files:** Create `src/gguf_limit_bench/answer_scoring.py`; Test `tests/test_answer_scoring.py`.

- [ ] **Step 1 — failing test.**

```python
# tests/test_answer_scoring.py
from gguf_limit_bench.answer_scoring import extract_answer, normalize_exact, score_answer
from gguf_limit_bench.packs import AnswerType

def test_extract_mc_letter_variants():
    assert extract_answer("...\nFinal Answer: C", AnswerType.MULTIPLE_CHOICE) == "C"
    assert extract_answer("\\boxed{D}", AnswerType.MULTIPLE_CHOICE) == "D"
    assert extract_answer("nothing here", AnswerType.MULTIPLE_CHOICE) is None

def test_extract_exact_and_normalize():
    assert extract_answer("Final Answer: 3.", AnswerType.EXACT) == "3"
    assert normalize_exact("Three") == "3"
    assert normalize_exact(" 9.9 ") == "9.9"

def test_score_exact_accepts_variants_and_containment():
    # canonical "his son"; model answered a phrase containing it
    assert score_answer("The person is his son.", "his son", AnswerType.EXACT, accept=()) is True
    assert score_answer("Final Answer: sister", "his son", AnswerType.EXACT, accept=("sister",)) is True
    assert score_answer("Final Answer: 4", "3", AnswerType.EXACT, accept=()) is False

def test_score_mc_exact_letter():
    assert score_answer("Final Answer: B", "B", AnswerType.MULTIPLE_CHOICE, accept=()) is True
    assert score_answer("Final Answer: A", "B", AnswerType.MULTIPLE_CHOICE, accept=()) is False
```

- [ ] **Step 2 — verify red.**

- [ ] **Step 3 — implement.** `extract_answer(text, answer_type)`: MC → reuse the hardened logic from `simple_bench.py` (`Final Answer:\s*([A-F])`, `\boxed{X}`, `**X**`, letter-alone-on-line) returning an uppercase letter or `None`. EXACT → capture the text after the last `Final Answer:` up to end-of-line; if none, return `None`. `normalize_exact(s)`: lowercase, strip surrounding whitespace/punctuation (`.,!?;:"'`), collapse internal whitespace, map number words 0–20 ↔ digits via a small dict. `score_answer(response, expected, answer_type, accept)`: MC → `extract_answer(...) == expected`. EXACT → extract the candidate; if `None` return `False`; build the candidate set from the extracted token AND (for robustness with phrase answers) the whole normalized response; expected set = `{normalize_exact(expected)} | {normalize_exact(a) for a in accept}`; return `True` if any expected value equals the normalized candidate OR is a whitespace-bounded substring of the normalized full response. (This is the audit fix for brittle riddle phrasings.)

- [ ] **Step 4 — verify green.**

- [ ] **Step 5 — commit:** `git commit -m "feat: pack-aware answer extraction + exact/MC scoring with accept-variants"`

---

### Task 3: Question selection (`question_selection.py`)

**Files:** Create `src/gguf_limit_bench/question_selection.py`; Test `tests/test_question_selection.py`.

- [ ] **Step 1 — failing test.** `select_questions(questions, size, mode, *, seed=None, cursor=0) -> (chosen, next_cursor)`. Sequential: returns the next `size` in order from `cursor`, wraps around, and `next_cursor` advances modulo len. Random: deterministic for a fixed seed; returns `size` distinct items; size > len clamps to len without error.

```python
# tests/test_question_selection.py
from gguf_limit_bench.question_selection import select_questions

ITEMS = list(range(10))

def test_sequential_advances_and_wraps():
    chosen, nxt = select_questions(ITEMS, 5, "sequential", cursor=0)
    assert chosen == [0,1,2,3,4] and nxt == 5
    chosen2, nxt2 = select_questions(ITEMS, 5, "sequential", cursor=8)
    assert chosen2 == [8,9,0,1,2] and nxt2 == 3

def test_random_is_seeded_and_distinct():
    a, _ = select_questions(ITEMS, 5, "random", seed=42)
    b, _ = select_questions(ITEMS, 5, "random", seed=42)
    assert a == b and len(set(a)) == 5

def test_size_larger_than_pool_clamps():
    chosen, _ = select_questions(ITEMS, 99, "sequential", cursor=0)
    assert sorted(chosen) == ITEMS
```

- [ ] **Step 2 — verify red.**
- [ ] **Step 3 — implement** using `random.Random(seed)` for the random mode; pure, no I/O.
- [ ] **Step 4 — verify green.**
- [ ] **Step 5 — commit:** `git commit -m "feat: sequential/seeded-random question selection"`

---

### Task 4: Outcome taxonomy on result dataclasses (`simple_bench.py`)

**Files:** Modify `src/gguf_limit_bench/simple_bench.py`; Test `tests/test_simple_bench.py` (add cases).

- [ ] **Step 1 — failing test.** `SimpleBenchQuestionResult` gains `outcome: str = "wrong"` (one of `"correct"|"wrong"|"incomplete"`); `SimpleBenchBatchResult` gains `incomplete: int = 0` and `completion_rate: float = 0.0`. Existing constructions still work (defaults). Add a test asserting the fields exist and `to_dict()` includes them.
- [ ] **Step 2 — verify red.**
- [ ] **Step 3 — implement:** add fields with defaults so existing call-sites keep working; include in `to_dict()`.
- [ ] **Step 4 — verify green:** run the whole existing `tests/test_simple_bench.py` to confirm no regression.
- [ ] **Step 5 — commit:** `git commit -m "feat: correct/wrong/incomplete outcome taxonomy on results"`

---

### Task 5: Pack runner + forced-final follow-up (`simple_bench_runner.py`)

**Files:** Modify `src/gguf_limit_bench/simple_bench_runner.py`; Test `tests/test_simple_bench_runner.py` (add cases with a fake client).

- [ ] **Step 1 — failing test.** With a fake chat client: (a) a response containing `Final Answer: B` for an MC question scores `outcome=="correct"` when expected is B; (b) a long response with NO final answer triggers exactly one forced-final follow-up; if the follow-up returns `Final Answer: B`, outcome becomes `correct`; (c) if the follow-up still has no answer, outcome is `incomplete` (not `wrong`) and the batch `incomplete` counter increments. Mock the HTTP/chat call so no server is needed.
- [ ] **Step 2 — verify red.**
- [ ] **Step 3 — implement:** generalize the runner to take a `QuestionPack` + selected questions + `answer_max_tokens`. Per question: ask; `extract_answer`; if `None`, send ONE follow-up turn appending the prior assistant text + "Reply with ONLY your final answer in the form 'Final Answer: X'." capped at 64 tokens; re-extract. Classify outcome via `score_answer`/extraction: extracted+match→correct, extracted+mismatch→wrong, none→incomplete. Aggregate `correct/total` accuracy and `(correct+wrong)/total` completion_rate and `incomplete` count. Keep a thin `run_simple_bench(...)` wrapper that calls the generalized path with the `simple-bench` pack so existing callers/tests pass.
- [ ] **Step 4 — verify green:** run `tests/test_simple_bench_runner.py` and the existing runner tests.
- [ ] **Step 5 — commit:** `git commit -m "feat: pack runner with forced-final follow-up + outcome scoring"`

---

### Task 6: GPU profiles (`gpu_profiles.py`)

**Files:** Create `src/gguf_limit_bench/gpu_profiles.py`; Test `tests/test_gpu_profiles.py`.

- [ ] **Step 1 — failing test.** `recommended_always_on(gpu_name)` → for a name containing "4090" returns flags including `--flash-attn on`, `--cache-type-k q8_0`, `--cache-type-v q8_0`; `recommended_parallel(gpu_name)`==4 for 4090. Unknown GPU → conservative set (`--flash-attn on` only) + `recommended_parallel`==1. `describe(gpu_name)` returns a human string.

```python
# tests/test_gpu_profiles.py
from gguf_limit_bench.gpu_profiles import recommended_always_on, recommended_parallel

def test_4090_profile():
    flags = recommended_always_on("NVIDIA GeForce RTX 4090")
    assert "--cache-type-k" in flags and "q8_0" in flags
    assert recommended_parallel("NVIDIA GeForce RTX 4090") == 4

def test_unknown_gpu_conservative():
    assert recommended_parallel("Some Future GPU") == 1
```

- [ ] **Step 2 — verify red.** **Step 3 — implement** (a small dict of profiles keyed by substring match; flags returned as a tuple of strings). **Step 4 — verify green.** **Step 5 — commit:** `git commit -m "feat: GPU recommended-always-on flag profiles (RTX 4090)"`

---

### Task 7: Lifetime stats + cursor in `state_db.py`

**Files:** Modify `src/gguf_limit_bench/state_db.py`; Test `tests/test_state_db.py` (add cases).

- [ ] **Step 1 — failing test.** New functions: `record_question_attempt(db, model_key, pack_id, question_id, outcome, ts)`; `lifetime_pack_stats(db, model_key, pack_id) -> {seen, correct, accuracy, last_seen}` where `seen` counts distinct question_ids and `correct` counts those whose latest outcome is `correct`; `get_selection_cursor(db, model_key, pack_id) -> int` / `set_selection_cursor(...)` default 0. Use an in-memory/temp sqlite db in the test.
- [ ] **Step 2 — verify red.** **Step 3 — implement:** `question_attempts(model_key, pack_id, question_id, outcome, ts)` table + a `selection_cursor(model_key, pack_id, cursor)` table; create-if-not-exists alongside existing schema init. **Step 4 — verify green.** **Step 5 — commit:** `git commit -m "feat: per-model lifetime question stats + sequential cursor in state_db"`

---

### Task 8: Apply data audit fixes to packs

**Files:** Modify `src/gguf_limit_bench/data/packs/easy-gotcha.json`; Test `tests/test_packs.py` (extend).

- [ ] **Step 1 — failing test.** Assert the ambiguous `staircase-steps` question is gone, and that riddle/phrase questions carry an `accept` list of variants (e.g. `sister-riddle` accepts `["his son","son"]`, `doctor-riddle` accepts `["mother","the doctor is his mother","his mom"]`, `rooster-egg` accepts `["roosters don't lay eggs","none","it doesn't","trick question"]`). Assert `load_pack("easy-gotcha")` exposes `q.accept` for those ids.
- [ ] **Step 2 — verify red.**
- [ ] **Step 3 — implement:** remove `staircase-steps`; add an `"accept": [...]` array to each phrase/riddle question (numeric ones need none). Ensure `packs.py` reads `accept` into `PackQuestion.accept`.
- [ ] **Step 4 — verify green.**
- [ ] **Step 5 — commit:** `git commit -m "fix: harden easy-gotcha scoring (accept-variants; drop ambiguous staircase)"`

---

### Task 9: results_report.py (results.md / results.json)

**Files:** Create `src/gguf_limit_bench/results_report.py`; Test `tests/test_results_report.py`.

- [ ] **Step 1 — failing test.** Given a small fixture of per-pack batch results (pack_id, tier, asked, correct, wrong, incomplete, accuracy, median_tps, median_ttft_ms, per-question records with prompt/expected/predicted/outcome), `render_results_markdown(payload)` returns a string containing each pack id, the score `X/Y`, the word "incomplete" when any, and at least one question's actual predicted answer. `build_results_payload(...)` returns a JSON-serializable dict (round-trips through `json.dumps`).
- [ ] **Step 2 — verify red.** **Step 3 — implement** pure render functions (no I/O); a separate `write_results(run_dir, payload)` writes both files. **Step 4 — verify green.** **Step 5 — commit:** `git commit -m "feat: per-run results.md/results.json rendering"`

---

### Task 10: Wire into autoresearch + CLI + config

**Files:** Modify `autoresearch.py`, `cli.py`, `config.py`; Test `tests/test_cli.py` (extend), `tests/test_config.py`.

- [ ] **Step 1 — failing test.** `config` exposes `question_sample_size` (default 5) and `question_selection` (default "sequential"), overridable by env `PILOTBENCH_QUESTION_SELECTION` and CLI. `autoresearch --help` shows non-empty descriptions; new `--sample-size` and `--selection` options parse. A unit test on the autoresearch wiring (with the existing fakes) asserts a `results.json` is written into the run dir and references the selected packs.
- [ ] **Step 2 — verify red.**
- [ ] **Step 3 — implement:** add config fields (mirror `forced_server_args` pattern); thread `packs`, `sample_size`, `selection`, seed, and the GPU recommended flags through `_run_one_autoresearch`; after the batch, call `write_results`, update `state_db` attempts + cursor. Add docstrings to the `survey`, `quick`, `autoresearch`, `autoresearch-all`, `tui` commands (fix blank help). Apply the 4090 recommended-always-on flags as the baseline when `forced_server_args` is unset.
- [ ] **Step 4 — verify green:** run `tests/test_cli.py tests/test_config.py`.
- [ ] **Step 5 — commit:** `git commit -m "feat: wire packs/selection/results into autoresearch + CLI; fix blank command help"`

---

### Task 11: TUI scoreboard + lifetime line

**Files:** Modify `src/gguf_limit_bench/tui.py`; Test `tests/test_tui.py` (extend if present, else add a pure-function test).

- [ ] **Step 1 — failing test.** Add a pure helper `format_scoreboard(per_pack)` → e.g. `"simple-bench 2/5 · easy-gotcha 4/5 · easy-mc 3/5 (1 incomplete)"`, and `format_lifetime_line(stats)` → `"lifetime: easy-gotcha 38/50 seen · 31 correct (62%)"`. Test these pure formatters.
- [ ] **Step 2 — verify red.** **Step 3 — implement** the formatters + call them from `_show_champion`/dashboard update; read lifetime via `state_db`. **Step 4 — verify green.** **Step 5 — commit:** `git commit -m "feat: TUI per-pack scoreboard + lifetime line"`

---

### Task 12: End-to-end validation on the 4B model

**Files:** none (validation); then docs.

- [ ] **Step 1** — full gate: `uv run --extra dev python -m pytest -q ; uv run --extra dev ruff check . ; uv run --extra dev mypy src ; uv run --extra dev python -m compileall src`.
- [ ] **Step 2** — real run via the helper: `_scratch_apb.ps1 autoresearch --model "<4B Qwen3.5-4B-THINKING Q8 path>" --budget-minutes 8 --sample-size 5 --selection random`. Confirm: `results.md` exists with the model's ACTUAL answers across `simple-bench`, `easy-gotcha`, `easy-mc`; the easy packs show a non-zero gradient (small model gets several right); at least one `incomplete` is classified distinctly from `wrong` if a thinking answer runs long; the run prints/leaves a per-pack scoreboard. Kill any llama-server afterwards.
- [ ] **Step 3** — update `CHANGELOG.md` ([Unreleased]) and the memory file `C:\Users\psychofanPLAYS\.claude\projects\G--\memory\pilotbenchy-project.md` + `MEMORY.md` with Phase A shipped status + evidence.
- [ ] **Step 4 — commit:** `git commit -m "docs: Phase A validation evidence + CHANGELOG"`

---

## Self-Review notes

- **Spec coverage:** A1 packs → Tasks 1,2,8; A2 selection → Task 3 + config/wiring Task 10; A3 let-it-think → Tasks 4,5; A4 4090 flags → Tasks 6,10; A5 results+lifetime+TUI → Tasks 7,9,10,11. Blank-help fix → Task 10. Audit fixes → Task 8 + Task 2 (containment/accept). End-to-end → Task 12.
- **Back-compat:** `simple-bench` keeps a thin wrapper (Tasks 1,5); result dataclass fields are additive with defaults (Task 4) so existing tests don't break.
- **Type consistency:** `AnswerType`, `PackQuestion(.accept)`, `score_answer(...accept=)`, `outcome` strings, `incomplete` counter, and `recommended_always_on/recommended_parallel` names are used consistently across tasks.
