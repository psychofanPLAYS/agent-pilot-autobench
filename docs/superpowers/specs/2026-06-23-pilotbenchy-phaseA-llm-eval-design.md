# pilotBENCHY Phase A — Good LLM Testing + Visible Results

Date: 2026-06-23
Status: DESIGN — approved direction (A1–A5 in scope, gotcha pack = mix), pending spec review.
Author: Claude (with Dawid).

Relationship to prior work: successor to
`2026-06-22-pilotbenchy-real-benchmark-design.md` (which made the cockpit actually
ask SimpleBench questions) and the `2026-06-23` handoff. This spec is **Phase A** of the
A→B→C plan agreed with the owner:

- **Phase A (this spec):** make single-model LLM evaluation genuinely good and make the
  results visible/valuable. Question packs beyond SimpleBench, "let the model think",
  5-per-set with random/sequential selection, a 4090-tuned recommended-flags profile, and
  per-run + lifetime results display in the TUI.
- **Phase B (later spec):** context-ladder sweep 16k→256k + VRAM-headroom guard + staged
  flag×context search (no naive cross-product).
- **Phase C (later spec):** results web dashboard spawned from the TUI.

Embedder/reranker/query-expansion **detection and packs are explicitly out of scope** for
now. They are already kept out of the candidate set by the filename/path exclusion shipped
on 2026-06-23 (`discovery._is_non_generative`); that is sufficient for "ignore them for
now." Robust GGUF-metadata detection + their own packs are a later effort.

---

## 1. Problem & goals

### Problem
1. The only question set is SimpleBench (10 adversarially-hard reasoning questions). Small
   local models score near zero → results look broken and give no gradient.
2. Reasoning models can spend the whole token budget thinking and never emit a
   `Final Answer:` line. Today that scores as **wrong** (`predicted=None`), conflating
   "ran out of room to think" with "got it wrong". Verified live on Qwen3.5-4B-THINKING:
   Q1 generated 3890 tokens still mid-reasoning, scored None.
3. There is no surfaced, human-readable record of *what the model actually answered*, and
   no cross-session ("lifetime") view per model. The value proposition — "stop loading
   model after model by hand just to see tok/s and whether it even loads" — isn't realized
   because the results aren't legible.

### Goals (Phase A)
- **A1** Multiple question packs via one `QuestionPack` interface; ship `simple-bench`
  (existing 10) and a new mixed easy-tier pack (~25 well-known LLM gotchas + ~25 frozen
  real-dataset MC questions). Support both `multiple_choice` and `exact` answer types.
- **A2** Per-run selection of **N=5** questions per pack with `sequential | random` modes,
  reproducible (recorded seed); lifetime stats accumulate coverage across runs.
- **A3** "Let the model think": generous token budget, a single cheap **forced-final**
  follow-up when no answer is found, and a distinct **`incomplete`** outcome separate from
  `wrong`.
- **A4** A **4090-tuned recommended-always-on flag profile** (detected GPU → recommended
  flags, e.g. `--cache-type-k/v q8_0`, flash-attn on, `--parallel 4`), surfaced in results.
- **A5** **Display results**: per-run `results.md` + `results.json` (per pack: score; per
  question: prompt, expected, the model's actual answer, correct/incomplete, tps/ttft), a
  compact per-pack scoreboard in the TUI, and a per-model **lifetime** line.

### Non-goals (Phase A)
- Context-ladder sweep / VRAM guard / flag×context search (Phase B).
- Web dashboard (Phase C).
- Embedder/reranker/QE detection or packs.
- Live web access at run time. Datasets are **fetched once and frozen** into the repo;
  runs are fully offline and deterministic.

---

## 2. A1 — Question packs

### 2.1 Interface
Generalize the SimpleBench dataclasses (`simple_bench.py`) into a pack abstraction. A
`QuestionPack` is a frozen dataclass plus a JSON asset under
`src/gguf_limit_bench/data/packs/<pack_id>.json`.

```python
class AnswerType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"   # extract Final Answer: [A-F]
    EXACT = "exact"                       # extract Final Answer: <token>, normalized

@dataclass(frozen=True)
class PackQuestion:
    question_id: int | str
    prompt: str
    answer: str                  # canonical expected answer (letter, or normalized token)
    choices: list[str] | None = None   # present for MULTIPLE_CHOICE rendering
    tags: tuple[str, ...] = ()

@dataclass(frozen=True)
class QuestionPack:
    pack_id: str                 # "simple-bench", "easy-gotcha-mix"
    title: str
    tier: str                    # "easy" | "medium" | "hard"
    answer_type: AnswerType
    system_prompt: str
    questions: tuple[PackQuestion, ...]
```

JSON asset schema (one file per pack):
```json
{
  "pack_id": "easy-gotcha",
  "title": "Well-known LLM gotchas (exact answers)",
  "tier": "easy",
  "answer_type": "exact",
  "system_prompt_ref": "system_prompt_exact.txt",
  "source": "curated + ARC-Easy/OpenBookQA/CommonsenseQA frozen sample",
  "license": "see data/packs/LICENSES.md",
  "questions": [
    {"question_id": "straw-r", "prompt": "How many times does the letter R appear in the word STRAWBERRY?", "answer": "3", "tags": ["counting"]}
  ]
}
```

### 2.2 Answer extraction (generalizes `scorers.extract_answer`)
- Both types require the canonical `Final Answer:` convention (matches the official
  SimpleBench harness and the existing hardened extractor in `simple_bench.py`).
- `MULTIPLE_CHOICE`: `Final Answer:\s*([A-F])`, plus the existing fallbacks
  (`\boxed{X}`, `**X**`, letter-alone-on-line).
- `EXACT`: capture the token(s) after `Final Answer:`, then **normalize**: lowercase, strip
  punctuation/whitespace, map number words 0–20 ↔ digits (so "three" == "3"). The expected
  answer in the JSON is stored pre-normalized; comparison normalizes both sides.
- Extraction returns `None` when nothing matches → triggers the A3 forced-final follow-up.

### 2.3 Bundled packs (Phase A)
- **`simple-bench`** — the existing 10 (`data/simple_bench_public.json`), `hard`,
  `multiple_choice`. Unchanged data; re-expressed through the pack interface.
- **`easy-gotcha-mix`** — `easy`, `exact` for the gotcha half / `multiple_choice` handled
  per-question is messy, so this pack is split into **two packs** for type cleanliness:
  - **`easy-gotcha`** (`exact`, ~25): well-documented LLM gotchas with verifiable short
    answers — letter/character counts (R's in STRAWBERRY), decimal comparison (9.11 vs
    9.9), simple family-relation riddles, unit/conversion traps, basic logic. Curated +
    web-sourced (fetched and frozen).
  - **`easy-mc`** (`multiple_choice`, ~25): frozen samples drawn from ARC-Easy,
    OpenBookQA, and CommonsenseQA (permissive licenses; attribution + license text under
    `data/packs/LICENSES.md`). Fetched once via the owner's resources and committed.
  - Together these are the "pool of 50 more" the owner asked for.

### 2.4 Registry
`packs.py`: `load_pack(pack_id)`, `available_packs()`, `DEFAULT_PACKS = ("simple-bench",
"easy-gotcha", "easy-mc")`. The runner asks each selected pack and aggregates.

---

## 3. A2 — Selection: 5 per set, random/sequential

New config (`[benchmark]`), CLI flags, and TUI control:
- `question_sample_size: int = 5` — questions asked per pack per run.
- `question_selection: "sequential" | "random" = "sequential"`.
  - **sequential**: questions are asked in file order, resuming where the *previous run for
    this model+pack* left off (cursor persisted in `state_db`), wrapping at the end. Gives
    deterministic, complete coverage over consecutive runs.
  - **random**: a per-run seed (recorded in the receipt as `selection_seed`) selects N
    questions. Reproducible from the receipt; lifetime stats accumulate coverage.
- Surfacing: `[benchmark]` config keys, `--sample-size` / `--selection` CLI options on
  `autoresearch`, and a TUI key (proposed `g`) cycling sequential↔random shown in the
  status line. The selected mode + size are recorded in each receipt and `results.json`.

`selection.py` (new helper, distinct from the existing model-`selection.py` — name it
`question_selection.py`): pure function `select_questions(pack, size, mode, *, seed=None,
cursor=0) -> (chosen, next_cursor)`. Fully unit-testable, no I/O.

---

## 4. A3 — Let the model think

In the pack runner (generalized `simple_bench_runner.py`):
1. **Budget**: `max_tokens` default stays 4096, configurable via existing
   `--simple-bench-max-tokens` (renamed/aliased `--answer-max-tokens`). No mid-stream cut
   beyond the budget; the model gets room to finish.
2. **Forced-final fallback**: if extraction returns `None`, issue exactly **one** more
   stateless chat turn appending the model's prior answer plus:
   `"Reply with ONLY your final answer in the form 'Final Answer: X'."` capped at a small
   budget (≤ 64 tokens). Re-extract.
3. **Outcome taxonomy** (replaces today's bool-only `correct`):
   - `correct` — extracted answer matches expected.
   - `wrong` — extracted answer present but mismatched.
   - `incomplete` — no answer even after the forced-final follow-up (e.g. truncated mid
     reasoning). Reported separately; **does not** count as correct, but is tallied apart
     from `wrong` so a verbose-but-capable model is visible as "ran long", not "dumb".
   Add `outcome: Literal["correct","wrong","incomplete"]` to the per-question result and
   `incomplete: int` to the batch result. Accuracy = `correct / total`; a separate
   `completion_rate = (correct+wrong)/total` is reported.
4. (Deferred, noted) optional self-consistency majority vote (`num_responses`) — the
   official robustness lever; default off (cost multiplier), real flag added in a later
   phase.

---

## 5. A4 — Recommended-always-on flags (4090-tuned)

New `gpu_profiles.py`:
- Detect GPU name via the existing `telemetry` path (nvidia-ml-py). Match to a profile.
- **RTX 4090 / Ada (24 GB)** profile `recommended_always_on`:
  `--flash-attn on`, `--cache-type-k q8_0`, `--cache-type-v q8_0` (Ada fp8 tensor cores
  favor 8-bit KV math over q4), full GPU offload (`--gpu-layers 99`), and `--parallel 4`
  recorded as the **validated concurrency capability** (not a single-stream speed claim).
- These feed the existing `forced_server_args` mechanism as the **baseline always-on set**
  when no explicit override is given, and are echoed in results as
  *"Recommended for your GPU (RTX 4090): …"*.
- Unknown GPU → a conservative generic profile (flash-attn on, full offload, parallel 1)
  and a note that tuned recommendations aren't available yet for that card.
- Deep search over flags/context to *confirm/beat* these stays in Phase B; Phase A only
  establishes the recommended baseline and shows it.

---

## 6. A5 — Display results

### 6.1 Per-run artifacts (in the run receipt dir)
- **`results.json`** — machine form: model identity, selection mode+seed+size, GPU profile
  used, recommended flags, and per pack `{pack_id, tier, asked, correct, wrong, incomplete,
  accuracy, median_tps, median_ttft_ms}` plus the full per-question records (prompt,
  expected, **predicted/actual answer text**, outcome, tps, ttft, generated_tokens).
- **`results.md`** — human form rendered from the same data: a per-pack table and, under
  each pack, every question with the model's actual answer and outcome. This is the
  "actually display the answers we got" deliverable for Phase A (rich web view = Phase C).

### 6.2 Lifetime stats (per model)
Extend `state_db.py` with a `question_attempts` table keyed by
`(model_identity, pack_id, question_id)` recording every attempt's outcome + timestamp.
After each run, upsert attempts and compute per-model lifetime aggregates:
`seen`, `correct`, `accuracy`, `last_seen`, per pack. Used for the sequential cursor (A2)
and the lifetime line (below).

### 6.3 TUI
- After a run, the dashboard shows a **compact scoreboard**: one line per pack
  `simple-bench  2/5   easy-gotcha 4/5   easy-mc 3/5   (1 incomplete)` plus the champion
  verdict and the recommended-flags line.
- A **lifetime line** per finished model from `state_db`:
  `lifetime: easy-gotcha 38/50 seen · 31 correct (62%)`.
- Points the user to `results.md` for the full answers.

---

## 7. Components / files

New:
- `packs.py` — `QuestionPack`, `PackQuestion`, `AnswerType`, registry, loaders.
- `question_selection.py` — pure selection (sequential/random, seed, cursor).
- `gpu_profiles.py` — GPU detection → recommended-always-on flags.
- `data/packs/easy-gotcha.json`, `data/packs/easy-mc.json`, `data/packs/LICENSES.md`,
  and any extra system prompts (`system_prompt_exact.txt`).
- `results_report.py` — render `results.md`/`results.json` from batch results.

Changed:
- `simple_bench.py` / `simple_bench_runner.py` — generalize to packs + outcome taxonomy +
  forced-final follow-up; keep a thin `simple-bench` compatibility path.
- `state_db.py` — `question_attempts` table + lifetime aggregates + sequential cursor.
- `config.py` — `question_sample_size`, `question_selection`; recommended-flags baseline.
- `cli.py` — `--sample-size`, `--selection`, results rendering, pack selection;
  non-blank command help (fix the blank `autoresearch`/`survey`/`quick`/`tui` descriptions
  noticed during dogfooding).
- `tui.py` — selection-mode key, scoreboard + lifetime line.
- `autoresearch.py` — thread packs/selection/outcomes through attempts and the receipt.

---

## 8. Testing strategy

- **`question_selection.py`**: sequential wrap + cursor advance; random determinism for a
  fixed seed; size > pool clamps.
- **Extraction**: MC letter, `\boxed`, bold, letter-alone; exact normalization
  (three==3, case/punct); `None` when absent.
- **Outcome taxonomy**: correct/wrong/incomplete classification; forced-final path
  upgrades an initially-None response when the follow-up yields an answer (mock the
  client).
- **Packs**: every bundled pack JSON loads, validates (answer present, MC choices/letters
  sane), and round-trips.
- **gpu_profiles**: 4090 name → expected flags; unknown → generic + note.
- **results_report**: renders deterministic md/json from a fixed batch fixture.
- **state_db**: attempt upsert + lifetime aggregate + cursor persistence.
- Live smoke on the 4B model: a real short run produces `results.md` with actual answers
  and a non-zero gradient on the easy packs.

## 9. Data sourcing & licensing

- Fetch real-dataset samples (ARC-Easy, OpenBookQA, CommonsenseQA) once via the owner's
  resources; freeze a fixed N into `easy-mc.json`. Record dataset name, version/split, and
  license in `data/packs/LICENSES.md` (all three are research-permissive; verify before
  commit).
- Gotchas are curated/web-sourced facts with verifiable answers; no dataset license
  applies, but note provenance in the pack `source` field.

## 10. Rollout / validation

Build in plan order (writing-plans next). Each unit lands test-first. The release gate
(`pytest`, ruff, mypy, compileall) stays green. Final acceptance: a real run on the 4B
model from the TUI shows per-pack scores with actual answers and a lifetime line, and the
easy packs produce a meaningful non-zero gradient — i.e. the owner can *see* benchmark
results without hand-loading models.
