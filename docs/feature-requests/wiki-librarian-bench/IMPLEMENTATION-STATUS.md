# Implementation status

Last updated: 2026-06-24 · branch `codex/wiki-librarian-bench`

## v0 — deterministic job-task layer (DONE)

Each job is a pure, seed-deterministic generator that returns a
`gguf_limit_bench.packs.QuestionPack` graded by the existing scorer in
`answer_scoring.py`. No server, no LLM-judge. Files under
`src/gguf_limit_bench/librarian/`.

| Pack id | Module | Answer type | Q @seed0 | What it measures |
|---------|--------|-------------|----------|------------------|
| `librarian-write-entry` | `write_entry.py` | EXACT | 16 | memory `type` classification + kebab slug formatting |
| `librarian-triage` | `triage.py` | EXACT | 16 | keep/drop salience + durable-fact count extraction |
| `librarian-dedupe` | `dedupe.py` | MC | 12 | duplicate / related / new classification |
| `librarian-gate` | `gate.py` | MC | 11 | inject vs skip (incl. distractor + stale cases) |
| `librarian-rerank` | `rerank.py` | MC | 14 | pick the snippet that answers the query |
| `librarian-compress` | `compress.py` | MC | 16 | pick the faithful, complete summary |
| `librarian-contradiction` | `contradiction.py` | MC | 14 | confirms / contradicts / unrelated |

Total: 99 gold-labeled questions at seed 0 (each generator yields 10–16 per seed,
deterministic per seed, varying across seeds).

Shared scaffolding (authored centrally, not by the parallel workers):
`librarian/_common.py` (helpers + scorer-contract notes), `librarian/registry.py`
(id → builder map), `librarian/__init__.py` (exports the registry).

Integration: `packs.available_packs()` lists all 7 ids and `packs.load_pack(id)`
builds them (seed 0), via lazy imports that avoid the circular dependency.

## v0.1 — hardening applied (DONE)

Applied the [VALIDITY] fix from [09-hardening-spec.md](09-hardening-spec.md) section C:
`dedupe`, `gate`, and `contradiction` now randomize the label->letter mapping per
question via the new `_common.shuffle_choices`, so the answer letter no longer encodes
the class. Verified empirically across 60 seeds — gold-letter distribution is ~uniform
and every class now appears at every letter position (e.g. dedupe `duplicate/related/new`
each -> `ABC`; gate `inject/skip` each -> `AB`). `compress`/`rerank` already randomized
position and were left unchanged. Remaining hardening items (preflight gates, adversarial
subtypes, negative controls, two-regime stability) are still open in the spec checklist.

## Verification (all green)

- `pytest tests/test_librarian_*.py` → 94 tests pass
- full repo `pytest -q` → 598 passed, 1 skipped
- `ruff format --check` and `ruff check src tests` → clean
- `mypy` on `src/gguf_limit_bench/librarian` → clean
  (the 2 mypy errors in `hf_catalog.py` are pre-existing and untouched by this work)

## How to use it now

```powershell
uv run --extra dev python -c "from gguf_limit_bench import packs; print(packs.load_pack('librarian-gate').questions[0].prompt)"
uv run --extra dev python -m pytest tests/test_librarian_*.py -q
```

The packs are runnable by the existing pack runner / cockpit because they are
ordinary `QuestionPack`s. Pointing a live `llama-server` model at them, sweeping
the knobs, and aggregating scores is the next phase (not yet built).

## Decisions made while building (reversible — confirm or correct)

- Q1 (inject gate): built `librarian-gate` as a model-driven inject/skip decision,
  including correct-skip on keyword distractors and stale/deprecated memories.
- Q2 (query understanding / HyDE): NOT built yet. `librarian-rerank` covers
  retrieval reranking; a dedicated query-rewrite/HyDE job is still open.
- write_entry was scoped to EXACT single-token answers (type + slug) rather than
  full-JSON emission, because the EXACT scorer normalizes punctuation/case and
  full-JSON exact-match is fragile. Full-frontmatter emission can be a later job
  with a dedicated schema validator.

## Next phases (not started)

1. Query-understanding / HyDE job (resolve Q2).
2. Knob-sweep harness: thinking on/off, chat template (froggeric v19 vs stock),
   sampling/seed determinism repeats — produce cube cells.
3. `librarian_bench_score` aggregation + Pareto recommendation over the new knobs.
4. Web dashboard (FastAPI + two-way WebSocket, auto-open) and static SVG receipts.
5. SSOT export/sync grounded on HF slugs.
