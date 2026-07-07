# Implementation status

Last updated: 2026-06-28 - branch `codex/huggingface_recommended_settings_auto-rec+pull`
(see [SESSION-2026-06-28-handoff](../../superpowers/SESSION-2026-06-28-handoff.md))

## v0 - deterministic job-task layer (DONE)

Each job is a pure, seed-deterministic generator that returns a
`gguf_limit_bench.packs.QuestionPack` graded by the existing scorer in
`answer_scoring.py`. No server, no LLM judge. Files live under
`src/gguf_limit_bench/librarian/`.

| Pack id | Module | Answer type | Q @seed0 | What it measures |
|---------|--------|-------------|----------|------------------|
| `librarian-write-entry` | `write_entry.py` | EXACT | 16 | memory `type` classification + kebab slug formatting |
| `librarian-triage` | `triage.py` | EXACT | 16 | keep/drop salience + durable-fact count extraction |
| `librarian-dedupe` | `dedupe.py` | MC | 12 | duplicate / related / new classification |
| `librarian-gate` | `gate.py` | MC | 11 | inject vs skip (incl. distractor + stale cases) |
| `librarian-query` | `query.py` | MC | 12 | query expansion / HyDE payload selection without answering |
| `librarian-rerank` | `rerank.py` | MC | 14 | pick the snippet that answers the query |
| `librarian-compress` | `compress.py` | MC | 16 | pick the faithful, complete summary |
| `librarian-contradiction` | `contradiction.py` | MC | 14 | confirms / contradicts / unrelated |

Total: 111 gold-labeled questions at seed 0. Each generator yields 10-16 questions
per seed, deterministic per seed and varying across seeds.

Integration: `packs.available_packs()` lists all 8 ids and `packs.load_pack(id)`
builds them through the librarian registry.

## v0.1 - validity hardening (DONE)

Applied the section C validity fix from [09-hardening-spec.md](09-hardening-spec.md):
`dedupe`, `gate`, and `contradiction` randomize the label-to-letter mapping per
question via `_common.shuffle_choices`, so the answer letter no longer encodes the
semantic class. `compress` and `rerank` already randomized position.

## v0.2 - runnable preset artifacts (DONE, preflight-sensitive)

Bundled benchmark-suite plans exist:

- `benchmarks/plans/wiki-librarian-gemma4-26b-a4b-thinking.plan.json`
- `benchmarks/plans/wiki-librarian-qwen3-moe-thinking.plan.json`

They call `python -m gguf_limit_bench.librarian_suite`, split the eight librarian
packs across general/agentic suite phases, and emit `librarian_bench_score`,
`agent_bench_score`, per-pack JSON, TSV, Markdown, and suite summaries. See
[10-runnable-presets.md](10-runnable-presets.md).

Important honesty note: these plans are runnable artifacts, not proof that a cell is
valid. The live run now performs preflight first. If the model identity/template/BOS/
thinking/answer-channel evidence is missing or broken, the preset run writes a
`preflight_fail` receipt instead of scoring the librarian packs.

## v0.3 - preflight gates in real run paths (DONE)

The five fail-fast gates from section A of [09-hardening-spec.md](09-hardening-spec.md)
now run before librarian questions are scored in both active paths:

- `champion_eval.evaluate_champion_packs()` for integrated benchmark/cockpit mode.
- `librarian_suite.run_librarian_suite()` for direct benchmark-suite plan tasks.

Implemented gates:

| Gate | Current behavior |
|------|------------------|
| Identity | Requires the model to resolve to HF-style identity evidence before scoring. |
| Single-BOS (Gemma) | Calls `/tokenize` with and without special tokens and requires exactly one added token. Non-Gemma cells skip this gate. |
| Template-load | Requires `--jinja`; hashes `--chat-template-file` when provided. |
| Thinking-sanity (Qwen) | With explicit `enable_thinking`, checks that thinking-on emits `<think>` and thinking-off does not. Qwen cells without an explicit thinking knob are recorded as skipped. |
| Answer-channel | Runs a warmup MC prompt and requires a parseable `Final Answer:` / MC letter. |

Failure behavior: a failed gate writes `preflight.json` plus the normal result
surface (`results.json/results.md` or `librarian-suite-summary.json/.tsv/.md`) with
`failure_class: preflight_fail`, `status: preflight_fail`, and `asked: 0`. It does
not call `run_pack_questions()` and does not record the cell as a zero-quality model
score.

## v0.3.1 - recommendation-grade score gate (DONE)

Raw librarian scores are now separated from recommendation-grade
`agent_bench_score`. A librarian receipt must have at least 3 scored packs and 30
scored attempts before pilotBENCHY treats the score as hard recommendation
evidence. Smaller samples still keep their raw `librarian_bench_score`,
per-pack accuracy, TSV/Markdown receipts, and audit trail, but they are marked as
`weak_sample`, do not populate `agent_bench_score`, and cannot win the model
comparison by raw accuracy alone.

This encodes the live-run lesson below: two-pack samples were useful smoke tests,
but not discriminating enough for model recommendations.

## v0.4 - first real end-to-end model-vs-model run (DONE, live)

Scored two real GGUFs end-to-end through `champion_eval` (each model self-served via
`llama_server_session`, preflight-gated, full 7-pack), with the froggeric chat template
applied per-model by the new `template_recommend` recommender:

| Pack | Qwen3.5-9B Q8 | Qwen3.5-4B Q8 (thinking) |
|------|---------------|--------------------------|
| write-entry | 0.812 | 0.125 |
| triage | 0.812 | 0.250 |
| dedupe | 1.000 | 1.000 |
| gate | 0.909 | 0.909 |
| rerank | 1.000 | 1.000 |
| compress | 1.000 | 0.688 |
| contradiction | 1.000 | 1.000 |
| **agent quality (mean)** | **0.933** | **0.710** |

Findings: the 9B is the better librarian (wins write-entry/triage/compress, ties on
retrieval). Chat template matters: the same 9B scored compress 0.688 with builtin
chatml vs 1.000 with froggeric. `results.html` renders this as an OpenRouter-style
per-pack matrix with a winner verdict. Two-pack samples were not discriminating (false
tie); the full suite separated the models by 22 points.

Also fixed: the answer-channel preflight warmup budget (64 -> 1024 tokens) so reasoning
models can emit their `<think>` block and still reach `Final Answer`.

## Verification

- 2026-06-28: full repo `.\.venv\Scripts\python.exe -m pytest -q` -> 644 passed, 1
  skipped; `ruff check` clean; `mypy src` clean (71 files). Live 7-pack runs above.
- 2026-06-28:
  `.\.venv\Scripts\python.exe -m pytest tests\test_librarian_preflight.py tests\test_champion_eval.py tests\test_librarian_suite.py -q`
  -> 13 passed.
- 2026-06-28:
  `.\.venv\Scripts\ruff.exe check src\gguf_limit_bench\librarian\preflight.py src\gguf_limit_bench\champion_eval.py src\gguf_limit_bench\results_report.py src\gguf_limit_bench\librarian_suite.py tests\test_librarian_preflight.py tests\test_champion_eval.py tests\test_librarian_suite.py`
  -> clean.

Earlier 2026-06-24 full validation remains historical evidence only: the branch had
`pytest tests/test_librarian_*.py` passing, full repo `pytest -q` at 598 passed / 1
skipped, Ruff clean, and librarian mypy clean with unrelated `hf_catalog.py` errors.

## How to use it now

```powershell
uv run --extra dev python -c "from gguf_limit_bench import packs; print(packs.load_pack('librarian-gate').questions[0].prompt)"
uv run --extra dev python -m pytest tests/test_librarian_*.py -q
```

For live runs, serve a real GGUF through llama.cpp with the intended template flags,
then run the preset plan. A missing identity/template/thinking contract now blocks
the cell with `preflight_fail`.

## Decisions made while building

- Q1 (inject gate): built `librarian-gate` as a model-driven inject/skip decision,
  including correct-skip on keyword distractors and stale/deprecated memories.
- Q2 (query understanding / HyDE): built as `librarian-query`. It asks the model to
  choose a retrieval payload with a lexical vector and HyDE-style synthetic document
  while rejecting direct-answer, wrong-intent, and keyword-spam payloads.
- `write_entry` is scoped to EXACT single-token answers (type + slug), not full JSON.

## Remaining work

1. Query-understanding / HyDE job.
2. Per-letter accuracy reporting at score time.
3. Adversarial subtypes and negative-control cases.
4. Full knob-sweep harness: thinking on/off, chat template, sampling, seed repeats.
5. Two-regime stability protocol.
6. Format-adherence and grounding as scored gates.
7. Seed rotation and seed recorded in every SSOT row.
8. Difficulty tags and balanced-distribution asserts.
9. SSOT export/sync grounded on HF slugs.
