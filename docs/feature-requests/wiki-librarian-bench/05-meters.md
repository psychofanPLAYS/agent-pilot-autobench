# 05 — Meters: deterministic scorers

Status: draft, WIP

A meter is a dependent variable: something we measure about a cell's result.

## Non-negotiable: deterministic scoring only

No LLM-as-judge in the scoring path. Every task ships with:

- a **gold dataset**: `(input, gold_answer/labels)` pairs, and
- a **programmatic scorer**: exact match, schema validation, set/ranking metrics,
  or substring/entity coverage.

Where "quality of a summary" would tempt an LLM judge, we instead pre-extract a
**gold fact list** and check coverage via required substrings/entities. This keeps
the whole cube reproducible and crowd-mergeable.

## Per-job scorer map

| Job | Primary scorer | Metric(s) |
|-----|----------------|-----------|
| understand query | retrieval hit on gold via the rewritten query | recall@k, MRR |
| rerank + gate (rank) | order vs gold relevance order | nDCG, Kendall tau |
| rerank + gate (gate) | inject/don't-inject vs gold | precision, recall, **specificity** (correct abstention), confusion matrix |
| compress + inject | gold-fact coverage + token budget check | fact-coverage %, budget-adherence (0/1) |
| triage + extract | salience class vs gold + extracted-set overlap | accuracy, set-F1 |
| dedupe / merge | decisions vs labeled duplicate set | precision, recall, F1 |
| write entry | schema/frontmatter/link validation | exact pass/fail, field-level error count |
| consolidate / prune | merge/delete decisions vs gold | precision, recall, false-delete rate |

## Cross-cutting meters (recorded on every cell)

- **format adherence** — output parses against the required contract (JSON /
  frontmatter). Binary + violation class.
- **grounding** — every emitted entity/fact is a subset of the source/retrieved
  set. Hallucination count.
- **contradiction & staleness** — accuracy on a labeled set of new-vs-old conflicts.
- **output stability** — repeat the cell N times at fixed seed; report
  exact-output-match rate and score stddev. (Where thinking-on may regress.)
- **calibration** — when the model emits a confidence, Brier score / ECE vs
  correctness. (Only for jobs where we ask for confidence.)

## System meters (reuse existing machinery)

- TTFT (cold/warm, p90/p99), TPS (generation + prefill), generation-speed stddev.
- context falloff (tok/s retention as context grows).
- failure classes: timeout, model-load failure, crash, GPU OOM, mem-alloc — plus
  **new librarian classes**: format-violation, hallucination, over-injection
  (injected when it should have abstained), under-injection (abstained when it
  should have injected).

## The composite score

Add `librarian_bench_score` alongside the existing `agent_bench_score`, combining:

- quality (weighted job scores)
- grounding + format adherence (gates: a cell that violates format is heavily penalized)
- stability bonus / variance penalty
- system viability (TTFT/TPS/context/stability)
- failure penalties

Exact weights are an open question (open question 5). The Pareto recommender then
optimizes this score against latency/throughput to recommend knob-settings.

## Gold dataset sourcing

The suite needs labeled gold sets per job. Options (open question 6):

- hand-authored seed sets (small, high-quality, deterministic) — recommended start.
- synthesized from the owner's real vault/logs (most representative, but must be
  scrubbed of secrets before it can be shared).
- public memory/RAG datasets adapted to the schema.

Recommendation: start with small hand-authored deterministic seed sets per job so
the harness is exercised end-to-end, then grow from scrubbed real data.
