# 06 — Open questions

Status: open · owner: psychofanPLAYS

These block finalizing the spec. Codex: do not decide these unilaterally; gather
options and trade-offs, but leave the choice to the owner.

## Job-shape confirmation (blocks [03-jobs.md](03-jobs.md))

1. **Inject gate** — does the local model decide *whether* to inject (and we test
   that decision, including correct abstention), or does the pipeline always
   inject top-k and let Claude sort it out? Changes a whole test family.
2. **Query understanding / HyDE** — does the model rewrite/expand/HyDE the query,
   or does raw Claude task text go straight into retrieval?
3. **Missing jobs** — anything the librarian does that 03 omits? Candidates:
   auto-linking/cross-referencing, write-time retrieval summaries (indexing-time
   model work), self-confidence scoring, scheduled re-summarization.

## Sweep / scope

4. **Default model + quant matrix** — exact list to ship in the default sweep
   (which current Gemma 4 build if any, which qwen3.5/3.6 MoE builds, which quants).
5. **Score weights** — how to weight job scores, grounding, stability, and system
   metrics into `librarian_bench_score`. Are some jobs gates (auto-fail on format
   violation) vs weighted contributors?
6. **Gold dataset source** — hand-authored seeds, scrubbed real vault/logs, public
   datasets, or a mix? Who authors/maintains the gold sets?

## SSOT / sharing (blocks [00-vision-ssot.md](00-vision-ssot.md))

7. **Sync transport** — data repo, GitHub releases, or a lightweight API for the
   cross-user DB?
8. **Hardware class buckets** — how coarse (GPU family + VRAM tier)? Must not be a
   fingerprint.
9. **Merge weighting + poisoning resistance** — how to aggregate conflicting rows
   across users and resist bad/poisoned submissions.

## Architecture / UI (relates to [01-architecture.md](01-architecture.md))

10. **Frontend approach** — embedded JS charting lib vs server-rendered SVG for
    the dashboards (the static receipts likely stay SVG either way).
11. **Live vs receipt parity** — should the live website render the exact same
    diagrams that get written as static receipts, or a lighter live view?

## Decided-by-building (2026-06-24, reversible — see IMPLEMENTATION-STATUS.md)

- Q1 (inject gate): built as a model-driven inject/skip job (`librarian-gate`),
  with distractor and stale cases. Confirm this matches the real pipeline.
- Q2 (query understanding / HyDE): built as `librarian-query`; `librarian-rerank`
  remains the separate retrieval reranking job.
- write_entry scoped to EXACT type + slug answers (not full-JSON emission) due to
  the EXACT scorer's normalization; full-frontmatter emission deferred to a job
  with a dedicated schema validator.

## Resolved (kept for history)

- UI: simple website + two-way WebSocket + auto-open browser, Python-backed,
  CLI-first core. (Owner decision, 2026-06-24.)
- Primary key: canonical OEM HF slug. (Owner decision, 2026-06-24.)
- Scoring: deterministic only, no LLM-judge. (Owner non-negotiable.)
- Skills, if installed, go repo-level. (Owner instruction.)
