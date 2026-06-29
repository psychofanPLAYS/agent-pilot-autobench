# Session handoff — 2026-06-28

Branch: `codex/huggingface_recommended_settings_auto-rec+pull` (PR
[#17](https://github.com/psychofanPLAYS/agent-pilot-autobench/pull/17), merged to
`main`).

## TL;DR

The Agent Pilot benchmark now produces a **real, useful, model-vs-model agent-quality
result end-to-end** — the thing that had never worked before. It scores any local GGUF
on the 7 wiki-librarian packs against a live llama.cpp endpoint, renders an
OpenRouter-style comparison page, and answers "which model is the better librarian".

## What is proven (live, this session)

Full 7-pack librarian suite, both models served through llama.cpp with the froggeric
chat template, on a 4090:

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
| gen tok/s | ~75 | ~146 |

Verdict: **the 9B is the better vault librarian.** They tie on pure retrieval
(dedupe/gate/rerank/contradiction); the 9B wins decisively on the jobs that define a
memory worker (write-entry, triage, compress). The 4B is 2x faster but collapses on
write/triage — only viable if restricted to retrieval.

Key finding: **chat template matters a lot.** The same 9B scored compress 0.688 with
builtin chatml vs 1.000 with the froggeric template. This is exactly the failure the
preflight gates exist to catch.

## What shipped

- Librarian preflight gates + suite runner; failed cells write `preflight_fail`
  receipts, never a misleading quality-zero.
- Removed the hardcoded "must pick one Gemma AND one Qwen" cockpit lock — any model(s)
  now run.
- OpenRouter-style `results.html`: per-pack accuracy matrix ranked by agent quality,
  plain-English winner verdict. Reads `results.json` (champion_eval) / falls back to
  `librarian-suite-summary.json`.
- Per-model template/flag recommender (`template_recommend.py`): Qwen -> `--jinja
  --chat-template-file <froggeric>` auto-discovered from
  `<model-root>/Qwen-Fixed-Chat-Templates/chat_template.jinja`; Gemma -> `--jinja`.
- HF settings pull (`model_recommendations.py`): parse author-recommended llama.cpp
  settings from a model's HF README/aux files by resolved slug.
- Preflight answer-channel warmup 64 -> 1024 tokens (thinking models were falsely
  failing the gate before reaching `Final Answer`).
- Full `pilotBENCHY` -> `Agent Pilot` rebrand; Gemma-vs-Qwen framing removed from copy.

## Architecture pointers

- **Two pack-run paths.** `champion_eval.evaluate_champion_packs()` is the integrated
  path: it launches its own llama-server (`server_session.llama_server_session`), runs
  preflight, scores packs, writes `results.json`. The cockpit Start -> autoresearch ->
  `_run_champion_pack_eval` uses this (with `champion_pack_ids = LIBRARIAN_PACK_IDS`
  for `librarian_bench` mode). The `benchmark_suite` plan path instead shells out to
  `librarian_suite` against an assumed-running `127.0.0.1:8080` — it does NOT start a
  server itself.
- Reports: `reports.build_leaderboard()` scans `*/best-settings.json`, now also reads
  `results.json` for the agent-quality score + per-pack map.
- `select_questions(size=N)` clamps to available; `size=0` selects ZERO (gotcha — the
  `librarian_suite` path treats 0 as "full", `champion_eval` does not).

## How to reproduce a comparison

A throwaway driver lived in the session scratchpad (not committed). To re-run: call
`champion_eval.evaluate_champion_packs(model=<gguf>, llama_server=<exe>,
best_settings=AutoresearchSettings(context_size=8192, gpu_layers=99,
flash_attention=True, extra_server_args=template_recommend.recommended_model_flags(
model, search_roots=(Path("G:/AI/models"),))), run_dir=<dir>, pack_ids=<7 packs>,
sample_size=16, gpu_name="RTX 4090")` per model, then `reports.write_leaderboard(
Path("_runs"))`. Open `_runs/results.html`.

## Open work / next steps (priority order)

1. **Verify live cockpit streaming** during a real `apb`-launched run against a real
   backend (the one piece only ever proven with a fake backend).
2. **Thinking-model speed/discipline** — long `<think>` blocks make the suite slow; no
   per-pack max-token cap. Consider a thinking budget or a thinking-off scoring mode.
3. **Stability protocol** (section B of `09-hardening-spec.md`) — N=5 repeats,
   `score_stddev`, seed recorded per row. Not yet implemented.
4. **Coverage**: run the Gemma path live (single-BOS gate, double-BOS risk unproven).
5. Remaining `09-hardening-spec.md` checklist: per-letter accuracy reporting,
   adversarial subtypes, negative controls, difficulty tags + balance asserts.
6. HF recommendation parsing robustness (README formats vary).

See the Sonnet code review attached to the PR / session for a deeper critique.
