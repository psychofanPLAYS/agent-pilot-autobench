# Feature request: wiki-librarian benchmark suite

Status: **v0 generators integrated + preflight gates implemented** (2026-06-28)
Owner: psychofanPLAYS · Contributors welcome (Codex, CC)

## Implemented so far (v0)

The deterministic job-task layer is built, integrated, and green. See
[IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) for the full snapshot.

- 7 librarian job generators under `src/gguf_limit_bench/librarian/`, each a pure
  seed-deterministic `build(seed) -> QuestionPack` graded by the existing EXACT /
  MULTIPLE_CHOICE scorer (no LLM-judge, no server): `write_entry`, `triage`,
  `dedupe`, `gate`, `rerank`, `compress`, `contradiction`.
- A registry (`librarian/registry.py`) and wiring into `packs.py` so the packs are
  discoverable via `available_packs()` and loadable via `load_pack()`.
- 99 gold-labeled questions at seed 0.
- Librarian preflight gates now run before scoring in the direct suite and champion
  eval paths. Failed identity, template, Gemma BOS, Qwen thinking, or answer-channel
  checks write explicit `preflight_fail` receipts instead of misleading zero scores.

Not yet built (next): query-understanding/HyDE job (open question 2), full
knob-sweep coverage (thinking on/off, chat template, seed repeats), adversarial and
negative-control expansions, the web dashboard, and SSOT sync.

## What this is

A new capability suite for Agent Pilot Autobench that measures how
well a **local model performs the jobs of a memory/RAG "librarian"** that serves
Claude Code / Codex.

The "wiki" here is **not** human-facing prose. It is a local-model-powered memory
layer: markdown notes + a vector store, retrieved with a hybrid **RRF** fusion of
vector and full-text search over `sqlite-vec`, with **autonomous injection** of
relevant memories / lessons / logs into the *agent's* context. Any local model
(for example gemma3-27b, or a qwen3.5 / qwen3.6 ~35B MoE, and other variants) is
the librarian doing the cognitive work. Agent Pilot is the instrument that
captures **every conceivable data point** about that model's librarian
performance across every knob we can turn, then draws trends and a
recommendation out of them.

## The core mental model: the measurement cube

Everything reduces to three independent dimensions. See
[02-measurement-cube.md](02-measurement-cube.md).

```
                Meters (what we measure)
                  ^
                  |
                  |        . cell = one run
                  |       /
                  +-------------------> Knobs (what we vary)
                 /
                /
           Jobs (what we ask the model to do)
```

- A single benchmark run = one **cell**: `(Job, Knob-setting) -> Meter values`.
- "All conceivable data points" = fill the cube.
- "Trends" = slice the cube along one axis (e.g. score vs thinking on/off).
- "Recommendation" = the existing Pareto layer reading the cube.

## Files in this folder

| File | What it covers | State |
|------|----------------|-------|
| [00-vision-ssot.md](00-vision-ssot.md) | The endgame: shared cross-user DB, GitHub sync, HF-slug-grounded single source of truth, recommended flags | draft |
| [01-architecture.md](01-architecture.md) | UI (simple website + two-way WebSocket + auto-open), planes, CLI-first core, repo integration points | draft |
| [02-measurement-cube.md](02-measurement-cube.md) | The Jobs x Knobs x Meters framing | draft |
| [03-jobs.md](03-jobs.md) | The librarian's job map (the atoms we test) | draft |
| [04-knobs.md](04-knobs.md) | The control axes we sweep | draft, WIP |
| [05-meters.md](05-meters.md) | The deterministic scorers / metrics | draft, WIP |
| [06-open-questions.md](06-open-questions.md) | Decisions still needed from the owner | open |
| [07-model-plan-gemma3-27b.md](07-model-plan-gemma3-27b.md) | Per-model plan: Gemma 3 27B (dense, no-thinking, double-BOS, QAT) | grounded |
| [08-model-plan-qwen3-moe.md](08-model-plan-qwen3-moe.md) | Per-model plan: Qwen3.5/3.6 35B-A3B (MoE, thinking x template, MTP) | grounded |
| [09-hardening-spec.md](09-hardening-spec.md) | Hardening the whole suite: preflight gates, determinism, MC validity, adversarial cases | actionable |
| [10-runnable-presets.md](10-runnable-presets.md) | Runnable Gemma/Qwen plan artifacts and first live command path | runnable |
| [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) | What's built vs not, verification snapshot | live |

## How to contribute (Codex / CC)

1. Pick one file. Each file is a self-contained unit with a clear scope.
2. Honor the two non-negotiables (see below). If a change violates them, stop and
   raise it in `06-open-questions.md` instead.
3. Mark anything speculative as `WIP` and add it to `06-open-questions.md` rather
   than silently deciding for the owner.
4. Keep it design-only for now. No code until the owner approves the spec and we
   move to an implementation plan.
5. If you install any Claude Code skills/plugins for this work, install them
   **repo-level**, not global.

## Two non-negotiables

1. **Deterministic scoring.** Every task has a programmatic scorer (exact match,
   schema validation, set/ranking metrics against a gold label). No LLM-as-judge
   in the scoring path. See [05-meters.md](05-meters.md).
2. **HF slug is the primary key.** Every result is grounded to a canonical OEM
   Hugging Face model slug so results merge cleanly across users into the SSOT.
   See [00-vision-ssot.md](00-vision-ssot.md).

## Glossary

- **SSOT** — single source of truth (the aggregated cross-user results DB).
- **RRF** — reciprocal rank fusion; the algorithm that merges vector + FTS hits.
- **HyDE** — hypothetical document embeddings; generate an answer, embed *that* to retrieve.
- **librarian / wiki worker** — the local model acting as the memory layer's brain.
- **job / knob / meter** — the three cube dimensions (task / variable / metric).
- **thinking on/off** — the model's reasoning toggle; a primary knob.
- **froggeric v19** — a chat-template variant for Qwen models; a primary knob.
- **serving-Claude** — the layer exists to feed Claude Code / Codex, powered locally.
