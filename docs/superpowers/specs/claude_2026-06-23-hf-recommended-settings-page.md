# claude_2026-06-23 — TUI page: HF dev-recommended settings, cached per model

> **Status:** Research-aid design from Claude (Opus 4.8). Codex owns code.
> Defines (1) a TUI page that shows the *model author's own recommended inference
> settings* pulled from Hugging Face, matched by slug, and (2) the per-model
> artifact cache that backs it. Claude is populating the initial cache via Sonnet
> lookups (artifacts under `src/gguf_limit_bench/data/model_recs/`).

## Why this matters (ties to "worth money")

Right now every benchmark starts from a generic forced baseline. But the model
*author* usually publishes the settings the model was tuned for (temp, top_p,
top_k, min_p, rep-penalty, context, chat template, thinking mode). Seeding each
run from the **dev-recommended baseline** means:

1. The benchmark measures the model at its *intended* operating point, not a
   generic guess — fairer and more trustworthy numbers.
2. It gives the buyer an instant answer ("the author says temp 0.6 / 256k YaRN")
   *before* a single benchmark runs — immediate value.
3. It becomes the seed/anchor for the flag-ablation and Pareto programs: deviate
   from the dev baseline only with measured justification.

This is the "settings intelligence" layer beneath the measurement layer.

## The "poor readme" problem (David's flag)

Most local GGUFs are community finetunes/quants (DavidAU, mradermacher, unsloth,
llmfan46, …) with thin or missing inference docs. Strategy:

- **Resolve to the base family first** (Qwen3.6-35B-A3B, Gemma-4-31B-it, …). The
  *official* base author (Qwen, Google) almost always documents recommended
  sampling + template. Use that as the authoritative baseline.
- **Layer finetune-specific overrides** when the finetune readme provides them
  (DavidAU in particular publishes detailed sampler params; uncensored/"heretic"
  variants sometimes change temp/rep-pen/system-prompt).
- **Record confidence + gaps + sources** in every artifact so a thin readme
  produces an honest "candidate/low" entry, never a confident guess.

## Slug → HF repo matching (reuse existing code)

`model_identity.resolve_path_identity(path)` already yields a `repo_id`
(`publisher/repo`) from `LM_Studio-gguf/<publisher>/<repo>/...` paths
(confidence `candidate`). Extend, don't reinvent:

1. If `resolve_path_identity` returns a `repo_id` → that's the HF repo.
2. Else parse a **family key** from the filename (strip quant suffix
   `-Q8_0/-UD-Q4_K_XL/.i1-…`, drop finetune adjectives) → e.g.
   `Qwen3.6-35B-A3B`, `gemma-4-31B-it`, `Gemma-4-E2B`.
3. Look up the artifact cache by `repo_id` first, then by `model_family` key
   (fuzzy/normalized). Embedders & rerankers are skipped (reuse
   `discovery.is_non_generative_gguf`).

## Artifact cache — one JSON per family

Path: `src/gguf_limit_bench/data/model_recs/<family-slug>.json`
(new dir; conflict-free with Codex's in-flight files). Schema v1:

```json
{
  "schema_version": 1,
  "model_family": "Qwen3.6-35B-A3B",
  "base_repo_id": "Qwen/Qwen3.6-35B-A3B",
  "aliases": ["Qwen3.6-35B-A3B", "qwen3.6-35b-a3b"],
  "matched_local_dirs": ["LM_Studio-gguf/lmstudio-community/Qwen3.6-35B-A3B-GGUF", "..."],
  "release_estimate": "2025-Q4",
  "architecture": {"type": "moe", "total_params": "35B", "active_params": "3B", "reasoning": true},
  "recommended_sampling": {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
    "notes": "thinking-mode values; non-thinking differs"
  },
  "recommended_context": {"default": 32768, "max": 262144, "yarn_notes": "..."},
  "chat_template": {"family": "chatml", "jinja_recommended": true, "system_prompt_support": true, "notes": "..."},
  "thinking": {"supported": true, "how": "<think> / enable_thinking", "recommended": "on for reasoning"},
  "llama_cpp_notes": "flash-attn ok; MTP draft heads present → speculative decode; ...",
  "finetune_overrides": [
    {"repo": "DavidAU/Qwen3.5-9B-Claude-4.6-...", "notes": "temp 0.8, rep-pen 1.05, ...", "source_url": "https://huggingface.co/..."}
  ],
  "sources": ["https://huggingface.co/Qwen/...", "..."],
  "confidence": "strong",
  "gaps": "what was missing or assumed",
  "researched_by": "claude-sonnet via HF lookup",
  "researched_on": "2026-06-23"
}
```

## TUI page

A new "Model settings" page (key e.g. `s` from the picker, or a panel on the
model-detail strip):

```
┌ Recommended settings — Qwen3.6-35B-A3B ───────────────── source: Qwen (strong) ┐
│ Sampling   temp 0.6 · top_p 0.95 · top_k 20 · min_p 0 · rep-pen 1.0            │
│ Context    default 32k · max 256k (YaRN) · q8_0 KV recommended                 │
│ Template   chatml · --jinja · system prompt supported                          │
│ Thinking   on (reasoning) · <think> tags                                       │
│ Notes      MTP draft heads → try --spec-type draft-mtp                         │
│ Finetune   DavidAU variant: temp 0.8, rep-pen 1.05 (overrides base)            │
│ [b] use as benchmark baseline   [r] refresh from HF   [o] open model card      │
└────────────────────────────────────────────────────────────────────────────────┘
```

- `[b] use as benchmark baseline` — seeds the program's sampling + context +
  template into the locked baseline for this model's run (the dev baseline
  becomes the anchor that ablation deviates from).
- `[r] refresh from HF` — see refresh flow below.
- Missing artifact → show "no dev recommendation cached — using generic 4090
  baseline" and offer `[r]`.

## Refresh flow (the app cannot spawn Claude)

Two tiers, honest about each:

1. **Shipped cache (authoritative).** Curated artifacts (these, produced by
   Sonnet HF lookups) ship in `data/model_recs/`. This is what the page reads.
2. **Live HF fallback (`apb model-recs refresh --model X`).** Hits the HF API for
   the resolved `repo_id`, downloads the model card + `generation_config.json` /
   `tokenizer_config.json` (chat template), and a heuristic parser extracts
   sampling defaults. Lower confidence; writes a `confidence: "candidate"`
   artifact and flags fields it could not find. This is a *parser*, not an LLM —
   so it can live in the app. Deep "read the prose recommendation" curation stays
   a Claude/Sonnet side task that refreshes the shipped cache.

Do **not** claim the TUI can call an LLM to read readmes live. It reads the cache;
curation of the cache is an offline Sonnet job (repeatable as new models drop).

## Initial cache population (in progress)

Claude is dispatching one Sonnet researcher per generative base family found in
David's local inventory (`s2_llama_models`), writing artifacts to
`data/model_recs/`. Families: Qwen3.6-35B-A3B (MoE, daily driver), Qwen3.6-27B
(dense), Qwen3.5-9B, Qwen3.5-4B, Gemma-4-31B-it, Gemma-4-26B-A4B-it (MoE),
Gemma-4-E2B. Embedders/rerankers excluded. Each artifact records confidence,
sources, and gaps per the "poor readme" strategy above.

## For Codex (when ready, after the program work)

- Add `model_recs.py`: load/resolve artifact by `repo_id` then family key;
  `is_non_generative_gguf` skip; return `None` cleanly when uncached.
- Wire the TUI page + `[b] use as baseline` into the program baseline builder.
- Add `apb model-recs refresh` (HF API parser tier). Keep the curated cache
  authoritative over the parser tier.
- Tests: slug→family resolution incl. quant-suffix stripping; cache hit by
  repo_id and by family; non-generative skip; graceful uncached `None`.
