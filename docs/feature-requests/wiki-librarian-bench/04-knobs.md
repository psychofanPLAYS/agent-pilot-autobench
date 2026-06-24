# 04 — Knobs: the control axes we sweep

Status: draft, WIP

A knob is an independent variable. The full set of a run's knob values is hashed
into a **knob-signature** that is part of the SSOT composite key.

Two classes:

- **Model-facing knobs** — change what/how the model thinks. These are the point
  of the study.
- **Pipeline knobs** — change the retrieval plumbing. Held fixed when scoring the
  model on a job; swept separately to study the pipeline itself.

## Model-facing knobs

| Knob | Values (initial) | Notes / why it matters |
|------|------------------|------------------------|
| model identity | gemma3-27b, qwen3.5-35B-MoE, qwen3.6-35B-MoE, variants | grounded to OEM HF slug (SSOT key) |
| quantization | e.g. Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ* | quality/speed Pareto axis |
| **thinking mode** | on / off (+ budget if supported) | primary axis; can help quality but hurt stability/latency |
| **chat template** | froggeric v19 / stock / other | huge effect on tool-call + format adherence; per-model |
| llama.cpp flags | existing flag ladder | reuse current ladder + Optuna study |
| sampling | temperature, top-p, top-k, min-p, rep-pen, **seed** | temp=0 + fixed seed for determinism runs; vary to study robustness |
| context size | 4K, 8K, 16K, 32K... | reuse existing context ladder |
| context placement | gold memory at start / middle / end | lost-in-the-middle sensitivity |
| system-prompt variant | librarian instruction phrasings | prompt sensitivity |
| shot count | zero-shot / few-shot | does the model need examples to hold format |

## Pipeline knobs (held fixed when scoring the model)

| Knob | Values | Notes |
|------|--------|-------|
| retrieval k | e.g. 5 / 10 / 20 | candidate pool size |
| RRF weights | vector vs FTS weighting | fusion balance |
| chunk size | e.g. 256 / 512 / 1024 tokens | affects recall and budget |
| embed model | fixed choice | not the chat model; held constant |

## Sweep strategy

Full factorial across every knob is exponential and wasteful. Tiered approach:

1. **Determinism pass first** — temp=0, fixed seed, establish each model's
   baseline and stability before varying anything stochastic.
2. **Primary 2-axis grids** — the axes the owner cares about most
   (thinking on/off x chat template) run as small dense grids per job, because
   their interaction is the headline trend.
3. **Ladder the rest** — reuse the existing ordered flag ladder and context
   ladder (keep-if-better), so we never blow up the search.
4. **Optuna for the long tail** — on long/overnight runs, let the existing
   persistent per-model study converge the remaining continuous knobs (sampling,
   flags) against the librarian score.
5. **Warm-start from SSOT** — once the SSOT exists, seed the search with its
   current recommendation for that model.

## Determinism note

Two senses of "deterministic" both matter and are kept distinct:

- **Scorer determinism** — the metric is programmatic (see [05-meters.md](05-meters.md)).
- **Output determinism** — the model's own reproducibility, measured as a meter
  (`output stability`) by repeating a cell N times at fixed seed.

## Open items

- Exact model/quant list to ship in the default sweep (open question 4).
- Whether thinking "budget" is a separate axis for models that support it.
- froggeric v19 template source/pinning so the knob value is reproducible.
