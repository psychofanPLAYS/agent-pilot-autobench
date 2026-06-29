# 07 — Model plan: Gemma 3 27B (dense, no-thinking)

Status: draft (grounded against HF, 2026-06-24)

This is the per-model execution plan for `google/gemma-3-27b-it`. It specializes the
cube (jobs x knobs x meters) to this model's real characteristics. The companion
plan for the Qwen MoE line is [08-model-plan-qwen3-moe.md](08-model-plan-qwen3-moe.md);
shared hardening is [09-hardening-spec.md](09-hardening-spec.md).

## Identity / SSOT grounding

| Field | Value |
|-------|-------|
| Canonical HF slug | `google/gemma-3-27b-it` |
| Architecture | `gemma3` (dense, decoder-only, multimodal image-text-to-text) |
| Parameters | 27.43B (dense — all active every token) |
| License | `gemma` (gated — download requires acceptance; record this for SSOT provenance) |
| Context | 128K (uses sliding-window attention; see KV notes) |
| Base | `google/gemma-3-27b-pt` |
| Reasoning | **No native thinking mode** (this is the defining asymmetry vs Qwen) |

Quant artifacts to benchmark (each a distinct SSOT row):
`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, plus Google's **QAT** (quantization-aware-trained)
int4 checkpoint — QAT typically holds quality far better at 4-bit than naive PTQ, so
it is a first-class knob value, not a footnote.

## The thinking-mode asymmetry (read this first)

Gemma 3 has no `enable_thinking` switch and no `<think>` channel. A naive "thinking
on/off" knob is undefined here. To keep the cross-model comparison fair we replace it
with a **reasoning-elicitation knob** that both models can honor:

- `direct` — system prompt asks for the answer with no scratchpad.
- `prompted-cot` — system prompt invites brief step-by-step reasoning before the
  required `Final Answer:` line.

For Gemma, `prompted-cot` is the only way to get reasoning; for Qwen it is a third
point alongside its native thinking switch. This lets us answer "does reasoning help
this job?" symmetrically, while still reporting Qwen's native-thinking results
separately. Expectation: Gemma's `direct` mode will be unusually **stable** (low
variance) — likely a Gemma advantage on the stability meter.

## Serving / template gotchas (preflight gates)

1. **Double BOS** — the Gemma template emits `<bos>`; llama.cpp may also prepend BOS.
   Two BOS tokens measurably degrade output. Preflight MUST assert exactly one BOS.
2. **No system role** — Gemma folds the system prompt into the first user turn. The
   librarian system prompt must be injected that way; verify it survives templating.
3. **Turn format** — `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`.
4. **Vision tower** — use a text-only GGUF for librarian work; do not pay VRAM for the
   vision encoder we never exercise.
5. **No native tool-call format** — irrelevant for our MC/EXACT jobs, but means the
   `Final Answer:` contract is the only output discipline we can lean on.

## Recommended sampling (Gemma team)

- Author default: `temp=1.0, top_k=64, top_p=0.95, min_p=0.0`.
- Determinism regime (stability meter): `temp=0` greedy is allowed for Gemma (unlike
  Qwen) — use it for the repro-stability run.
- Also sweep a low-temp operational point (`temp=0.3`) since author default 1.0 is hot
  for a classifier-style librarian and may hurt format adherence.

## llama.cpp flag plan

- Dense model, **no MTP** — skip the speculative `draft-mtp` rung entirely.
- Reuse the standard flag ladder (kv-unified, RAM cache, cache reuse, q8 KV, thread
  sweep). 27B dense at Q4_K_M is ~16 GB — single-GPU friendly.
- Sliding-window attention changes KV growth at long context: the context/KV-falloff
  ladder is especially meaningful here. Watch q8-KV quality interaction with SWA.

## Per-job hypotheses (what to expect, so we can be surprised on purpose)

| Job | Hypothesis for Gemma 3 27B |
|-----|----------------------------|
| write-entry (type/slug) | Strong — excellent instruction following; slug formatting near-perfect |
| triage (keep/drop) | Strong, very stable in `direct` mode |
| dedupe | Strong on clear cases; watch paraphrase/negation edge cases |
| gate (inject/skip) | **Risk: over-eager inject** (low precision, weak abstention) — dense instruct models tend to be helpful; the gate's true-negative rate is the key thing to measure |
| rerank | Strong; 128K context helps; test lost-in-the-middle |
| compress | Strong faithfulness; watch budget adherence at `temp=1.0` |
| contradiction | Good; `prompted-cot` likely helps the subtle temporal cases |
| stability (cross-cutting) | **Likely best-in-class** in `direct` mode — surface this |

## Knob grid for Gemma (tiered, not full factorial)

```
model:        gemma-3-27b-it  x {Q4_K_M, Q5_K_M, Q6_K, Q8_0, QAT-int4}
reasoning:    {direct, prompted-cot}
sampling:     {author(1.0/64/0.95), low-temp(0.3), greedy(0) for stability}
flags:        standard ladder (no MTP)
context:      {4k, 8k, 16k, 32k, 128k}
jobs:         all 7 librarian packs
seeds:        N=5 repeats per cell for the stability meter
```

Tiering: determinism pass first (greedy) -> reasoning x sampling 2-axis grid per job
-> quant ladder -> context ladder. Log any caps; no silent truncation.

## Success criteria for "Gemma is a good librarian"

- gate specificity (correct-skip rate) >= a threshold we set after the first run —
  this is the make-or-break metric for a memory layer (precision over recall).
- write-entry + triage near-ceiling with high stability.
- compress grounding: zero hallucinated facts in the chosen summary.
- A clear, defensible Pareto point (quant x reasoning) vs the Qwen line on the
  quality/latency frontier.

## Risks specific to Gemma

- Gated download (license:gemma) — provenance + access friction for crowd SSOT.
- Double-BOS silently degrading every score if preflight is skipped.
- Hot author-default sampling (1.0) hurting strict-format jobs.
- SWA + KV-quant interaction at long context confounding the context-falloff meter.

## References

- HF: `google/gemma-3-27b-it` (gated, gemma3, 27.43B).
- Gemma-team sampling: temp 1.0 / top_k 64 / top_p 0.95 / min_p 0; double-BOS warning.
  (HF model discussions; Unsloth Gemma 3 run guide.)
