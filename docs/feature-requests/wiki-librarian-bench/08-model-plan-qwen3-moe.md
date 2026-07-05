# 08 — Model plan: Qwen3.5 / Qwen3.6 35B-A3B (MoE, hybrid-reasoning)

Status: draft (grounded against HF, 2026-06-24)

Per-model execution plan for the Qwen MoE line: `Qwen/Qwen3.5-35B-A3B` and
`Qwen/Qwen3.6-35B-A3B`. This is the richer of the two plans because the model has the
most knobs that actually move librarian quality: native thinking, a template that can
silently break that thinking, MoE speed behavior, and MTP speculative decoding.
Current comparison should be Qwen3.6-first, with Gemma 4 as the Google-family
challenger if needed. Shared hardening: [09-hardening-spec.md](09-hardening-spec.md).

## Identity / SSOT grounding

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|-------|-----------------|-----------------|
| Canonical HF slug | `Qwen/Qwen3.5-35B-A3B` | `Qwen/Qwen3.6-35B-A3B` |
| Architecture | `qwen3_5_moe` | `qwen3_5_moe` |
| Total / active params | 35.95B / ~3B (A3B) | 35.95B / ~3B (A3B) |
| License | apache-2.0 | apache-2.0 |
| Context | 256K | 256K |
| Reasoning | hybrid (thinking + non-thinking) | hybrid (improved tool-calling vs 3.5) |

Treat 3.5 and 3.6 as **separate SSOT rows** (separate slugs) and report the delta —
3.6's headline improvement is tool-calling/format accuracy, which should show up in
our format-adherence meter and the structured jobs (write-entry, gate).

Quant artifacts: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, plus an `IQ4` variant. MoE quant
sensitivity is **not** like a dense model — experts quantize unevenly, so do not
assume the dense-model quality/size curve transfers. Measure it.

MTP GGUFs (e.g. froggeric MTP builds) enable speculative decoding — see flag plan.

## The two crown-jewel knobs: thinking x template

These interact, and the interaction is the single most important trend this whole
suite exists to surface.

### Thinking (native)

- `enable_thinking=True` / `/think` — extended chain-of-thought in a `<think>` block.
- `enable_thinking=False` / `/no_think` — straight to the answer.
- Sampling MUST follow the mode:
  - thinking: `temp=0.6, top_p=0.95, top_k=20, min_p=0`. **Never greedy** — greedy
    decoding causes repetition/degradation in thinking mode.
  - non-thinking: `temp=0.7, top_p=0.8, top_k=20, min_p=0`.
  - `presence_penalty` 0–2 can curb repetition, but high values cause language mixing —
    treat as a cautious knob, default 0.

### Template (the silent killer)

- `froggeric-v21.3` — current pinned custom template for Qwen3.5/3.6; preserves
  thinking, supports the native XML tool-call format, and keeps the template path
  reproducible. Load with `--jinja`, `--chat-template-kwargs
  '{"enable_thinking":true,"preserve_thinking":true}'`, `--reasoning on`, and
  `--reasoning-format deepseek`.
- `stock-embedded` — the template baked into the official GGUF's tokenizer_config.
- `chatml` — **NEGATIVE CONTROL**: `--chat-template chatml` silently disables thinking
  mode. We include it precisely because it should demonstrably tank the thinking-on
  results — if our harness can't detect that breakage, the harness is wrong.

Template x thinking matrix to run per job:

```
                 thinking=on        thinking=off
froggeric-v21.3  A (expected best)  B
stock-embedded   C                  D
chatml           E (should BREAK    F
                 thinking -> ~= D)
```

If cell E does not collapse toward F, the preflight thinking-sanity gate failed.

## XML tool-call format

Native format (v16+ in froggeric, and the trained format):
`<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`,
parser `qwen3_coder`. Our v0 librarian jobs are MC/EXACT (no tool calls), but when we
add a tool-selection job this is the contract to score against — and it is where 3.6
should beat 3.5.

## MoE behavior (speed + determinism caveats)

- ~3B active params -> generation speed comparable to a 3–4B dense model while holding
  35B of knowledge. Expect strong tok/s; the win vs Gemma is on the speed axis.
- Expert routing + server batching can introduce small run-to-run nondeterminism even
  at fixed seed. The stability meter must account for this (see hardening spec) and not
  attribute MoE/batch jitter to the model "being unstable."

## llama.cpp flag plan

- **MTP speculative**: when using MTP GGUFs and llama.cpp b9180+, add the speculative
  rung `--spec-type draft-mtp --spec-draft-n-max N`. This is a major MoE speed axis the
  dense Gemma plan cannot use — give it first-class coverage.
- Standard ladder otherwise (kv-unified, RAM cache, cache reuse, q8 KV, threads).
- q8 KV at 256K is attractive but verify it does not erode long-context quality
  (couple it with the perplexity/needle falloff ladder).

## Per-job hypotheses

| Job | Hypothesis (thinking-on vs off) |
|-----|--------------------------------|
| write-entry | off wins — fast structured output; thinking adds variance, little gain. 3.6 > 3.5 |
| triage | off wins — cheap classification; thinking wastes latency |
| dedupe | mixed — subtle paraphrase/negation cases may benefit from thinking |
| gate | **on likely wins on the hard abstention/distractor/stale cases**; off may over-inject. The crossover here is the headline result |
| rerank | on helps hard negatives + lost-in-the-middle; off fine on easy |
| compress | on helps faithfulness on multi-fact notes; watch budget blowups when thinking |
| contradiction | **on wins** — temporal/numeric supersession is reasoning-heavy |
| stability | off >> on (thinking inflates variance) — quantify the cost of thinking |
| latency | off and MTP both cut TTFT/total; thinking multiplies tokens — the quality/latency Pareto is the deliverable |

## Knob grid for Qwen (tiered)

```
model:     {Qwen3.5-35B-A3B, Qwen3.6-35B-A3B} x {Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ4}
thinking:  {on, off}                      # native enable_thinking
template:  {froggeric-v21.3, stock-embedded, chatml(neg-control)}
sampling:  mode-matched (0.6/0.95/20 thinking; 0.7/0.8/20 non-thinking) + low-temp(0.1)
mtp:       {off, draft-mtp}               # MTP GGUF + recent llama.cpp only
flags:     standard ladder + q8 KV
context:   {4k, 8k, 16k, 32k, 64k, 128k, 256k}
jobs:      all 7 librarian packs
seeds:     N=5 repeats per cell (stability) — sampling on, fixed seed
```

This grid is large. Apply the tiered sweep ruthlessly (see knobs doc):
1. Preflight + determinism pass (thinking-sanity, template-sanity, single answer check).
2. **thinking x template 2-axis dense grid per job** (the crown jewel) at one quant.
3. Quant ladder + MTP at the winning (thinking, template).
4. Context/needle ladder.
5. Optuna for the continuous tail. Log every cap.

## Success criteria for "Qwen is a good librarian"

- A clear per-job thinking on/off recommendation (not one global setting).
- froggeric-v21.3 demonstrably >= stock on format adherence, and chatml demonstrably
  breaks thinking (validates the harness).
- gate precision under thinking-on beats Gemma's gate precision (the abstention test).
- MTP delivers real tok/s gain without a quality drop beyond a set tolerance.
- 3.6 >= 3.5 on the structured/tool jobs; quantify the upgrade.

## Risks specific to Qwen

- **Greedy ban** complicates determinism: the repro-stability run cannot use temp=0.
  Use temp=0.1 + fixed seed and LABEL it as quasi-deterministic; report operational
  stability at recommended sampling.
- chatml silently disabling thinking — caught only by the preflight gate.
- MoE/batch nondeterminism contaminating the stability meter.
- MTP requires a recent llama.cpp build (b9180+) — version-gate the rung.
- Quant sensitivity of experts — don't extrapolate the dense quality curve.

## References

- HF: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.6-35B-A3B` (qwen3_5_moe, 35.95B/A3B, 256K).
- Qwen sampling + thinking switches: Qwen docs / model cards (thinking 0.6/0.95/20 no
  greedy; non-thinking 0.7/0.8/20; /think /no_think; presence_penalty caveat).
- froggeric/Qwen-Fixed-Chat-Templates v21.3 (thinking preserved, XML tool calls;
  chatml kills thinking; use llama.cpp's DeepSeek reasoning parser).
