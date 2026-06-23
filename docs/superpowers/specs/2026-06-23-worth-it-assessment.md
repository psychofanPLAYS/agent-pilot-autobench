# 2026-06-23 — Is pilotBENCHY worth paying for vs. manual flag trial-and-error?

> Honest assessment by Claude (Opus 4.8) holding the dev lane. No marketing.
> Written after landing the decision layer (serving metrics, procedural
> long-context, Pareto recommender, recommendation adapter).

## The thing we're competing against

A competent llama.cpp user tuning a new GGUF on a 4090, by hand:

1. Copies flags from the model card / community wisdom (flash-attn, q8_0 KV,
   kv-unified, jinja, gpu-layers 99).
2. Eyeballs tok/s in llama-server or runs `llama-bench` once.
3. Bumps `--ctx-size` until it OOMs, backs off.
4. Stops at the first setting that "feels fine."

Cost: ~20-40 min of fiddling. **Strengths:** fast to "good enough", free, no
setup, uses tools already installed. **Blind spots:** no quality measurement
(did q8_0 KV hurt accuracy? unknown), no real ablation (which flag actually
helped?), no long-context falloff, fooled by run-to-run noise, not reproducible,
no record, and it finds *a* setting — not the *optimum*.

## What pilotBENCHY does that manual can't

| Capability | Manual | pilotBENCHY | Status |
|---|---|---|---|
| Systematic one-variable flag ablation | ✗ guess | ✓ flag ladder | shipped |
| VRAM-aware ascending context search (no OOM crash, SWA-aware) | ✗ trial OOM | ✓ `context_limit` + `vram` | shipped |
| Honest partial results (not "crash") | n/a | ✓ partial reporting | shipped |
| Per-GGUF memory (champion + ctx ceiling across runs) | ✗ in your head | ✓ `state_db` | shipped |
| Seed from the author's recommended settings | ✗ maybe | ✓ `model_recs` (7 families) | shipped |
| **Pareto "use these flags" recommendation** | ✗ | ✓ `recommendation.py` | **shipped — wired into every autoresearch run (`recommendation.md`/`.json`)** |
| Serving profile (TPOT/ITL/p99/goodput) | ✗ | ✓ `serving_metrics.py` | built (library); speed-path emission is next |
| Uncontaminated long-context falloff (RULER needle/var-tracking) | ✗ | ✓ `procedural_packs.py` | **built, registry-wired** |
| CI-backed intelligence (MMLU/GSM8K via lm-eval) | ✗ | ◐ designed (uvx backend) | **not built yet** |
| Sustained-load / thermal-throttle truth | ✗ (you'd notice) | ✗ | not built |
| Reproducible receipts you can defend / diff across builds | ✗ | ✓ `_runs/` | shipped |

## Honest weaknesses (where it does NOT beat manual today)

1. **Time-to-first-answer.** A real flag×context sweep with questions runs 20 min
   to hours per model. Manual "good enough" is faster. The tool only wins when
   you value the *optimum* or need *evidence*, or run *many* models.
2. **The accuracy axis is still weakly grounded.** Until the lm-eval CI backend
   lands, "accuracy" comes from a small fixture and is noisy — so the
   accuracy dimension of the Pareto recommendation is provisional. This is the
   biggest honesty caveat. (Procedural long-context tasks help; full academic
   CIs are the fix.)
3. **Partly resolved:** the Pareto recommendation is now wired into every
   autoresearch run (`recommendation.md` + `recommendation.json`), so the output
   *is* a decision card, not just a leaderboard. Still library-only:
   `serving_metrics` (the speed path doesn't emit TPOT/p99 yet). So speed is
   measured but not yet shown as a full serving profile.
4. **Speed can be optimistic.** No sustained-load/thermal program yet, so a short
   probe may overstate all-day tok/s. A manual user running the model for hours
   would feel a throttle the bench might miss.
5. **Setup friction.** It's an app to install and run; manual is the llama-server
   you already have open.

## Verdict

**Worth paying for? Conditionally — and the gap to an unqualified "yes" is now
small and well-defined.**

- It **already beats manual** on rigor, reproducibility, OOM-safe context
  discovery, cross-run memory, and finding the *optimum* rather than "good
  enough." For someone who tunes many GGUFs and squeezes a specific GPU (the
  owner's profile: dozens of models, cares about the 4090), the math favors the
  tool: cost-of-a-suboptimal-setting × number-of-models > the minutes manual
  saves.
- It **loses to manual** on time-to-first-answer and setup for the casual
  "does this one model run okay" case.
- The remaining gap to "clearly worth money" is three concrete, mostly-designed
  steps: **(a)** wire the built decision layer (serving metrics + Pareto
  recommendation) into the live run so the output is a *decision*; **(b)** land
  the lm-eval CI backend so the accuracy axis is trustworthy; **(c)** add the
  sustained-load/thermal program so speed reflects reality.

**One-line:** today it's a rigorous measurement harness with a freshly-built
decision layer and a credible, mostly-designed path to a product — genuinely
useful for power users now, and worth paying for once (a)-(c) land, which is
weeks of focused work, not a rewrite.

## Recommended next slices (priority order)

1. ~~Wire the recommendation into the live run~~ — **DONE** (this session).
2. **Speed path emits serving metrics** (TPOT/ITL/p99/goodput) — finishes the
   serving-profile story. (Low risk; library already built + tested.)
3. **lm-eval CI backend** (uvx) — trustworthy accuracy axis. (Biggest remaining
   honesty fix.)
4. **Sustained-load/thermal program** — honest all-day speed.
5. Then BFCL/inspect agentic backends; build-over-build regression tracking.
