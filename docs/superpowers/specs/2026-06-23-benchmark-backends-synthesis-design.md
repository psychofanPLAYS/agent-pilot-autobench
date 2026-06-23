# 2026-06-23 — Benchmark backends: borrow the best, own the tuning

> **Status:** Authoritative design (Claude Opus 4.8, now holding the dev lane;
> Codex out of context). Goal: stop hand-rolling "programs" from scratch; adopt
> cutting-edge harnesses from the field, filter each to its best part, and glue
> them with pilotBENCHY's unique moat. Pairs with the order-of-operations and
> worth-money docs.

## The strategic move

pilotBENCHY should **not** be a benchmark author. It should be a **tuning
orchestrator** that drives best-in-class measurement harnesses and turns their
results into a per-GGUF × per-GPU flag/context **decision**. Nobody else does
llama.cpp flag tuning with rigorous eval — that's the moat. Everything else is
borrowable.

```
                 pilotBENCHY (owns: search + decision + memory + UI)
   discover models/flags ─ autoresearch loop ─ Pareto recommender ─ receipts ─ TUI/dashboard
                                   │  delegates measurement to ↓
   ┌──────────────┬───────────────┬─────────────────┬──────────────┬──────────────┐
   │ INTELLIGENCE │ LONG-CONTEXT  │ SERVING/SPEED    │ AGENTIC       │ QUALITY      │
   │ lm-eval-     │ RULER         │ vLLM/GuideLLM    │ BFCL +        │ llama-       │
   │ harness      │ generators    │ metric method    │ inspect_ai    │ perplexity   │
   └──────────────┴───────────────┴─────────────────┴──────────────┴──────────────┘
```

## Field survey — who's best at what

| Player | What it's best at | Best part to take | License |
|---|---|---|---|
| **EleutherAI lm-evaluation-harness** | Academic intelligence (MMLU/GSM8K/ARC/HellaSwag/…), **native stderr/CI**, OpenAI-endpoint adapter (`local-completions`/`local-chat-completions`) | The whole intelligence backend + its confidence intervals | MIT |
| **NVIDIA RULER** | Synthetic, **uncontaminated, length-controllable** long-context tasks (NIAH single/multi-key/value/query, Variable Tracking, Aggregation, QA) | Vendor its task **generators** (pure-python, no heavy deps) | Apache-2.0 |
| **vLLM `bench serve` / GuideLLM** | Serving perf methodology: **TTFT, TPOT, ITL, E2EL, percentiles (p50/p90/p99), goodput under SLO, Poisson arrival/burstiness, request-rate sweeps** | Borrow the **metric definitions** + load model natively | Apache-2.0 |
| **vLLM `auto_tune`** | SLO-constrained server-config **search** (throughput s.t. latency SLO) | Borrow the **search methodology** for our flag×context loop | Apache-2.0 |
| **Berkeley BFCL** | Tool/function-calling correctness (AST + executable) | Optional agentic suite via subprocess | Apache-2.0 |
| **UK AISI inspect_ai** | Agentic/task eval framework, solvers+scorers | Optional agentic backend (`inspect eval`) | MIT |
| **Stanford HELM** | Holistic **scenario × metric × CI** reporting philosophy | Adopt the reporting *shape*, not the code | Apache-2.0 |
| **Karpathy autoresearch** | The disciplined fixed-budget keep/discard/crash loop | Already adapted (our autoresearch.py) | MIT |

## Filtered synthesis — best part of each, under our constraints

pilotBENCHY is **local-first** (Windows + 4090, offline-capable). That constrains
*how* we adopt, not *whether*:

1. **Vendor the lightweight, high-value generators.** RULER's synthetic
   long-context tasks are essentially deterministic text assembly + exact-match
   scoring — no torch. **Vendor/adapt them into our `packs` system** as a new
   procedural pack type. Big differentiation, ~zero dependency cost. Powers the
   `long-context-dropoff` program AND the Stage-1 needle climb.

2. **Compute serving metrics natively.** We already capture TTFT + gen TPS. Add
   **TPOT, ITL, E2EL, p50/p90/p99, and goodput-under-SLO** ourselves (a few
   arithmetic helpers + percentiles) using GuideLLM/vLLM's *definitions*. No new
   dependency; upgrades `speed` from single-stream tok/s to a real serving
   profile, and enables `concurrency` (request-rate sweep, Poisson arrival).

3. **Delegate heavy academic/agentic suites via `uvx` subprocess** — the repo
   already designed this (`docs/AUTORESEARCH-PROGRAM.md` calls
   `uvx --from lm-eval lm-eval run` and `inspect eval`; `benchmark_suite.py`
   exists). So lm-eval/BFCL/inspect run **ephemerally, never as hard deps**, and
   the app degrades gracefully (status `backend_unavailable`) when offline. This
   is the "don't reinvent MMLU" win without bloating the local install.

4. **Own the glue.** The autoresearch loop, the Pareto recommender, per-GGUF
   memory, the dev-recommended baseline seeding (`model_recs`), TUI/dashboard,
   and receipts stay ours. Borrowed harnesses feed it; they don't replace it.

## The unifying interface — `EvaluatorBackend`

One protocol so every borrowed harness plugs in the same way and the loop stays
backend-agnostic:

```python
class EvaluatorBackend(Protocol):
    id: str            # "lm-eval", "ruler", "serving", "bfcl", "perplexity", "simplebench"
    def available(self) -> BackendAvailability: ...        # installed? offline-ok?
    def run(self, *, base_url: str, settings: AutoresearchSettings,
            budget_seconds: int, receipt_dir: Path) -> EvalResult: ...

@dataclass(frozen=True)
class EvalResult:
    backend: str
    task: str
    score: float | None          # primary comparable scalar (None on failure)
    stderr: float | None         # standard error of score (CI half-width source)
    ci_low: float | None
    ci_high: float | None
    n: int                       # effective sample size
    status: str                  # complete | partial | backend_unavailable | crash
    raw_receipt: str             # path to the harness's own JSON output
    extra: dict                  # backend-specific metrics (tpot, niah@depth, ...)
```

This is the seam that makes the keep/discard/partial/crash decision (already in
`autoresearch.py`) work uniformly — and gives every program a real CI, killing
the "5-question coin flip" problem.

## Program → backend map

| Program | Primary backend | Context | Comparable score |
|---|---|---|---|
| fit (+characterize) | `ruler` (needle) + native serving | climb 16k→256k | max served ctx + quality@ctx |
| speed | native serving metrics | 16k | tok/s + TPOT/p99 |
| intelligence | `lm-eval` (uvx) → fallback `simplebench` | 64k, 1 Q/window | accuracy ± CI |
| long-context-dropoff | `ruler` generators | 16k/64k/128k/256k | NIAH/VT accuracy vs ctx |
| concurrency | native serving (request-rate sweep) | 16k | goodput @ SLO |
| flag-ablation | any of the above, one flag off | 16k | Δscore w/ CI |
| quality | `perplexity` (have it) | ladder | ppl delta |

## First implementation slice (proposed)

**Slice 1 — RULER-style procedural long-context pack (no new deps).**
- New `procedural_packs.py`: deterministic, seeded generators —
  `needle_single`, `needle_multikey`, `variable_tracking`, `word_aggregation` —
  each parameterized by target token length + difficulty, with exact-match
  scorers. Integrate into the existing `packs`/`pack_runner` registry.
- Powers `long-context-dropoff` and the Stage-1 fit needle immediately.
- TDD: generators are pure functions → fully unit-testable (right length band,
  answer embedded at requested depth, scorer exact-match, seed reproducibility).

**Slice 2 — native serving metrics** (TPOT/ITL/E2EL/percentiles/goodput) added to
the speed measurement path.

**Slice 3 — `EvaluatorBackend` interface + `LmEvalBackend` (uvx subprocess)**,
opt-in, graceful when unavailable. Then BFCL/inspect the same way.

Slices are additive and independently shippable; each keeps the gate green.

## Risks / decisions

- **Integration depth for heavy suites** (the one real fork): `uvx` ephemeral
  subprocess (recommended — zero hard dep, needs network on first use) vs a
  pinned optional extra (`pip install pilotbenchy[eval]`, offline-capable but
  heavy) vs reimplement (rejected — that's the reinvention we're avoiding).
- **Licensing:** all sources are MIT/Apache-2.0 → vendoring RULER generators and
  borrowing metric definitions is clean with attribution (add `NOTICE`/source
  headers like the existing SimpleBench/ARC notices).
- **Offline:** vendored RULER + native serving metrics work fully offline; only
  the uvx academic/agentic suites need network on first fetch.
