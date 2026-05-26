# Benchmark Suite Phase

This phase is required before Agent Pilot Autobench can honestly call a model
or setting production-ready.

Speed, TTFT, context, and warmup behavior are necessary system metrics. They
are not enough. A local agent model also needs repeatable usefulness evidence
from general-purpose and agentic benchmarks.

## Karpathy Autoresearch Contract

The project should copy the operating contract from Andrej Karpathy's
`karpathy/autoresearch`, adapted to local GGUF serving:

1. Use a fixed experiment budget.
2. Keep the evaluation harness stable.
3. Use one comparable score for keep/discard decisions.
4. Append every attempt to a TSV ledger.
5. Keep improvements, discard regressions, and mark crashes honestly.
6. Keep raw logs out of the main context; write receipts and inspect summaries.

Karpathy's original loop optimizes `val_bpb` for a small training setup. This
repo's equivalent target must become an `agent_bench_score` built from stable
benchmark phases, not from vibes or ad hoc prompts.

## Required Phases

### Phase 0: System Viability

Already partly implemented.

- Load at explicit 4K context first.
- Climb 8K, 16K, 32K, and higher contexts.
- Record server-ready time, cold TTFT, warm TTFT, TPS, prompt-cache behavior,
  and per-question serving rows.
- Write history to `runs\autoresearch-results.tsv` and
  `runs\serving-metrics.tsv`.
- Write every attempted setting to `runs\autoresearch-attempts.tsv` with a
  `keep`, `discard`, or `crash` decision.

### Phase 1: General-Purpose Benchmarks

This phase must use an existing harness where possible.

Executable wrapper now exists:

- `agent-autobench benchmark-suite-template`
- `agent-autobench benchmark-suite --plan benchmark-suite.plan.json`
- installed optional dependency: `lm-eval`

Primary harness:

- EleutherAI `lm-evaluation-harness`

Candidate tasks:

- MMLU or MMLU-Pro style knowledge and reasoning
- GSM8K or equivalent math reasoning
- HellaSwag / ARC style common-sense and science reasoning
- TruthfulQA style hallucination resistance
- HumanEval / MBPP / BigCodeBench style coding, when the local serving adapter
  supports stable code-generation scoring

Output requirement:

- `runs\benchmark-suite.tsv`
- one row per benchmark task group
- include model, context, settings, benchmark id, score, pass/fail, runtime,
  receipt path, and failure class

### Phase 2: Agentic Benchmarks

This phase must focus on local-agent deployment usefulness, not only raw model
knowledge.

Executable wrapper now exists:

- command-based `agentic` plan tasks
- installed optional dependency: `inspect-ai`
- output ledger: `runs\agentic-suite.tsv`

Candidate suites:

- Berkeley Function Calling Leaderboard (BFCL) for tool/function calling
- SWE-bench or SWE-bench Lite/Verified style coding-agent tasks
- tau-bench, tau2-bench, or tau3-bench style user-simulator task completion
- repo-local deterministic tasks for command safety, receipt inspection,
  JSON repair, tool selection, and multi-step planning

Output requirement:

- `runs\agentic-suite.tsv`
- one row per task
- include model, context, settings, task id, pass/fail, score, latency, tool
  validity, receipt path, and failure class

### Phase 3: Autoresearch Keep/Discard Loop

Only after Phase 1 and Phase 2 exist should the optimizer call a setting a real
candidate.

The keep/discard score should combine:

- system viability: TTFT, TPS, context, stability
- general benchmark score
- agentic benchmark score
- failure penalties
- complexity/safety penalties where applicable

The current executable benchmark-suite runner writes the combined scalar to:

```text
runs\agent-bench-score.tsv
```

The project should use git history the same way Karpathy's loop does:

- experiment branch per campaign
- commit each candidate settings/harness change before running
- append result to TSV
- keep the commit only if the comparable score improves
- revert or discard losing changes

## Non-Negotiable Labeling

Until this benchmark suite phase exists:

- A run may be `slow`.
- A run may be `speed_only`.
- A run may be `serving_measured`.
- A run may be `context_unproven`.
- A run may be `workflow_unproven`.
- A run may be `workflow_weak`.
- A run may be `workflow_smoke`.
- A run must not be called production-ready.

Production-ready requires system metrics plus general-purpose benchmark evidence
plus agentic benchmark evidence.
