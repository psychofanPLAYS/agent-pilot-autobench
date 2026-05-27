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
benchmark phases, not from guesswork or ad hoc prompts.

## Required Phases

### Phase 0: System Viability

Already partly implemented.

- Load at explicit 4K context first.
- Climb 8K, 16K, 32K, and higher contexts.
- Record server-ready time, cold TTFT, warm TTFT, TPS, prompt-cache behavior,
  and per-question serving rows.
- Write history to `_runs\autoresearch-results.tsv` and
  `_runs\serving-metrics.tsv`.
- Write every attempted setting to `_runs\autoresearch-attempts.tsv` with a
  `keep`, `discard`, or `crash` decision.

### Phase 1: General-Purpose Benchmarks

This phase must use an existing harness where possible.

Executable wrapper now exists:

- `agent-autobench benchmark-suite-template`
- `agent-autobench benchmark-suite --plan benchmark-suite.plan.json`
- `agent-autobench autoresearch --model MODEL --benchmark-suite-plan benchmark-suite.plan.json`
- bundled plan files under `benchmarks\plans\`
- external harness command: `uvx --from lm-eval lm-eval`

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

- `_runs\benchmark-suite.tsv`
- one row per benchmark task group
- include model, context, settings, benchmark id, score, pass/fail, runtime,
  receipt path, and failure class

### Phase 2: Agentic Benchmarks

This phase must focus on local-agent deployment usefulness, not only raw model
knowledge.

Executable wrapper now exists:

- command-based `agentic` plan tasks
- repo-local Inspect task: `benchmarks\inspect_tasks\json_repair.py`
- Inspect score extractor: `python -m gguf_limit_bench.inspect_score`
- installed optional dependency: `inspect-ai`
- isolated BFCL CLI venv: `.venv-bfcl\Scripts\bfcl.exe`
- output ledger: `_runs\agentic-suite.tsv`

Candidate suites:

- Berkeley Function Calling Leaderboard (BFCL) for tool/function calling
- SWE-bench or SWE-bench Lite/Verified style coding-agent tasks
- tau-bench, tau2-bench, or tau3-bench style user-simulator task completion
- repo-local deterministic tasks for command safety, receipt inspection,
  JSON repair, tool selection, and multi-step planning

Bundled plans:

- `benchmarks\plans\local-openai-smoke.plan.json`: external `lm-eval` via
  `uvx` plus repo-local Inspect JSON repair.
- `benchmarks\plans\local-bfcl-smoke.plan.json`: external `lm-eval` via `uvx`
  plus isolated BFCL `simple_python`.
- `benchmarks\plans\external-agentic-heavy.plan.json`: explicit SWE-bench and
  tau2-bench integration plan. It should be treated as incomplete until those
  external harnesses and their Docker/API requirements are installed and a real
  receipt passes.

Output requirement:

- `_runs\agentic-suite.tsv`
- one row per task
- include model, context, settings, task id, pass/fail, score, latency, tool
  validity, receipt path, and failure class

### Phase 3: Autoresearch Keep/Discard Loop

Only autoresearch runs with `--benchmark-suite-plan` should call a setting a real
production-readiness candidate. Without that option, the loop remains a system
viability scout.

The keep/discard score is `agent_bench_score` when a benchmark-suite plan is
provided. The suite score combines:

- system viability: TTFT, TPS, context, stability
- general benchmark score
- agentic benchmark score
- failure penalties
- complexity/safety penalties where applicable

The current executable benchmark-suite runner writes the combined scalar to:

```text
_runs\agent-bench-score.tsv
```

Autoresearch also copies the suite outcome into:

```text
_runs\autoresearch-attempts.tsv
_runs\autoresearch-results.tsv
```

Those rows include `agent_bench_score`, general score, agentic score, suite
status, suite receipt, and suite failure.

The project should use git history the same way Karpathy's loop does:

- experiment branch per campaign
- commit each candidate settings/harness change before running
- append result to TSV
- keep the commit only if the comparable score improves
- revert or discard losing changes

## Non-Negotiable Labeling

Until a run uses this benchmark suite phase:

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
