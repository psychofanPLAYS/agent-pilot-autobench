# Agent Pilot Autobench Autoresearch Program

Adapted from Andrej Karpathy's `karpathy/autoresearch` `program.md` pattern.
Upstream inspected locally at:

```text
G:\AI\_codex_projects\_upstream\karpathy-autoresearch
```

Upstream commit inspected:

```text
228791fb499afffb54b46200aca536f79142f117
```

## Contract

The loop should stay simple:

1. Use a fixed time budget.
2. Keep the benchmark harness stable.
3. Measure one ground-truth score per attempt.
4. Record every attempt in machine-readable receipts and TSV ledgers.
5. Keep better evidence and discard worse evidence.
6. Mark crashes separately from regressions.
7. Never call a speed-only run agent-ready.

Karpathy's original metric is `val_bpb`, where lower is better. This repo's
metric is `agent_bench_score`, where higher is better, but the control idea is
the same: fixed budget, one comparable number, durable receipts.

This is only the full Karpathy-style contract when the run includes the
benchmark-suite phase from `docs\BENCHMARK-SUITE-PHASE.md`. Without
`--benchmark-suite-plan`, the repo is running a system-viability loop, not a
production-ready autoresearch loop.

## In-Scope Files

For GGUF agent-pilot research, the stable harness is:

- `src\gguf_limit_bench\autoresearch.py`
- `src\gguf_limit_bench\reports.py`
- `src\gguf_limit_bench\workflows.py`
- `src\gguf_limit_bench\telemetry.py`
- `src\gguf_limit_bench\learning.py`

Generated evidence belongs under:

```text
runs\
```

Local experiment data belongs under:

```text
db\
```

Do not delete benchmark runs, local databases, or model files without explicit
approval.

## Evidence Labels

The loop must separate evidence levels:

- `slow`: model loaded, but generation speed is below the useful local-agent target.
- `speed_only`: model loaded and decoded, but context/workflow usefulness is not proven.
- `serving_measured`: real local serving TTFT exists, but useful context/workflow
  evidence is not proven.
- `context_unproven`: context target is too low for agent-pilot use.
- `workflow_unproven`: speed/context evidence exists, but no workflow checks passed.
- `workflow_weak`: only smoke-level workflow checks passed.
- `workflow_smoke`: local smoke workflow checks passed, but full benchmark-suite
  evidence is still missing.
- `failed` / `partial`: run failed or did not produce complete evidence.

There is no current `agent_ready` shortcut. Production readiness is not a
status this Phase 0 loop may emit.

## Results TSV

Each run should append a tab-separated row to:

```text
runs\autoresearch-results.tsv
```

Columns:

```text
run_id	model	score	status	context	generation_tps	prompt_tps	serving_ttft_ms	serving_warm_ttft_ms	serving_warmup_penalty_ms	serving_server_ready_ms	serving_cold_start_to_first_token_ms	serving_tps	agent_bench_score	benchmark_suite_general_score	benchmark_suite_agentic_score	benchmark_suite_status	benchmark_suite_receipt	benchmark_suite_failure	receipt	description
```

This is the local GGUF equivalent of Karpathy's `results.tsv`: compact,
append-only, easy to grep, and not a replacement for full JSON receipts.

Per-question serving metrics append to:

```text
runs\serving-metrics.tsv
```

That ledger is for charts over time. It writes one row per serving question with
stable `question_index` and `question_id` values.

Per-attempt keep/discard/crash decisions append to:

```text
runs\autoresearch-attempts.tsv
```

This ledger is closer to Karpathy's `results.tsv` than the final best-settings
receipt. It writes one row for every attempted setting with `decision` equal to
`keep`, `discard`, or `crash`, plus the evidence status, comparable score,
git branch/commit metadata, settings JSON, and receipt path.

The benchmark-suite phase writes these ledgers before a run can be called
production-ready:

```text
runs\benchmark-suite.tsv
runs\agentic-suite.tsv
runs\agent-bench-score.tsv
```

`runs\benchmark-suite.tsv` is for general-purpose benchmark scores. The first
integration target should be EleutherAI `lm-evaluation-harness`, because it is
an existing broad benchmark harness instead of a homegrown replacement.

`runs\agentic-suite.tsv` is for agent deployment usefulness: tool/function
calling, coding-agent tasks, task completion, JSON repair, command safety, and
multi-step planning. Candidate external suites are BFCL, SWE-bench, and
tau2-bench/tau3-bench. Repo-local deterministic tasks should fill the gaps where
external suites are too heavy for quick local runs.

The command runner for these ledgers is:

```text
agent-autobench benchmark-suite --plan benchmark-suite.plan.json
```

Bundled plans live under:

```text
benchmarks\plans\
```

Use `benchmarks\plans\local-openai-smoke.plan.json` for the first real local
OpenAI-compatible endpoint proof. Use `benchmarks\plans\local-bfcl-smoke.plan.json`
when BFCL function-calling evidence is required. The heavy SWE-bench/tau2 plan
is intentionally explicit but not passing evidence until those external harnesses
are installed and configured.

The plan calls real harness commands, currently expected to start with
`uvx --from lm-eval lm-eval run` for general-purpose scores and `inspect eval`
for agentic scores. Missing harnesses, crashes, and scoreless runs are written
as failed evidence. Two-step harnesses such as BFCL can use sequential
`commands` so generation and evaluation stay in one scored task receipt.

To make autoresearch optimize settings by the suite result, pass the same plan
into the loop:

```text
agent-autobench autoresearch --model G:\AI\models\path\to\model.gguf --benchmark-suite-plan benchmark-suite.plan.json
```

With that option enabled, every successful speed/TTFT attempt runs the general
and agentic suite. The attempt's `agent_bench_score` becomes the comparable
score used for `keep`, `discard`, and `crash` decisions. Raw tokens/sec stays in
the receipt, but it no longer decides the winner.

## Loop

For a model or model batch:

1. Establish a 4K context baseline run.
2. Run a fixed-budget benchmark attempt.
3. Parse speed, server-ready time, cold serving TTFT, warm serving TTFT, warmup penalty, context, workflow, failure class, and telemetry evidence.
4. Write full JSON/Markdown receipts under `runs\`.
5. Append one row to `runs\autoresearch-results.tsv`.
6. Append one row per attempted setting to `runs\autoresearch-attempts.tsv`.
7. Mark each attempted setting as `keep`, `discard`, or `crash`.
8. Ask the same serving questions in the same order by context tier: 1 at 4K, 2 at 8K, 3 at 16K, and 5 at 32K and above. Append those rows to `runs\serving-metrics.tsv`.
9. Keep the best measured settings for that model only when score improves.
10. Mark results honestly if they are speed-only, workflow-weak, or workflow-smoke.

The loop is useful only when the label matches the evidence.
