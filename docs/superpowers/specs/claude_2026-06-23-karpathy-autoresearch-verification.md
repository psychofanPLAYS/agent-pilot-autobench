# claude_2026-06-23 — Karpathy autoresearch fidelity verification

> **Status:** Research-aid handoff from Claude (Opus 4.8). Codex owns all code.
> Nothing here has been implemented by Claude. This is a read of the existing
> code against the upstream Karpathy contract, plus one concrete bug + fix.

## What I checked against

Upstream pinned in `docs/AUTORESEARCH-PROGRAM.md`:
`karpathy/autoresearch` @ commit `228791fb499afffb54b46200aca536f79142f117`
(`program.md` + repo `README.md`).

Upstream contract, as written:

- **Fixed time budget per experiment** (Karpathy: 5 min wall-clock *training*).
  The point is that an equal budget makes experiments directly comparable.
- **One ground-truth metric per attempt** — Karpathy `val_bpb` (lower better),
  vocab-size-independent so architectural changes compare fairly.
- **Durable, append-only ledger** — `results.tsv`:
  `commit | val_bpb | memory_gb | status | description`.
- **Keep / discard / crash** are the three statuses.
- **Crashes are handled separately from regressions**: inspect stack trace, fix
  if fixable, otherwise log as `crash` — explicitly NOT the same as a worse score.
- Never pause; iterate until stopped.

## Verdict: contract faithful, mechanism legitimately different

The repo **adapts the contract correctly**:

| Karpathy | This repo |
|---|---|
| Agent edits `train.py` | Loop searches llama.cpp **flag profiles** (`AutoresearchSettings`) + optional Optuna learner |
| Minimize `val_bpb` | Maximize `simple_bench_score` / `agent_bench_score` |
| Fixed 5-min training budget | `budget_seconds` + `round_seconds` cap (`autoresearch.py:309-313`) |
| `results.tsv` | `_runs/autoresearch-results.tsv` + `-attempts.tsv` + `serving-metrics.tsv` |
| keep/discard/crash | `_decision_for_attempt` → keep/discard/crash (`autoresearch.py:1213`) |

The substitution (flag search instead of code editing) is a sound, honest
adaptation. `docs/AUTORESEARCH-PROGRAM.md` cites the upstream accurately and is
appropriately careful ("only the full Karpathy contract when the benchmark-suite
phase runs; otherwise it is a system-viability loop").

## Bug #1 (root cause of the 9B run flaw): `not ok → crash`

`src/gguf_limit_bench/autoresearch.py:1213`

```python
def _decision_for_attempt(result, previous_best):
    if not result.ok:
        return "crash"          # collapses three different outcomes
    if previous_best is None:
        return "keep"
    return "keep" if result.score() > previous_best.score() else "discard"
```

This violates the upstream invariant **"mark crashes separately from
regressions."** It collapses THREE distinct outcomes into `crash`:

1. **True crash** — segfault / `gpu_oom` / `model_load` / server died.
2. **Regression** — ran fine, lower score (correctly `discard`, but only
   reachable when `ok`).
3. **Partial** — budget expired after N graded questions already completed with
   real data. `simple_bench_runner._failed_result` (`simple_bench_runner.py:233`)
   zeroes `simple_bench_accuracy`, `tps`, and returns `ok=False` → this path then
   labels it `crash`.

Outcome #3 is exactly what made the 20-min 9B run file useful partial data as
"crashes / budget exhausted" and hide it from the top-level report.

### Evidence trail in the code

- `simple_bench_runner.py:120-146` — questions loop; on `TimeoutError` from
  `_remaining_timeout_seconds` (`:411-415`, message `"SimpleBench attempt budget
  exhausted"`), control jumps to the `except TimeoutError` handler.
- `:171-186` — that handler calls `_failed_result(...)`.
- `:233-274` `_failed_result` — discards completed `question_results`
  (accuracy/score/tps set to `None`/`0.0`, `ok=False`). The completed transcript
  *was* written to `transcript.jsonl`/`summary.json` (`:144`, `:212-222`), but the
  returned `AttemptResult` no longer carries it.
- `autoresearch.py:1217` — `not ok` → `crash` in the attempts ledger.
- Note: `evidence.py:30-33` `evidence_status` *would* return `PARTIAL` for a
  non-crash failure — so the status vocabulary already supports `partial`; the
  decision ledger and the runner are what throw the data away.

## Recommended fix (for Codex — this is item #4 in the task brief)

Add a real `partial` outcome end-to-end. Restores Karpathy fidelity.

1. **Runner** (`simple_bench_runner.py`): on `TimeoutError` when
   `len(question_results) >= 1` and ≥1 was scored, build a **partial**
   `AttemptResult` from `combine_simple_bench_results(question_results)`:
   - keep `simple_bench_accuracy` = accuracy over **completed** questions,
   - `generation_tokens_per_second` / serving TPS = median over completed,
   - `failure = "budget_exhausted"` (NOT classified to crash),
   - `ok = False` (so it never gets crowned best), but data preserved.
   - Write `summary.json` with `status: "partial"`, `completed_questions`,
     `attempted_questions`, `accuracy`, `median_tps`, `reason: "budget_exhausted"`.
   - Only fall through to `_failed_result` (true crash) when **zero** questions
     completed.

2. **Decision** (`autoresearch.py:_decision_for_attempt`): distinguish partial
   from crash. Suggested:
   ```python
   if not result.ok:
       if result.failure == "budget_exhausted" and result.simple_bench_accuracy is not None:
           return "partial"
       return "crash"
   ```
   Keep `crash` strictly for `model_load` / `gpu_oom` / `memory_allocation` /
   `crash` / `no_successful_attempt`.

3. **Pure helper to unit-test** (no server needed) — put in `simple_bench.py`:
   `summarize_simplebench_outcome(attempted, question_results, budget_exhausted,
   server_crashed=False) -> dict` returning
   `{status, completed_questions, attempted_questions, accuracy, median_tps, reason}`.
   - `completed >= attempted and not budget_exhausted` → `complete`
   - `completed >= 1` (budget_exhausted) → `partial`
   - `completed == 0` → `crash`

4. **Report** (`reports.py` / `run_report.py`): surface partial rows with their
   completed-question accuracy + median TPS instead of hiding them. `reports.py:83`
   already special-cases `status == "partial"` — extend it to print the partial
   metrics block.

### Suggested tests (conflict-free; new test file)

- partial SimpleBench results summarize as `partial`, not `crash`
- `summarize_simplebench_outcome` returns `crash` only when 0 completed
- a budget-exhausted attempt with ≥1 scored question yields
  `simple_bench_accuracy is not None` and `decision == "partial"`

## Bug #2 (conceptual, lower urgency): one shared budget across the whole ladder

Karpathy gives **each** experiment an **equal** fixed budget. The repo shares
**one** global `budget_seconds` across the entire flag ladder
(`autoresearch.py:365` while-loop; `:397-403` carves remaining time per attempt,
capped by `round_seconds` if set). Late profiles therefore starve and "exhaust
budget" — which, combined with Bug #1, is why "most attempts" looked like crashes.

The product fix is the **separated-program architecture** (see
`claude_2026-06-23-order-of-operations-and-programs.md`): each program has its own
explicit evidence contract and context floor, so a profile is never silently
starved by an unrelated profile earlier in the queue.

## Summary for Codex

- Keep the current adaptation; it is a faithful Karpathy contract.
- Land item #4 (`partial`) — it is not cosmetic, it restores "crashes separate
  from regressions."
- The separated programs fix the shared-budget starvation conceptually.
