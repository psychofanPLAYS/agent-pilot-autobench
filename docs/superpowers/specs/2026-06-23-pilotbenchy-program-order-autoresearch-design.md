# pilotBENCHY Program Order and Autoresearch Contract

Status: coordination draft for Codex and Claude Code.

This document is intentionally docs-only. Claude Code currently has the write
lane for `programs.py`, `cli.py`, `flag_ladder.py`, and related tests. Use this
as the concrete target before editing scheduler code.

## Source Baseline

Karpathy's `autoresearch` is not mainly a bag of flags. It is a disciplined
experiment loop:

- One fixed time budget per experiment.
- One stable harness.
- One main editable surface.
- One comparable scalar metric.
- A TSV ledger with `keep`, `discard`, and `crash`.
- Raw output redirected to logs, then summarized from receipts.
- Losing changes are discarded; improving changes advance the branch.

Sources checked on 2026-06-23:

- https://github.com/karpathy/autoresearch
- https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md
- https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md

The important transfer to pilotBENCHY is the contract, not the exact training
code. pilotBENCHY should run one benchmark program at a time with one evidence
contract, then compose the results in a report.

## Current Mismatch From The 9B Run

Evidence receipt:
`_runs\20260623-161636-Qwen3-5-9B-Claude-4-6-AWARE_UNCENSORED-Q8_0`

What went wrong:

1. The flag ladder ran serious questions at `4096` context.
2. Fit discovery, speed testing, intelligence testing, and flag ablation were
   blended into one noisy schedule.
3. Budget exhaustion after completed questions was surfaced as crash/failure.
4. Top-level reports hid useful partial rows that existed under
   `simplebench-*`.
5. The output did not answer the buyer question: "What settings should I use,
   and why is this better than manual trial and error?"

Useful evidence still existed:

- Standard flags were present:
  `--flash-attn on --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --jinja --gpu-layers 99`.
- Root `results.md` showed easy-gotcha `5/5`, simple-bench `1/5` with 4
  incomplete, and easy-mc `0/5` with 2 incomplete.
- Per-profile summaries showed roughly 80-86 decode tok/s and partial accuracy.

Conclusion: the implementation produced data, but the program order was wrong.

## Standard Baseline

Every program starts from the same locked baseline unless the user explicitly
asks for an ablation:

```text
--flash-attn on
--kv-unified
--cache-type-k q8_0
--cache-type-v q8_0
--jinja
--gpu-layers 99
```

If the user supplies a custom template, such as `--chat-template-file`, that
template becomes part of the locked baseline for every program. Template testing
is a separate future mode.

The baseline key for lifetime stats should include:

- model slug
- quant/file path identity
- llama.cpp version/build
- standard flag signature
- template signature
- GPU name/VRAM

## Concrete Program Order

### 0. Preflight

Purpose: prove the run is reproducible before spending GPU time.

Actions:

1. Resolve model slug and model path.
2. Capture machine snapshot: OS, CPU, RAM, GPU, VRAM, driver, llama.cpp build,
   command path, current Git commit/dirty state.
3. Resolve standard baseline flags and template signature.
4. Check prior lifetime stats by model slug and flag/template signature.
5. Write a run manifest before launching llama-server.

Output:

- `run-manifest.json`
- `machine-snapshot.json`
- first event in `events.jsonl`

Keep/discard metric: none. This is setup evidence.

### 1. Fit Program

Purpose: find the largest usable context before intelligence tests.

Important distinction:

- `16k` is the absolute serious floor.
- Fit search starts at `32k` because lower contexts are not useful enough for
  the target workflow.

Algorithm:

1. Start at `32k`.
2. Increase by `32k`: `32k -> 64k -> 96k -> 128k -> ...`.
3. Each load must run a gradable probe, not just a health ping.
4. On OOM or fatal allocation failure, try `failed_context - 16k`.
5. If that also fails, return to last working context and refine upward by `8k`.
6. Stop at the highest working context that completes the probe.
7. Persist this max context to lifetime stats under model slug plus flag/template
   signature.

Fit probe shape:

- It should take meaningful generation time, aiming for about 15 seconds on a
  fast local model, without being absurdly expensive on slow models.
- The task must be gradable by exact constraints. Example: require exactly three
  numbered observations, exactly two risks, and a final sentinel line.
- If a model is very slow, cap by dynamic time budget and mark `slow_partial`
  rather than hanging the campaign.

Primary metrics:

- max working context
- server-ready time
- TTFT
- generated tokens
- decode TPS
- prompt TPS when available
- peak VRAM/RAM
- OOM/failure class

Keep/discard metric:

- This is not a quality champion program.
- Keep the largest context that serves and completes the fit probe.

### 2. Speed Program

Purpose: measure throughput separately from intelligence.

Context:

- Minimum `16k`.
- Also run at one practical context derived from fit, such as min(max_fit,
  `64k`) or a user-selected deployment context.

Prompts:

- Use repeatable generation prompts, not SimpleBench.
- Example: generate a 500-word poem or structured report with an exact final
  sentinel.
- Use the same prompt for every comparable speed profile.

Primary metrics:

- TTFT
- prompt eval TPS
- decode TPS
- generated tokens
- wall time
- stop reason
- output completeness
- CPU/RAM/GPU/VRAM samples
- llama.cpp `/metrics` snapshot when available

Keep/discard metric:

- For pure speed: valid completion plus median decode TPS, with TTFT and prompt
  TPS as secondary columns.
- Do not mix SimpleBench accuracy into this score.

### 3. Intelligence Program

Purpose: test model reasoning/answer quality at the context size where David
actually wants to use the model.

Context:

- Default `64k`.
- One question per fresh server/session/window.
- Use fit result to confirm that `64k` is possible; if not, mark the model as
  `intelligence_context_unavailable`.

Question packs:

- SimpleBench at `64k`.
- Easy gotcha pack.
- Easy MC pack.
- User-added packs from the repo data directory.

Prompt rule:

- The system prompt travels with the question pack.
- Let models think; do not truncate reasoning with tiny generation limits.
- Score only the final answer, but preserve reasoning transcripts.

Primary metrics:

- accuracy
- incomplete count
- wrong count
- per-question TTFT/TPS
- generated tokens
- timeout/budget class
- exact prompt/system/template signature

Keep/discard metric:

- Accuracy first.
- Speed only breaks ties after accuracy.
- Incomplete is not crash; it is an outcome.

### 4. Flag Ablation Program

Purpose: test whether a single optional change improves the already locked
baseline.

Rules:

- Standard flags stay on.
- Change exactly one variable per comparison.
- Template stays locked unless this is explicitly a template-ablation program.
- Run speed probe first. Run a small intelligence sample only for changes that
  pass speed/stability thresholds.

Examples:

- baseline vs cache-reuse
- baseline vs RAM cache
- baseline vs thread count
- baseline vs MTP draft setting, if supported
- baseline vs parallel slots, treated as concurrency capacity, not single-stream
  speed

Keep/discard metric:

- For speed-only ablations: valid completion plus speed/TTFT improvement.
- For quality-sensitive ablations: require no accuracy regression on the
  selected intelligence sample.

### 5. Long-Context Dropoff Program

Purpose: measure whether intelligence or speed falls apart as context grows.

Context tiers:

- Start at `16k` only as the floor/reference tier.
- Use `64k` as the main intelligence tier.
- Continue through fit-proven tiers, for example `96k`, `128k`, `160k`, etc.

Workload:

- Same prompt pack and same questions across tiers.
- One question per fresh window for scored reasoning.
- A separate repeatable speed prompt for throughput.

Keep/discard metric:

- This is a curve, not a champion selector.
- Report accuracy retention, TPS retention, TTFT growth, and OOM boundary.

### 6. Report Program

Purpose: answer the buyer question.

The report must say:

1. What max context worked?
2. What context should I use today?
3. What speed should I expect?
4. How smart was the model at 64k?
5. Which flag changes helped, hurt, or were inconclusive?
6. Was the run complete, partial, crash, or not comparable?
7. What should the next run test?

Partial reporting rule:

- If a budget ends after completed questions, report `partial`.
- Surface `completed_questions`, `attempted_questions`, accuracy over completed
  scored questions, median TPS, TTFT, and reason `budget_exhausted`.
- Do not promote partial rows to champion unless the program defines an evidence
  threshold and the threshold is met.

Buyer verdict rule:

- "Worth paying for" requires that the report is easier and more informative
  than manual trial and error.
- The 2026-06-23 9B run fails this bar because the user had to inspect subfolders
  to recover the useful data.

## Implementation Priority

Do not implement all programs at once. The smallest useful order is:

1. Fix partial-result reporting so useful question rows are no longer called
   crash.
2. Make fit program real: 32k start, 32k climb, 16k/8k refinement, gradable
   probe, lifetime max-context write.
3. Split speed probe away from SimpleBench.
4. Run SimpleBench at 64k, one question per fresh window.
5. Update the TUI to show program choice and lifetime stats.

## Acceptance Criteria For The Next Real 9B Run

The next run is not accepted unless:

1. It starts with preflight and machine snapshot.
2. Fit search starts at `32k`, not `4k` or `16k`.
3. Each fit tier runs a gradable probe.
4. Speed prompt is not SimpleBench.
5. SimpleBench intelligence runs at `64k`, one question per fresh window.
6. Standard flags and template signature are printed in the report.
7. Partial data is visible in the top-level report.
8. The final report can answer whether the app beat manual flag trial and error.

