# Agent Pilot Autobench Command Board

This is the compact command map. The public command is `agent-autobench`; `apb` is the short alias; `pilotbench` remains as an older compatibility command.

## Start

For Windows users:

```text
Double-click FIRST_RUN.bat once
```

For terminal users:

```powershell
uv run --extra dev --extra bench agent-autobench --first-run
```

After setup adds `_bin` to PATH:

```powershell
apb --start
agent-autobench --start
```

Plain `apb` opens the local pilotBENCHY web UI at `http://127.0.0.1:36939/`. The
browser is the primary workflow for model selection, Flight Plan selection, live
WebSocket run progress, telemetry, and receipt links. Benchmark-suite `.plan.json`
files remain available as advanced/source-controlled run artifacts under the
Flight Plan layer. `apb tui` remains available as a fallback terminal TUI.

## Checks

```powershell
agent-autobench doctor
agent-autobench doctor --strict
agent-autobench doctor --json-out
```

## Model Discovery

```powershell
agent-autobench survey
agent-autobench survey --qwen-only
agent-autobench survey --qwen-35b-only
agent-autobench survey --qwen-35b-only --mtp-only
```

## Model Web Evidence Cache

```powershell
agent-autobench models scan --model-root "_models"
agent-autobench models enrich --model-root "_models" --cache-root "_db\catalog"
agent-autobench models recommendations "Qwen3-4B-Q4_K_M.gguf"
agent-autobench models export "_db\github-sync"
```

`scan` is local-only. `enrich` learns candidate Hugging Face repos from folder
names, GGUF filenames, LM Studio-style owner/repo paths, and readable GGUF
metadata, then searches/ranks HF matches before fetching the selected model
card. It pins model-card evidence by HF revision, caches README/config files on
disk, and extracts useful llama.cpp settings when the model card or Hub config
contains them. The GitHub-sync seed is `_db\catalog\recommendations.json`; the
matching audit trail is `_db\catalog\hf-match-decisions.json`.

Default paths are repo-relative and can be overridden in `_CONFIG.toml`:

- `_models`
- `_llama`
- `_runs`
- `_db`
- `_bin`

## Results

```powershell
agent-autobench results
```

Important files:

- `_runs\leaderboard.md`
- `_runs\champion.json`
- `_runs\results.html`
- `_runs\autoresearch-results.tsv`
- `_runs\autoresearch-attempts.tsv`
- `_runs\serving-metrics.tsv`
- `_runs\benchmark-suite.tsv`
- `_runs\agentic-suite.tsv`
- `_runs\agent-bench-score.tsv`

## One Model

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --budget-minutes 5 --parallel-max 4
```

Learning is on by default. To avoid updating the local Optuna memory:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --no-learning
```

## Corrected Program Order

This is the target order for the next useful Agent Pilot campaign. The current
legacy commands are still listed below, but the 2026-06-23 9B run proved that a
single 4k SimpleBench flag ladder is not a useful benchmark program.

1. **Preflight**: capture model slug, machine snapshot, llama.cpp version, GPU,
   RAM/VRAM, Git state, standard flags, and template signature.
2. **Fit**: start at `32k`, climb by `32k`, and run a gradable generation probe
   at every tier. After OOM, try `failed_context - 16k`; if needed, refine from
   the last working context by `8k`.
3. **Speed**: use repeatable generation prompts, not SimpleBench. Measure TTFT,
   prompt eval TPS, decode TPS, generated tokens, wall time, GPU/RAM/VRAM, and
   llama.cpp `/metrics` when available.
4. **Intelligence**: run SimpleBench and question packs at `64k`, one question
   per fresh server/session/window, with the pack's system prompt loaded and
   reasoning left untruncated.
5. **Flag ablation**: keep the standard flags on and change one variable at a
   time. Template choice remains locked unless the program is explicitly testing
   templates.
6. **Long-context dropoff**: run matched speed/intelligence probes across
   fit-proven context tiers and report retention curves instead of picking a
   single champion.
7. **Report**: answer what context/settings to use, whether the data is complete
   or partial, what insight was gained, and whether it beat manual trial and
   error.

Standard baseline flags:

```text
--flash-attn on
--kv-unified
--cache-type-k q8_0
--cache-type-v q8_0
--jinja
--gpu-layers 99
```

If the user supplies a custom Jinja/template argument, lock it into the baseline
for the whole campaign. Template testing is a separate future program.

## Context & VRAM

Predict which context sizes fit in VRAM before running (sliding-window aware):

```powershell
apb vram-plan --model "path\to\model.gguf" --kv-bits 8
```

Find the largest context the model can actually serve. The current fit finder
starts at `32k`, climbs by `32k`, uses `q8_0` KV by default, runs a gradable
generation probe at each launched tier, recognises out-of-memory failures, then
backs off/refines with the `16k`/`8k` rule and remembers the ceiling:

```powershell
apb context-limit --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe"
```

## Speed Probe

Run the repeatable speed program at a serious context floor. If you ask for less
than `16k`, the command bumps the run to `16k` instead of wasting a model load.
It uses the 2026 standard server baseline (`flash-attn`, unified `q8_0` KV,
Jinja) and writes a receipt under `_runs`:

```powershell
apb serve-probe --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --context-size 16384
```

Lock a custom template into the speed probe:

```powershell
apb serve-probe --model "path\to\model.gguf" --llama-server-extra-arg=--chat-template-file --llama-server-extra-arg="path\to\template.jinja"
```

## SimpleBench Flag Ladder

Dry-run the useful autoresearch ladder first. This writes the exact
`llama-server` commands and starts no model process:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --flag-ladder --dry-run
```

Run the real 10-question SimpleBench flag comparison:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --llama-server "path\to\llama-server.exe" --flag-ladder --budget-minutes 20 --parallel-max 6
```

The ladder asks the same 10 SimpleBench public questions for every flag profile,
scores accuracy first and speed second, then writes the best observed profile to
`best-settings.json`. A completed ladder names a champion; a budget-limited or
attempt-limited ladder is explicitly marked partial and names only a provisional best.
Partial ladders are not eligible for promotion to the project-wide `champion.json`.

Transition note: this legacy ladder is no longer the recommended full benchmark
program. It should be split into separate fit, speed, intelligence, and ablation
programs before the next buyer-style 9B acceptance run.

Add experimental llama.cpp flags without changing code:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --flag-ladder --llama-server-extra-arg=--dry
```

Use `--simple-bench` or `--simple-bench-system-prompt` to point at a different
local benchmark file or prompt. The default answer budget is 1024 tokens so
strict `Final Answer: X` scoring does not guess from truncated reasoning.

Each supported flag is tested as an independent ablation against the same base,
and `flag-ladder-results.md` reports its TPS slowdown versus `L0-baseline`.
Models whose filename contains `MTP` automatically receive three additional
native speculative rungs using `--draft-max 8`, `16`, and `32`.

## Queue

```powershell
agent-autobench autoresearch-all --qwen-only --budget-minutes 5 --parallel-max 4
agent-autobench autoresearch-all --qwen-35b-only --total-budget-minutes 30 --budget-minutes 5 --parallel-max 4 --workflow-eval
agent-autobench autoresearch-all --qwen-35b-only --mtp-only --total-budget-minutes 20 --budget-minutes 5 --workflow-eval
```

## Serving Probe

```powershell
agent-autobench serve-probe --model "path\to\model.gguf" --context-size 4096
```

The probe records server ready time, cold TTFT, warm TTFT, warmup penalty, and serving tokens/sec. Context tiers ask the same ordered question suite each time so `_runs\serving-metrics.tsv` stays comparable.

## Flight Plans

Flight Plans are the beginner contract: choose model(s), choose the goal, then
start. The browser shows them in the Run menu, and the CLI can list the same
plans for agents, scripts, and troubleshooting:

```powershell
agent-autobench flight-plans
agent-autobench flight-plans --json-out
```

Use Flight Plans for normal runs. Use benchmark-suite `.plan.json` files only
when you need the lower-level, source-controlled task artifact behind a run.

The Flight Plan UX follows the durable patterns used by current LLM evaluation
tools:

- [Inspect AI](https://inspect.aisi.org.uk/eval-logs.html): keep eval logs as
  the primary record and make eval sets resumable.
- [promptfoo](https://www.promptfoo.dev/docs/usage/command-line/): keep a
  simple eval command plus a browsable result surface.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):
  keep task/model configuration declarative and runnable from one command.

pilotBENCHY applies those ideas locally: a Flight Plan is the user-facing preset,
`resolved-plan.json` is the replay-safe resolved configuration, `command.txt` is
the readable command copy, and `status.json` is the small dashboard/agent status
file. The web UI should preserve the two-button path: `Select all`, then
`Start librarian benchmark`.

## Benchmark Suite

Install harness extras:

```powershell
uv sync --extra dev --extra bench
```

List bundled plans:

```powershell
agent-autobench benchmark-suite-plans
```

The pilotBENCHY web UI also lists bundled benchmark-suite plans so a normal run can
start from the same source-controlled plan files without hand-typing the path.

Run a smoke plan:

```powershell
agent-autobench benchmark-suite --plan benchmarks\plans\local-openai-smoke.plan.json
```

Use the suite as the optimizer target:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --benchmark-suite-plan benchmarks\plans\local-openai-smoke.plan.json
```

Profile context falloff:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --context-ladder 4096 --context-ladder 8192 --context-ladder 16384 --context-ladder 32768
```

Profile perplexity falloff:

```powershell
agent-autobench autoresearch --model "path\to\model.gguf" --perplexity-corpus "path\to\corpus.txt" --perplexity-context 4096 --perplexity-context 8192 --perplexity-context 16384
```

## Receipts

Every run writes a folder under `_runs\<timestamp>-<model>`.

Receipt files:

- `events.jsonl`: attempt events, settings, telemetry, and failure class
- `resolved-plan.json`: replay-safe run plan with exact command `argv` arrays
- `command.txt`: human-readable command copy from `resolved-plan.json`
- `status.json`: latest run status for dashboards and agents
- `summary.md`: plain-English result
- `itemized-report.md`: attempt-by-attempt report
- `report.html`: browser report for the run
- `report.json`: machine-readable itemized report with metric coverage
- `context-profile.md`: context ladder report when `--context-ladder` is used
- `context-profile.tsv`: grep/chart-friendly context ladder rows
- `context-profile.json`: machine-readable context ladder rows
- `perplexity-profile.md`: quality falloff report when `--perplexity-corpus` is used
- `perplexity-profile.tsv`: grep/chart-friendly perplexity ladder rows
- `perplexity-profile.json`: machine-readable perplexity ladder rows
- `simplebench-<profile>\summary.json`: SimpleBench accuracy, speed, settings, and launch command for one flag profile
- `simplebench-<profile>\transcript.jsonl`: one row per SimpleBench question and model answer
- `simplebench-<profile>\warnings.log`: short warning/error-only log
- `simplebench-<profile>\server-tail.log`: bounded final server log lines
- `flag-ladder-plan.json`: dry-run command plan when `--flag-ladder --dry-run` is used
- `flag-ladder-results.md`: completion status, champion or provisional best, and the attempted profile table
- `flag-ladder-results.tsv`: chart-friendly profile comparison
- `flag-ladder-results.json`: full settings, commands, receipts, and comparison data
- `best-settings.json`: best settings and score
- `learning.json`: best Optuna settings when learning is enabled
- `recovery.json`: latest recovery status

Export the resolved plan from a receipt:

```powershell
agent-autobench export-plan --run "_runs\20260629-example" --json-out
agent-autobench export-plan --run "_runs\20260629-example" --output "_runs\replay-plan.json"
```

Folder-level reports:

- `_runs\leaderboard.md`: run-level ranking
- `_runs\model-comparison.md`: best-known result grouped by model
- `_runs\results.html`: browser-friendly results dashboard

Open the browser report:

```powershell
agent-autobench results --open-browser
```

Serve the report folder on localhost:

```powershell
agent-autobench results --serve
```

Receipts should stay small, deterministic, and inspectable. They should record the settings, score, failure class, and short command evidence needed to understand a result later.
