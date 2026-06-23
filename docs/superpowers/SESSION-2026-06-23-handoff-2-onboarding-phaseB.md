# Session Handoff — 2026-06-23 (onboarding + Phase B + eval depth)

Audience: the next engineer/agent picking this up cold. This **supersedes**
`SESSION-2026-06-23-handoff.md` (which covered the earlier Phase-1 cockpit work).
Read that first only for the deep mental model of the autoresearch loop.

Branch: `codex/pilotbenchy-first-run-reports` · Draft PR: #9 ·
Repo: https://github.com/psychofanPLAYS/agent-pilot-autobench
Gate at handoff: **412 passed / 1 skipped**, ruff + mypy clean. All commits local
on the branch — **not yet pushed** (push when the owner asks).

## TL;DR of this session's growth

Three themes landed, each verified live on the owner's real RTX 4090:

1. **One-command install + zero-flag launch.** `INSTALL.bat`/`install.ps1` is a
   single command (auto-installs `uv`, builds env, PATH). Plain `apb` now *opens
   the app* instead of a help wall; first run self-bootstraps; first run also
   **auto-detects** model folders + llama.cpp binaries and saves them to user
   env, so a fresh machine works without hand-setting `PILOTBENCH_*`.
2. **Phase B — context scaling done right.** Ascending, OOM-resilient
   context-limit search (`apb context-limit`): now starts at 32k, climbs by 32k, q8_0 KV
   default, recognises a CUDA OOM and backs off instead of crashing, and a
   sliding-window-aware VRAM estimate (Gemma 3/4) decides what to even attempt.
   The discovered ceiling is remembered per model across runs.
3. **Eval depth.** Models are no longer truncated mid-thought (`n_predict:-1`
   default); question packs are self-contained (system prompt inline) and
   user-extendable from one folder.

## What shipped (commits, newest last)

- `f280999` one-command install; bare `apb` launches; first-run self-setup
  (`installer.is_setup_complete`/`mark_setup_complete`, `.apb-setup-complete`).
- `d3190e7` first-run path **auto-detection** (`autodetect.py`:
  `find_model_roots`/`find_llama_binaries`) + `installer.persist_user_env`;
  `cli._autoconfigure_paths` (gated on `add_to_path`, fills only missing paths).
- `e28d918` VRAM guard foundation: `gguf_metadata.py` (dependency-free GGUF
  header reader) + `vram.py` (`detect_vram_mb`, `kv_cache_bytes`,
  `plan_context_fit`) + `apb vram-plan`.
- `18ea8ab` ascending OOM-resilient search (`context_limit.py`, `oom.py`,
  `apb context-limit`) + **Gemma SWA VRAM fix** (read per-layer
  `sliding_window_pattern`; only global layers scale with full context →
  256k q8 KV = 1.8 GB not 8.8 GB).
- `ad9f000` remember each model's context ceiling (`state_db.model_context_limit`,
  `record_context_limit`/`get_context_limit`; `context-limit` recalls + persists).
- `b15c623` let models think (`n_predict:-1` default in `pack_runner` &
  `simple_bench_runner`); self-contained packs (inline `system_prompt`);
  `data/packs/README.md`.

## New commands (also add to docs/COMMAND-BOARD.md)

- `apb` — bare, opens the cockpit (first run sets itself up).
- `apb vram-plan --model X [--kv-bits 8|16]` — predict which context tiers fit.
- `apb context-limit --model X [--kv-cache-type q8_0] [--max-context N]` — climb
  from 32k by 32k, refine after OOM, find the real max served context, remember it.

## Verified live this session (evidence, not claims)

- Bare `apb` launches the TUI (screen-takeover codes observed), not the help wall.
- Fresh-machine sim (wiped `PILOTBENCH_*`): auto-detect found `G:\AI\models` +
  llama-server on PATH → doctor READY.
- `vram-plan` + `context-limit` on Gemma-4-E2B: earlier real ascending launches
  16k→256k all served; max 256k; SWA math matches the owner's real 256k usage.
  Current worktree updates the fit contract to start at 32k and climb by 32k.
- Learn loop: run 1 stored 32k → run 2 printed "Previously found max context: 32k".
- Let-them-think: Gemma reasoned 424 tokens through the STRAWBERRY gotcha (not
  truncated) and answered correctly with a `Final Answer:` line.

## How the new pieces fit (mental model additions)

- **GGUF arch** (`gguf_metadata.read_model_arch`) → `ModelArch` carries SWA
  fields. **VRAM** (`vram.py`) sizes KV: SWA models only count global layers at
  full context. Both are advisory; the real launch is ground truth.
- **Context search** (`context_limit.find_context_limit`) takes an injectable
  `attempt(ctx)->LaunchOutcome`; the CLI wires it to
  `server_probe.probe_llama_server_ttft` (q8_0 KV). OOM detection
  (`oom.is_oom_failure`) reads llama-server stderr; `server_probe` now relabels
  early-exit OOM as `oom` (was masquerading as `timeout`).
- **Packs** (`packs.py`): each `data/packs/*.json` is self-contained
  (`system_prompt` inline + questions + answers + `accept`). `simple-bench` is
  still special-cased (loads from `data/simple_bench_public.json`).

## 2026-06-23 Codex user-run verdict: useful raw data, wrong program

Codex ran the app as a user for a real 20+ minute 9B session:

```powershell
.venv\Scripts\apb.exe autoresearch `
  --model G:\AI\models\LM_Studio-gguf\DavidAU\Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED\Qwen3.5-9B-Claude-4.6-AWARE_UNCENSORED-Q8_0.gguf `
  --flag-ladder --budget-minutes 20 --parallel-max 4 `
  --sample-size 5 --selection sequential
```

Receipt:
`_runs\20260623-161636-Qwen3-5-9B-Claude-4-6-AWARE_UNCENSORED-Q8_0`

Evidence from that run:

- Standard forced flags were correctly present in the champion eval:
  `--flash-attn on --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --jinja --gpu-layers 99`.
- The flag ladder ran at `4096` context, which is not a useful starting point
  for David's target workflow. This is the core schedule bug.
- `flag-ladder-results.md` and `itemized-report.md` label the ladder attempts as
  failures/crashes (`SimpleBench attempt budget exhausted`) and show no useful
  TPS/accuracy.
- The individual `simplebench-*` summaries contradict the top-level report: they
  contain usable partial accuracy/TPS data, for example L0 baseline `2/8`
  accuracy with median decode around `85.77 tok/s`, and L2 kv-unified `2/8`
  accuracy around `80.96 tok/s`.
- Root `results.md` gives a real champion eval snapshot: simple-bench `1/5`
  with 4 incomplete, easy-gotcha `5/5`, easy-mc `0/5` with 2 incomplete.

Honest buyer assessment:

- Would Codex pay for this benchmark as-is? **No, not yet.**
- Was it easy to use? **Partly.** The TUI and standard flags are improving, and
  the app does produce artifacts, but the user must dig through subfolders to
  find the real evidence.
- Was it useful? **The raw data was useful; the product conclusion was not.**
  The run proved that the 9B model handles easy gotchas well, struggles or
  truncates on SimpleBench/MC under this schedule, and serves around 80-86
  tok/s in the tested path. The report failed to surface that.
- Is it better than manual flag trial-and-error? **Not yet.** It can become
  better because it records transcripts, flags, and speeds, but only after the
  scheduler separates fit, speed, intelligence, and flag ablation programs.

Immediate correction already started in the worktree:

- `src/gguf_limit_bench/programs.py` defines explicit programs and context
  floors: serious/speed `16k`, fit search starts at `32k`, intelligence `64k`.
- Fit search contract: climb by `32k`; after OOM, try `-16k`; if needed,
  refine upward/downward by `8k`; keep the last working size. Fit probes should
  run a gradable task that takes meaningful generation time, not waste a load on
  a tiny readiness ping.
- Flag-ladder CLI defaults now start at `16k` instead of `4096`.
- Focused verification reached `50 passed` for program/CLI/GPU/TUI tests, and
  Ruff passed on the touched slice.

Claude Code was delegated a parallel task to continue the deeper scheduler and
partial-result reporting work. Avoid overlapping broad rewrites until that work
is reconciled.

## Owner directives still OPEN (next build, roughly prioritized)

These came straight from the owner and are the active roadmap:

1. **TUI sizing / full visibility (HIGH).** The owner wants every panel fully
   visible, nothing cut off. Probed headlessly: at 80×24 nothing is region-clipped,
   but **model names are truncated** in the table (e.g. `…HauhauCS-Ag`). Needs a
   responsive layout pass (prioritise the model-name column; trim/shrink chrome at
   small heights). Verify with `BenchTui.run_test(size=...)` + `export_screenshot`.
2. **"Standard 2026 flags" as forced/always-on**, NOT brute-tested: flash-attn ON,
   **q8_0 KV**, kv-unified. Move these into the forced set
   (`benchmark.forced_server_args` / config defaults) so the programs run them by
   default and the ladder only brute-forces the *meaningful* variables.
3. **Pre-run settings + flag-selection UI.** Before a run, let the user adjust the
   standing settings and pick *which* flags to brute-test. Add **chat-template
   testing** to the lineup (test multiple templates for speed + intelligence).
4. **Single source-of-truth lifetime stats page**, keyed by model **slug**
   (later: aggregate into a shared DB across users). `state_db` already has
   `lifetime_pack_stats` + `model_context_limit` to build on.
5. **Per-window question asking.** Ask SimpleBench (and the MC/easy packs) **one
   question per context/session window**; owner expects ~64k ctx for SimpleBench.
6. **Named "programs" for everything** (review `modes.py`): (a) **long-context +
   intelligence drop-off** (accuracy vs context tier), (b) **speed + intelligence**
   combined into one "useful throughput" number (owner flagged this as the part
   he's unsure how to define — needs design).

## Gotchas (still true)

- Do NOT put machine paths in tracked `_CONFIG.toml` — breaks tests. Use
  `PILOTBENCH_*` user env (conftest isolates tests). Auto-detect persists there.
- The VRAM estimate is a guide; SWA is modelled but shared-KV layers
  (`shared_kv_layers`) are not — it stays conservative. Real launch is truth.
- `context-limit` / `vram-plan` need `--llama-server` if `PILOTBENCH_LLAMA_SERVER`
  isn't in the shell env (it's a USER env var, not always inherited).

## Key new files

`autodetect.py`, `gguf_metadata.py`, `vram.py`, `oom.py`, `context_limit.py`,
`installer.py` (marker + `persist_user_env`), `state_db.py` (`model_context_limit`),
`data/packs/README.md`, `install.ps1` / `INSTALL.bat`.

## Resume / run

```powershell
apb                                   # cockpit (first run self-sets-up)
apb vram-plan --model <gguf>          # what context fits
apb context-limit --model <gguf> --llama-server <exe>   # find + remember max ctx
uv run --extra dev python -m pytest -q ; uv run --extra dev ruff check . ; uv run --extra dev mypy src
```
