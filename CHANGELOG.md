# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
semantic versioning for published versions.

## [Unreleased]

### Added

- **Phase B (foundation): VRAM-headroom guard for the context ladder.** A new
  dependency-free GGUF header reader (`gguf_metadata.py`) extracts the real
  architecture fields (block count, GQA KV-head count, explicit key/value lengths —
  e.g. Gemma's 512, not embedding/heads) needed to size a KV cache, and `vram.py`
  estimates per-tier VRAM and plans which context sizes fit. New `apb vram-plan
  --model X [--kv-bits 8|16]` prints, for the detected GPU, which context tiers
  (16k→256k) are predicted to fit — so the ladder can skip tiers that would OOM-crash
  llama-server. The estimate is a conservative dense upper bound (sliding-window
  attention not yet modelled, so SWA models use less). Verified on a real RTX 4090:
  Gemma-4-E2B reaches 256k at q8_0 KV vs 224k at f16.
- **First-run path auto-detection.** On a fresh machine `apb` setup now scans common
  locations (PATH, LM Studio dirs, `?:\AI\models`, `?:\AI\llama.cpp`, ...) for GGUF model
  folders and llama.cpp binaries it cannot already resolve, and saves what it finds to
  your user environment — so a brand-new user gets a working app instead of "Something is
  missing", without hand-setting `PILOTBENCH_*` env vars. Detection only fills in paths
  that are not already configured, so an existing setup is left untouched. New
  `autodetect.py` (`find_model_roots`, `find_llama_binaries`) and
  `installer.persist_user_env`.
- **One-command install, then just `apb`.** `INSTALL.bat` / `install.ps1` is a single
  command that auto-installs `uv` if missing, builds the local environment, pulls
  dependencies, and puts `apb` on PATH — no flags, nothing else to run.
- **Plain `apb` opens the app.** Bare `apb` (no subcommand, no flags) now launches the
  model picker instead of printing a help wall. The first run on a machine detects it is
  not set up yet and runs setup automatically before opening; every run after that goes
  straight to the app. A `.apb-setup-complete` marker records the installed state, and a
  wiped `.venv` re-triggers the one-time setup. Power-user subcommands are unchanged
  (`apb --help`).

### Changed

- `FIRST_RUN.bat` now delegates to the one-command installer; `START.bat` just launches
  `apb`. README "Start" is now two steps: install once, then type `apb`.

- **Phase A — multi-pack LLM evaluation with visible results.** Question packs beyond
  SimpleBench via a `QuestionPack` interface and registry: `simple-bench` (10, hard),
  `easy-gotcha` (~24 verified gotchas, exact-answer with accept-variants), and `easy-mc`
  (~26 frozen ARC-Easy/OpenBookQA/CommonsenseQA questions with authoritative answer keys;
  see `data/packs/LICENSES.md`). Two answer types (`multiple_choice`, `exact` with
  number-word normalization and phrase/containment scoring).
- "Let the model think": a correct/wrong/**incomplete** outcome taxonomy with one cheap
  forced-final follow-up, so a reasoner that runs out of token budget mid-thought is
  scored `incomplete` rather than silently `wrong`.
- Per-run `results.md` / `results.json` showing the model's actual answer for every
  question across packs, plus a 5-per-pack `sequential`/seeded-`random` selection setting
  (`--sample-size`, `--selection`, `PILOTBENCH_QUESTION_*`) and per-model lifetime stats
  in the experiment DB (a sequential cursor accumulates coverage across runs).
- GPU-aware recommended-always-on flags (`gpu_profiles.py`): RTX 4090 → `--cache-type-k/v
  q8_0` (Ada fp8 KV), flash-attn, 4 parallel slots, surfaced in results.
- TUI per-pack scoreboard + lifetime line; non-blank `--help` for the `survey`, `quick`,
  `autoresearch`, `autoresearch-all`, and `tui` commands.
- Non-generative GGUFs (embedding/reranker/query-expansion/imatrix/mmproj) are excluded
  from model discovery so they can never be benchmarked as chat models.
- Independent llama-server flag ablations with SimpleBench accuracy, TTFT, throughput,
  slowdown, warning, command, and receipt evidence.
- MTP draft-profile experiments for models explicitly identified as MTP-capable.
- Package-contained, MIT-licensed SimpleBench public data with upstream provenance.
- Architecture/code map, contributor guide, security policy, and release-quality CI job.
- Goal-shaped cockpit run modes (Quick check, Find best settings, How flags affect
  speed, Context limits, Deep / overnight, Custom), cycled with the `M` key, plus a
  champion verdict line shown when a run finishes.
- Prefill/prompt-processing throughput, generation-speed variance (stddev), and TTFT
  p90/p99 tail latency on the question path, with a server warmup before the scored batch.
- `Lmin-stripped` bare-minimum ladder baseline (no flash-attn, no continuous batching)
  to test whether removing standard flags helps or hurts; `--parallel` is now tested
  last as a concurrency-capability axis (`Lpar-2`, `Lpar-3`).
- `forced_server_args` config option (and `PILOTBENCH_FORCED_SERVER_ARGS`) to force raw
  llama-server flags on for every benchmark profile.
- Overnight convergence: after the ordered flag ladder, a persistent per-model Optuna
  study keeps searching (accuracy-first) until the time budget and warm-starts the next
  session.

### Changed

- Autoresearch inputs now reject invalid datasets, impossible numeric values, and extra
  arguments that could override benchmark-managed model or network bindings.
- One SimpleBench attempt now shares a bounded deadline across its question batch.
- Package metadata and CLI help now use a consistent public product description.
- The codebase is checked by MyPy in addition to Ruff, pytest, compilation, and builds.
- Asking the SimpleBench questions through `llama-server` is now the default; the fast
  synthetic `llama-bench` scout is opt-in via `--speed-scout`.
- Autoresearch follows Karpathy's fixed 5-minute round; each attempt is capped at the
  round, and cockpit-mode budgets are expressed as round multiples.
- Speed flags are measured at single stream; `--parallel` is evaluated last as
  concurrency capability rather than mixed into the single-stream tok/s ranking.
- Default SimpleBench generation budget raised to 4096 tokens, the answer extractor
  accepts more real-world answer formats, and the system prompt demands the final-answer
  line — so reasoning models actually reach and record an answer.
- `--dry-run` now plans the flag ladder in the benchmark default and is only rejected
  when combined with `--speed-scout`.
- Learning studies are tagged by objective (accuracy vs throughput) and the score
  version is bumped to `score-v4-simplebench`.

### Fixed

- The cockpit and default `autoresearch` run now actually ask the model the benchmark
  questions; previously the TUI and default path ran the synthetic `llama-bench` probe
  and asked nothing.
- Prefill/prompt throughput is captured from `llama-server` timings instead of being
  hardcoded to `0.0`.
- Tests are isolated from a developer's machine `PILOTBENCH_*` environment variables so a
  local setup (real model/llama paths) can no longer break the suite.
- Revised final answers are scored using the latest explicit answer marker.
- Accuracy now always outranks throughput; speed is only a bounded tie-breaker.
- Budget-limited ladders are marked partial, suppress a champion, and bound each
  server attempt to the remaining run budget. Partial evidence is excluded from
  project-wide champion promotion.
- Thread-sweep profiles inherit the documented q8 KV settings.
- Exact server launch arguments are stored as non-executable JSON instead of a `.cmd` file.
- The optional benchmark dependency lock now uses patched `aiohttp 3.14.1`.
- Built wheels include the default SimpleBench dataset and system prompt.
- Duplicate helper definitions and new runner typing regressions were removed.
