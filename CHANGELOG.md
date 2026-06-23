# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
semantic versioning for published versions.

## [Unreleased]

### Added

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
