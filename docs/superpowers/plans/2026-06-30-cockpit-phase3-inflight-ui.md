# Cockpit Phase 3 — in-flight UI + llama-paths hardening

Status: approved (2026-06-30). Branch `claude/focused-kepler-21f76d`.
Realizes §6 of `specs/2026-06-30-inflight-cockpit-design.md`. Aesthetic chosen by
owner: **Live mission-control** (dark telemetry flight-deck; the streaming
reasoning is the glowing centerpiece).

This phase has two independently-verifiable workstreams that share no source files,
so they can proceed in parallel:

- **W-A (agent): llama-paths hardening** — `cli.py` + `webui.py` seam + tests.
- **W-B (me): structured live_events + the cockpit UI** — `webui.py` only,
  prototyped standalone first, then ported.

The seam between them is the `run-spec.json` `paths` block (the contract below).

---

## Contract: `run-spec.json` `paths` block

The web server writes the **resolved** executable paths it was launched with into
the run spec; the engine prefers them over `_CONFIG.toml`. Shape:

```json
{
  "models": [ {"path": "...", "has_mtp": false} ],
  "mode": "librarian_bench",
  "options": { ... unchanged ... },
  "paths": {
    "llama_server": "G:\\AI\\llama.cpp\\cuda12\\llama-server.exe",
    "llama_bench": "G:\\AI\\llama.cpp\\cuda12\\llama-bench.exe",
    "llama_cli": "G:\\AI\\llama.cpp\\cuda12\\llama-cli.exe",
    "llama_perplexity": "G:\\AI\\llama.cpp\\cuda12\\llama-perplexity.exe",
    "runs_root": "G:\\AI\\_codex_projects\\_agent-pilot-autobench\\_runs"
  }
}
```

Rules:
- Every value is a string path **or `null`**. `null`/absent ⇒ engine falls back to
  its own resolved config (today's behavior). Back-compat: a spec with no `paths`
  key behaves exactly as before.
- The engine prefers a non-null spec path over config/flags for that one executable.
- Paths are written verbatim as the web server resolved them (already absolute on
  this machine).

---

## W-A — llama-paths hardening (AGENT)

**Problem.** `_start_app` (cli.py) accepts `--llama-server/--llama-bench/--llama-cli/
--llama-perplexity/--runs-root` but passes only `root`+`runs_root` to `serve_webui`,
so the detached `engine` resolves llama paths from `_CONFIG.toml` defaults
(relative `_models`/`_llama`, which don't exist here). Launching from the cockpit
therefore can't find the 4090 binaries.

**Fix (end-to-end, TDD):**

1. `webui.serve_webui` gains optional `llama_server, llama_bench, llama_cli,
   llama_perplexity: Path | None = None` params and forwards them into
   `WebUiState`.
2. `WebUiState.__init__` stores a `llama_paths: dict[str, str | None]` (str() of
   each provided Path, else None). Add a frozen helper if cleaner.
3. `_spec_payload(...)` (or `start_run`) emits the `paths` block per the contract,
   including `runs_root` (str of `self.runs_root`).
4. `_start_app` (cli.py) passes its resolved `llama_server/bench/cli/perplexity`
   into `serve_webui`.
5. The `engine` CLI command: after `read_spec`, if `spec["paths"]` has non-null
   entries, prefer them when building `config.paths` for `run_model` (per-exe
   override; null ⇒ keep config). Keep existing `--llama-*` CLI flags working
   (explicit flag still wins if you like, but spec-over-config is the key fix).

**Tests (add, keep all 680+ green, ruff clean):**
- `_spec_payload` includes a `paths` block with the provided resolved paths; nulls
  when not provided.
- `serve_webui`/`WebUiState` threads provided paths into the spec on `start_run`
  (use the injectable `spawn_engine` to capture the written spec without launching).
- engine path-preference: a spec with `paths.llama_server=X` makes `run_model`
  receive `X` even when config default differs; absent/null ⇒ config default.
- back-compat: a spec with no `paths` key runs unchanged.

**Do NOT touch:** `INDEX_HTML`, `state_payload`, `_tail_live_events`, or any
render/JS in `webui.py` — those are W-B. Commit only the files you changed
(`git add` explicit paths; do not `git add -A`). Run the suite with the repo's
`uv`. Report the commit hash.

---

## W-B — structured live_events + cockpit UI (ME)

### B1. Structured live records in the payload
`state_payload` currently appends only flattened `{at, kind, message}` (via
`_tail_live_events`). Add `run_payload["live_events"] = _tail_live_records(active)`
returning the raw `[{at, type, data}]` from `live.jsonl` (cap ~400 lines; keep the
flattened `events` for the legacy feed/back-compat). The cockpit renders purely
from `live_events` + `status` + `telemetry`.

### B2. The cockpit (INDEX_HTML rewrite), mission-control aesthetic
Reuse the existing CSS variables (`--bg/--panel/--teal/--good/--bad/...`) and the
`shell/aside/main` frame. Pre-flight selection stays; when `run.phase === "running"`
(or reattached) the main stage **transforms** into the cockpit; on
`complete/stopped/failed` it settles with receipt/report links; otherwise it shows
pre-flight.

**Top command bar:** model name · phase pill (gate→reasoning→librarian→complete) ·
`model i/N · question j/M` · elapsed + ETA (`remaining_q × rolling_latency`) ·
heartbeat dot · **STOP after current** + **ABORT**.

**Left ~60% — Reasoning Terminal:** current question card (pack badge, `Qj/M`,
prompt); stacked **THINKING** (dim mono, glow, from `question_progress.thinking`)
and **ANSWER** (bright, from `.answer`) with a blink cursor while live; score
resolves in place `grading… → 0.86 PASS / 0.00 FAIL` (+ ttft, tok/s). Completed
questions collapse into a scrollable history (id · pass/fail dot · score · latency ·
tok/s). External-harness tasks render as honest task rows (no fake token stream).

**Right ~40% — Mission Control:** live score gauge 0–100 labeled **"LIVE · partial"**
(+ correct/answered, coverage); telemetry gauges w/ sparklines (tok/s, TTFT, VRAM
used/total + peak watermark, GPU util, GPU power, CPU, RAM); phase pipeline stepper;
sequential **model queue** (done w/ score · running pulsing · queued).

Honesty: running score is the live approximation, never the final Agent Index; all
displayed numbers rounded; render is a pure function of the snapshot.

### B3. Cadence
While `phase === "running"`, tighten the client refresh (~700–900 ms) for a smooth
stream; relax to the current 2.5 s otherwise. (Server re-sends full state incl.
tailed `live_events`; `question_progress` carries cumulative thinking/answer so the
client just replaces text — no delta reassembly.)

### Build method (verify visually, no GPU)
1. Prototype the full cockpit as a standalone HTML in scratch, fed by a mock
   `state` derived from `_runs/phase2-verify/live.jsonl` (real Qwen3.5-4B thinking
   run). Iterate via Preview MCP screenshots until genuinely beautiful.
2. Port into `INDEX_HTML`; wire `live_events`; drive the real web UI with the
   **replay engine** (`pilotbench engine-replay --run-dir <d> --source
   _runs/phase2-verify/live.jsonl`) and confirm in Preview.
3. Real acceptance: launch from the cockpit on the 4090 (needs W-A landed), watch
   real thinking→answer→score stream, **stop mid-run, confirm zero orphaned
   llama-server**.

**Out of scope (Phase 4):** the parallel pressure-test concurrency lane.

---

## Integration / done criteria
- All tests green, ruff clean, on `claude/focused-kepler-21f76d`.
- W-A: cockpit launch resolves the real 4090 binaries via the spec.
- W-B: cockpit verified against replay (Preview) **and** a real short run; refresh
  mid-run reattaches and rebuilds from `live.jsonl`; stop/abort leave no orphans.
