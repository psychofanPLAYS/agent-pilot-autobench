# In-flight cockpit — design

Status: approved (2026-06-30). Build order: hygiene backbone first.
Branch: `claude/focused-kepler-21f76d`.

## 1. Purpose

When a user launches a benchmark, the pre-flight selection page should transform
into a beautiful live "in-flight" cockpit that shows **literally everything the
engine is doing** — the question being asked, the model's thinking as it streams,
the answer as it forms, the score the moment it lands, plus live speed and
hardware metrics, the phase pipeline, and the model queue.

This is the cockpit run-flow redesign flagged in
`memory/inflight-page-and-auto-planner-vision.md`. The results pages
(`results.html`, `report.html`) are a separate, already-shipped effort.

## 2. Governing principles (owner directives, 2026-06-30)

1. **The engine is the only thing that evaluates.** It runs models on disk and
   writes the canonical record. It does all the work itself.
2. **The web UI is a thin client.** Two jobs only: (a) pass instructions to the
   engine, (b) render what the engine wrote. It never computes a score, never
   queries a model, never holds evaluation state.
3. **The run directory on disk is the only seam** between the two halves.
4. **Impeccable process & window hygiene** — no orphaned engine or llama-server
   processes, no duplicate servers/ports, clean start/stop/teardown, and a
   browser refresh or close must never kill or orphan an in-flight run.
5. **Show everything, honestly.** Stream what the engine can actually see. Never
   fabricate per-question detail for run paths that can't expose it.

## 3. Architecture

```
website side (thin)            run directory (the only seam)        engine side (detached)
┌─────────────────┐            ┌───────────────────────┐           ┌────────────────────┐
│ browser cockpit │ <── WS ──> │ control.json          │ <──────── │ engine             │
│ renders only    │            │ live.jsonl            │ ────────> │ sequential runner  │
├─────────────────┤            │ status.json           │           ├────────────────────┤
│ web server      │ ── r/w ──> │ telemetry.jsonl       │           │ llama-server       │
│ passes orders   │            │ metrics.json          │           │ own process group  │
└─────────────────┘            └───────────────────────┘           └────────────────────┘
```

- The web server **launches the engine as a detached subprocess** (its own
  process group) and returns immediately. It never runs evaluation in-process.
  (Today's code runs the benchmark in a daemon thread inside the web server —
  this is the central thing Phase 1 removes.)
- The engine writes an append-only `live.jsonl`, a `status.json` heartbeat, and
  `telemetry.jsonl`. It reads `control.json` for stop/abort.
- The UI tails `live.jsonl` from a cursor and renders. On reconnect/refresh it
  re-reads from offset 0 (or a checkpoint) — the file is the state.

### 3.1 Run directory contract

Per active run, under `runs_root/<run_id>/`:

- `control.json` — UI → engine. `{ "action": "run" | "stop" | "abort",
  "requested_at": ISO8601 }`. `stop` = finish current question then halt;
  `abort` = kill-tree now.
- `run-spec.json` — the immutable launch instruction (models, mode, options).
- `live.jsonl` — append-only event stream (§4). Superset of today's
  `events.jsonl`; existing event types are preserved so reports keep working.
- `status.json` — engine heartbeat: `{ "phase", "model", "model_index",
  "model_total", "question_index", "question_total", "pid", "alive_at" }`,
  rewritten every ≤2 s.
- `telemetry.jsonl` — periodic samples (§ telemetry).
- `metrics.json` — rolling derived metrics (running score, coverage); rewritten
  as signals land.
- `engine.pid` / `engine.lock` — single-writer lock + PID for reattach + cleanup.

## 4. Live event protocol (`live.jsonl`)

Schema unchanged: `{ "time": ISO8601, "type": str, "data": {...} }`.
Existing types (`autoresearch_*`, `benchmark_suite_*`, `command_*`,
`champion_pack_eval_*`, etc.) are kept. New per-question types:

| type | data | source |
|------|------|--------|
| `question_started` | `q_id, pack, phase, index, total, prompt` | `pack_runner._run_one_question` entry |
| `thinking_delta` | `q_id, text` (chunk) | `pack_runner._chat` stream, `reasoning_content` |
| `answer_delta` | `q_id, text` (chunk) | `pack_runner._chat` stream, `content` |
| `question_scored` | `q_id, answer, expected, outcome, score, ttft_ms, tok_s` | `_run_one_question` after `score_answer` |
| `running_score` | `quality_0_100, answered, correct, coverage` | recomputed as each `question_scored` lands |
| `phase_changed` | `phase, label` | engine phase transitions |

**Emission is via a callback the engine installs**, not a hard import — the pure
evaluation functions accept an optional `on_event` sink (default no-op) so unit
tests stay isolated and the library has no UI dependency. The engine process
wires `on_event` to an append-to-`live.jsonl` writer.

**Honesty rule:** `thinking_delta`/`answer_delta`/`question_scored` are emitted
only on the **in-process librarian/simple-bench path** (`pack_runner`,
`simple_bench_runner`), which streams `reasoning_content` + `content` from
llama-server and scores inline. The `benchmark_suite` path (inspect-ai etc.) runs
opaque subprocesses; it streams at **task** granularity (`task_started` →
`task_finished` with score) and the cockpit labels those tasks as "external
harness" rather than faking question detail.

### Telemetry
`telemetry.jsonl` samples `telemetry.sample_telemetry()` (CPU, RAM, GPU util,
VRAM used/total, GPU power) on a fixed cadence (≈1 s) plus at attempt/command
boundaries. The cockpit derives sparklines and a peak-VRAM watermark from this.

## 5. Process & window hygiene (Phase 1 — first)

Current gaps (from audit): benchmark runs in a daemon thread inside the web
server; llama-server spawned without process-group isolation (Windows orphan
risk); `stop` only halts between models, never aborts a running subprocess; no
PID lock; WS disconnect does nothing; no signal/atexit cleanup.

Requirements:

1. **Detached engine subprocess.** Web server spawns the engine with
   `CREATE_NEW_PROCESS_GROUP` (Windows) / `start_new_session=True` (POSIX). The
   engine owns the run; the web server can die and the run continues.
2. **Kill-tree on stop/abort.** Engine tracks every llama-server PID. `abort`
   terminates the whole tree: `taskkill /PID <pid> /T /F` on Windows,
   `os.killpg` on POSIX. `stop` lets the current question finish, then halts and
   flushes the receipt.
3. **Process-group isolation for llama-server** so a hard kill never orphans a
   GPU-holding process. Centralize llama-server spawn/teardown through one
   helper used by `server_session`, `server_probe`, `simple_bench_runner`.
4. **PID lockfile + single-writer.** One engine per run dir; the web server
   refuses to launch a second engine for a live run and instead **reattaches**.
5. **Reattach after refresh.** On UI load: if `status.json` is fresh
   (`alive_at` within ~10 s) → show live; if stale and PID dead → mark the run
   crashed, offer cleanup; never silently relaunch.
6. **Signal/atexit cleanup** in the engine: SIGINT/SIGTERM/atexit → stop
   llama-server tree, flush receipt, release lock.
7. **Cleanup on startup.** Scan for incomplete receipts (no `summary`/`status`
   complete) whose PID is dead; mark crashed.

## 6. The cockpit UI (Phase 3)

Two-pane "mission control" (owner choice):

- **Left — live transcript.** Scrolling history of questions. Current question
  expanded: prompt, thinking streaming in mono, answer streaming, score chip
  (`grading… → 0.86 pass`). Completed questions collapse to a one-line row
  (id, pass/fail, score, latency). External-harness tasks render as task rows.
- **Right — mission control.** Run header (model, phase pill, progress, ETA,
  real stop/abort). Live metric cards (running score, tok/s, TTFT, VRAM, GPU).
  Phase pipeline. **Model queue (strictly sequential).**

**Multi-model:** testing is **sequential** — one model fully evaluated before the
next; the focused view follows the running model, a queue strip shows done/next.

**Parallel pressure test (the one exception):** a deliberate test type that fires
**concurrent requests at a single model** to measure how it (and the GPU) behave
under load — aggregate tok/s, latency spread, and stability/degradation. The
cockpit shows a concurrency lane view for this mode only; it is never the default
and never runs two different models at once.

The UI is a single render function over the run-directory snapshot + the tailed
event stream. No business logic; round all displayed numbers.

## 7. Phasing

1. **Hygiene backbone** — detached engine, run-dir contract, control file, PID
   lock, kill-tree, reattach, signal cleanup. Removes the daemon-thread coupling.
2. **Rich emission** — `on_event` sink threaded through `_chat` +
   `_run_one_question`; engine writes `live.jsonl`; running-score + telemetry.
3. **Two-pane cockpit** — pre-flight → in-flight transform; render the stream.
4. **Parallel pressure test** — concurrency mode + cockpit lane view; polish, ETA.

Each phase is independently verifiable (TDD) and shippable.

## 8. Verification

- **Replay engine mode** — an engine that replays a recorded/scripted
  `live.jsonl` at realistic cadence, so the cockpit can be driven and visually
  verified (Preview/browser MCP) **without a GPU**. Doubles as a test fixture and
  a demo.
- **Real run (required)** — RTX 4090 + `G:\AI\llama.cpp\cuda12` binaries +
  `G:\AI\models\LM_Studio-gguf`. Baseline captured from a 4B Qwen3.5 *thinking*
  model (emits `reasoning_content`) to ground the design and seed the replay
  fixture. Final acceptance: launch from the cockpit, watch real thinking →
  answer → score stream, stop mid-run and confirm no orphaned llama-server.

## 9. Cross-cutting pointers (other "departments")

These are deliberately out of scope for the cockpit build but the run-directory +
event contract is designed to enable them. Captured here so they aren't lost.

### 9.1 Scoring / "winning strategy" formula
Goal: one canonical, hardware-independent, poolable **Agent Index (0–100)**
(see `memory/world-class-results-page.md`). Properties:
- **Gate must-pass**: a model that fails the gate pack is capped/flagged — no
  amount of downstream cleverness buys back a broken gate.
- **Composite of phase scores** (general reasoning, agentic tool-use, librarian
  synthesis/citation/memory) with fixed weights, not a moving average.
- **Report coverage %** (how much of the planned battery actually ran) and
  **N-repeat median + IQR** instead of a single temp=0 sample — directly fixes
  the validity gaps in `memory/benchmark-validity-gaps.md`.
- In-flight **running score** is the *live approximation*: running correct/answered
  scaled 0–100 within the current phase, clearly labelled "live, partial" so it's
  never confused with the final Agent Index. Upgrades to the canonical formula
  when `metrics.py` (on branch `claude/zen-bhaskara-c46989`) merges.

### 9.2 Testing algorithm / adaptive auto-planner
A planner that chooses, from run history + model metadata, **which packs to run,
how many repeats, and how long** to reach a target confidence on the Agent Index,
and exposes a live **ETA**. Sketch:
- Start each phase with a small fixed batch; after each question update a running
  mean + variance of the phase score.
- Stop a phase early when the confidence interval on its contribution is tighter
  than a threshold (enough signal), or extend (more repeats) when the CI is wide
  or near a gate boundary.
- Budget is allocated across phases by marginal information gain; ETA = remaining
  planned questions × observed per-question latency.
- Agent Index coverage + CI width is the global stopping criterion.
This is the natural source of the cockpit's ETA and progress; until it lands,
ETA is the simpler `remaining_questions × rolling_latency`.

### 9.3 Three-axis metrics model (context for the cockpit cards)
Quality (Agent Index), Speed (tok/s, TTFT, total-time-100, single vs parallel),
Efficiency (peak VRAM, tok/s-per-GB, tokens/Joule). The cockpit's right rail
surfaces the live slice of all three; the results page owns the historical view.

## 10. Risks / dependencies
- `metrics.py` / `charts.py` (Agent Index, K-sample stats) are **not on this
  branch**; the cockpit uses a self-contained live running-score and upgrades
  later. Not a blocker.
- inspect-ai subprocess opacity limits per-question streaming to in-process
  packs — accepted and surfaced honestly in the UI.
- Windows process-group/kill-tree behavior must be tested explicitly (primary
  target OS).
