# ADR 0001 — Detached engine, thin web client, run-directory seam

- Status: **Accepted** (2026-06-30; re-affirmed 2026-07-01)
- Deciders: owner + Claude
- SSOT: `docs/ARCHITECTURE.md` → "DESIGN DIRECTION". This ADR records the *why* and the
  rejected alternatives so they don't get re-litigated or re-built by mistake.

## Context

The web cockpit is the **primary surface** users interact with, and it shows a live
"in-flight" view of a benchmark run (streaming thinking/answer/score, telemetry). Runs
are long (minutes to hours), hold a GPU, and spawn `llama-server`. Development is done
by multiple AI agents across sessions.

## Decision

1. All evaluation happens in a **detached `engine` subprocess** (own process group),
   launched by the web server, which returns immediately.
2. The **web UI is a thin client**: it only writes instructions and renders what the
   engine wrote. It never evaluates, scores, or holds run state.
3. The **run directory on disk is the only seam** (`run-spec.json`, `control.json`,
   `status.json`, `live.jsonl`, `telemetry.jsonl`, receipts).
4. Hygiene: detached process groups, kill-tree on stop/abort, status heartbeat,
   reattach-on-refresh. Sequential by default.

## Alternatives considered — and REJECTED

- **In-process evaluation in the web server** (pass a `run_model` callback to
  `serve_webui`, run the benchmark in a daemon thread). *Rejected:* a browser refresh
  or web-server crash kills or orphans the run; no clean stop/abort of a running
  subprocess; GPU-holding `llama-server` processes orphan on Windows; the UI and eval
  become entangled and untestable in isolation. **A prior branch (`codex/FULL_TEST_RUN`)
  took this path and had to be discarded** — this is the concrete failure this ADR
  exists to prevent.
- **WebSocket-only, no on-disk record.** *Rejected:* no reattach/replay, no durable
  evidence, can't drive the cockpit GPU-free for testing.
- **CLI/TUI as the primary surface.** *Rejected:* the owner wants the web UI to be the
  primary, beautiful surface; CLI/TUI remain for power users and automation.

## Consequences

- The web server can die and the run continues; refresh reattaches from `live.jsonl`.
- Eval logic is testable without a UI; the cockpit is verifiable GPU-free via
  `engine-replay`.
- New web features (presets, sampler policy, N-repeat) are added by writing richer
  `run-spec.json` and teaching the engine to read it — **never** by moving eval back
  into the web server.
- Any future change to this boundary must update the SSOT + add an ADR superseding this
  one, with the owner's sign-off.
