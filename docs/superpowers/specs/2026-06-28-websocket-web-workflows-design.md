# WebSocket Web Workflows Design

Date: 2026-06-28
Status: approved for implementation planning

## Goal

Make the browser cockpit the primary human workflow for Agent Pilot Autobench while
keeping `apb tui` as a fallback terminal cockpit for now.

The web cockpit should cover the full normal path:

- discover local GGUF models
- choose one or more models
- choose a benchmark mode or benchmark-suite plan
- review safe llama.cpp server flags
- start a run
- watch live benchmark, telemetry, and receipt progress
- request stop-after-current when a run is too long
- open generated reports and receipts

The CLI remains the stable contract for agents, CI, scripts, and power users. The
website must call the existing Python core instead of reimplementing benchmark logic.

## Current State

`src/gguf_limit_bench/webui.py` already provides a useful browser cockpit, but it
uses the standard library `ThreadingHTTPServer`, `GET /api/state` polling, and
`POST /api/start`.

That is a good prototype, not the final app surface. It proves the workflow shape:
model discovery, run options, telemetry, event feed, and receipt links already have
tests in `tests/test_webui.py`.

The older Textual TUI still exists in `src/gguf_limit_bench/tui.py`. It should not
be deleted during this slice. It stays available through `apb tui`, but it is no
longer where new product workflow work should go.

## Options Considered

### Option A: WebSocket Foundation Only

Replace polling with WebSocket state events, keep the page mostly as-is, and defer
workflow changes.

Pros:

- smallest code change
- easy to test
- low risk to existing runs

Cons:

- still leaves benchmark-suite plan selection and stop-after-current outside the
  main browser workflow
- does not fully answer the requested product direction

### Option B: Full Web Workflow, Staged Internals

Introduce a local FastAPI/Starlette service with WebSocket events, then move the
complete run workflow into the page while still using the existing benchmark core.

Pros:

- matches the product direction
- reuses tested Python benchmark modules
- keeps implementation slices small
- creates a durable service boundary for future charts and richer dashboards

Cons:

- adds a runtime dependency and a new test shape
- needs careful migration so the current cockpit does not regress

### Option C: Website-Only Repositioning First

Mostly update docs and command flow so `apb` is website-first, then do WebSocket
work later.

Pros:

- fastest visible repositioning
- minimal technical risk

Cons:

- leaves the technical gap in place
- gives David a better promise before the app can fully deliver it

## Decision

Use Option B.

Build the full web workflow as the first-class human surface, but implement it in
small vertical slices:

1. Add a local web service layer.
2. Add WebSocket state, telemetry, and run-event streaming.
3. Add full run controls to the browser page.
4. Add benchmark-suite plan selection and receipt-aware result panels.
5. Update docs and command wording so TUI is clearly fallback.

## Architecture

```text
apb / agent-autobench start
  -> CLI resolves config and doctor checks
  -> local web service on 127.0.0.1
  -> browser cockpit
       -> HTTP for static assets and initial bootstrap JSON
       -> WebSocket for state, commands, telemetry, run events
  -> existing benchmark core
       -> autoresearch, benchmark_suite, simple_bench_runner, reports
  -> _runs receipts, ledgers, reports
```

The service should bind to `127.0.0.1` by default. No cloud service, account,
upload, or remote telemetry is introduced.

## Backend Design

Add a new service boundary without moving benchmark logic:

- `webui.py` can remain the public module name.
- Internally split the large module into smaller units when implementation starts:
  - `web_state.py`: state models, run queue state, payload shaping.
  - `web_service.py`: FastAPI/Starlette routes and WebSocket endpoint.
  - `web_assets.py`: static HTML/CSS/JS helpers, or package-data assets.
  - `web_commands.py`: start/stop command handling and validation.
- Keep compatibility helpers such as `WebRunOptions`, `WebUiState`, and
  `serve_webui` unless tests show a cleaner migration path.

The WebSocket should carry JSON messages with explicit `type` fields:

- server to browser:
  - `hello`: protocol version and initial summary
  - `state`: full or partial cockpit state
  - `telemetry`: CPU/RAM/GPU/VRAM sample
  - `run_event`: benchmark event with timestamp, phase, message, receipt path
  - `receipt`: new or updated receipt artifact
  - `error`: validation or backend failure safe for display
- browser to server:
  - `subscribe`: request current state
  - `start_run`: validated run start payload
  - `stop_after_current`: request graceful stop after the current model/profile
  - `refresh`: explicit state refresh

The first implementation may keep a narrow HTTP fallback for tests or non-WebSocket
clients, but the browser should use WebSocket as the normal live path.

## Run Control

The browser should support:

- select model rows
- select mode
- optionally select a benchmark-suite plan from bundled plans
- set budget minutes where allowed
- toggle only validated safe forced flags
- start run
- request stop-after-current
- show failed validation before any model server starts

Hard stop/kill is intentionally out of scope for the first slice. Graceful
stop-after-current is safer because benchmark subprocesses and receipts should
finish coherently.

## Benchmark-Suite Plan Selection

The website should list source-controlled plans under `benchmarks/plans/` with:

- filename
- title or description when available
- whether it is smoke/local/heavy
- warning text when a plan expects an already-running endpoint or external tool

The web start payload should pass the selected plan path through the same CLI/core
path as `--benchmark-suite-plan`. The plan file remains the source of truth.

## Frontend Design

Keep the frontend simple and local:

- no React/Vite/TypeScript build step in this slice
- plain HTML/CSS/JavaScript is acceptable
- stable dimensions for model table, run controls, status feed, telemetry, and
  receipt panels
- no marketing landing page; `apb` opens the actual cockpit
- use compact controls suitable for repeated benchmark work

Expected cockpit areas:

- model table
- run configuration panel
- benchmark-suite plan picker
- live status/event feed
- telemetry meters
- active receipt panel
- recent receipts/results panel

Charts can remain static-report work unless they are cheap to include. The first
WebSocket slice should prioritize correct workflow and live state over decorative
visuals.

## TUI Position

Keep `apb tui`.

Do not delete `src/gguf_limit_bench/tui.py` or its tests during this slice. The TUI
is now described as:

- fallback terminal cockpit
- useful when the browser cannot open
- not the main place for new workflow features

Future removal can be a separate decision after the web workflow is stable.

## Safety

- Bind only to `127.0.0.1` by default.
- Reject unsafe llama-server flags exactly as the existing web path does.
- Do not start heavyweight XTREME services as part of tests.
- Do not delete `_runs`, model files, databases, or benchmark receipts.
- Keep generated run artifacts in ignored local folders.
- Store server launch arguments as data, not executable scripts.
- Surface validation failures before any model server starts.

## Testing

Use narrow tests first:

- unit tests for WebSocket message validation and payload shaping
- tests that the service lists models, modes, plans, and receipts
- tests that `start_run` dispatches the existing backend callback with expected
  `WebRunOptions`
- tests that unsafe flags and unknown models are rejected
- tests that stop-after-current sets state without killing a process
- CLI tests that `apb` and `agent-autobench start` still call `serve_webui`
- compile check for `src` and `tests`

No live llama.cpp benchmark is required for the first service migration. Mocked
backend callbacks should prove the web service contract.

## Documentation Updates

Update docs after implementation:

- `README.md`: browser cockpit is the main workflow; TUI fallback wording.
- `docs/START-FOR-NORMAL-PEOPLE.md`: plain `apb` opens the website; `apb tui` is
  only for fallback.
- `docs/ARCHITECTURE.md`: FastAPI/Starlette + WebSocket service in the feature map.
- `docs/COMMAND-BOARD.md`: include web workflow commands and benchmark-suite plan
  examples.

## Rollback

Keep `serve_webui` as the CLI integration point. If the new service has issues,
the implementation can temporarily route `serve_webui` back to the current HTTP
server behavior while keeping the benchmark core untouched.

## Out Of Scope

- deleting the TUI
- replacing the CLI
- replacing llama.cpp tools
- cloud dashboards
- user accounts
- remote telemetry
- hard process kill controls
- new benchmark science beyond exposing existing benchmark-suite plans

## Acceptance Criteria

- `apb` opens a browser workflow that uses WebSocket for live state.
- The page can start a mocked benchmark run from selected models.
- The page can request stop-after-current.
- The page lists bundled benchmark-suite plans.
- Existing CLI run paths still work.
- `apb tui` still exists.
- Tests cover the web service contract and run validation.
- Docs clearly say the website is primary and TUI is fallback.
