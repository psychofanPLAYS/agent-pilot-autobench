# pilotBENCHY Web UI Visual Refresh + Charts

Date: 2026-07-07
Status: implemented same session (owner directive: "make it simpler, easier to read, more pleasant — and where are the charts?")

## Goals

1. Reduce visual noise: the page presented every control and all sixteen report
   links at once, with equal weight.
2. Make the primary flow obvious: pick models → pick a flight plan → start.
3. Add charts — the app had zero data visualization on the control page.

## What changed (all inside `src/gguf_limit_bench/webui.py`; no new dependencies)

### Layout / readability
- Base type 14px→15px, looser line height, 10px card radius, more padding,
  max-width main column, sticky table header with a scrollable model table.
- Numbered step badges: "1 Pick your models", "2 Pick a test, then start".
- Progressive disclosure with `<details class="fold">`:
  - the 16 report links fold into "All report links" under Previous results;
  - mode / suite plan / sampler / questions-per-pack / repeats / forced flags /
    websocket toggles fold into "Advanced settings" (flight plan + budget +
    run summary + Start stay visible);
  - Librarian bot jobs fold closed.
- Run status is a proper titled panel in the right column.

### Charts (dependency-free, DIV bars, live via existing state refresh)
- **Best score by model** and **Generation speed (tok/s)** bar charts, fed by a
  new `leaderboard_entries` field in the state payload
  (`_leaderboard_chart_entries`: best run per model, ranked, top 8 — deduped
  because the raw leaderboard has one entry per run).
- Telemetry (CPU/RAM/GPU/VRAM) got live gauge bars under the numbers, with
  warm/hot coloring at 70/90%.

### Data hygiene fixed along the way
- `recent_receipts` now only lists directories that contain a receipt marker
  file, so the persistent Optuna store `_runs/learning` no longer appears as a
  fake "Previous result".

## Constraints honored
- All existing element IDs and test-pinned strings kept (tests/test_webui.py
  passes unchanged except no changes were needed).
- Both themes (default dark, sepia) still work.
- No JS/CSS frameworks; page stays a single self-contained string.

## Verified
- 838-test suite green; drove a real quick-check run end-to-end through the
  redesigned page (model checkbox → flight plan → Start → live status +
  charts render with real `_runs` data).
