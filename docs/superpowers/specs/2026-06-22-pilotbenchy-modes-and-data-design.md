# PilotBENCHY Modes, Data Capture & Learning — Design (north star)

Date: 2026-06-22
Status: DESIGN (vision + first build slice). Successor to
`2026-06-22-pilotbenchy-real-benchmark-design.md`.
Driver: the TUI is the product. A non-technical user drives it with
arrows/space/enter/escape, picks a goal-shaped **mode**, and pilotBENCHY runs the
right workflow, captures everything, graphs what it learned, and remembers it.

## Principles

1. **TUI-first.** Double-click `START.bat` → arrow/space/enter cockpit. CLI is for
   power users and automation only; the novice never types a flag.
2. **Goal-shaped modes, not knobs.** The user picks *what they want to learn*, not
   server arguments. Each mode maps deterministically to run settings.
3. **Capture everything.** Both the app's measurements (accuracy, tok/s, TTFT) AND
   live hardware telemetry sampled *during* each profile (GPU util, VRAM, power,
   temp, CPU, RAM), time-aligned with generation so we can see what actually
   bottlenecks a model.
4. **Show, don't dump.** Results are graphs + a one-line verdict, not TSVs.
5. **Learn across sessions.** Persist per-model champion + per-flag effects keyed to
   model+hardware; surface "last time" on re-select; warm-start to converge faster.
6. **Build on upstream.** Intelligence/agentic scoring uses inspect-ai (already a
   dep), lm-eval, BFCL, and HF datasets — not homegrown reimplementations.

## Mode catalog

Each mode is a `RunMode` mapping a friendly name to run parameters: budget, which
ladder, whether questions are asked, context ladder on/off, intelligence on/off.

| id | label | budget | ladder | questions | context ladder | intelligence |
|----|-------|--------|--------|-----------|----------------|--------------|
| `quick` | Quick check | ~2 min | baseline only | no (speed scout) | no | no |
| `best_settings` | Find best settings | ~10 min | full flag ladder | yes | no | no |
| `flag_effect` | How flags affect speed | ~8 min | full ladder + stripped baseline | yes (few) | no | no |
| `context_limits` | Context limits | ~12 min | best profile | yes (1/tier) | yes | no |
| `deep` | Deep / overnight | budgeted | full | yes | yes | yes |
| `custom` | Custom (15/30/45/60) | user-set | full | yes | optional | optional |

Intelligence add-on is an independent toggle layered on any mode (more rounds).

## The "fewer flags = faster" experiment (stripped baseline)

The current ladder is *additive* (L0 baseline → +one flag per rung), so it only shows
what ADDING a flag costs. To test the user's hypothesis ("less flags = quicker") we
add a truly stripped rung BEFORE L0:

- `Lmin-stripped`: `--gpu-layers 99` only-essentials — `flash_attention=False`,
  `cont_batching=False`, `parallel=1`, `kv_unified=False`, default batch. (GPU
  offload stays on; without it the model falls to CPU and the comparison is
  meaningless, not a "flag" effect.)

This makes the ladder show, per flag, whether it COSTS or BUYS speed vs truly bare —
e.g. the live run already showed `--kv-unified` is ~19% FASTER while `--parallel`
is slightly slower. The `flag_effect` mode surfaces this as a readable table/graph.

## Data capture (during the bench, not just after)

Reuse `telemetry.py`'s sampler thread but run it *for the duration of each profile's
question batch*, writing a per-profile `telemetry.csv` time series and rolling up
peak/median GPU util, VRAM, power, temp, CPU, RAM. Align with per-question tok/s and
TTFT so a report can answer "was this model GPU-bound, memory-bound, or spilling?".
This is the raw material for the graphs and for smarter learning.

## Visualization

Generate charts into the existing `report.html` (and a top-level `results.html`):
- per-flag tok/s & TTFT bars (flag effect)
- context tier vs tok/s falloff line (context limits)
- VRAM/GPU-util vs tok/s during a run (bottleneck view)
- accuracy-vs-speed scatter across profiles (the trade-off)
Charts are self-contained (inline SVG / a tiny JS lib), openable with no server.

## Learning / memory

Per `model + hardware_fingerprint + score_version`, persist to `_db`:
champion settings, per-flag effect deltas, max usable context, and accuracy. On
re-select in the TUI, show "Last time: best = L2-kv-unified, 194 tok/s, ctx 16k,
3/10 (2026-06-22)". Warm-start the ladder from the prior champion; deprioritize rungs
that lost/crashed last time. (Optuna study already persists for the settings search;
this adds ladder-level memory — Unit H of the prior spec.)

## Big-think / later

- **Meta-optimization:** use the Karpathy loop not just to tune settings but to learn
  WHICH tests are most informative per model class (stop running tests that never
  change the ranking), shrinking time-to-answer over sessions.
- **Custom-budget planner:** given "I have 30 minutes," plan the most informative
  sequence of profiles/tiers/packs that fits, using prior learning to skip the
  settled questions.
- **Upstream intelligence packs:** inspect-ai / lm-eval / BFCL adapters for the
  reasoning + agentic add-on.

## First build slice (this iteration)

1. `Lmin-stripped` rung added to the flag ladder (+ tests).
2. `modes.py`: `RunMode` catalog mapping the table above to run parameters (+ tests).
3. TUI mode selector: arrow/enter pick among modes, active mode + description shown,
   wired into the run (+ tests).

Subsequent slices: in-bench telemetry capture → charts → learning/memory surface →
intelligence add-on → custom-budget planner.
