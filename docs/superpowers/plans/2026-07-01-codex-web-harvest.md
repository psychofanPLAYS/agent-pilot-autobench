# Harvest codex's web work onto the detached-engine cockpit

Status: planned (2026-07-01). Owner directive: the web UI is the **primary surface**
and must be a *work of art + functional*; codex's work must be **worth something**.
Codex built valuable features on the **wrong (in-process) architecture** — we keep the
detached-engine architecture (`docs/ARCHITECTURE.md` SSOT) and **port the good ideas
onto it**, not the plumbing. Source branches: `codex/FULL_TEST_RUN` (committed),
`codex/flight-plans-wip` (preserved WIP, `58b986b`).

## What to harvest (ranked by value)

### 1. Eval-rigor params — fixes KNOWN validity gaps (highest value)
Memory `benchmark-validity-gaps` flags: temp=0-only, **no N-repeat**, divergent score
formulas. Codex addressed these:
- **`champion_repeats`** (default 3) — run each question N times → median + IQR instead
  of one temp=0 sample. This is the N-repeat fix. **Port first.**
- **`sampler_policy`** (`hf_recommended` | `runtime_defaults` | `hf:<preset>`) +
  `_sampler_flags_for_policy(model, policy)` — correct per-model sampler settings
  (directly relevant to the Qwen froggeric/template gap).
- **`champion_sample_size`** (default 5) — questions per pack (already partly on main
  via `champion_eval.sample_size`; expose + thread consistently).

**Port target (detached arch):** thread these through **`run-spec.json` options** →
the `engine` command's `run_model` → `_run_one_autoresearch` / `evaluate_champion_packs`.
The web UI sets them in the spec; the engine consumes them. **Never** re-add
`run_model` injection into `serve_webui`.

### 2. Flight plans — curated pre-flight presets (web UX / "work of art")
`flight_plans.py`: `FlightPlan` dataclass + presets `quick_check`,
`find_best_settings`, `librarian_benchmark`, `overnight_campaign` (each: mode, budget,
evidence_goal, workflow, start_label, recommended/advanced). Replaces raw-config knobs
with a beautiful "pick a mission" pre-flight.
**Port target:** surface flight plans in the cockpit pre-flight; on launch, a chosen
plan writes the corresponding `run-spec.json` (mode + options incl. the rigor params
above). Pure client→spec; engine unchanged in shape.

### 3. Already harvested
- **pilotBENCHY branding** — done (`8e6a53e`, web UI only, matches codex's scope).

## How (reliable, per the SSOT + incoming self-upgrade workflows)
Do this as a proper cycle, not a bulk merge:
1. Spec the port (this doc + a short design) → confirm it honors the ARCHITECTURE SSOT.
2. TDD each piece; verify with a **real run** (repeats/sampler visibly change results)
   + the cockpit rendering the new pre-flight.
3. Cross-check scores against `metrics.py` (Agent Index) now that it's on main.

## Do NOT
- Do not merge `codex/FULL_TEST_RUN` wholesale (it predates cockpit+metrics; would
  delete ~5,900 lines). Cherry-pick ideas, re-implement on the detached seam.
- Do not reintroduce in-process web evaluation.
