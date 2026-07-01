# Self-Upgrade Cycle for pilotBENCHY: Research Report

**Date:** 2026-07-01
**Scope:** How to run a repeatable, self-improving development loop for pilotBENCHY (FastAPI cockpit + detached llama.cpp engine, developed by Claude Code / Codex agents) without repeating the "codex diverged onto the wrong architecture" incident.
**Constraint under design:** detached engine does all evaluation; the web UI is the PRIMARY surface but a THIN client; the on-disk run directory is the only seam between them.

---

## (a) Executive Summary

pilotBENCHY already has the bones of an agent-governed repo — `AGENTS.md` and `docs/ARCHITECTURE.md` exist and are read by both Claude Code and Codex. The problem that caused the prior drift incident is not "no docs exist," it's that **the docs on disk describe the old architecture** (a CLI-first "gguf_limit_bench" tool with a WebSocket cockpit bolted on) rather than the new decision (detached engine / thin web client / run-directory-as-seam). An agent reading `docs/ARCHITECTURE.md` today would reasonably build the old shape — that's exactly what happened.

2025–2026 tooling converged on an answer to this class of problem: **AGENTS.md as the one true instruction file** (now a Linux Foundation standard, symlinked from `CLAUDE.md` so there is one file, not two that drift), backed by **spec-driven development** (GitHub Spec Kit, AWS Kiro, BMAD-METHOD) where a written, human-approved spec/plan gates implementation, and **ADRs** for the "why" that survives even after the spec is superseded. Separately, the self-improving-agent literature (Ralph Wiggum loops, SICA, "Compound" loops, Anthropic's own Claude Code "Superpowers" pattern) converges on: bounded task loops, mandatory verification before claiming done, and a persistent knowledge file the agent itself updates after every cycle — not a one-time document, a living one.

The single highest-leverage fix for pilotBENCHY is to **stop treating architecture docs as descriptive and start treating them as prescriptive contracts that gate work** — i.e., turn `docs/ARCHITECTURE.md` into the actual SSOT for the new engine/UI/run-dir design (today it describes the pre-refactor shape), write one ADR that records the detached-engine decision and why the CLI-first alternative was rejected, and add a repo-local skill/gate that forces any agent touching `src/` to read the SSOT and the run-directory contract *first*. Everything else in this report — eval-driven verification, retrospective notes, subagent review — compounds on top of that fix but does not substitute for it.

---

## (b) Best Ideas, Ranked

### 1. Make the architecture doc prescriptive and current — the actual root-cause fix
**Why it matters:** The prior "codex diverged" incident happened because the design intent wasn't "stated loudly enough in the docs." Concretely: `docs/ARCHITECTURE.md` as it stands today describes a CLI+WebSocket-cockpit tool, not the detached-engine/thin-client/run-dir-seam design the owner has since decided on. An agent that reads the current file and does exactly what it says will rebuild the wrong thing again — the doc itself is the drift vector.
**How to apply here:** Rewrite `docs/ARCHITECTURE.md` (or add a new top section) to state, in the first 10 lines, in imperative language: "The engine is a DETACHED process. It owns all evaluation. The web UI is the PRIMARY user surface and MUST be a THIN client — it renders and issues run requests only; it must never contain evaluation/scoring logic. The run directory on disk is the ONLY integration seam between engine and UI: no shared Python imports of engine internals from `webui.py`, no in-process calls into the engine from the FastAPI process." Put a one-line rule at the very top of `AGENTS.md`: "Before touching `src/` under `engine/` or `webui/`, read `docs/ARCHITECTURE.md` §Design Intent. It is normative, not descriptive." This is the single change that would have prevented the original divergence. Docs-as-contracts research (ICSE 2026 architectural-conformance study) backs this: architectural documentation measurably improves conformance in LLM-generated code, but only when it's current and directive, not historical.

### 2. AGENTS.md as literal single source of truth, CLAUDE.md as a stub
**Why it matters:** AGENTS.md became a Linux Foundation–backed cross-tool standard in 2025 (OpenAI, Google, Cursor, Sourcegraph, Factory) precisely because projects using Claude Code *and* Codex *and* others were maintaining near-duplicate instruction files that silently diverged — exactly the two-agent situation here (Claude Code + Codex). The fix pattern used industry-wide: one file, one or more filenames pointing at it.
**How to apply here:** pilotBENCHY already has `AGENTS.md` — good. Verify there is no separate, independently-edited `CLAUDE.md`; if one exists or gets created, make it a one-line stub (`See @AGENTS.md`) or a symlink, never a parallel copy. Any project-specific Claude Code guidance (skills, hooks) should live under `.claude/` and *reference* AGENTS.md rather than restate its rules, so there is exactly one place to update the architecture rule from idea (1).

### 3. Spec → plan → implement → verify, with a human-approved gate before code
**Why it matters:** GitHub Spec Kit, AWS Kiro, and BMAD-METHOD all converged on the same shape in 2025–2026: write requirements → generate a technical design/plan → get explicit approval → only then generate code, with each phase gated. The research explicitly frames this as the direct response to "intent drift," where an underspecified instruction lets the model pick defaults the team didn't want — again, the exact mechanism behind the original codex divergence.
**How to apply here:** For any change that touches the engine/UI boundary or the run-directory schema, require: (1) a short spec stating what run-dir artifacts change and why, (2) a plan reviewed against `docs/ARCHITECTURE.md` for conformance, (3) implementation, (4) verification against real run output. The existing `superpowers:writing-plans` / `superpowers:executing-plans` skills already available in this environment implement this shape — use them for engine/UI-boundary work specifically, not just "any task."

### 4. ADRs for "why," separate from specs for "what"
**Why it matters:** Specs and plans get superseded and deleted; the *reasoning* for a decision needs to outlive the artifact. The 2026 pattern explicitly pairs OpenSpec-style specs (what to build) with ADRs (why this and not the alternative) so future proposals — including a future agent's — can see that "CLI-first" or "in-process engine" were considered and rejected, and why, without having to re-derive it from git archaeology.
**How to apply here:** Write one ADR now: "ADR-0001: Detached engine, thin web client, run-directory-as-seam." Body should state the alternatives actually rejected (in-process FastAPI calling engine functions directly; engine importing webui internals for progress reporting) and the concrete failure mode of each (couples process lifecycles, breaks "engine survives UI restart," makes the run directory not a real contract). This is cheap insurance: it's the document that would have let a reviewer catch the codex divergence in one paragraph instead of a full rebuild.

### 5. Eval-driven / verification-before-completion loops, not "looks right" loops
**Why it matters:** Across the self-improving-agent literature (SICA, the Ralph Wiggum technique, Claude Code's own "Superpowers" skill set already installed in this environment), the common failure mode isn't bad code generation — it's agents claiming success without running anything. The corrective pattern is uniform: pick task → implement → run real validation (tests, compiler, or in pilotBENCHY's case, a real llama.cpp run) → only then commit/claim done.
**How to apply here:** pilotBENCHY's own memory already flags this risk ("Real-run budget finding" — the engine can silently produce a "no useful result" with too small a budget and *look* like success). The verification step for this project cannot be `pytest` alone; it must include a real-run smoke test against the actual RTX 4090 + GGUF model + llama.cpp path (see idea #9 below), because the class of bug that bit this project before (budget too small → useless run that still "succeeds") is invisible to unit tests. Use the `superpowers:verification-before-completion` skill already available, but scope its "evidence" requirement specifically to include a fresh real-run receipt, not just green CI.

### 6. Adversarial / second-agent code review before merging engine-boundary changes
**Why it matters:** 2025–2026 patterns (Claude Code's Superpowers two-stage review subagent; SCOUT/GUARD subagent splits; Kiro's reviewer + security-reviewer agents) consistently use an agent with a *different* prompt and *no investment in the change* to review it — because the implementing agent is structurally bad at noticing its own architectural drift (it "wrote the plan," so of course the code matches the plan; the question is whether the plan matched the SSOT).
**How to apply here:** For engine/UI-boundary and run-directory-schema changes specifically, require a review pass by a fresh subagent (or the other tool — Codex reviewing Claude's diff, or vice versa) whose prompt is literally "does this diff violate the SSOT in docs/ARCHITECTURE.md? Does it add any in-process coupling between engine and webui, or logic outside the run directory as the seam?" This is cheap (one subagent call) and directly targets the recurrence of the original failure mode. The `superpowers:requesting-code-review` / `superpowers:receiving-code-review` skills already installed implement the mechanics of this.

### 7. Persistent, agent-updated knowledge log (not a one-shot doc)
**Why it matters:** The self-improving-agent research (Ralph's "four channels of memory": git history, progress log, task-state file, and a semantic knowledge file the agent itself edits after each cycle) treats "lessons learned" as compounding infrastructure, not a report written once. The explicit framing: "each improvement should make future improvements easier" — discovered gotchas get written down so the *next* agent (possibly a different tool, days later) doesn't re-learn them by re-breaking them.
**How to apply here:** This project's memory system (MEMORY.md index with topic files) is already doing roughly this at the *user* level. Extend the same idea into the repo itself: a short `docs/LEARNINGS.md` or a `## Recent Learnings` section in `AGENTS.md` that agents append to after any cycle that surfaces a non-obvious gotcha (e.g. "Qwen3.5/3.6 need `--jinja` + froggeric template," "small budgets silently produce useless runs," "engine must never import from `webui.py`"). Keep it short and pruned — stale entries are worse than none (see idea #8).

### 8. Guard against drift *of the guardrails themselves*
**Why it matters:** Recent 2026 research on "constraint drift" in multi-agent systems (Li et al., arXiv:2605.10481) makes the point that safety/architecture constraints decay as they pass through memory, delegation, and repeated re-statement unless something actively *maintains* them — an instruction stated once in a doc six months ago has no operational force unless it's re-checked. A constraint only has force when it is available at the point of action, checkable against the actual effect, and reconstructable after the fact.
**How to apply here:** Don't treat AGENTS.md / ARCHITECTURE.md as "written once, done." Build the periodic self-check described in the playbook below (step 6): a recurring pass (could literally be the existing `superpowers:consolidate-memory`-style reflective pass, or a scheduled `/loop`) that re-reads the SSOT against the actual code shape and flags where they've silently diverged — the same mechanism a lint rule provides for code style, applied to architecture.

### 9. Real-run verification as a first-class gate, not an afterthought
**Why it matters:** This is specific to pilotBENCHY, not generic literature, but it's the concrete instantiation of idea #5. The project's own memory records that the engine can "work" (127 tok/s, completes without error) while still producing a useless result because the budget didn't fit the model — a failure class that only a real run against real hardware surfaces, and one that's uniquely dangerous for a benchmark tool since a silently-bad run pollutes the results DB that the whole cockpit is built to present.
**How to apply here:** Any change to `engine/` (the detached process) or to the run-directory schema should require, before being called done, a fresh run against a real small/fast GGUF model on the actual RTX 4090 path (the paths documented under "Run environment" in memory), inspected for (a) the run directory actually being written in the new/expected shape, (b) the webui actually rendering it as a thin client (no client-side scoring), (c) a sane, non-degenerate score. This is the pilotBENCHY-specific analog of "run the compiler and the test suite" from the generic eval-driven-dev pattern.

### 10. Harvest-from-divergent-branches as a deliberate, bounded workflow
**Why it matters:** The prior codex branch that diverged onto the wrong architecture isn't pure waste — multi-agent literature on parallel/planner-worker setups (Cursor's planner-worker-judge model, Kiro's parallel task groups) treats a rejected branch as a source of candidate ideas to mine under supervision, not something to silently discard. The risk is doing this *unsupervised*, where the SSOT violation quietly re-enters through a "good idea" cherry-pick.
**How to apply here:** Build this as an explicit, bounded workflow (see (d) below) rather than ad hoc "let's look at what codex did." The workflow's contract must be: read the divergent branch, extract *ideas* (UI layout choices, scoring tweaks, test cases) into a plain-text list, evaluate each idea against the current SSOT independently, and only port code for ideas that pass — never merge or rebase the divergent branch directly.

---

## (c) The pilotBENCHY Self-Upgrade Cycle Playbook

Numbered steps an agent (Claude Code or Codex) runs each cycle. This is the operational loop; it assumes idea (1) — a current, prescriptive `docs/ARCHITECTURE.md` — is already true. If it isn't, step 0 fixes that first and nothing else proceeds.

1. **SSOT gate.** Before touching any file under `src/` (engine or webui), read `AGENTS.md` in full and the "Design Intent" section of `docs/ARCHITECTURE.md`. If the task touches the engine/UI boundary or the run-directory schema, also read the relevant ADR(s) under `docs/adr/`. State in the first line of your working notes which SSOT constraints apply to this task (e.g. "engine stays detached; UI must not import engine internals; run dir is the only seam").

2. **Spec the change.** For anything bigger than a one-line fix: write a short spec (what run-dir artifact or UI surface changes, and why) and a plan, checked explicitly against step 1's constraints. Use `superpowers:writing-plans`. Get human approval for anything that changes the run-directory schema, the process boundary, or public CLI/API surface — those are the categories where the original drift happened.

3. **Implement with TDD, narrow context.** Use `superpowers:test-driven-development`. Keep the engine and webui changes in separate, reviewable commits so the seam boundary is visible in the diff, not just in prose.

4. **Verify with evidence, including a real run.** Run the narrow test/lint checks from `AGENTS.md` §Verification. For engine or run-directory changes, additionally do a real-run smoke test against actual llama.cpp/GGUF/RTX 4090 (per idea #9) and inspect the resulting run directory by hand — don't trust "the process exited 0." Use `superpowers:verification-before-completion` and require the evidence (command + output, not just a claim) before saying done.

5. **Adversarial review on boundary changes.** If the change touched the engine/UI boundary or run-directory schema, get a second, differently-prompted pass (a fresh subagent, or hand off to the other tool) whose only job is to check the diff against the SSOT — not general code quality. Use `superpowers:requesting-code-review`.

6. **Update the living knowledge log.** Append any non-obvious gotcha discovered this cycle to the "Recent Learnings" section (idea #7). Prune anything now stale or superseded — a wrong learning left in place is worse than a missing one.

7. **Periodic SSOT self-audit (not every cycle — e.g. weekly or every N cycles).** Re-read `docs/ARCHITECTURE.md` against the actual shape of `src/`: does the engine still avoid importing anything UI-side; does the webui still avoid holding evaluation logic; is the run directory still the only integration point? This is the "constraint drift" check (idea #8) — write findings as a short note, and if drift is found, that becomes the next cycle's task, prioritized above new features.

8. **Close the loop.** Commit, and if a PR/merge is requested, follow the finishing-a-development-branch skill. Do not silently accumulate unmerged divergent branches — if a branch is abandoned mid-cycle, either finish it, explicitly mark it "harvest candidate only" (see workflow below), or delete it. An abandoned-but-not-marked branch is exactly how the last divergence went unnoticed for as long as it did.

---

## (d) Prioritized Workflows/Skills to Build

Ordered by leverage (highest first). Each is scoped to be buildable as a Claude Code skill (`.claude/skills/`) or a lightweight hook/gate.

| # | Name | Trigger | What it does |
|---|------|---------|---------------|
| 1 | **`ssot-gate`** | Any edit to a file under `src/engine/**` or `src/webui/**` (or equivalent boundary dirs) — ideally enforced as a pre-edit hook, not just a convention | Blocks/warns unless the agent has stated, in this session, that it read `docs/ARCHITECTURE.md` §Design Intent and the relevant ADR. This is idea (1) made mechanical instead of aspirational — the exact gate that would have caught the codex divergence before code was written. |
| 2 | **`real-run-verify`** | Any change to `engine/` internals, run-directory writer, or scoring code; also invoked manually as "verify this against a real run" | Launches the detached engine against a small/fast real GGUF model on the documented RTX 4090 path, runs one short benchmark, then asserts: run directory has the expected shape, webui renders it without client-side scoring, and the score isn't degenerate (not zero/null/suspiciously uniform — the "budget too small" failure mode from memory). Fails loudly rather than reporting "process exited 0" as success. |
| 3 | **`adr-writer`** | Any decision that rejects an architectural alternative (detected by the agent itself, e.g. "I considered X and chose Y because...") or explicitly requested ("write an ADR for this") | Creates `docs/adr/NNNN-title.md` in a fixed template (context, decision, alternatives rejected and why, consequences). Keeps this separate from specs/plans so the "why" survives after the plan file is deleted. |
| 4 | **`boundary-review`** | Any diff touching both `engine/` and `webui/`, or the run-directory schema, before merge | Spawns a fresh subagent (different prompt, no investment in the change) whose sole brief is: does this diff violate the SSOT (detached engine, thin client, run-dir-only seam)? Report pass/fail with specific line references, not general code review. This is the adversarial-review idea (6) scoped narrowly so it's cheap enough to run every time, not just on request. |
| 5 | **`learnings-append`** | End of any cycle where a non-obvious gotcha was discovered (agent self-identifies: "this cost me time / surprised me") | Appends a dated, one-paragraph entry to `docs/LEARNINGS.md` (or an `AGENTS.md` section). Small, bounded — this is NOT a place for full retrospective essays; it's the four-line "here's the trap" note the next agent needs. |
| 6 | **`ssot-drift-audit`** | Scheduled (weekly, via the `/loop` or `scheduled-tasks` mechanism already available in this environment) or manually invoked | Re-reads `docs/ARCHITECTURE.md` and diffs its claims against the actual module boundaries in `src/` (e.g., grep for cross-imports between engine and webui packages; check the run-directory writer/reader are the only shared contract). Produces a short drift report; if drift found, files it as the top-priority task for the next cycle. This operationalizes idea (8) — constraints decay unless actively re-checked. |
| 7 | **`harvest-divergent-branch`** | Explicitly invoked when a stale/rejected branch (like the original codex divergence) needs mining for salvageable ideas | Reads the divergent branch's diff and commits, extracts a plain list of *ideas* (not code), evaluates each against the current SSOT independently, and proposes which ideas are worth a fresh, SSOT-conformant implementation. Never merges or cherry-picks code directly from the divergent branch — output is a decision list, ported by hand through the normal spec→plan→implement loop. |
| 8 | **`results-page-thin-client-check`** (project-specific extension of #4) | Any change under the webui/results-page code | Specifically checks that no scoring/aggregation/evaluation logic was added client-side or in `webui.py` — only rendering of what the engine already wrote to the run directory. Directly protects the "thin client" half of the SSOT, which is the half most tempting to violate incrementally (it's easy to add "just a little" computed metric in the UI layer). |

Build order: **1 and 2 first** (they are the direct fix for the root cause and the direct fix for the "looks done but isn't" failure mode already seen in this project). **3 and 4 next** (cheap, high-leverage, prevent recurrence). **5 and 6** once the loop is running for a few cycles (they need real cycles to have content). **7** only when there's actually a divergent branch worth mining. **8** can be folded into 4's prompt initially and split out only if it needs to fire more often than boundary changes alone would trigger it.

---

## (e) Sources

- [The Kitchen Loop: User-Spec-Driven Development for a Self-Evolving Codebase](https://arxiv.org/pdf/2603.25697)
- [A Self-Improving Coding Agent (SICA)](https://arxiv.org/html/2504.15228v2) / [ICLR 2025 workshop version](https://openreview.net/pdf?id=rShJCyLsOr)
- [Closing the Loop: Coding Agents, Telemetry, and the Path to Self-Improving Software — Arize](https://arize.com/blog/closing-the-loop-coding-agents-telemetry-and-the-path-to-self-improving-software/)
- [A Survey of Self-Evolving Agents](https://arxiv.org/pdf/2507.21046)
- [AddyOsmani.com — Self-Improving Coding Agents](https://addyosmani.com/blog/self-improving-agents/) (Ralph Wiggum technique, Compound loops, Planner-Worker-Judge model, AGENTS.md-as-knowledge-base pattern)
- [awesome-harness-engineering (GitHub)](https://github.com/ai-boost/awesome-harness-engineering)
- [Agents.md: an open standard for AI coding agents — Tessl](https://tessl.io/blog/the-rise-of-agents-md-an-open-standard-and-single-source-of-truth-for-ai-coding-agents/) (symlink/stub pattern, SSOT argument)
- [AGENTS.md vs CLAUDE.md: The AI Developer's Guide to Context Standards — Hivetrail](https://hivetrail.com/blog/agents-md-vs-claude-md-cross-tool-standard)
- [How to Build Your AGENTS.md (2026) — Augment Code](https://www.augmentcode.com/guides/how-to-build-agents-md)
- [CLAUDE.md and AGENTS.md, In Depth — redreamality](https://redreamality.com/blog/claude-md-agents-md-deep-dive/)
- [What Is Spec-Driven Development? — Augment Code](https://www.augmentcode.com/guides/what-is-spec-driven-development)
- [Spec-Driven Development (SDD): The Definitive 2026 Guide — BCMS](https://thebcms.com/blog/spec-driven-development)
- [The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture](https://arxiv.org/pdf/2606.27045)
- [Architectural Decision Records with Spec-Driven Development using OpenSpec](https://intent-driven.dev/blog/2026/04/29/spec-driven-development-with-adr/)
- [ADR vs Spec-Driven Development: Why, What, and Using Both](https://ceaksan.com/en/adr-openspec-decision-spec-management)
- [Kiro: Move beyond AI coding to agentic engineering](https://kiro.dev/) / [Kiro Specs docs](https://kiro.dev/docs/specs/)
- [Comprehensive Guide to Spec-Driven Development: Kiro, GitHub Spec Kit, and BMAD-METHOD — Medium](https://medium.com/@visrow/comprehensive-guide-to-spec-driven-development-kiro-github-spec-kit-and-bmad-method-5d28ff61b9b1)
- [Superpowers Framework: TDD Methodology for AI Coding Agents 2026](https://baeseokjae.github.io/posts/superpowers-framework-ai-coding-2026/)
- [Spec + TDD: The Combination That Actually Produces Shippable AI Code — Augment Code](https://www.augmentcode.com/guides/spec-tdd-shippable-ai-generated-code)
- [TDAD: Test-Driven Agentic Development](https://arxiv.org/html/2603.17973v2)
- [Orchestrating AI Agents: A Subagent Architecture](https://clouatre.ca/posts/orchestrating-ai-agents-subagent-architecture/)
- [Safe Multi-Agent Behavior Must Be Maintained, Not Merely Asserted: Constraint Drift in LLM-Based Multi-Agent Systems (arXiv:2605.10481)](https://arxiv.org/abs/2605.10481)
- [Codex vs Claude Code: The Divergence in Subagent Design Philosophy — SmartScope](https://smartscope.blog/en/blog/codex-vs-claude-code-subagent-architecture-2026/)
- [Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems (GitHub / arXiv writeup)](https://github.com/VILA-Lab/Dive-into-Claude-Code)
- [Loop Engineering: The Quiet Revolution in How We Work with AI](https://www.alphamatch.ai/blog/loop-engineering-ai-coding-2026)

---

*Note on repo state observed while researching (context only, not part of the literature review): as of this writing, `AGENTS.md` and `docs/ARCHITECTURE.md` in this repo both describe the pre-refactor "gguf_limit_bench" CLI/WebSocket-cockpit shape, not the detached-engine/thin-client/run-dir-seam design referenced in this task's brief. Recommendation #1 above is written against that observed gap.*
