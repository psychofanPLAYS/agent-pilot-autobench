# AGENTS Instructions For Agent Pilot Autobench

## Purpose

This repository is a local-first benchmarking lab for finding useful llama.cpp/GGUF
settings for agent workflows. It should produce evidence, receipts, and champion
settings rather than vibes.

## Operating Rule

For any large, multi-file, benchmark-heavy, local-AI, or multi-session task in this repo, use sidecar/subagent dispatch early when the tool is available.

Do not only write a plan saying "subagents could help." Actually dispatch a bounded sidecar unless:

- the task is tiny
- the subagent tool is unavailable
- the next step is a small local inspection that would be slower to delegate
- the work requires immediate user clarification
- write scopes would overlap unsafely

If no sidecar is used for a big task, state the concrete reason.

## Preferred Sidecars

- `repo_reader`: map the codebase, CLI commands, tests, run receipts, config, and likely edit points.
- `local AI probe`: inspect llama.cpp paths, GGUF models, benchmark commands, GPU/offload evidence, and receipt folders.
- `monitor_worker`: watch long benchmark/test/server output and report only meaningful state changes.
- `reviewer_worker`: critique diffs, benchmark assumptions, Windows path handling, and receipt correctness.
- `docs_writer`: update README, handoffs, beginner docs, command boards, and release notes.
- `dependency_doctor`: inspect `uv`, Python, package, PATH, and lockfile issues.
- `windows_ops`: inspect Windows launch scripts, environment variables, services, and PowerShell behavior.

Use only agent capabilities available in the current development environment.

## Context Workflow

- Start `/goal` for large benchmark campaigns, multi-session changes, or work that needs master/subagent coordination.
- Always use context engineering, scaled to task size: research code-derived truth first, plan when more than a tiny one-shot, implement from narrow context, verify, then summarize.
- Use sidecars/subagents for context isolation and delegation when the work fits and tool policy allows it. Choose the proper level agent for the lane: light scout/monitor for cheap bounded work, medium/strong implementation or debugger for hard slices, reviewer for critique, docs writer for receipts.
- Keep raw benchmark logs out of the main context. Summarize the command, model, settings, evidence path, and result.
- Keep long sessions token-efficient: use targeted `rg`/path-specific reads, avoid dumping long server logs into chat, and keep sidecar prompts/results compact.
- Recycle sidecars around roughly 75% context use or when logs/research get noisy.
- Require receipts before closing workers: assigned task, files inspected/changed, commands run, findings, verification, blockers, and recommended next action.

## Runtime And Dependency Policy

- Use the newest Python version that makes sense for the main app, after confirming the repo dependencies support it.
- Isolate compatibility-only benchmark harnesses in their own venv when newer Python breaks upstream wheels or native builds.
- Document any older compatibility lane. Example: BFCL may use `.venv-bfcl` on Python 3.11 when newer Python tries to build `tree-sitter` locally.
- Install first-party tooling only when a current failing check proves it is needed; record why it was installed and where it writes data.
- Keep optional local-AI helpers outside the product contract. Tests and documented
  commands must not depend on private services or a particular workstation.

## Git Hygiene

- Before ending substantial work, make git reviewable: fetch, inspect status/diff, run the narrow verification checks, stage the intended scope, commit, and publish only when requested.
- If the branch is `main`, create a `codex/...` branch before committing new work unless direct commits were explicitly requested.
- Prove local/remote state after push with `git rev-list --left-right --count <branch>...origin/<branch>` and `git status --short --branch`.
- Keep generated benchmark receipts, model files, databases, and caches out of commits unless a specific small public artifact is intentionally approved.

## Project Constraints

- Keep heavy artifacts in the configured project-local ignored directories or another
  explicitly selected data drive.
- Do not place large downloads or benchmark output in system-managed directories.
- Prefer existing llama.cpp tools such as `llama-bench`, `llama-cli`, and later `llama-server` over custom reinvention.
- Preserve Windows usability: `.bat` launchers should be readable, reversible, and safe to double-click.
- Keep generated run evidence in `runs\` and local experiment data in `db\`.
- Do not delete benchmark runs, model files, or local databases without explicit approval.

## Verification

Use the narrowest practical checks:

```powershell
uv run --extra dev python -m pytest -q
uv run --extra dev python -m compileall src tests
uv run --extra dev agent-autobench first-run
uv run --extra dev agent-autobench results
```

For launcher changes, inspect and test the `.bat` path on Windows instead of assuming shell behavior.
