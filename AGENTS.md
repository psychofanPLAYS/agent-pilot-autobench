# AGENTS Instructions For Agent Pilot Autobench

## Purpose

This repo is David's local-first autobenchmarking lab for finding useful llama.cpp/GGUF settings for agent workflows. It should produce evidence, receipts, and champion settings rather than vibes.

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
- `local AI probe`: inspect XTREME llama.cpp paths, GGUF models, benchmark commands, GPU/offload evidence, and receipt folders.
- `monitor_worker`: watch long benchmark/test/server output and report only meaningful state changes.
- `reviewer_worker`: critique diffs, benchmark assumptions, Windows path handling, and receipt correctness.
- `docs_writer`: update README, handoffs, beginner docs, command boards, and release notes.
- `dependency_doctor`: inspect `uv`, Python, package, PATH, and lockfile issues.
- `windows_ops`: inspect Windows launch scripts, environment variables, services, and PowerShell behavior.

Use `G:\_codex_global\docs\sidecar-catalogue.md` for the full catalogue.

## Context Workflow

- Start `/goal` for large benchmark campaigns, multi-session changes, or work that needs master/subagent coordination.
- Use Research -> Plan -> Implement for medium or larger changes.
- Keep raw benchmark logs out of the main context. Summarize the command, model, settings, evidence path, and result.
- Keep long sessions token-efficient: use targeted `rg`/path-specific reads, avoid dumping long server logs into chat, and keep sidecar prompts/results compact.
- Recycle sidecars around roughly 75% context use or when logs/research get noisy.
- Require receipts before closing workers: assigned task, files inspected/changed, commands run, findings, verification, blockers, and recommended next action.

## Runtime And Dependency Policy

- Use the newest Python version that makes sense for the main app, after confirming the repo dependencies support it.
- Isolate compatibility-only benchmark harnesses in their own venv when newer Python breaks upstream wheels or native builds.
- Document any older compatibility lane. Example: BFCL may use `.venv-bfcl` on Python 3.11 when newer Python tries to build `tree-sitter` locally.
- Install first-party tooling only when a current failing check proves it is needed; record why it was installed and where it writes data.

## Git Hygiene

- Before ending substantial work, make git reviewable: fetch, inspect status/diff, run the narrow verification checks, stage the intended scope, commit, push, and open a draft PR when David asked for a publish flow.
- If the branch is `main`, create a `codex/...` branch before committing new work unless David explicitly wants direct commits on `main`.
- Prove local/remote state after push with `git rev-list --left-right --count <branch>...origin/<branch>` and `git status --short --branch`.
- Keep generated benchmark receipts, model files, databases, and caches out of commits unless David explicitly asks to publish a specific receipt artifact.

## Project Constraints

- Keep heavy artifacts on `G:\`, especially models, runs, caches, and virtual environments.
- Do not put large downloads or benchmark output on `C:\`.
- Prefer existing llama.cpp tools such as `llama-bench`, `llama-cli`, and later `llama-server` over custom reinvention.
- Preserve Windows usability: `.bat` launchers should be readable, reversible, and safe for David to double-click.
- Keep generated run evidence in `runs\` and local experiment data in `db\`.
- Do not delete benchmark runs, model files, or local databases without explicit approval.

## Verification

Use the narrowest practical checks:

```powershell
uv run --extra dev pytest -q
uv run --extra dev python -m compileall src tests
uv run --extra dev agent-autobench first-run
uv run --extra dev agent-autobench results
```

For launcher changes, inspect and test the `.bat` path on Windows instead of assuming shell behavior.
