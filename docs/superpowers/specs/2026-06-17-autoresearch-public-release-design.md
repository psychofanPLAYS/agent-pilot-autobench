# Autoresearch Public Release Design

## Objective

Turn the existing `codex/pilotbenchy-first-run-reports` branch and draft PR #9 into a
public, portfolio-grade release candidate for Agent Pilot Autobench. The work must
remain local-first, preserve existing benchmark evidence, and make the repository
easy for an evaluator to understand, run, verify, and trust.

## Publication Strategy

- Keep the existing branch and PR #9 as the single publication vehicle.
- Preserve ignored local models, databases, environments, and benchmark receipts.
- Stage only reviewed source, tests, documentation, metadata, and small public assets.
- Use thematic commits so the PR tells a coherent engineering story.
- Retitle and rewrite PR #9 after the final diff is known.
- Do not merge, tag, or create a GitHub release without a separate explicit decision.

## Release Gates

### Product and portfolio

- The README first screen explains the problem, audience, first command, and proof.
- Beginner and automation-oriented workflows are both copy-pasteable.
- A compact architecture/code map connects commands, modules, outputs, and tests.
- Public metadata accurately describes the package, supported Python versions,
  license, repository, keywords, and project status.
- Limitations and benchmark interpretation are explicit; no unsupported performance
  or production-readiness claims are allowed.

### Engineering

- The full test suite passes on supported Python versions through GitHub Actions.
- Ruff formatting and linting, MyPy, compilation, package build, wheel installation,
  and CLI smoke checks pass locally and in CI where practical.
- The SimpleBench flag ladder has deterministic answer parsing, bounded timeouts,
  validated inputs, failure receipts, and package-safe bundled assets.
- Runtime output remains under ignored project-local folders on `G:` by default.

### Safety and reproducibility

- No secrets, private hostnames, user-profile paths, heavyweight artifacts, or local
  benchmark receipts are committed.
- User-supplied llama.cpp arguments remain explicit in receipts and never use a shell.
- Server defaults remain bound to localhost.
- The release documents exact verification commands and generated artifact paths.

### GitHub

- PR #9 contains intentional commits and an evidence-backed description.
- All required GitHub Actions checks pass after the final push.
- The PR is marked ready for review only after local and remote gates agree.

## Architecture

The existing Python package remains the product boundary. `cli.py` orchestrates
commands; `autoresearch.py` owns the search loop and receipts; `flag_ladder.py` owns
independent llama.cpp profiles; `simple_bench.py` owns dataset/scoring semantics; and
`simple_bench_runner.py` owns one benchmark-controlled server lifecycle. Generated
receipts remain the audit trail, while public docs explain how those pieces connect.

## Error Handling

Invalid datasets and invalid numeric options fail before a model server starts. A
server launch, readiness, request, stream, timeout, or scoring failure produces a
bounded failure result and receipt instead of silently selecting a broken profile.
Interrupted or failed profiles do not erase successful earlier evidence.

## Verification Evidence

Release readiness is proved by command output and GitHub state, not by prose:

- `uv run --extra dev python -m pytest -q`
- `uv run --extra dev ruff format --check .`
- `uv run --extra dev ruff check .`
- `uv run --extra dev mypy src`
- `uv run --extra dev python -m compileall -q src tests`
- `uv build`
- isolated wheel import and CLI smoke checks
- public-safety scans and `git diff --check`
- PR #9 check rollup after the final push

## Out of Scope

- Uploading models or benchmark receipts.
- Publishing to PyPI.
- Changing repository visibility, branch protection, or GitHub secrets.
- Merging PR #9, tagging a version, or creating a GitHub release without explicit
  approval after the release-candidate audit.
