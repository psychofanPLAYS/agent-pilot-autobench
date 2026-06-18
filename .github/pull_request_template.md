## What changed

Describe the user-visible behavior and why this is the smallest correct change.

## Evidence

- [ ] Tests added or updated for behavior changes
- [ ] `uv run --extra dev python -m pytest -q`
- [ ] `uv run --extra dev ruff format --check .`
- [ ] `uv run --extra dev ruff check .`
- [ ] `uv run --extra dev mypy src`
- [ ] `uv build`

## Safety and publication check

- [ ] No models, receipts, databases, virtual environments, secrets, or private paths are included
- [ ] New server behavior stays bound to localhost by default
- [ ] Benchmark claims identify their evidence and limitations
- [ ] Documentation and generated-artifact formats are updated when needed
