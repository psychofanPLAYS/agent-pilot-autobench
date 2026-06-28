# Agent Pilot To Do

## Active Release Goal

- [x] Harden SimpleBench input validation, timeouts, receipts, and package assets.
- [x] Add MyPy, package build, isolated-wheel, and Linux/Python CI gates.
- [x] Add a Windows launcher and CLI smoke job.
- [x] Add public architecture, security, contribution, support, and community files.
- [x] Add a sanitized dry-run artifact without publishing private benchmark history.
- [ ] Run the live flag ladder after the shared LLM services are available.
- [ ] Push the final branch, verify fresh GitHub checks, and mark PR #9 ready for review.

## Deferred Live Acceptance

Start with a short controlled run using one representative model:

```powershell
apb autoresearch --model "G:\models\model.gguf" --llama-server "G:\llama.cpp\llama-server.exe" --flag-ladder --budget-minutes 20 --parallel-max 6
```

Keep the generated receipt and confirm accuracy, TTFT, throughput, warnings, failure
classification, and exact launch commands before considering a stable release.

## Later UX Polish

- Add a custom Windows icon for the first-run launcher.
- Add `apb -q` / `apb --quick` as a short scout mode for the last selected models.
- Add `apb -a` / `apb --all` as a short way to benchmark all discovered models.
- Add a richer campaign mode for "10 random models" or "all models in a given family/size".
- Add a sample public corpus recommendation for perplexity profiling.
- Add screenshots or a small GIF after a sanitized live report exists.
