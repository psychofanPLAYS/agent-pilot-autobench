# Start

Use one first-run path, then one everyday start command.

On Windows, first double-click:

```text
FIRST_RUN.bat
```

That file sets up the app, installs the `apb` command, and opens the browser cockpit when the machine checks pass.

## Terminal Path

From the repo folder:

```powershell
uv run --extra dev --extra bench agent-autobench --first-run
```

After setup adds `_bin` to your user PATH, open a new terminal and run:

```powershell
apb --start
```

Plain `apb` also opens the browser cockpit at `http://127.0.0.1:36939/` after
setup. The browser is the primary workflow for model selection,
benchmark-suite plan selection, live run progress, telemetry, and receipt links.
Use `apb tui` only when you specifically want the older fallback terminal cockpit.

In the browser cockpit, pick models, choose the test type, review the recommended
forced flags, and start the run. The page uses a local WebSocket connection to
show live activity, telemetry, the current best model for this machine, and links
to the receipts when the run finishes.

The longer command also works:

```powershell
agent-autobench --start
```

## What Setup Does

- syncs the local `.venv`
- creates `_bin\agent-autobench.bat`
- creates `_bin\apb.bat`
- adds `_bin` to your Windows user PATH when supported
- prepares `_db`
- prepares `_runs`
- checks model and llama.cpp paths

If setup says something is missing, run:

```powershell
agent-autobench doctor
```

## Where Results Go

The app writes receipts and reports under:

```text
_runs\
```

The browser cockpit links to `_runs\results.html`, `_runs\leaderboard.md`, and
recent per-run reports directly. You can still open `_runs\results.html` by hand
for the cross-run browser report or `_runs\leaderboard.md` for the compact text
version.

You can also ask the app to open the report:

```powershell
agent-autobench results --open-browser
```

Each individual run folder also has `report.html` and `itemized-report.md`.
