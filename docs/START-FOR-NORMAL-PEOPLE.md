# Start Here

This is the easy path.

The new short command is:

```text
agent-autobench
```

After installing shortcuts, the tiny command also works:

```text
apb
```

The easiest command is:

```text
agent-autobench first-run
```

## If You Are On Windows

Double-click this file:

```text
START-HERE.bat
```

That is the start button.

It will:

1. Open a black command window.
2. Use the local `.venv` command when it is already installed.
3. Fall back to `uv` when the local command is not installed yet.
4. Run the first-time installer.
5. Create the command shortcuts under `G:\_codex_global\bin`.
6. Open the model picker when the checks are good.

If something is missing, it will tell you what is missing in normal language.

## Best First Command

Double-click this:

```text
START-HERE.bat
```

That is easier than typing commands by hand.

For a terminal, run:

```powershell
.\.venv\Scripts\agent-autobench.exe first-run
```

If the local `.venv` is not installed yet, use:

```powershell
uv run --extra dev agent-autobench first-run
```

## Check Only

This checks the computer without opening the model picker:

```powershell
.\.venv\Scripts\agent-autobench.exe --start --check-only
```

Or through `uv`:

```powershell
uv run --extra dev agent-autobench --start --check-only
```

## If You Already Know Terminals

The direct local command is:

```powershell
.\.venv\Scripts\agent-autobench.exe first-run
```

The older compatibility command also works:

```powershell
.\.venv\Scripts\pilotbench.exe --start
```

## Make The Command Work From Anywhere

Double-click:

```text
INSTALL-COMMAND.bat
```

That creates a small Windows command file named:

```text
G:\_codex_global\bin\agent-autobench.bat
G:\_codex_global\bin\apb.bat
```

The `first-run` command already creates those files. The `.bat` installer is the
double-click helper for adding that folder to your user PATH.

PATH is the Windows list of folders where commands can be found. If you say yes,
open a new terminal and run:

```powershell
agent-autobench first-run
```

Or the shorter version:

```powershell
apb first-run
```

## What To Do In The Picker

Use the picker to choose a model from your GGUF model folder.

The app is meant to write small receipt files under:

```text
runs\
```

Those receipts are the proof of what happened.

## If It Says `uv` Is Missing

That means the local `.venv` command is not installed yet either.
Install `uv` first:

```text
https://docs.astral.sh/uv/getting-started/installation/
```

Then double-click `START-HERE.bat` again.
