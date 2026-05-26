# Start Here

This is the easy path.

The new short command is:

```text
agent-autobench
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
2. Run the beginner startup check.
3. Open the model picker when the checks are good.

If something is missing, it will tell you what is missing.

## If You Already Know Terminals

Open a terminal in this folder and run:

```powershell
uv run --extra dev agent-autobench first-run
```

Check only, without opening the picker:

```powershell
uv run --extra dev agent-autobench --start --check-only
```

The older compatibility command also works:

```powershell
uv run --extra dev pilotbench --start
```

## Make The Command Work From Anywhere

Double-click:

```text
INSTALL-COMMAND.bat
```

That creates a small Windows command file named:

```text
G:\_codex_global\bin\agent-autobench.bat
```

The installer asks before it changes your user PATH. PATH is the Windows list of
folders where commands can be found. If you say yes, open a new terminal and run:

```powershell
agent-autobench first-run
```

## What To Do In The Picker

Use the picker to choose a model from your GGUF model folder.

The app is meant to write small receipt files under:

```text
runs\
```

Those receipts are the proof of what happened.

## If It Says `uv` Is Missing

Install `uv` first:

```text
https://docs.astral.sh/uv/getting-started/installation/
```

Then double-click `START-HERE.bat` again.
