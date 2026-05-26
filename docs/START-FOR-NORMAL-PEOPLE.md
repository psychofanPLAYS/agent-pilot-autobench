# Start Here

This is the easy path.

The new short command is:

```text
pilotbench
```

The old command, `gguf-limit-bench`, still works for older notes and scripts.

## If You Are On Windows

Double-click this file:

```text
START-HERE.bat
```

That is the start button.

It will:

1. Open a black command window.
2. Check that the app can run.
3. Open the model picker.

If something is missing, it will tell you what is missing.

## If You Already Know Terminals

Open a terminal in this folder and run:

```powershell
uv run --extra dev pilotbench --start
```

Check only, without opening the picker:

```powershell
uv run --extra dev pilotbench --start --check-only
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
