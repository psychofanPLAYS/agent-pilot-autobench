# Agent Pilot one-command installer.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Installs everything a fresh machine needs, then leaves you with a single
# command: `apb`. It auto-installs uv if missing, builds the local virtual
# environment, pulls dependencies, and puts `apb` on your PATH.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host ""
Write-Host "Agent Pilot setup" -ForegroundColor Cyan
Write-Host "================="
Write-Host ""

# 1. Ensure uv (the Python toolchain) is available.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (one time)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer drops uv here; make it usable in this same session.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 2. Build the virtual environment and pull dependencies.
Write-Host "Building the environment and pulling dependencies..."
uv sync --extra dev --extra bench

# 3. Install the `apb` command, add it to PATH, and prepare local state.
Write-Host "Installing the apb command..."
uv run apb setup

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Open a NEW terminal and type:  apb"
Write-Host ""
