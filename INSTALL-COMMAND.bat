@echo off
setlocal
cd /d "%~dp0"

set "REPO_DIR=%CD%"
set "SHIM_DIR=G:\_codex_global\bin"
set "SHIM_FILE=%SHIM_DIR%\agent-autobench.bat"

echo.
echo Install agent-autobench command
echo ===============================
echo.
echo This will create:
echo %SHIM_FILE%
echo.
echo The shim will run this repo:
echo %REPO_DIR%
echo.
echo No admin rights are needed.
echo This script will ask before changing your user PATH.
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo I could not find uv on this computer.
  echo.
  echo Please install uv first:
  echo https://docs.astral.sh/uv/getting-started/installation/
  echo.
  pause
  exit /b 1
)

if not exist "%SHIM_DIR%" mkdir "%SHIM_DIR%"
if errorlevel 1 (
  echo Could not create:
  echo %SHIM_DIR%
  echo.
  pause
  exit /b 1
)

(
  echo @echo off
  echo cd /d "%REPO_DIR%"
  echo uv run --extra dev agent-autobench %%*
) > "%SHIM_FILE%"
if errorlevel 1 (
  echo Could not write:
  echo %SHIM_FILE%
  echo.
  pause
  exit /b 1
)

echo Created command shim.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir = $env:SHIM_DIR; $old = [Environment]::GetEnvironmentVariable('Path', 'User'); if (($old -split ';') -contains $dir) { exit 0 } exit 1"
if not errorlevel 1 (
  echo Your current PATH already includes:
  echo %SHIM_DIR%
  echo.
  echo You can run:
  echo agent-autobench first-run
  echo.
  pause
  exit /b 0
)

echo To make agent-autobench work from any terminal folder, Windows needs this
echo folder in your user PATH:
echo %SHIM_DIR%
echo.
choice /C YN /N /M "Add it to your user PATH now? [Y/N] "

if errorlevel 2 (
  echo.
  echo PATH was not changed.
  echo You can still run the shim directly:
  echo %SHIM_FILE% first-run
  echo.
  pause
  exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir = $env:SHIM_DIR; $old = [Environment]::GetEnvironmentVariable('Path', 'User'); if ([string]::IsNullOrWhiteSpace($old)) { $new = $dir } elseif (($old -split ';') -contains $dir) { $new = $old } else { $new = $old.TrimEnd(';') + ';' + $dir }; [Environment]::SetEnvironmentVariable('Path', $new, 'User')"
if errorlevel 1 (
  echo.
  echo Could not update the user PATH.
  echo The shim was still created here:
  echo %SHIM_FILE%
  echo.
  pause
  exit /b 1
)

echo.
echo User PATH updated.
echo Close and reopen your terminal, then run:
echo agent-autobench first-run
echo.
pause
exit /b 0
