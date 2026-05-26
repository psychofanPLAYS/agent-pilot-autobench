@echo off
setlocal
cd /d "%~dp0"

echo.
echo Agent Pilot Autobench
echo =====================
echo.
echo This opens the easy first-run flow.
echo You can close this window any time.
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

uv run --extra dev agent-autobench --help >nul 2>nul
if "%ERRORLEVEL%"=="0" (
  uv run --extra dev agent-autobench first-run
  if not errorlevel 1 (
    uv run --extra dev agent-autobench --start
  )
) else (
  echo The agent-autobench command is not available in this checkout.
  echo Falling back to the older pilotbench startup command.
  echo.
  uv run --extra dev pilotbench --start
)
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo Something stopped before the picker could open.
  echo Read the message above. It usually tells you what file or folder is missing.
) else (
  echo All done.
)
echo.
pause
exit /b %EXIT_CODE%
