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

set "LOCAL_AGENT=%CD%\.venv\Scripts\agent-autobench.exe"
if exist "%LOCAL_AGENT%" (
  set "AGENT_CMD="%LOCAL_AGENT%""
) else (
  where uv >nul 2>nul
  if errorlevel 1 (
    echo I could not find uv, and the local .venv command is not installed yet.
    echo.
    echo Please install uv first:
    echo https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
  )
  set "AGENT_CMD=uv run --extra dev --extra bench agent-autobench"
)

call %AGENT_CMD% --help >nul 2>nul
if "%ERRORLEVEL%"=="0" (
  call %AGENT_CMD% first-run
  if not errorlevel 1 (
    call %AGENT_CMD% --start
  )
) else (
  echo The agent-autobench command is not available in this checkout.
  echo.
  if exist "%CD%\.venv\Scripts\pilotbench.exe" (
    "%CD%\.venv\Scripts\pilotbench.exe" --start
  ) else (
    echo Falling back to the older pilotbench startup command through uv.
    echo.
    uv run --extra dev --extra bench pilotbench --start
  )
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
