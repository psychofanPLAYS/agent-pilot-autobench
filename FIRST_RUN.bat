@echo off
setlocal
cd /d "%~dp0"

echo.
echo pilotBENCHY first run
echo =====================
echo.
echo This installs the local command, adds apb to PATH, then opens the model picker.
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

call %AGENT_CMD% --first-run
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo First run stopped before benchmarking could begin.
) else (
  echo All done. Next time you can run: apb --start
)
echo.
pause
exit /b %EXIT_CODE%
