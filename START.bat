@echo off
setlocal
cd /d "%~dp0"

echo.
echo pilotBENCHY
echo ===========
echo.
echo This opens the model picker. For a new install, run FIRST_RUN.bat first.
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

call %AGENT_CMD% --start
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo Something stopped before the picker finished.
) else (
  echo All done.
)
echo.
pause
exit /b %EXIT_CODE%
