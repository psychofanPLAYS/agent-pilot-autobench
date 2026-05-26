@echo off
setlocal
cd /d "%~dp0"

echo.
echo Agent Pilot Autobench
echo =====================
echo.
echo This opens the easy model picker.
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

uv run --extra dev pilotbench --start
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
