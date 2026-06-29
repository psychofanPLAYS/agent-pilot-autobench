@echo off
setlocal
cd /d "%~dp0"

echo.
echo Installing Agent Pilot. This sets up everything one time.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo Setup stopped before it finished. See the messages above.
) else (
  echo All set. Open a new terminal and type:  apb
)
echo.
pause
exit /b %EXIT_CODE%
