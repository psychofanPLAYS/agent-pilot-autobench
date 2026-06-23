@echo off
setlocal
cd /d "%~dp0"

echo.
echo pilotBENCHY first run
echo =====================
echo.
echo This installs everything one time, then you just type:  apb
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo First run stopped before setup could finish. See the messages above.
) else (
  echo All done. From any terminal, just type:  apb
)
echo.
pause
exit /b %EXIT_CODE%
