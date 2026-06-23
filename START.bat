@echo off
setlocal
cd /d "%~dp0"

echo.
echo pilotBENCHY
echo ===========
echo.
echo Opening the app. If this is a brand new copy, run INSTALL.bat first.
echo.

set "LOCAL_APB=%CD%\.venv\Scripts\apb.exe"
if exist "%LOCAL_APB%" (
  "%LOCAL_APB%"
) else (
  uv run --extra dev --extra bench apb
)
set EXIT_CODE=%ERRORLEVEL%

echo.
pause
exit /b %EXIT_CODE%
