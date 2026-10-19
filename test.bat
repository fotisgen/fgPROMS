@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title fgPROMS tests

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found. Run run.bat once first.
  pause
  exit /b 1
)
set "VENVPY=.venv\Scripts\python.exe"

%VENVPY% -c "import pytest, httpx" >nul 2>nul
if errorlevel 1 (
  echo Installing test dependencies...
  %VENVPY% -m pip install --disable-pip-version-check -q -r requirements-dev.txt
)

rem ---- Chromium for the GUI tests (one time) ----
if not exist "%LOCALAPPDATA%\ms-playwright" (
  echo Installing Chromium for GUI tests ^(one time, ~150 MB^)...
  %VENVPY% -m playwright install chromium
)

echo.
echo   ==========================================
echo     fgPROMS - automated test suite
echo   ==========================================
echo.
%VENVPY% -m pytest tests -v --tb=short
echo.
pause
