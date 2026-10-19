@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title fgPROMS

echo.
echo   ==========================================
echo     fgPROMS - process modelling studio
echo   ==========================================
echo.

rem ---- locate Python ----
set "PYCMD="
py -3.14 -c "import sys" >nul 2>nul && set "PYCMD=py -3.14"
if not defined PYCMD py -3.10 -c "import sys" >nul 2>nul && set "PYCMD=py -3.10"
if not defined PYCMD python -c "import sys" >nul 2>nul && set "PYCMD=python"
if not defined PYCMD (
  echo [ERROR] Python was not found on this computer.
  echo Please install Python 3 from https://www.python.org/downloads/
  pause
  exit /b 1
)
echo Python launcher: %PYCMD%

rem ---- virtual environment ----
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYCMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create the virtual environment.
    pause
    exit /b 1
  )
)
set "VENVPY=.venv\Scripts\python.exe"

rem ---- dependencies ----
%VENVPY% -c "import fastapi, uvicorn, scipy, numpy" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies ^(first run only, please wait^)...
  %VENVPY% -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Package installation failed. Check your internet connection.
    pause
    exit /b 1
  )
)

rem ---- start ----
echo Starting fgPROMS...
%VENVPY% launch.py
pause
