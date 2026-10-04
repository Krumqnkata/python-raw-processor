@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Install 64-bit Python 3.12 and try again.
        pause
        exit /b 1
    )
)
if not exist ".venv\raw-studio-deps-v3" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check the message above.
        pause
        exit /b 1
    )
    type nul > ".venv\raw-studio-deps-v3"
)
".venv\Scripts\python.exe" raw_processor.py
if errorlevel 1 pause
endlocal
