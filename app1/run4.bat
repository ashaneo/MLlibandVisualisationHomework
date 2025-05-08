@echo off
REM ───────────────────────────────────────────────────────────────
REM run.bat — Create/activate venv, install deps, and launch Streamlit
REM ───────────────────────────────────────────────────────────────
setlocal

REM 1) If no virtual-env folder, check Python and create it
if not exist "venv" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python and add it to PATH.
        pause
        exit /b 1
    )
    echo [INFO] Creating virtual environment…
    python -m venv venv
)

REM 2) Activate the venv
call venv\Scripts\activate.bat

REM 3) Install requirements quietly
if exist "requirements.txt" (
    echo [INFO] Installing Python dependencies…
    pip install -q --disable-pip-version-check -r requirements.txt
) else (
    echo [WARN] requirements.txt not found...\
)

REM 4) Ensure setuptools is installed
pip install -q --disable-pip-version-check setuptools

REM 5) Launch the Streamlit app
echo [INFO] Starting Streamlit…
@REM cd /d front-end\src
streamlit run webui.py

endlocal
