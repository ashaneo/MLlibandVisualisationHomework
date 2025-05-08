@echo off
REM ───────────────────────────────────────────────────────────────
REM run.bat — setup venv, install deps, align PySpark Python, launch Streamlit
REM ───────────────────────────────────────────────────────────────
setlocal

REM 1) Create the virtual-env folder if it doesn’t exist
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

REM 3) Install (or upgrade) dependencies with logs
echo [INFO] Installing Python dependencies…
pip install --upgrade pip
pip install --disable-pip-version-check -r requirements.txt

REM 4) Ensure setuptools is present
pip install --disable-pip-version-check setuptools

REM 5) Align Spark’s Python (driver and workers) to this same interpreter
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)"') do (
    set "PYTHON_EXE=%%P"
)
echo [INFO] Using Python interpreter at: %PYTHON_EXE%
set PYSPARK_DRIVER_PYTHON=%PYTHON_EXE%
set PYSPARK_PYTHON=%PYTHON_EXE%

REM 6) Launch the Streamlit front-end
echo [INFO] Starting Streamlit app…
streamlit run webui.py

endlocal
