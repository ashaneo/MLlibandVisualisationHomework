@echo off
rem ───── run.bat that boots conda env automatically ─────────────
set "ENV_NAME=music-genre"
set "ENV_PY=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe"
if not exist "%ENV_PY%" (
    echo [ERROR] Cannot find python for env %ENV_NAME%.
    echo         Please install / rename the env or run this batch
    echo         from an activated Anaconda Prompt.
    pause
    exit /b 1
)

rem 1) install deps inside the env (idempotent)
"%ENV_PY%" -m pip install --quiet numpy streamlit pyspark

rem 2) point Spark to that interpreter
set PYSPARK_DRIVER_PYTHON=%ENV_PY%
set PYSPARK_PYTHON=%ENV_PY%

rem 3) launch Streamlit
"%ENV_PY%" -m streamlit run "%~dp0webui.py" --server.headless true
