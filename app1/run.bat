@echo off
rem ══════════════════════════════════════════════════════════════════════
rem   run.bat  –  start Streamlit front-end for the Spark ML lyrics model
rem ══════════════════════════════════════════════════════════════════════
setlocal

rem 1) Path of the active conda-env python
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)"') do (
    set "PYTHON_EXE=%%P"
)

rem 2) Make sure required packages exist (quiet install is fine)
%PYTHON_EXE% -m pip install --quiet --disable-pip-version-check numpy streamlit

rem 3) Tell Spark (inside webui.py) to use *this* interpreter
set "PYSPARK_DRIVER_PYTHON=%PYTHON_EXE%"
set "PYSPARK_PYTHON=%PYTHON_EXE%"

rem 4) Launch the Streamlit server (headless flag avoids extra prompt)
%PYTHON_EXE% -m streamlit run "%~dp0webui.py" --server.headless true

endlocal
