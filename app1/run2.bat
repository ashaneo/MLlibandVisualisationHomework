@echo off
REM ───────────────────────────────────────────────────────────────
REM  run.bat – Launch the Dockerized Spark-ML Lyrics UI
REM ───────────────────────────────────────────────────────────────
setlocal

REM 1) Make sure Docker is running
docker version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker doesn^t seem to be running. Please start Docker Desktop.
  pause
  exit /b 1
)

REM 2) Try pulling from Docker Hub; fallback to building locally
set "IMAGE=ashansubodha1/200623p-lyrics-classify:latest"
echo Pulling %IMAGE% ...
docker pull %IMAGE% 2>nul || (
  echo [INFO] Pull failed – building image locally as %IMAGE% ...
  docker build -t %IMAGE% "%~dp0"
)

REM 3) Remove any old container & run the new one
docker rm -f lyrics-ui 2>nul
echo Starting container on http://localhost:8506 ...
docker run -d --name lyrics-ui -p 8501:8501 %IMAGE%

REM 4) Open the browser
start http://localhost:8501

endlocal
