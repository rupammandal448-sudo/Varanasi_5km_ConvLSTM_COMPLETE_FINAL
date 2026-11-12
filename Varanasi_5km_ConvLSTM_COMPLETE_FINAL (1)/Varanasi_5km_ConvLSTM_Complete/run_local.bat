@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo   VARANASI NOWCAST - LOCAL WEB SERVER
echo ==========================================
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.10 or newer first.
  pause
  exit /b 1
)
py -c "import flask, requests, dotenv" >nul 2>nul
if errorlevel 1 (
  echo Required Python packages are missing.
  echo Run this command once from this folder:
  echo   py -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Starting the website at http://127.0.0.1:8000
echo Keep this window open while using the site.
echo Press Ctrl+C to stop the server.
echo.
start "" cmd /c "timeout /t 2 ^>nul ^& start http://127.0.0.1:8000"
py server.py
if errorlevel 1 (
  echo.
  echo The server stopped with an error. Check the messages above.
  pause
)
endlocal
