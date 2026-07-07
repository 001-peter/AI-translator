@echo off
title Antigravity Translation Studio Launcher
echo ======================================================================
echo             🔮 Antigravity NMT Translation Studio Launcher
echo ======================================================================
echo.

:: 1. Launch FastAPI Server in a separate Command Prompt window
echo [SYSTEM] Launching local Translation Server backend on http://127.0.0.1:8000 ...
start "Antigravity Translation Backend" cmd /k ".\python-portable\tools\python.exe server.py"

:: 2. Give the model/API a few seconds to start initializing
echo [SYSTEM] Waiting 5 seconds for backend initialization...
timeout /t 5 /nobreak > nul

:: 3. Launch Streamlit UI in the current window
echo [SYSTEM] Launching Streamlit Web UI ...
echo [SYSTEM] Press Ctrl+C in the backend window to stop the server when finished.
echo.
.\python-portable\tools\python.exe -m streamlit run app.py

pause
