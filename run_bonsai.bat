@echo off
title Bonsai Micro-AI - Local Web Chat
color 0A

echo ====================================================================
echo    ____   ____  _   _  _____   ___   ___ 
echo   (  _ \ / ___)( ) ( )( ____) / _ \ / _ \
echo    ) _ ( \___ \| \_/ | |__   ( (_) | (_) )
echo   (____/ (____/ \___/ (____)  \___/ \___/ 
echo.
echo   The art of cultivating massive foundation models into pocket-sized
echo   intelligences on consumer silicon.
echo ====================================================================
echo.

cd /d "D:\bonsai"

echo [1/3] Checking Python environment...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found on PATH! Please install Python 3.10+
    pause
    exit /b 1
)

echo [2/3] Initializing Bonsai Micro-AI onto NVIDIA RTX 2050 GPU...
echo       Model: Qwen2.5-Coder-0.5B-Instruct (Zero-OOM Protected)
echo       Adapter: D:\bonsai\bonsai_pruned\adapter
echo.
echo [3/3] Launching Web UI at http://localhost:7860 ...
echo       (Press Ctrl+C in this window anytime to stop the server)
echo.

python -m bonsai.web_server

pause
