@echo off
title NeuroTrader API Adapter (:8000)
cd /d "%~dp0"
echo =======================================================
echo Starting NeuroTrader API Adapter on http://127.0.0.1:8000
echo =======================================================
.\api_env\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
