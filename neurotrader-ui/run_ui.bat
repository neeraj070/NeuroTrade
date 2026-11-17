@echo off
title NeuroTrader Dashboard UI (:3000)
cd /d "%~dp0"
echo =======================================================
echo Starting NeuroTrader UI Frontend on http://localhost:3000
echo =======================================================
npm run dev -- --port 3000
pause
