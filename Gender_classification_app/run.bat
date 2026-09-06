@echo off
echo Starting Gender Classification App...
cd /d "%~dp0backend"
C:\Users\LENOVO\Anaconda3\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
