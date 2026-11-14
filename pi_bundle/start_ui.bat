@echo off
cd /d "%~dp0"
start "" http://localhost:8000
"..\.venv\Scripts\python.exe" ui_server.py 8000
pause
