@echo off
cd /d C:\workspace\chatbot_project
call venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
