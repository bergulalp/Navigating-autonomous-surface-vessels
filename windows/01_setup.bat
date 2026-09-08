@echo off
setlocal
cd /d "%~dp0.."
echo 5.9.1. MARIN - create a local Python 3.12 environment
py -3.12 -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -m pip install -r requirements-tested.txt
if errorlevel 1 goto fail
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
python -m pip install -r requirements-rl-tested.txt
if errorlevel 1 goto fail
python -m pip install -e . --no-deps
if errorlevel 1 goto fail
echo Setup finished. Next run windows\02_check.bat
exit /b 0
:fail
echo Setup stopped. Read the error above and docs\WINDOWS_RL_START.md.
exit /b 1
