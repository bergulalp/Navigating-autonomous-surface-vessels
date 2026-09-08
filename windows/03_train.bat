@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe goto missing
call .venv\Scripts\activate.bat
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set MARIN_STEPS=%~1
set MARIN_WORKERS=%~2
set MARIN_SEED=%~3
if "%MARIN_STEPS%"=="" set MARIN_STEPS=2048
if "%MARIN_WORKERS%"=="" set MARIN_WORKERS=1
if "%MARIN_SEED%"=="" set MARIN_SEED=0
python scripts/windows_rl.py train --steps %MARIN_STEPS% --workers %MARIN_WORKERS% --seed %MARIN_SEED% --out runs/windows_seed%MARIN_SEED%
exit /b %errorlevel%
:missing
echo Run windows\01_setup.bat first.
exit /b 1
