@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe goto missing
call .venv\Scripts\activate.bat
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
python -m usv_research.ppo check --config configs/train.json --out runs/windows_check
if errorlevel 1 exit /b 1
echo Checks passed. Next: windows\03_train.bat 2048 1 0
exit /b 0
:missing
echo Run windows\01_setup.bat first.
exit /b 1
