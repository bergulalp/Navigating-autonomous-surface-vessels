@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe goto missing
call .venv\Scripts\activate.bat
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set MARIN_TAG=%~1
if "%MARIN_TAG%"=="" set MARIN_TAG=pilot
python -m usv_bio.learning_eval --config configs/bio_short.json --out runs/test_%MARIN_TAG% --split test --selection runs/selection_%MARIN_TAG%.json --checkpoints runs/windows_seed0/final.zip --seeds 20 --seed-start 19000 --workers 1
if errorlevel 1 exit /b 1
python scripts/analyze_v4_learning.py --data runs/test_%MARIN_TAG% --selection runs/selection_%MARIN_TAG%.json --out runs/test_%MARIN_TAG%/analysis
exit /b %errorlevel%
:missing
echo Run windows\01_setup.bat first.
exit /b 1
