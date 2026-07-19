@echo off
REM Credit Scoring Model - Windows Batch Runner

echo ==========================================
echo Credit Scoring Model Pipeline
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo.
echo [1/4] Checking dependencies...
pip install -q -r requirements.txt

REM Generate sample data if not exists
echo.
echo [2/4] Generating sample data...
python scripts/generate_data.py -n 2000 -o data/raw/credit_data.csv

REM Run pipeline
echo.
echo [3/4] Running pipeline...
python -m src.pipeline.run_pipeline --data data/raw/credit_data.csv --output results/experiment_1

REM Test predictions
echo.
echo [4/4] Testing predictions...
python scripts/generate_data.py -n 10 -o data/raw/new_applications.csv
python -m src.pipeline.run_pipeline --predict data/raw/new_applications.csv --model results/experiment_1/best_model.pkl --preprocessor results/experiment_1/preprocessor.pkl

echo.
echo ==========================================
echo Pipeline completed successfully!
echo Results saved to: results/experiment_1/
echo ==========================================
pause