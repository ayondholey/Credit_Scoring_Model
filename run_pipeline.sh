#!/bin/bash
# Credit Scoring Model - Linux/Mac Shell Runner

set -e

echo "=========================================="
echo "Credit Scoring Model Pipeline"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found in PATH"
    exit 1
fi

echo ""
echo "[1/4] Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "[2/4] Generating sample data..."
python scripts/generate_data.py -n 2000 -o data/raw/credit_data.csv

echo ""
echo "[3/4] Running pipeline..."
python -m src.pipeline.run_pipeline --data data/raw/credit_data.csv --output results/experiment_1

echo ""
echo "[4/4] Testing predictions..."
python scripts/generate_data.py -n 10 -o data/raw/new_applications.csv
python -m src.pipeline.run_pipeline --predict data/raw/new_applications.csv --model results/experiment_1/best_model.pkl --preprocessor results/experiment_1/preprocessor.pkl

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved to: results/experiment_1/"
echo "=========================================="