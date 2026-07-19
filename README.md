# Credit Scoring Model

A credit scoring pipeline built with Python that loads or generates credit data, preprocesses and engineers features, trains classification models, evaluates their performance, and saves the best model and preprocessing artifacts.

## Features

- Data loading and validation
- Missing value imputation and outlier handling
- Feature engineering and WOE encoding
- Class imbalance handling
- Model training with cross-validation
- Hyperparameter tuning support
- Model evaluation and visualization
- Prediction helper for new data

## Requirements

- Python 3.10+ (project was tested with Python 3.10)
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`
- `imblearn`
- `optuna`
- `matplotlib`
- `seaborn`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

- `src/` - main application code
  - `data/` - data loading and preprocessing
  - `features/` - feature engineering logic
  - `models/` - model training and management
  - `pipeline/` - end-to-end pipeline runner and prediction
  - `evaluation/` - model evaluation metrics and plotting
- `scripts/` - test scripts and utilities
- `config/` - configuration file(s)
- `data/` - raw and processed datasets
- `models/` - saved model artifacts
- `results/` - evaluation outputs, plots, and metrics

## Usage

### Run the full pipeline

```bash
python -m src.pipeline.run_pipeline
```

### Run the pipeline with a custom output directory

```bash
python -m src.pipeline.run_pipeline --output results/experiment_1
```

### Run the pipeline with specific models only

```bash
python -m src.pipeline.run_pipeline --models logistic_regression random_forest --no-tune
```

### Run training and test pipeline script

```bash
python scripts/test_pipeline.py
```

## Predict with a saved model

```bash
python -m src.pipeline.run_pipeline --predict data/new_data.csv --model models/best_model.pkl --preprocessor models/preprocessor.pkl
```

## Notes

- The project can generate sample data if `data/raw/credit_data.csv` is missing.
- The pipeline currently uses WOE encoding, feature selection, and model comparison.
- The main training implementation is in `src/models/train.py`.

## Troubleshooting

If VS Code underlines the `sklearn` imports, confirm the interpreter is using the same Python environment where `scikit-learn` is installed:

1. Open the Python interpreter selection in VS Code.
2. Choose the environment with `scikit-learn` installed.
3. Reload the window if needed.

If you need help cleaning up warnings or improving model performance, inspect `src/models/train.py` and the preprocessing pipeline in `src/data/preprocessing.py`.
