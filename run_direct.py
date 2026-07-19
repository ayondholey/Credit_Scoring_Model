"""Direct pipeline run for testing."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
# Disable optional models for faster testing
config._config.setdefault('models', {})
for m in ['xgboost', 'lightgbm', 'catboost']:
    if m in config._config['models']:
        config._config['models'][m]['enabled'] = False

from src.pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    results = run_pipeline(
        models_to_train=["logistic_regression", "random_forest", "gradient_boosting"],
        tune_hyperparams=False,
        output_dir="results/test_run"
    )
    print("Pipeline completed!")
    print(f"Best Model: {results['summary']['best_model']}")
    print(f"Test ROC-AUC: {results['summary']['test_roc_auc']:.4f}")
    print(f"Test F1: {results['summary']['test_f1']:.4f}")