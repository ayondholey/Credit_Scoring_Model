"""Test script for Credit Scoring Pipeline."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.preprocessing import load_and_preprocess_data, DataLoader, Preprocessor
from src.features.feature_engineering import FeatureEngineer, create_credit_features
from src.models.train import ModelTrainer, train_all_models
from src.evaluation.evaluator import ModelEvaluator, evaluate_model, compare_models
from src.pipeline.run_pipeline import run_pipeline, predict
from src.config import config
from src.logging_config import setup_logging

setup_logging()


def test_data_generation():
    """Test data loading and generation."""
    print("=" * 60)
    print("TESTING DATA GENERATION")
    print("=" * 60)
    
    # Test data loader
    loader = DataLoader()
    df = loader.load_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target distribution:\n{df['credit_risk'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Test validation
    df = loader.validate_data(df)
    X_train, X_test, y_train, y_test = loader.split_data(df)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("=" * 60)
    print("TESTING PREPROCESSING")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = test_data_generation()
    
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    print(f"Selected features: {preprocessor.selected_features}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def test_feature_engineering():
    """Test feature engineering."""
    print("=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, preprocessor = test_preprocessing()
    
    engineer = FeatureEngineer()
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    print(f"Features after engineering: {X_train_fe.shape[1]}")
    print(f"Derived features: {engineer.derived_features}")
    
    # Test IV calculation
    iv_summary = engineer.get_iv_summary()
    print(f"\nInformation Value Summary:\n{iv_summary}")
    
    return X_train_fe, X_test_fe, y_train, y_test, preprocessor, engineer


def test_model_training():
    """Test model training."""
    print("=" * 60)
    print("TESTING MODEL TRAINING")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, preprocessor, engineer = test_feature_engineering()
    
    # Train a few models
    trainer = ModelTrainer()
    
    # Only test a subset to save time
    models_config = {
        "logistic_regression": trainer.models_config.get("logistic_regression", {"enabled": True, "params": {}}),
        "random_forest": trainer.models_config.get("random_forest", {"enabled": True, "params": {}}),
    }
    
    results = trainer.train_all_models(X_train, y_train, models_config=models_config, tune_hyperparams=False)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  CV Scores: {result.cv_scores}")
        print(f"  Train Time: {result.train_time:.2f}s")
        if result.feature_importance is not None:
            print(f"  Top 5 Features:\n{result.feature_importance.head()}")
    
    best_name, best_result = trainer.select_best_model()
    print(f"\nBest model: {best_name}")
    
    return best_result.model, X_test, y_test


def test_evaluation():
    """Test model evaluation."""
    print("=" * 60)
    print("TESTING MODEL EVALUATION")
    print("=" * 60)
    
    model, X_test, y_test = test_model_training()
    
    evaluator = ModelEvaluator()
    
    # Evaluate on test set
    results = evaluator.evaluate(model, X_test, y_test, "test")
    
    print("\nTest Metrics:")
    for k, v in results['metrics'].items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Test comparison
    print("\nComparison DataFrame:")
    comparison = evaluate_model(model, X_test, y_test)
    print(comparison)
    
    return model, X_test, y_test


def test_full_pipeline():
    """Test the complete pipeline."""
    print("=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    results = run_pipeline(
        models_to_train=["logistic_regression", "random_forest", "gradient_boosting"],
        tune_hyperparams=False,
        output_dir="results/test_run"
    )
    
    print(f"\nPipeline Results:")
    print(f"Best Model: {results['summary']['best_model']}")
    print(f"Test ROC-AUC: {results['summary']['test_roc_auc']:.4f}")
    print(f"Test F1: {results['summary']['test_f1']:.4f}")
    print(f"Total Time: {results['summary']['total_pipeline_time']:.2f}s")
    
    return results


def test_prediction():
    """Test prediction on new data."""
    print("=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    
    # First run pipeline to get models
    results = test_full_pipeline()
    
    # Create some test data
    loader = DataLoader()
    df = loader.load_data()
    test_data = df.head(5).drop(columns=['credit_risk'])
    
    print(f"\nTest data shape: {test_data.shape}")
    
    # Make predictions
    predictions = predict(
        model_path="models/best_model.pkl",
        preprocessor_path="models/preprocessor.pkl",
        data=test_data
    )
    
    print("\nPredictions:")
    print(predictions[['probability_bad', 'credit_score', 'predicted_credit_risk']])
    
    return predictions


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)
    
    try:
        test_data_generation()
        print("\n✓ Data generation test passed")
    except Exception as e:
        print(f"\n✗ Data generation test failed: {e}")
        return False
    
    try:
        test_preprocessing()
        print("✓ Preprocessing test passed")
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False
    
    try:
        test_feature_engineering()
        print("✓ Feature engineering test passed")
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        return False
    
    try:
        test_model_training()
        print("✓ Model training test passed")
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False
    
    try:
        test_evaluation()
        print("✓ Evaluation test passed")
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False
    
    try:
        test_full_pipeline()
        print("✓ Full pipeline test passed")
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False
    
    try:
        test_prediction()
        print("✓ Prediction test passed")
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)