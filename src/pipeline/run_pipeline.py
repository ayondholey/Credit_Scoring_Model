"""Main pipeline runner for Credit Scoring Model."""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib

from src.config import config
from src.data.preprocessing import DataLoader, Preprocessor, load_and_preprocess_data
from src.features.feature_engineering import FeatureEngineer, create_credit_features
from src.models.train import ModelTrainer, train_all_models
from src.evaluation.evaluator import ModelEvaluator, compare_models, evaluate_model
from src.logging_config import get_logger

logger = get_logger(__name__)


def run_pipeline(data_path: str = None,
                 config_path: str = None,
                 models_to_train: list = None,
                 tune_hyperparams: bool = None,
                 output_dir: str = None) -> Dict[str, Any]:
    """Run the complete credit scoring pipeline.
    
    Args:
        data_path: Path to input CSV data file
        config_path: Path to config YAML file
        models_to_train: List of model names to train
        tune_hyperparams: Whether to run hyperparameter tuning
        output_dir: Output directory for results
        
    Returns:
        Dictionary with pipeline results
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("STARTING CREDIT SCORING MODEL PIPELINE")
    logger.info("=" * 60)
    
    # Load config if provided
    if config_path:
        config.load_config(config_path)
    
    # Override output directory
    if output_dir:
        config._config.setdefault("output", {})
        config._config["output"]["model_path"] = f"{output_dir}/best_model.pkl"
        config._config["output"]["preprocessor_path"] = f"{output_dir}/preprocessor.pkl"
        config._config["output"]["metrics_path"] = f"{output_dir}/metrics.json"
        config._config["output"]["plots_path"] = f"{output_dir}/plots/"
        config._config["output"]["logs_path"] = f"{output_dir}/logs/"
    
    # Create output directories
    for key in ['model_path', 'preprocessor_path', 'metrics_path', 'plots_path', 'logs_path']:
        path = Path(config.get(f"output.{key}", ""))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Load data and do basic preprocessing (NO WOE encoding)
        logger.info("Step 1: Loading and basic preprocessing (no WOE)...")
        loader = DataLoader()
        df = loader.load_data(data_path)
        df = loader.validate_data(df)
        X_train, X_test, y_train, y_test = loader.split_data(df)
        
        # Basic preprocessing without WOE
        preprocessor = Preprocessor(apply_woe=False)
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        
        results['data_shapes'] = {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'y_train': y_train.shape,
            'y_test': y_test.shape
        }
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        engineer = FeatureEngineer()
        X_train_fe = engineer.fit_transform(X_train, y_train)
        X_test_fe = engineer.transform(X_test)
        results['feature_engineering'] = {
            'original_features': X_train.shape[1],
            'engineered_features': X_train_fe.shape[1],
            'derived_features': engineer.derived_features
        }
        logger.info(f"Features after engineering: {X_train_fe.shape[1]}")
        
        # Step 3: Apply WOE encoding to ALL categorical features (original + engineered)
        logger.info("Step 3: Applying WOE encoding to all categorical features...")
        # Find all categorical columns in engineered data
        cat_cols = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Categorical features to encode: {cat_cols}")
        
        woe_encoder = Preprocessor(apply_woe=True)
        # Fit WOE encoder on engineered train data
        woe_encoder._identify_features(X_train_fe)
        woe_encoder._fit_woe_encoder(X_train_fe, y_train)
        X_train_woe = woe_encoder._transform_woe(X_train_fe.copy())
        X_test_woe = woe_encoder._transform_woe(X_test_fe.copy())
        
        logger.info(f"Features after WOE: {X_train_woe.shape[1]}")
        
        # Step 4: Handle imbalance
        logger.info("Step 4: Handling class imbalance...")
        X_train_woe, y_train = preprocessor.handle_imbalance(X_train_woe, y_train)
        
        # Step 5: Feature selection (refit on WOE encoded data)
        logger.info("Step 5: Feature selection...")
        if preprocessor.config.get("feature_selection", True):
            # We need to refit feature selector on WOE encoded data
            woe_encoder._fit_feature_selector(X_train_woe, y_train)
            X_train_final = woe_encoder._transform_feature_selection(X_train_woe)
            X_test_final = woe_encoder._transform_feature_selection(X_test_woe)
        else:
            X_train_final = X_train_woe
            X_test_final = X_test_woe
        
        logger.info(f"Final features: {X_train_final.shape[1]}")
        
        # Step 6: Train models
        logger.info("Step 6: Training models...")
        trainer = ModelTrainer()
        
        # Set the preprocessor and woe_encoder for later saving
        trainer.preprocessor = preprocessor
        trainer.woe_encoder = woe_encoder
        trainer.feature_engineer = engineer
        
        models_config = config.get_section("models")
        if models_to_train:
            models_config = {k: v for k, v in models_config.items() if k in models_to_train}
        
        model_results = trainer.train_all_models(
            X_train_final, y_train,
            models_config=models_config,
            tune_hyperparams=tune_hyperparams
        )
        results['models'] = {name: {
            'cv_scores': res.cv_scores,
            'best_params': res.best_params,
            'train_time': res.train_time
        } for name, res in model_results.items()}
        
        # Step 7: Evaluate models
        logger.info("Step 7: Evaluating models...")
        evaluator = ModelEvaluator()
        
        comparison_df = evaluator.compare_models(
            {name: res.model for name, res in model_results.items()},
            X_test_final, y_test
        )
        logger.info(f"\nModel Comparison:\n{comparison_df}")
        results['model_comparison'] = comparison_df.to_dict('records')
        
        # Step 8: Select best model
        best_model_name = comparison_df.iloc[0]['model']
        best_model = model_results[best_model_name].model
        logger.info(f"Best model: {best_model_name}")
        
        # Detailed evaluation of best model
        best_model_results = evaluator.evaluate(best_model, X_test_final, y_test, f"{best_model_name}_test")
        
        # Train set evaluation
        y_train_pred = best_model.predict(X_train_final)
        y_train_prob = best_model.predict_proba(X_train_final)[:, 1] if hasattr(best_model, 'predict_proba') else None
        train_metrics = evaluator._calculate_metrics(y_train, y_train_pred, y_train_prob)
        
        results['best_model'] = {
            'name': best_model_name,
            'test_metrics': best_model_results['metrics'],
            'train_metrics': train_metrics,
            'feature_importance': model_results[best_model_name].feature_importance.to_dict('records') 
                                 if model_results[best_model_name].feature_importance is not None else None
        }
        
        # Step 9: Save model and artifacts
        logger.info("Step 9: Saving model and artifacts...")
        model_path = config.get("output.model_path", "models/best_model.pkl")
        preprocessor_path = config.get("output.preprocessor_path", "models/preprocessor.pkl")
        
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Save preprocessor with feature engineer and woe encoder
        preprocessor_data = {
            'preprocessor': preprocessor,
            'feature_engineer': engineer,
            'woe_encoder': woe_encoder
        }
        joblib.dump(preprocessor_data, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Step 7: Generate plots
        logger.info("Step 7: Generating evaluation plots...")
        plots_path = config.get("output.plots_path", "results/plots/")
        Path(plots_path).mkdir(parents=True, exist_ok=True)
        
        evaluator.confusion_matrix_plot(
            best_model, X_test_final, y_test,
            save_path=f"{plots_path}confusion_matrix.png"
        )
        evaluator.roc_curve_plot(
            {name: res.model for name, res in model_results.items()},
            X_test_final, y_test,
            save_path=f"{plots_path}roc_curves.png"
        )
        evaluator.precision_recall_curve_plot(
            {name: res.model for name, res in model_results.items()},
            X_test_final, y_test,
            save_path=f"{plots_path}pr_curves.png"
        )
        evaluator.calibration_plot(
            best_model, X_test_final, y_test,
            save_path=f"{plots_path}calibration.png"
        )
        evaluator.feature_importance_plot(
            model_results[best_model_name].feature_importance,
            top_n=20,
            save_path=f"{plots_path}feature_importance.png"
        )
        evaluator.score_distribution_plot(
            best_model, X_test_final, y_test,
            save_path=f"{plots_path}score_distribution.png"
        )
        
        # Step 8: Save metrics
        metrics_path = config.get("output.metrics_path", "results/metrics.json")
        evaluator.save_results(best_model_results, metrics_path)
        
        # Add summary to results
        results['summary'] = {
            'best_model': best_model_name,
            'test_roc_auc': best_model_results['metrics']['roc_auc'],
            'test_f1': best_model_results['metrics']['f1'],
            'test_accuracy': best_model_results['metrics']['accuracy'],
            'test_precision': best_model_results['metrics']['precision'],
            'test_recall': best_model_results['metrics']['recall'],
            'train_time': model_results[best_model_name].train_time,
            'total_pipeline_time': time.time() - start_time
        }
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Test ROC-AUC: {best_model_results['metrics']['roc_auc']:.4f}")
        logger.info(f"Test F1: {best_model_results['metrics']['f1']:.4f}")
        logger.info(f"Total Time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results['error'] = str(e)
        raise
    
    return results


def predict(model_path: str, preprocessor_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """Make predictions on new data using trained model.
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        data: New data for prediction
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor_data = joblib.load(preprocessor_path)
    
    preprocessor = preprocessor_data['preprocessor']
    engineer = preprocessor_data['feature_engineer']
    woe_encoder = preprocessor_data.get('woe_encoder')
    
    # Transform data using the same feature engineering and selection steps as training
    X = preprocessor.transform(data)
    X = engineer.transform(X)
    if woe_encoder is not None:
        X = woe_encoder.transform_woe(X)
        if getattr(woe_encoder, "feature_selector", None) is not None:
            X = woe_encoder._transform_feature_selection(X)
    
    # Predict
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Create results
    results = data.copy()
    results['predicted_credit_risk'] = ['bad' if p == 1 else 'good' for p in y_pred]
    if y_prob is not None:
        results['probability_bad'] = y_prob
        results['probability_good'] = 1 - y_prob
        results['credit_score'] = (1 - y_prob) * 1000  # Scale to 0-1000
    
    logger.info(f"Predictions generated for {len(results)} samples")
    return results


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Credit Scoring Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.run_pipeline                    # Run with defaults
  python -m src.pipeline.run_pipeline --data data/raw/credit_data.csv
  python -m src.pipeline.run_pipeline --tune-hyperparams --models logistic_regression random_forest xgboost
  python -m src.pipeline.run_pipeline --config config/custom_config.yaml --output results/experiment_1
        """
    )
    
    parser.add_argument(
        "--data", "-d", type=str, default=None,
        help="Path to input data CSV file"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--models", "-m", type=str, nargs="+", default=None,
        help="Models to train (logistic_regression, decision_tree, random_forest, xgboost, lightgbm, catboost)"
    )
    parser.add_argument(
        "--tune-hyperparams", action="store_true",
        help="Enable hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Disable hyperparameter tuning (overrides config)"
    )
    parser.add_argument(
        "--predict", "-p", type=str, default=None,
        help="Path to new data for prediction (requires --model and --preprocessor)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model for prediction"
    )
    parser.add_argument(
        "--preprocessor", type=str, default=None,
        help="Path to preprocessor for prediction"
    )
    
    args = parser.parse_args()
    
    # Handle prediction mode
    if args.predict:
        if not args.model or not args.preprocessor:
            parser.error("--predict requires --model and --preprocessor")
        
        logger.info(f"Loading prediction data from {args.predict}")
        new_data = pd.read_csv(args.predict)
        
        results = predict(args.model, args.preprocessor, new_data)
        
        output_path = f"predictions_{Path(args.predict).stem}.csv"
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        return
    
    # Run pipeline
    tune = config.get("hyperparameter_tuning.enabled", True)
    if args.no_tune:
        tune = False
    if args.tune_hyperparams:
        tune = True
    
    results = run_pipeline(
        data_path=args.data,
        config_path=args.config,
        models_to_train=args.models,
        tune_hyperparams=tune,
        output_dir=args.output
    )
    
    # Print summary
    if 'summary' in results:
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {results['summary']['best_model']}")
        print(f"Test ROC-AUC: {results['summary']['test_roc_auc']:.4f}")
        print(f"Test F1: {results['summary']['test_f1']:.4f}")
        print(f"Test Accuracy: {results['summary']['test_accuracy']:.4f}")
        print(f"Test Precision: {results['summary']['test_precision']:.4f}")
        print(f"Test Recall: {results['summary']['test_recall']:.4f}")
        print(f"Total Time: {results['summary']['total_pipeline_time']:.2f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()