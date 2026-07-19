"""Configuration management module."""
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class Config:
    """Configuration manager for the credit scoring project."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: str = "config/config.yaml") -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = self._get_default_config()
            return
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "data": {
                "raw_path": "data/raw/credit_data.csv",
                "processed_path": "data/processed/credit_data_processed.csv",
                "target_column": "credit_risk",
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True
            },
            "preprocessing": {
                "missing_value_strategy": "median",
                "categorical_encoding": "woe",
                "scaling_method": "standard",
                "handle_imbalance": True,
                "imbalance_method": "smote",
                "outlier_method": "iqr",
                "feature_selection": True,
                "feature_selection_method": "mutual_info",
                "n_features": 20
            },
            "features": {
                "categorical_features": [
                    "checking_account", "savings_account", "employment_duration",
                    "property", "housing", "purpose", "sex", "housing"
                ],
                "numerical_features": [
                    "duration", "credit_amount", "installment_rate",
                    "age", "existing_credits", "num_dependents"
                ],
                "target": "credit_risk"
            },
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "params": {
                        "C": 1.0, "penalty": "l2", "solver": "lbfgs",
                        "max_iter": 1000, "class_weight": "balanced", "random_state": 42
                    }
                },
                "decision_tree": {
                    "enabled": True,
                    "params": {
                        "max_depth": 10, "min_samples_split": 20,
                        "min_samples_leaf": 10, "class_weight": "balanced", "random_state": 42
                    }
                },
                "random_forest": {
                    "enabled": True,
                    "params": {
                        "n_estimators": 200, "max_depth": 15,
                        "min_samples_split": 20, "min_samples_leaf": 10,
                        "class_weight": "balanced", "random_state": 42, "n_jobs": -1
                    }
                },
                "xgboost": {
                    "enabled": True,
                    "params": {
                        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "scale_pos_weight": 1, "random_state": 42, "n_jobs": -1
                    }
                },
                "lightgbm": {
                    "enabled": True,
                    "params": {
                        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1
                    }
                }
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "n_trials": 50, "timeout": 3600, "direction": "maximize", "metric": "roc_auc"
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "precision_recall_auc", "brier_score"],
                "cv_folds": 5, "stratify": True,
                "threshold_optimization": True, "threshold_metric": "f1"
            },
            "output": {
                "model_path": "models/best_model.pkl",
                "preprocessor_path": "models/preprocessor.pkl",
                "metrics_path": "results/metrics.json",
                "plots_path": "results/plots/",
                "logs_path": "logs/"
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                "file": "logs/credit_scoring.log",
                "rotation": "10 MB", "retention": "10 days"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})


config = Config()