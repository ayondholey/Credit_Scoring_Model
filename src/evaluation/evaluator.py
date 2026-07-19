"""Model evaluation module with comprehensive metrics and visualizations."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import json

from src.config import config
from src.logging_config import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for credit scoring."""
    
    def __init__(self):
        self.eval_config = config.get_section("evaluation")
        self.metrics_list = self.eval_config.get("metrics", [
            "accuracy", "precision", "recall", "f1", "roc_auc", 
            "precision_recall_auc", "brier_score", "log_loss",
            "matthews_corrcoef", "cohen_kappa"
        ])
        self.threshold_optimization = self.eval_config.get("threshold_optimization", True)
        self.threshold_metric = self.eval_config.get("threshold_metric", "f1")
        self.output_config = config.get_section("output")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all specified metrics."""
        metrics = {}
        
        # Basic classification metrics
        if "accuracy" in self.metrics_list:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        if "precision" in self.metrics_list:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        if "recall" in self.metrics_list:
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        if "f1" in self.metrics_list:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_prob is not None:
            if "roc_auc" in self.metrics_list:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            if "precision_recall_auc" in self.metrics_list:
                metrics["precision_recall_auc"] = average_precision_score(y_true, y_prob)
            if "brier_score" in self.metrics_list:
                metrics["brier_score"] = brier_score_loss(y_true, y_prob)
            if "log_loss" in self.metrics_list:
                metrics["log_loss"] = log_loss(y_true, y_prob)
        else:
            for m in ["roc_auc", "precision_recall_auc", "brier_score", "log_loss"]:
                if m in self.metrics_list:
                    metrics[m] = None
        
        # Additional metrics
        if "matthews_corrcoef" in self.metrics_list:
            metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
        if "cohen_kappa" in self.metrics_list:
            metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        
        # Derived metrics
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
        
        # KS Statistic
        if y_prob is not None:
            metrics["ks_statistic"] = self._calculate_ks(y_true, y_prob)
            metrics["gini"] = 2 * metrics.get("roc_auc", 0) - 1
        
        return metrics
    
    def _calculate_ks(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['cum_bad'] = (df['y_true'] == 1).cumsum()
        df['cum_good'] = (df['y_true'] == 0).cumsum()
        df['total_bad'] = df['cum_bad'].iloc[-1]
        df['total_good'] = df['cum_good'].iloc[-1]
        df['cum_bad_rate'] = df['cum_bad'] / df['total_bad']
        df['cum_good_rate'] = df['cum_good'] / df['total_good']
        df['ks'] = abs(df['cum_bad_rate'] - df['cum_good_rate'])
        return df['ks'].max()
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal threshold based on specified metric."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        if self.threshold_metric == "f1":
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        elif self.threshold_metric == "precision":
            best_idx = np.argmax(precision[:-1])  # Last is 1.0 by definition
        elif self.threshold_metric == "recall":
            best_idx = np.argmax(recall[:-1])
        elif self.threshold_metric == "youden":
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return roc_thresholds[best_idx]
        else:
            best_idx = np.argmax(f1_scores)
        
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                 dataset_name: str = "test") -> Dict[str, Any]:
        """Evaluate model on given data."""
        logger.info(f"Evaluating model on {dataset_name} set...")
        
        # Get predictions
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            from scipy.special import expit
            y_prob = expit(model.decision_function(X))
        elif hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba', None)):
            y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        results = {}
        results['dataset'] = dataset_name
        results['metrics'] = self._calculate_metrics(y, y_pred, y_prob)
        results['predictions'] = y_pred.tolist()
        results['probabilities'] = y_prob.tolist() if y_prob is not None else None
        
        # Threshold optimization
        if self.threshold_optimization and y_prob is not None:
            optimal_threshold = self._find_optimal_threshold(y, y_prob)
            y_pred_opt = (y_prob >= optimal_threshold).astype(int)
            results['optimal_threshold'] = float(optimal_threshold)
            results['optimized_metrics'] = self._calculate_metrics(y, y_pred_opt, y_prob)
            logger.info(f"Optimal threshold ({self.threshold_metric}): {optimal_threshold:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_normalized'] = (cm / cm.sum(axis=1, keepdims=True)).tolist()
        
        # Classification report
        results['classification_report'] = classification_report(y, y_pred, output_dict=True)
        
        # Calibration data
        if y_prob is not None:
            results['calibration'] = self._get_calibration_data(y, y_prob)
            results['roc_curve'] = self._get_roc_data(y, y_prob)
            results['pr_curve'] = self._get_pr_data(y, y_prob)
        
        logger.info(f"{dataset_name} Metrics: {self._format_metrics(results['metrics'])}")
        return results
    
    def _get_calibration_data(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              n_bins: int = 10) -> Dict[str, List[float]]:
        """Get calibration curve data."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    
    def _get_roc_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
        """Get ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def _get_pr_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
        """Get Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for logging."""
        return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                         if v is not None and not isinstance(v, (int, float))])
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare multiple models."""
        logger.info(f"Comparing {len(models)} models...")
        
        results = []
        for name, model in models.items():
            try:
                y_pred = model.predict(X)
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    from scipy.special import expit
                    y_prob = expit(model.decision_function(X))
                
                metrics = self._calculate_metrics(y, y_pred, y_prob)
                metrics['model'] = name
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            # Sort by primary metric
            primary_metric = self.eval_config.get("metric", "roc_auc")
            if primary_metric in df.columns:
                df = df.sort_values(primary_metric, ascending=False)
        
        return df
    
    def confusion_matrix_plot(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             labels: List[str] = None, save_path: str = None) -> plt.Figure:
        """Plot confusion matrix."""
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        labels = labels or ['Good (0)', 'Bad (1)']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Add normalized version as text
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})', 
                       ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def roc_curve_plot(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      save_path: str = None) -> plt.Figure:
        """Plot ROC curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    from scipy.special import expit
                    y_prob = expit(model.decision_function(X))
                else:
                    continue
                
                fpr, tpr, _ = roc_curve(y, y_prob)
                auc = roc_auc_score(y, y_prob)
                ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
                
            except Exception as e:
                logger.error(f"Error plotting ROC for {name}: {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def precision_recall_curve_plot(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                                    save_path: str = None) -> plt.Figure:
        """Plot Precision-Recall curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    from scipy.special import expit
                    y_prob = expit(model.decision_function(X))
                else:
                    continue
                
                precision, recall, _ = precision_recall_curve(y, y_prob)
                aps = average_precision_score(y, y_prob)
                ax.plot(recall, precision, label=f'{name} (AP={aps:.3f})', linewidth=2)
                
            except Exception as e:
                logger.error(f"Error plotting PR curve for {name}: {e}")
        
        # Baseline
        baseline = y.sum() / len(y)
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        return fig
    
    def calibration_plot(self, model: Any, X: pd.DataFrame, y: pd.Series,
                        n_bins: int = 10, save_path: str = None) -> plt.Figure:
        """Plot calibration curve."""
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not have predict_proba, skipping calibration plot")
            return None
        
        y_prob = model.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy='uniform')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")
        
        return fig
    
    def feature_importance_plot(self, feature_importance: pd.DataFrame,
                               top_n: int = 20, save_path: str = None) -> plt.Figure:
        """Plot feature importance."""
        if feature_importance is None or len(feature_importance) == 0:
            logger.warning("No feature importance data available")
            return None
        
        df = feature_importance.head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        bars = ax.barh(range(len(df)), df['importance'], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, df['importance'])):
            ax.text(val + max(df['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def score_distribution_plot(self, model: Any, X: pd.DataFrame, y: pd.Series,
                               save_path: str = None) -> plt.Figure:
        """Plot score distribution by class."""
        if not hasattr(model, 'predict_proba'):
            return None
        
        y_prob = model.predict_proba(X)[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        good_scores = y_prob[y == 0]
        bad_scores = y_prob[y == 1]
        
        ax.hist(good_scores, bins=30, alpha=0.6, label='Good (0)', color='green', density=True)
        ax.hist(bad_scores, bins=30, alpha=0.6, label='Bad (1)', color='red', density=True)
        
        ax.set_xlabel('Probability of Bad Credit')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Score distribution plot saved to {save_path}")
        
        return fig
    
    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save evaluation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove non-serializable items
        save_results = results.copy()
        if 'predictions' in save_results:
            save_results['predictions'] = None
        if 'probabilities' in save_results:
            save_results['probabilities'] = None
        
        with open(path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def generate_report(self, results: Dict[str, Any], path: str = None) -> str:
        """Generate text evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nDataset: {results.get('dataset', 'N/A')}")
        
        metrics = results.get('metrics', {})
        report.append(f"\n{'METRICS':^60}")
        report.append("-" * 60)
        for k, v in metrics.items():
            if v is not None:
                if isinstance(v, float):
                    report.append(f"  {k:25s}: {v:.4f}")
                else:
                    report.append(f"  {k:25s}: {v}")
        
        # Optimized metrics
        if 'optimized_metrics' in results:
            report.append(f"\n{'OPTIMIZED METRICS (threshold=' + str(results.get('optimal_threshold', 0))[:6] + ')':^60}")
            report.append("-" * 60)
            for k, v in results['optimized_metrics'].items():
                if v is not None and isinstance(v, float):
                    report.append(f"  {k:25s}: {v:.4f}")
        
        # Confusion Matrix
        cm = results.get('confusion_matrix', [])
        if cm:
            report.append(f"\n{'CONFUSION MATRIX':^60}")
            report.append("-" * 60)
            report.append(f"                    Predicted Good  Predicted Bad")
            report.append(f"  Actual Good           {cm[0][0]:6d}          {cm[0][1]:6d}")
            report.append(f"  Actual Bad            {cm[1][0]:6d}          {cm[1][1]:6d}")
        
        report_text = "\n".join(report)
        
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {path}")
        
        return report_text


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                   dataset_name: str = "test") -> Dict[str, Any]:
    """Convenience function to evaluate a single model."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate(model, X, y, dataset_name)


def compare_models(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Convenience function to compare multiple models."""
    evaluator = ModelEvaluator()
    return evaluator.compare_models(models, X, y)