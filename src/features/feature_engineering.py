"""Feature engineering module for credit scoring."""
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

from src.config import config
from src.logging_config import get_logger

logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer for credit scoring."""
    
    def __init__(self):
        self.features_config = config.get_section("features")
        self.numerical_features = self.features_config.get("numerical_features", [])
        self.categorical_features = self.features_config.get("categorical_features", [])
        self.derived_features = []
        self.category_mappings = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        """Fit feature engineer."""
        self._identify_features(X)
        self._fit_category_mappings(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        X = X.copy()
        
        # Create derived features
        X = self._create_derived_features(X)
        
        # Create interaction features
        X = self._create_interaction_features(X)
        
        # Create ratio features
        X = self._create_ratio_features(X)
        
        # Create binned features
        X = self._create_binned_features(X)

        # Convert categorical features to numeric codes for sklearn compatibility
        X = self._encode_categorical_features(X)
        
        logger.info(f"Feature engineering complete. Shape: {X.shape}")
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_iv_summary(self) -> pd.DataFrame:
        """Return an empty IV summary for compatibility with the training scripts."""
        return pd.DataFrame(columns=['Feature', 'IV'])
    
    def _identify_features(self, X: pd.DataFrame) -> None:
        """Auto-detect feature types if not in config."""
        if not self.numerical_features:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numerical_features = [c for c in self.numerical_features if c in X.columns]
        
        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.categorical_features = [c for c in self.categorical_features if c in X.columns]
        
        logger.info(f"Identified numerical: {self.numerical_features}")
        logger.info(f"Identified categorical: {self.categorical_features}")

    def _fit_category_mappings(self, X: pd.DataFrame) -> None:
        """Learn stable category mappings for categorical features."""
        self.category_mappings = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            values = X[col].astype(str).fillna('missing').tolist()
            unique_values = list(dict.fromkeys(values))
            self.category_mappings[col] = {
                value: idx for idx, value in enumerate(unique_values)
            }

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric codes for sklearn models."""
        X = X.copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in self.category_mappings:
                X[col] = (
                    X[col]
                    .astype(str)
                    .fillna('missing')
                    .map(self.category_mappings[col])
                    .fillna(-1)
                    .astype(float)
                )
            else:
                X[col] = X[col].astype('category').cat.codes.astype(float)
        return X
    
    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for credit scoring."""
        
        # Credit utilization indicators
        if 'credit_amount' in X.columns and 'duration' in X.columns:
            X['credit_amount_per_month'] = X['credit_amount'] / (X['duration'] + 1)
            self.derived_features.append('credit_amount_per_month')
        
        if 'installment_rate' in X.columns and 'credit_amount' in X.columns:
            X['installment_to_credit'] = X['installment_rate'] / (X['credit_amount'] + 1)
            self.derived_features.append('installment_to_credit')
        
        # Age related features
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['young', 'young_adult', 'adult', 'middle_age', 'senior'])
            X['is_young'] = (X['age'] < 25).astype(int)
            X['is_senior'] = (X['age'] > 55).astype(int)
            self.derived_features.extend(['age_group', 'is_young', 'is_senior'])
        
        # Employment stability
        if 'employment_duration' in X.columns:
            emp_map = {'unemployed': 0, '<1': 1, '1-4': 2, '4-7': 3, '>7': 4}
            X['employment_stability'] = X['employment_duration'].map(emp_map).fillna(0)
            self.derived_features.append('employment_stability')
        
        # Credit history quality
        if 'credit_history' in X.columns:
            hist_map = {'critical': 0, 'delayed': 1, 'paid_duly': 2, 'no_credits': 3}
            X['credit_history_score'] = X['credit_history'].map(hist_map).fillna(0)
            self.derived_features.append('credit_history_score')
        
        # Savings strength
        if 'savings_account' in X.columns:
            sav_map = {'unknown': 0, '<100': 1, '100-500': 2, '500-1000': 3, '>1000': 4}
            X['savings_strength'] = X['savings_account'].map(sav_map).fillna(0)
            self.derived_features.append('savings_strength')
        
        # Checking account status
        if 'checking_account' in X.columns:
            chk_map = {'<0': 0, '0-200': 1, '>200': 2, 'none': 1}
            X['checking_status'] = X['checking_account'].map(chk_map).fillna(1)
            self.derived_features.append('checking_status')
        
        # Debt burden indicators
        if 'existing_credits' in X.columns:
            X['multiple_credits'] = (X['existing_credits'] > 1).astype(int)
            self.derived_features.append('multiple_credits')
        
        # Dependency ratio
        if 'num_dependents' in X.columns and 'age' in X.columns:
            X['dependency_ratio'] = X['num_dependents'] / (X['age'] / 18)
            self.derived_features.append('dependency_ratio')
        
        # Housing stability
        if 'housing' in X.columns:
            hous_map = {'rent': 1, 'own': 2, 'free': 0}
            X['housing_stability'] = X['housing'].map(hous_map).fillna(1)
            self.derived_features.append('housing_stability')
        
        # Job stability
        if 'job' in X.columns:
            job_map = {'unemployed': 0, 'unskilled_resident': 1, 'skilled': 2, 'highly_skilled': 3}
            X['job_level'] = X['job'].map(job_map).fillna(1)
            self.derived_features.append('job_level')
        
        # Purpose risk
        if 'purpose' in X.columns:
            high_risk_purposes = ['radio_tv', 'education', 'furniture']
            X['high_risk_purpose'] = X['purpose'].apply(
                lambda x: 1 if x in high_risk_purposes else 0
            )
            self.derived_features.append('high_risk_purpose')
        
        logger.info(f"Created derived features: {self.derived_features}")
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        interactions = []
        
        # Age x Credit amount interaction
        if 'age' in X.columns and 'credit_amount' in X.columns:
            X['age_x_credit'] = X['age'] * X['credit_amount'] / 1000
            interactions.append('age_x_credit')
        
        # Duration x Credit amount
        if 'duration' in X.columns and 'credit_amount' in X.columns:
            X['duration_x_credit'] = X['duration'] * X['credit_amount'] / 1000
            interactions.append('duration_x_credit')
        
        # Employment x Credit amount
        if 'employment_stability' in X.columns and 'credit_amount' in X.columns:
            X['emp_stability_x_credit'] = X['employment_stability'] * X['credit_amount'] / 1000
            interactions.append('emp_stability_x_credit')
        
        # Savings x Credit amount
        if 'savings_strength' in X.columns and 'credit_amount' in X.columns:
            X['savings_x_credit'] = X['savings_strength'] * X['credit_amount'] / 1000
            interactions.append('savings_x_credit')
        
        if interactions:
            logger.info(f"Created interaction features: {interactions}")
        
        return X
    
    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features."""
        ratios = []
        
        # Credit amount to age ratio
        if 'credit_amount' in X.columns and 'age' in X.columns:
            X['credit_to_age'] = X['credit_amount'] / (X['age'] + 1)
            ratios.append('credit_to_age')
        
        # Installment to income proxy (credit_amount as proxy)
        if 'installment_rate' in X.columns and 'credit_amount' in X.columns:
            X['installment_burden'] = X['installment_rate'] * X['credit_amount'] / 100
            ratios.append('installment_burden')
        
        # Duration to age ratio
        if 'duration' in X.columns and 'age' in X.columns:
            X['duration_to_age'] = X['duration'] / (X['age'] + 1)
            ratios.append('duration_to_age')
        
        if ratios:
            logger.info(f"Created ratio features: {ratios}")
        
        return X
    
    def _create_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of numerical features."""
        binned = []
        
        # Credit amount bins
        if 'credit_amount' in X.columns:
            X['credit_amount_bin'] = pd.qcut(X['credit_amount'], q=4, 
                                              labels=['low', 'medium', 'high', 'very_high'],
                                              duplicates='drop')
            binned.append('credit_amount_bin')
        
        # Duration bins
        if 'duration' in X.columns:
            X['duration_bin'] = pd.cut(X['duration'], bins=[0, 12, 24, 36, 72],
                                        labels=['short', 'medium', 'long', 'very_long'])
            binned.append('duration_bin')
        
        # Age bins (if not already created)
        if 'age' in X.columns and 'age_group' not in X.columns:
            X['age_bin'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 100],
                                   labels=['very_young', 'young', 'middle', 'senior', 'very_senior'])
            binned.append('age_bin')
        
        if binned:
            logger.info(f"Created binned features: {binned}")
        
        return X


class WoETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence transformer for categorical variables."""
    
    def __init__(self, min_samples: int = 30, reg_param: float = 0.5):
        self.min_samples = min_samples
        self.reg_param = reg_param
        self.woe_maps = {}
        self.iv_values = {}
        self.feature_names = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WoETransformer':
        """Fit WOE transformer."""
        for col in X.columns:
            if X[col].dtype in ['object', 'category'] or X[col].nunique() < 20:
                woe_map, iv = self._calculate_woe_iv(X[col], y)
                self.woe_maps[col] = woe_map
                self.iv_values[col] = iv
                self.feature_names.append(col)
        
        logger.info(f"WOE fitted for {len(self.woe_maps)} features")
        logger.info(f"Information Values: {self.iv_values}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features to WOE."""
        X = X.copy()
        
        for col, woe_map in self.woe_maps.items():
            if col in X.columns:
                X[col] = X[col].map(woe_map).fillna(0)
                X = X.rename(columns={col: f"{col}_woe"})
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def _calculate_woe_iv(self, feature: pd.Series, target: pd.Series) -> tuple:
        """Calculate WOE and IV for a feature."""
        df = pd.DataFrame({'feature': feature, 'target': target})
        df = df.dropna()
        
        # Bin numerical features if too many unique values
        if feature.nunique() > 20 and feature.dtype in ['int64', 'float64']:
            df['feature'] = pd.qcut(df['feature'], q=10, duplicates='drop')
        
        grouped = df.groupby('feature')['target'].agg(['count', 'sum']).rename(
            columns={'count': 'total', 'sum': 'goods'}
        )
        grouped['bads'] = grouped['total'] - grouped['goods']
        
        # Regularization
        eps = self.reg_param
        total_goods = grouped['goods'].sum() + eps
        total_bads = grouped['bads'].sum() + eps
        
        grouped['dist_goods'] = (grouped['goods'] + eps) / total_goods
        grouped['dist_bads'] = (grouped['bads'] + eps) / total_bads
        
        grouped['woe'] = np.log(grouped['dist_goods'] / grouped['dist_bads'])
        grouped['iv'] = (grouped['dist_goods'] - grouped['dist_bads']) * grouped['woe']
        
        woe_map = grouped['woe'].to_dict()
        iv = grouped['iv'].sum()
        
        return woe_map, iv
    
    def get_iv_summary(self) -> pd.DataFrame:
        """Get Information Value summary."""
        if not self.iv_values:
            return pd.DataFrame()
        return pd.DataFrame(list(self.iv_values.items()), 
                           columns=['Feature', 'IV']).sort_values('IV', ascending=False)


def create_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create credit scoring features."""
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)


def get_feature_importance_iv(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Get feature importance using Information Value."""
    woe = WoETransformer()
    woe.fit(X, y)
    return woe.get_iv_summary().head(top_n)