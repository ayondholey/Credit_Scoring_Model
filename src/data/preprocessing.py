"""Data loading and preprocessing module."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy import stats
from scipy.stats import iqr
import warnings
warnings.filterwarnings('ignore')

from src.config import config
from src.logging_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loading and initial validation."""
    
    def __init__(self):
        self.config = config.get_section("data")
        self.preprocessing_config = config.get_section("preprocessing")
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = file_path or self.config.get("raw_path", "data/raw/credit_data.csv")
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File {file_path} not found. Generating sample data...")
            return self._generate_sample_data()
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample German Credit-like dataset for demonstration."""
        logger.info(f"Generating sample data with {n_samples} samples")
        np.random.seed(config.get("data.random_state", 42))
        
        n_bad = int(n_samples * 0.3)
        n_good = n_samples - n_bad
        
        data = {
            'checking_account': np.random.choice(['<0', '0-200', '>200', 'none'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
            'duration': np.random.randint(6, 72, n_samples),
            'credit_history': np.random.choice(['no_credits', 'paid_duly', 'delayed', 'critical'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'purpose': np.random.choice(['car_new', 'car_used', 'furniture', 'radio_tv', 'education', 'business'], n_samples),
            'credit_amount': np.random.randint(250, 20000, n_samples),
            'savings_account': np.random.choice(['<100', '100-500', '500-1000', '>1000', 'unknown'], n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.15]),
            'employment_duration': np.random.choice(['unemployed', '<1', '1-4', '4-7', '>7'], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.2]),
            'installment_rate': np.random.randint(1, 5, n_samples),
            'personal_status_sex': np.random.choice(['male_divorced', 'male_single', 'male_married', 'female_single', 'female_married'], n_samples),
            'other_debtors': np.random.choice(['none', 'co_applicant', 'guarantor'], n_samples, p=[0.6, 0.3, 0.1]),
            'residence_since': np.random.randint(1, 5, n_samples),
            'property': np.random.choice(['real_estate', 'savings', 'car', 'unknown'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'age': np.random.randint(19, 75, n_samples),
            'other_installment_plans': np.random.choice(['bank', 'stores', 'none'], n_samples, p=[0.2, 0.2, 0.6]),
            'housing': np.random.choice(['own', 'rent', 'free'], n_samples, p=[0.3, 0.6, 0.1]),
            'existing_credits': np.random.randint(1, 5, n_samples),
            'job': np.random.choice(['unskilled_resident', 'skilled', 'highly_skilled', 'unemployed'], n_samples, p=[0.3, 0.4, 0.1, 0.2]),
            'num_dependents': np.random.randint(1, 3, n_samples),
            'telephone': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'foreign_worker': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
            'credit_risk': np.array(['good'] * n_good + ['bad'] * n_bad)
        }
        
        np.random.shuffle(data['credit_risk'])
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        for col in df.select_dtypes(include=[np.number]).columns[:3]:
            missing_idx = np.random.choice(df.index, int(0.05 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        for col in df.select_dtypes(include=['object']).columns[:3]:
            missing_idx = np.random.choice(df.index, int(0.05 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        logger.info(f"Generated sample data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data structure and quality."""
        logger.info("Validating data...")
        
        target_col = self.config.get("target_column", "credit_risk")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        logger.info(f"Target distribution:\n{df[target_col].value_counts()}")
        logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        logger.info(f"Data types:\n{df.dtypes.value_counts()}")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        target_col = self.config.get("target_column", "credit_risk")
        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", 42)
        stratify = self.config.get("stratify", True)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if y.dtype == 'object':
            y = y.map({'good': 1, 'bad': 0, 'Good': 1, 'Bad': 0, 1: 1, 0: 0})
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test


class Preprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(self, apply_woe: bool = True):
        self.config = config.get_section("preprocessing")
        self.features_config = config.get_section("features")
        self.numerical_features = self.features_config.get("numerical_features", [])
        self.categorical_features = self.features_config.get("categorical_features", [])
        self.target = self.features_config.get("target", "credit_risk")
        self.scaler = None
        self.imputer_num = None
        self.imputer_cat = None
        self.feature_selector = None
        self.selected_features = None
        self.woe_encoder = None
        self.iv_values = {}
        self.is_fitted = False
        self.apply_woe = apply_woe
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Preprocessor':
        """Fit the preprocessor on training data."""
        logger.info("Fitting preprocessor...")
        
        X = X.copy()
        
        # Identify feature types
        self._identify_features(X)
        
        # Handle missing values
        self._fit_imputers(X)
        
        # Handle outliers
        self._fit_outlier_bounds(X)
        
        # Fit scaler
        self._fit_scaler(X)
        
        # Fit WOE encoder for categorical features (if enabled)
        if self.apply_woe:
            self._fit_woe_encoder(X, y)
        
        # Apply transforms to get the final feature set for feature selection
        X_transformed = self._transform_missing_values(X)
        X_transformed = self._transform_outliers(X_transformed)
        if self.apply_woe:
            X_transformed = self._transform_woe(X_transformed)
        X_transformed = self._transform_scaling(X_transformed)
        
        # Fit feature selector on transformed data (only if WOE encoding is applied)
        # Feature selection requires numerical features only - skip if WOE not applied
        if self.apply_woe and self.config.get("feature_selection", True):
            self._fit_feature_selector(X_transformed, y)
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        logger.info("Transforming data...")
        X = X.copy()
        
        # Handle missing values
        X = self._transform_missing_values(X)
        
        # Handle outliers
        X = self._transform_outliers(X)
        
        # Apply WOE encoding to categorical features (if enabled)
        if self.apply_woe:
            X = self._transform_woe(X)
        
        # Scale numerical features
        X = self._transform_scaling(X)
        
        # Select features
        if self.feature_selector is not None:
            X = self._transform_feature_selection(X)
        
        logger.info(f"Transformed data shape: {X.shape}")
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _identify_features(self, X: pd.DataFrame) -> None:
        """Identify numerical and categorical features."""
        all_numerical = self.numerical_features + [c for c in self.categorical_features if c in X.columns]
        all_categorical = self.categorical_features + [c for c in self.numerical_features if c in X.columns]
        
        # Auto-detect if not in config
        if not self.numerical_features:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numerical_features = [c for c in self.numerical_features if c in X.columns]
        
        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.categorical_features = [c for c in self.categorical_features if c in X.columns]
        
        logger.info(f"Numerical features: {self.numerical_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
    
    def _fit_imputers(self, X: pd.DataFrame) -> None:
        """Fit imputers for missing values."""
        strategy = self.config.get("missing_value_strategy", "median")
        
        if self.numerical_features:
            self.imputer_num = SimpleImputer(strategy=strategy)
            self.imputer_num.fit(X[self.numerical_features])
        
        if self.categorical_features:
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
            self.imputer_cat.fit(X[self.categorical_features])
    
    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values."""
        if self.imputer_num and self.numerical_features:
            X[self.numerical_features] = self.imputer_num.transform(X[self.numerical_features])
        
        if self.imputer_cat and self.categorical_features:
            X[self.categorical_features] = self.imputer_cat.transform(X[self.categorical_features])
        
        return X
    
    def _fit_outlier_bounds(self, X: pd.DataFrame) -> None:
        """Calculate outlier bounds using IQR method."""
        self.outlier_bounds = {}
        method = self.config.get("outlier_method", "iqr")
        
        if method == "iqr" and self.numerical_features:
            for col in self.numerical_features:
                if col in X.columns:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    self.outlier_bounds[col] = (lower, upper)
        
        elif method == "zscore" and self.numerical_features:
            for col in self.numerical_features:
                if col in X.columns:
                    mean = X[col].mean()
                    std = X[col].std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    self.outlier_bounds[col] = (lower, upper)
    
    def _transform_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers at bounds."""
        if not hasattr(self, 'outlier_bounds'):
            return X
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        
        return X
    
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit scaler on numerical features."""
        method = self.config.get("scaling_method", "standard")
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        if self.numerical_features:
            self.scaler.fit(X[self.numerical_features])
    
    def _transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        if self.scaler and self.numerical_features:
            X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X
    
    def _fit_woe_encoder(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Weight of Evidence encoder for all categorical features."""
        self.woe_encoder = {}
        self.iv_values = {}
        
        # Detect ALL categorical columns (object and category dtypes)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in X.columns:
                woe_map, iv = self._calculate_woe_iv(X[col], y)
                self.woe_encoder[col] = woe_map
                self.iv_values[col] = iv
        
        logger.info(f"Information Values: {self.iv_values}")
    
    def _calculate_woe_iv(self, feature: pd.Series, target: pd.Series) -> tuple:
        """Calculate Weight of Evidence and Information Value."""
        df = pd.DataFrame({'feature': feature, 'target': target})
        df = df.dropna()
        
        grouped = df.groupby('feature')['target'].agg(['count', 'sum']).rename(
            columns={'count': 'total', 'sum': 'goods'}
        )
        grouped['bads'] = grouped['total'] - grouped['goods']
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        total_goods = grouped['goods'].sum() + eps
        total_bads = grouped['bads'].sum() + eps
        
        grouped['dist_goods'] = grouped['goods'] / total_goods
        grouped['dist_bads'] = grouped['bads'] / total_bads
        
        grouped['woe'] = np.log((grouped['dist_goods'] + eps) / (grouped['dist_bads'] + eps))
        grouped['iv'] = (grouped['dist_goods'] - grouped['dist_bads']) * grouped['woe']
        
        woe_map = grouped['woe'].to_dict()
        iv = grouped['iv'].sum()
        
        return woe_map, iv
    
    def _transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WOE encoding to categorical features."""
        if not self.woe_encoder:
            return X
        
        X = X.copy()
        for col, woe_map in self.woe_encoder.items():
            if col in X.columns:
                # Convert to object if categorical to allow fillna with numeric
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype('object')
                X[col] = X[col].map(woe_map).fillna(0)
                # Rename to indicate WOE encoding
                X = X.rename(columns={col: f"{col}_woe"})
        
        return X

    def fit_woe_encoder(self, X: pd.DataFrame, y: pd.Series) -> 'Preprocessor':
        """Fit WOE encoder on new data (e.g., after feature engineering)."""
        logger.info("Fitting WOE encoder on engineered features...")
        self._identify_features(X)
        self._fit_woe_encoder(X, y)
        logger.info("WOE encoder fitted on engineered features")
        return self
    
    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WOE encoding to new data."""
        logger.info("Applying WOE encoding to engineered features...")
        X = X.copy()
        X = self._transform_woe(X)
        logger.info(f"WOE encoded data shape: {X.shape}")
        return X
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit feature selector."""
        method = self.config.get("feature_selection_method", "mutual_info")
        n_features = self.config.get("n_features", 20)
        
        n_features = min(n_features, X.shape[1])
        
        if method == "mutual_info":
            self.feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        elif method == "f_classif":
            self.feature_selector = SelectKBest(f_classif, k=n_features)
        else:
            self.feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        
        self.feature_selector.fit(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
    
    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        if self.feature_selector:
            return X[self.selected_features]
        return X
    
    def get_feature_importance_iv(self) -> pd.DataFrame:
        """Get Information Value for features."""
        if not self.iv_values:
            return pd.DataFrame()
        return pd.DataFrame(list(self.iv_values.items()), columns=['Feature', 'IV']).sort_values('IV', ascending=False)
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE or other methods."""
        if not self.config.get("handle_imbalance", True):
            return X, y
        
        method = self.config.get("imbalance_method", "smote")
        random_state = config.get("data.random_state", 42)
        
        logger.info(f"Handling class imbalance using {method}")
        logger.info(f"Before: {y.value_counts().to_dict()}")
        
        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)
        elif method == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=random_state)
        elif method == "smoteenn":
            sampler = SMOTEENN(random_state=random_state)
        elif method == "smotetomek":
            sampler = SMOTETomek(random_state=random_state)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            sampler = SMOTE(random_state=random_state)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        logger.info(f"After: {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled


def load_and_preprocess_data(file_path: str = None, apply_woe: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience function to load and preprocess data.
    
    Args:
        file_path: Path to data file
        apply_woe: Whether to apply WOE encoding during initial preprocessing.
                   Set to False if WOE will be applied after feature engineering.
    """
    loader = DataLoader()
    df = loader.load_data(file_path)
    df = loader.validate_data(df)
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    preprocessor = Preprocessor(apply_woe=apply_woe)
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)
    
    # Handle imbalance on training data only
    X_train, y_train = preprocessor.handle_imbalance(X_train, y_train)
    
    # Save processed data
    processed_path = config.get("data.processed_path", "data/processed/credit_data_processed.csv")
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['credit_risk'] = y_train
    test_df = X_test.copy()
    test_df['credit_risk'] = y_test
    
    pd.concat([train_df, test_df]).to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    # Save preprocessor
    import joblib
    preprocessor_path = config.get("output.preprocessor_path", "models/preprocessor.pkl")
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    return X_train, X_test, y_train, y_test