"""Generate sample credit scoring dataset (German Credit Data style)."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import config
from src.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def generate_sample_data(n_samples: int = 1000, 
                         n_good: int = None,
                         random_state: int = 42) -> pd.DataFrame:
    """Generate sample German Credit-like dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_good: Number of good credit samples (default: 70%)
        random_state: Random seed
        
    Returns:
        DataFrame with credit scoring features
    """
    np.random.seed(random_state)
    
    if n_good is None:
        n_good = int(n_samples * 0.7)
    n_bad = n_samples - n_good
    
    # Generate features with different distributions for good/bad
    data = {
        'checking_account': [],
        'duration': [],
        'credit_history': [],
        'purpose': [],
        'credit_amount': [],
        'savings_account': [],
        'employment_duration': [],
        'installment_rate': [],
        'personal_status_sex': [],
        'other_debtors': [],
        'residence_since': [],
        'property': [],
        'age': [],
        'other_installment_plans': [],
        'housing': [],
        'existing_credits': [],
        'job': [],
        'num_dependents': [],
        'telephone': [],
        'foreign_worker': [],
        'credit_risk': []
    }
    
    # Good credit applicants
    for _ in range(n_good):
        data['checking_account'].append(np.random.choice(
            ['0 <= DM < 200', '>= 200 DM', 'no checking account'],
            p=[0.4, 0.4, 0.2]
        ))
        data['duration'].append(np.random.randint(6, 48))
        data['credit_history'].append(np.random.choice(
            ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously'],
            p=[0.4, 0.3, 0.2, 0.1]
        ))
        data['purpose'].append(np.random.choice(
            ['car (new)', 'car (used)', 'furniture/equipment', 'radio/TV', 
             'domestic appliances', 'repairs', 'education', 'business', 'vacation'],
            p=[0.15, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05]
        ))
        data['credit_amount'].append(np.random.randint(500, 15000))
        data['savings_account'].append(np.random.choice(
            ['< 100 DM', '100 <= DM < 500', '500 <= DM < 1000', '>= 1000 DM', 'unknown'],
            p=[0.2, 0.25, 0.2, 0.2, 0.15]
        ))
        data['employment_duration'].append(np.random.choice(
            ['unemployed', '< 1 year', '1 <= years < 4', '4 <= years < 7', '>= 7 years'],
            p=[0.05, 0.15, 0.3, 0.25, 0.25]
        ))
        data['installment_rate'].append(np.random.randint(1, 5))
        data['personal_status_sex'].append(np.random.choice(
            ['male : divorced/separated', 'male : single', 'male : married/widowed',
             'female : single', 'female : married/widowed'],
            p=[0.05, 0.3, 0.4, 0.15, 0.1]
        ))
        data['other_debtors'].append(np.random.choice(
            ['none', 'co-applicant', 'guarantor'],
            p=[0.8, 0.15, 0.05]
        ))
        data['residence_since'].append(np.random.randint(1, 5))
        data['property'].append(np.random.choice(
            ['real estate', 'building society savings', 'car or other', 'unknown'],
            p=[0.2, 0.2, 0.3, 0.3]
        ))
        data['age'].append(np.random.randint(19, 65))
        data['other_installment_plans'].append(np.random.choice(
            ['bank', 'stores', 'none'],
            p=[0.1, 0.1, 0.8]
        ))
        data['housing'].append(np.random.choice(
            ['own', 'rent', 'free'],
            p=[0.4, 0.5, 0.1]
        ))
        data['existing_credits'].append(np.random.randint(1, 4))
        data['job'].append(np.random.choice(
            ['unemployed', 'unskilled - resident', 'skilled employee', 
             'management/self-employed/highly qualified'],
            p=[0.05, 0.25, 0.5, 0.2]
        ))
        data['num_dependents'].append(np.random.randint(1, 3))
        data['telephone'].append(np.random.choice(['yes', 'no'], p=[0.6, 0.4]))
        data['foreign_worker'].append(np.random.choice(['yes', 'no'], p=[0.05, 0.95]))
        data['credit_risk'].append('good')
    
    # Bad credit applicants - different distributions
    for _ in range(n_bad):
        data['checking_account'].append(np.random.choice(
            ['0 <= DM < 200', '>= 200 DM', 'no checking account'],
            p=[0.2, 0.1, 0.7]
        ))
        data['duration'].append(np.random.randint(12, 72))
        data['credit_history'].append(np.random.choice(
            ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously'],
            p=[0.1, 0.1, 0.2, 0.6]
        ))
        data['purpose'].append(np.random.choice(
            ['car (new)', 'car (used)', 'furniture/equipment', 'radio/TV', 
             'domestic appliances', 'repairs', 'education', 'business', 'vacation'],
            p=[0.05, 0.1, 0.1, 0.05, 0.05, 0.15, 0.1, 0.15, 0.2]
        ))
        data['credit_amount'].append(np.random.randint(1000, 20000))
        data['savings_account'].append(np.random.choice(
            ['< 100 DM', '100 <= DM < 500', '500 <= DM < 1000', '>= 1000 DM', 'unknown'],
            p=[0.4, 0.3, 0.15, 0.05, 0.1]
        ))
        data['employment_duration'].append(np.random.choice(
            ['unemployed', '< 1 year', '1 <= years < 4', '4 <= years < 7', '>= 7 years'],
            p=[0.2, 0.3, 0.3, 0.1, 0.1]
        ))
        data['installment_rate'].append(np.random.randint(2, 5))
        data['personal_status_sex'].append(np.random.choice(
            ['male : divorced/separated', 'male : single', 'male : married/widowed',
             'female : single', 'female : married/widowed'],
            p=[0.15, 0.4, 0.2, 0.15, 0.1]
        ))
        data['other_debtors'].append(np.random.choice(
            ['none', 'co-applicant', 'guarantor'],
            p=[0.6, 0.2, 0.2]
        ))
        data['residence_since'].append(np.random.randint(1, 5))
        data['property'].append(np.random.choice(
            ['real estate', 'building society savings', 'car or other', 'unknown'],
            p=[0.1, 0.1, 0.2, 0.6]
        ))
        data['age'].append(np.random.randint(19, 65))
        data['other_installment_plans'].append(np.random.choice(
            ['bank', 'stores', 'none'],
            p=[0.2, 0.15, 0.65]
        ))
        data['housing'].append(np.random.choice(
            ['own', 'rent', 'free'],
            p=[0.15, 0.7, 0.15]
        ))
        data['existing_credits'].append(np.random.randint(1, 5))
        data['job'].append(np.random.choice(
            ['unemployed', 'unskilled - resident', 'skilled employee', 
             'management/self-employed/highly qualified'],
            p=[0.2, 0.4, 0.3, 0.1]
        ))
        data['num_dependents'].append(np.random.randint(1, 3))
        data['telephone'].append(np.random.choice(['yes', 'no'], p=[0.3, 0.7]))
        data['foreign_worker'].append(np.random.choice(['yes', 'no'], p=[0.15, 0.85]))
        data['credit_risk'].append('bad')
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def save_sample_data(output_path: str = "data/raw/credit_data.csv",
                    n_samples: int = 1000) -> Path:
    """Generate and save sample data."""
    logger.info(f"Generating {n_samples} samples...")
    df = generate_sample_data(n_samples)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['credit_risk'].value_counts()}")
    
    return output_path


def load_or_generate_data(data_path: str = None, 
                         generate_if_missing: bool = True,
                         n_samples: int = 1000) -> pd.DataFrame:
    """Load data from file or generate sample data."""
    path = Path(data_path) if data_path else None
    
    if path and path.exists():
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    
    if generate_if_missing:
        default_path = "data/raw/credit_data.csv"
        logger.info(f"Data not found. Generating sample data at {default_path}")
        save_sample_data(default_path, n_samples)
        return pd.read_csv(default_path)
    
    raise FileNotFoundError(f"Data file not found: {data_path}")


if __name__ == "__main__":
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate sample credit scoring data")
    parser.add_argument("-n", "--n-samples", type=int, default=1000, 
                       help="Number of samples to generate")
    parser.add_argument("-o", "--output", type=str, default="data/raw/credit_data.csv",
                       help="Output CSV path")
    args = parser.parse_args()
    
    save_sample_data(args.output, args.n_samples)