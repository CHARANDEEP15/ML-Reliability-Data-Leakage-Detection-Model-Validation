import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataLoader:
    """
    Handles loading of datasets and generation of synthetic data for testing.
    """
    
    @staticmethod
    def load_data(filepath: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Loads data from a file.
        """
        if file_type == 'csv':
            return pd.read_csv(filepath)
        elif file_type == 'parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'parquet'.")

    @staticmethod
    def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, 
                   split_type: str = 'random', time_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into train and test sets.
        Supports 'random' and 'time' based splitting.
        """
        if split_type == 'random':
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(df, test_size=test_size, random_state=42)
            return train, test
        
        elif split_type == 'time':
            if time_col is None or time_col not in df.columns:
                raise ValueError("time_col must be provided and exist in dataframe for time-based split.")
            
            df = df.sort_values(by=time_col)
            split_idx = int(len(df) * (1 - test_size))
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
            return train, test
        
        else:
            raise ValueError("Unknown split_type. Use 'random' or 'time'.")

    @staticmethod
    def generate_synthetic_data(n_samples: int = 1000, leakage_type: str = 'none') -> pd.DataFrame:
        """
        Generates synthetic data for demonstration.
        leakage_type: 'none', 'target_leakage', 'temporal_leakage'
        """
        np.random.seed(42)
        
        # Base features
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(10, 2, n_samples),
            'feature_3': np.random.exponential(1, n_samples),
            'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Target variable (Linear combination + noise)
        df['target'] = (
            2 * df['feature_1'] + 
            0.5 * df['feature_2'] + 
            np.random.normal(0, 0.5, n_samples)
        )
        # Convert to binary classification for variety
        df['target_class'] = (df['target'] > df['target'].median()).astype(int)
        
        if leakage_type == 'target_leakage':
            # Create a feature that is a noisy version of the future target
            # e.g., 'invoice_status' which is only known after target 'is_churned'
            df['leaky_feature_target_proxy'] = df['target_class'] * 0.95 + np.random.normal(0, 0.05, n_samples)
            
        elif leakage_type == 'temporal_leakage':
            # Simulate time series where future data leaks into past
            # e.g., using a feature that aggregates future information
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
            # A feature that "peaks" ahead of the target in time (not real causality, but data error)
            # transform target to be dependent on next row's feature_1 (impossible in real time)
            df['leaky_feature_future_lookahead'] = df['feature_1'].shift(-1).fillna(0)
            
        elif leakage_type == 'none':
            pass
            
        return df
