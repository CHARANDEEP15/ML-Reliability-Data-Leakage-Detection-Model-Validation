import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any, Dict
from backend.app.integrations.base import DataSource
from backend.app.integrations.file_source import FileDataSource
from backend.app.integrations.snowflake import SnowflakeDataSource
from backend.app.integrations.bigquery import BigQueryDataSource
from sklearn.model_selection import train_test_split

class DataLoader:
    @staticmethod
    def get_source(source_type: str) -> DataSource:
        if source_type == 'file':
            return FileDataSource()
        elif source_type == 'snowflake':
            return SnowflakeDataSource()
        elif source_type == 'bigquery':
            return BigQueryDataSource()
        raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def load_data(source_identifier: str, source_type: str = 'file', config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Universal data loader.
        """
        source = DataLoader.get_source(source_type)
        source.connect(config or {})
        return source.fetch_data(source_identifier)

    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """
        Legacy wrapper for backward compatibility.
        """
        return DataLoader.load_data(file_path, 'file')

    @staticmethod
    def preprocess_data(df: pd.DataFrame, auto_convert: bool = False) -> pd.DataFrame:
        if not auto_convert:
            return df
            
        # Convert object columns to numeric (Label Encoding or coercion)
        for col in df.select_dtypes(include=['object', 'category']).columns:
            # Try numeric first (e.g. "1.5", "2")
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # If still object, encode
            if df[col].dtype == 'object':
                 df[col] = df[col].astype('category').cat.codes
        return df

    @staticmethod
    def split_data(df: pd.DataFrame, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data based on config (audit config object or dict).
        """
        # Handle dict or object safely
        if isinstance(config, dict):
            split_strategy = config.get('split_strategy', 'random')
            time_col = config.get('time_col')
        else:
            split_strategy = getattr(config, 'split_strategy', 'random')
            time_col = getattr(config, 'time_col', None)
        
        if split_strategy == 'time':
            if not time_col or time_col not in df.columns:
                raise ValueError("Time column required for time-based split")
            df = df.sort_values(by=time_col)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
        return train_df, test_df
