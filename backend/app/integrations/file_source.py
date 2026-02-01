import pandas as pd
from typing import Dict, Any
from .base import DataSource
import os

class FileDataSource(DataSource):
    """
    Handles loading from local files (CSV, Parquet).
    """
    
    def connect(self, config: Dict[str, Any]) -> bool:
        # File source doesn't need explicit connection, just check paths exist?
        # For uniformity, we return True
        return True

    def fetch_data(self, source_identifier: str) -> pd.DataFrame:
        if not os.path.exists(source_identifier):
            raise FileNotFoundError(f"File not found: {source_identifier}")
            
        if source_identifier.endswith('.parquet'):
            return pd.read_parquet(source_identifier)
        
        # Default to CSV
        return pd.read_csv(source_identifier)
