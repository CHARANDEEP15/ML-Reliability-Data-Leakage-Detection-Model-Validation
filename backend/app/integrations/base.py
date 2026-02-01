from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class DataSource(ABC):
    """
    Abstract Base Class for Data Integrations (File, Snowflake, BigQuery, etc.)
    """
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Establishes connection to the source.
        Returns True if successful.
        """
        pass

    @abstractmethod
    def fetch_data(self, source_identifier: str) -> pd.DataFrame:
        """
        Fetches data into a Pandas DataFrame.
        source_identifier: File path, Table name, or SQL Query.
        """
        pass
