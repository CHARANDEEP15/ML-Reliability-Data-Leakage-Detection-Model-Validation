from typing import Dict, Any
import pandas as pd
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

from .base import DataSource
import os

class BigQueryDataSource(DataSource):
    """
    Connects to BigQuery via Service Account or Default Credentials.
    """
    
    def __init__(self):
        self.client = None
        
    def connect(self, config: Dict[str, Any]) -> bool:
        if not bigquery:
            raise ImportError("google-cloud-bigquery is not installed.")
            
        try:
            # If key_path is provided, use it. Otherwise assume environment variable GOOGLE_APPLICATION_CREDENTIALS
            key_path = config.get('key_path')
            if key_path and os.path.exists(key_path):
                self.client = bigquery.Client.from_service_account_json(key_path)
            else:
                self.client = bigquery.Client()
            return True
        except Exception as e:
            print(f"BigQuery Connection Failed: {e}")
            raise e

    def fetch_data(self, source_identifier: str) -> pd.DataFrame:
        """
        source_identifier: The SQL Query.
        """
        if not self.client:
            raise ConnectionError("Not connected to BigQuery.")
            
        try:
            query_job = self.client.query(source_identifier)
            return query_job.to_dataframe()
        except Exception as e:
            raise RuntimeError(f"Failed to execute BigQuery query: {e}")
