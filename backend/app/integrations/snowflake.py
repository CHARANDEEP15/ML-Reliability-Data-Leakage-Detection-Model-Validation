from typing import Dict, Any
import pandas as pd
try:
    import snowflake.connector
except ImportError:
    snowflake = None

from .base import DataSource

class SnowflakeDataSource(DataSource):
    """
    Connects to Snowflake and fetches data via SQL query.
    Config requires: user, password, account, warehouse, database, schema.
    """
    
    def __init__(self):
        self.conn = None
        
    def connect(self, config: Dict[str, Any]) -> bool:
        if not snowflake:
            raise ImportError("snowflake-connector-python is not installed.")
            
        try:
            self.conn = snowflake.connector.connect(
                user=config['user'],
                password=config['password'],
                account=config['account'],
                warehouse=config.get('warehouse'),
                database=config.get('database'),
                schema=config.get('schema')
            )
            return True
        except Exception as e:
            print(f"Snowflake Connection Failed: {e}")
            raise e

    def fetch_data(self, source_identifier: str) -> pd.DataFrame:
        """
        source_identifier: The SQL Query to execute.
        """
        if not self.conn:
            raise ConnectionError("Not connected to Snowflake.")
            
        try:
            # simple fetch using pandas read_sql
            # Note: read_sql requires SQLAlchemy usually for cleaner syntax, 
            # but snowflake connector has a pandas method or cursor fetch.
            # Best practice with connector: curator -> fetch_pandas_all()
            
            cur = self.conn.cursor()
            cur.execute(source_identifier)
            df = cur.fetch_pandas_all()
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to execute Snowflake query: {e}")
        finally:
            # Should we close? Ideally yes, but maybe we want to reuse? 
            # For this 'audit' job, it's one-off.
            if self.conn:
                self.conn.close()
