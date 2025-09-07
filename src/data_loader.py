"""
Data Loader Module
Handles data extraction from various sources
"""

import pandas as pd
import requests
from pathlib import Path
import logging
from typing import Union, Dict, Any

class DataLoader:
    """Data loader for various data sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load data from various sources"""
        source_path = Path(source)

        if source_path.suffix == '.csv':
            return self._load_csv(source_path)
        elif source_path.suffix == '.parquet':
            return self._load_parquet(source_path)
        elif source_path.suffix in ['.xlsx', '.xls']:
            return self._load_excel(source_path)
        else:
            # Try to load from URL
            if str(source).startswith('http'):
                return self._load_from_url(str(source))
            else:
                raise ValueError(f"Unsupported data source: {source}")

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV file"""
        self.logger.info(f"Loading CSV from: {path}")
        return pd.read_csv(path)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        self.logger.info(f"Loading Parquet from: {path}")
        return pd.read_parquet(path)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Load Excel file"""
        self.logger.info(f"Loading Excel from: {path}")
        return pd.read_excel(path)

    def _load_from_url(self, url: str) -> pd.DataFrame:
        """Load data from URL"""
        self.logger.info(f"Loading data from URL: {url}")
        try:
            if url.endswith('.csv'):
                return pd.read_csv(url)
            else:
                response = requests.get(url)
                response.raise_for_status()
                # Assume CSV for simplicity
                from io import StringIO
                return pd.read_csv(StringIO(response.text))
        except Exception as e:
            self.logger.error(f"Failed to load from URL: {e}")
            raise
