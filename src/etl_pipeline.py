"""
ETL Pipeline for Data Processing
Author: CODTECH Internship
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sqlite3
import yaml
import joblib

from data_loader import DataLoader
from transformers import DataTransformer
from validators import DataValidator

class ETLPipeline:
    """Complete ETL Pipeline for data processing"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.data_loader = DataLoader(self.config)
        self.transformer = DataTransformer(self.config)
        self.validator = DataValidator(self.config)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'data': {
                    'source_path': 'data/raw/',
                    'output_path': 'data/processed/',
                    'formats': ['csv', 'parquet', 'sqlite']
                },
                'processing': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'scaling_method': 'standard'
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'logs/etl_pipeline.log'
                }
            }

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract(self, source_path: str = None) -> pd.DataFrame:
        """Extract data from source"""
        self.logger.info("Starting data extraction...")

        if source_path is None:
            source_path = self.config['data']['source_path']

        # Load sample data if no specific source provided
        if not Path(source_path).exists():
            self.logger.info("Creating sample dataset...")
            df = self._create_sample_data()
        else:
            df = self.data_loader.load_data(source_path)

        self.logger.info(f"Extracted {len(df)} rows and {len(df.columns)} columns")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform and preprocess data"""
        self.logger.info("Starting data transformation...")

        # Data validation
        if not self.validator.validate_data(df):
            raise ValueError("Data validation failed")

        # Apply transformations
        df_transformed = self.transformer.apply_transformations(df)

        # Feature engineering
        df_transformed = self.transformer.engineer_features(df_transformed)

        self.logger.info(f"Transformation completed. Shape: {df_transformed.shape}")
        return df_transformed

    def load(self, df: pd.DataFrame, output_path: str = None):
        """Load processed data to various formats"""
        self.logger.info("Starting data loading...")

        if output_path is None:
            output_path = self.config['data']['output_path']

        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # Save in different formats
        formats = self.config['data']['formats']

        if 'csv' in formats:
            csv_path = output_dir / 'processed_data.csv'
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Data saved to CSV: {csv_path}")

        if 'parquet' in formats:
            parquet_path = output_dir / 'processed_data.parquet'
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"Data saved to Parquet: {parquet_path}")

        if 'sqlite' in formats:
            sqlite_path = output_dir / 'processed_data.db'
            conn = sqlite3.connect(sqlite_path)
            df.to_sql('processed_data', conn, if_exists='replace', index=False)
            conn.close()
            self.logger.info(f"Data saved to SQLite: {sqlite_path}")

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'feature_1': np.random.normal(50, 15, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.uniform(0, 100, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }

        df = pd.DataFrame(data)

        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        df.loc[missing_indices, 'feature_1'] = np.nan

        # Save sample data
        sample_path = Path('data/sample_data.csv')
        sample_path.parent.mkdir(exist_ok=True)
        df.to_csv(sample_path, index=False)

        return df

    def run_full_pipeline(self, source_path: str = None, output_path: str = None):
        """Run the complete ETL pipeline"""
        try:
            self.logger.info("Starting ETL Pipeline...")

            # Extract
            df = self.extract(source_path)

            # Transform
            df_transformed = self.transform(df)

            # Load
            self.load(df_transformed, output_path)

            self.logger.info("ETL Pipeline completed successfully!")

            # Generate summary report
            self._generate_report(df, df_transformed)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _generate_report(self, original_df: pd.DataFrame, processed_df: pd.DataFrame):
        """Generate pipeline execution report"""
        report = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'columns_original': list(original_df.columns),
            'columns_processed': list(processed_df.columns),
            'missing_values_original': original_df.isnull().sum().to_dict(),
            'data_types_processed': processed_df.dtypes.to_dict()
        }

        report_path = Path('outputs/pipeline_report.yaml')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        self.logger.info(f"Pipeline report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='ETL Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--source', type=str, help='Source data path')
    parser.add_argument('--output', type=str, help='Output data path')
    parser.add_argument('--step', type=str, choices=['extract', 'transform', 'load', 'all'], 
                       default='all', help='Pipeline step to run')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ETLPipeline(args.config)

    if args.step == 'all':
        pipeline.run_full_pipeline(args.source, args.output)
    else:
        # Run individual steps (simplified for demo)
        df = pipeline.extract(args.source)
        if args.step in ['transform', 'load']:
            df = pipeline.transform(df)
        if args.step == 'load':
            pipeline.load(df, args.output)

if __name__ == "__main__":
    main()
