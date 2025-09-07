"""
Data Transformers Module
Handles data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, Any, List

class DataTransformer:
    """Data transformation and preprocessing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all data transformations"""
        df_transformed = df.copy()

        # Handle missing values
        df_transformed = self._handle_missing_values(df_transformed)

        # Encode categorical variables
        df_transformed = self._encode_categorical(df_transformed)

        # Scale numerical features
        df_transformed = self._scale_numerical(df_transformed)

        return df_transformed

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values...")

        # Numerical columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with median: {median_val}")

        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with mode: {mode_val}")

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        self.logger.info("Encoding categorical variables...")

        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col != 'target':  # Don't encode target if it's categorical
                unique_vals = df[col].nunique()

                if unique_vals <= 5:  # One-hot encode for low cardinality
                    encoded_cols = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), encoded_cols], axis=1)
                    self.logger.info(f"One-hot encoded {col} ({unique_vals} categories)")
                else:  # Label encode for high cardinality
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                    self.logger.info(f"Label encoded {col} ({unique_vals} categories)")

        return df

    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        self.logger.info("Scaling numerical features...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude target if it exists
        numeric_cols = [col for col in numeric_cols if col != 'target']

        if numeric_cols:
            scaling_method = self.config.get('processing', {}).get('scaling_method', 'standard')

            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                self.logger.warning(f"Unknown scaling method: {scaling_method}, using standard")
                scaler = StandardScaler()

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['numerical'] = scaler
            self.logger.info(f"Applied {scaling_method} scaling to {len(numeric_cols)} features")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        self.logger.info("Engineering new features...")

        # Example feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']

        if len(numeric_cols) >= 2:
            # Create interaction features
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if f'{col1}_x_{col2}' not in df.columns:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

            # Create polynomial features (squares)
            for col in numeric_cols[:3]:  # Limit to first 3 to avoid explosion
                if f'{col}_squared' not in df.columns:
                    df[f'{col}_squared'] = df[col] ** 2

        self.logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
