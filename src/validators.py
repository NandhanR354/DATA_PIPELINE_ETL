"""
Data Validators Module
Handles data quality checks and validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

class DataValidator:
    """Data validation and quality checks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Comprehensive data validation"""
        self.logger.info("Starting data validation...")

        validation_results = []

        # Check for empty dataframe
        validation_results.append(self._check_empty_dataframe(df))

        # Check data types
        validation_results.append(self._check_data_types(df))

        # Check for duplicates
        validation_results.append(self._check_duplicates(df))

        # Check data ranges
        validation_results.append(self._check_data_ranges(df))

        # Check missing values
        validation_results.append(self._check_missing_values(df))

        # Generate validation report
        self._generate_validation_report(df, validation_results)

        # Return True if all validations pass
        all_passed = all(validation_results)
        self.logger.info(f"Data validation {'PASSED' if all_passed else 'FAILED'}")

        return all_passed

    def _check_empty_dataframe(self, df: pd.DataFrame) -> bool:
        """Check if dataframe is empty"""
        is_valid = not df.empty
        if not is_valid:
            self.logger.error("Dataframe is empty")
        else:
            self.logger.info(f"Dataframe shape: {df.shape}")
        return is_valid

    def _check_data_types(self, df: pd.DataFrame) -> bool:
        """Check data types are as expected"""
        self.logger.info("Checking data types...")

        # Log data types
        for col, dtype in df.dtypes.items():
            self.logger.info(f"Column '{col}': {dtype}")

        # Basic validation - no columns should be all null
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            self.logger.warning(f"Columns with all null values: {null_columns}")
            return False

        return True

    def _check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate rows"""
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate rows")
            return False
        else:
            self.logger.info("No duplicate rows found")
            return True

    def _check_data_ranges(self, df: pd.DataFrame) -> bool:
        """Check if numerical data is within expected ranges"""
        self.logger.info("Checking data ranges...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.logger.error(f"Column '{col}' has {inf_count} infinite values")
                return False

            # Check for extremely large values (basic outlier detection)
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            outlier_count = ((df[col] > q99 * 10) | (df[col] < q01 * 10)).sum()

            if outlier_count > len(df) * 0.05:  # More than 5% outliers
                self.logger.warning(f"Column '{col}' has {outlier_count} potential outliers")

        return True

    def _check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check missing values pattern"""
        self.logger.info("Checking missing values...")

        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()

        if total_missing > 0:
            self.logger.info(f"Total missing values: {total_missing}")

            # Log missing values per column
            for col, missing_count in missing_summary[missing_summary > 0].items():
                missing_pct = (missing_count / len(df)) * 100
                self.logger.info(f"Column '{col}': {missing_count} missing ({missing_pct:.2f}%)")

                # Fail if more than 50% missing in any column
                if missing_pct > 50:
                    self.logger.error(f"Column '{col}' has too many missing values ({missing_pct:.2f}%)")
                    return False

        return True

    def _generate_validation_report(self, df: pd.DataFrame, validation_results: List[bool]):
        """Generate detailed validation report"""
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'validation_passed': all(validation_results),
            'validation_details': {
                'empty_check': validation_results[0],
                'data_types_check': validation_results[1],
                'duplicates_check': validation_results[2],
                'ranges_check': validation_results[3],
                'missing_values_check': validation_results[4]
            }
        }

        # Save report
        import yaml
        from pathlib import Path

        report_path = Path('outputs/validation_report.yaml')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        self.logger.info(f"Validation report saved: {report_path}")
