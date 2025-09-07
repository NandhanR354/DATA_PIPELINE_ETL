# Task 1: Data Pipeline Development

## Overview
This project implements an ETL (Extract, Transform, Load) pipeline using pandas and scikit-learn for data preprocessing, transformation, and loading.

## Features
- Data extraction from multiple sources (CSV, API)
- Data cleaning and preprocessing
- Feature engineering and transformation
- Data validation and quality checks
- Automated loading to different formats (CSV, Parquet, SQLite)
- Comprehensive logging and error handling

## Project Structure
```
Task1_Data_Pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data.csv
├── src/
│   ├── __init__.py
│   ├── etl_pipeline.py
│   ├── data_loader.py
│   ├── transformers.py
│   ├── validators.py
│   └── config.py
├── outputs/
├── logs/
├── requirements.txt
├── setup.py
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run the complete ETL pipeline
python src/etl_pipeline.py

# Run with custom configuration
python src/etl_pipeline.py --config config/custom_config.yaml

# Run specific pipeline step
python src/etl_pipeline.py --step extract
python src/etl_pipeline.py --step transform
python src/etl_pipeline.py --step load
```

## Configuration
Edit `src/config.py` to modify:
- Data source paths
- Transformation parameters
- Output formats and destinations
- Logging settings

## Output
- Processed data in multiple formats (CSV, Parquet, SQLite)
- Data quality report
- Pipeline execution logs
- Performance metrics

## Author
CODTECH Internship - Data Science Track
