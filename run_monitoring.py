#!/usr/bin/env python
"""
Monitoring Runner

This script runs the data drift monitoring to detect changes in data distribution.
It compares a reference dataset with a current dataset to identify when model retraining might be necessary.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from monitoring.data_drift import DataDriftMonitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the data drift monitoring."""
    # Path to the configuration file
    config_path = 'data/config/sample_monitoring_config.json'
    
    # Check if monitoring config exists, if not create a sample one
    if not os.path.exists(config_path):
        logger.info(f"Monitoring configuration not found. Creating sample configuration at {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Sample monitoring configuration
        sample_monitoring_config = {
            'reference_data_path': 'data/processed/reference_data.csv',
            'current_data_path': 'data/processed/current_data.csv',
            'drift_threshold': 0.05,  # p-value threshold for KS test
            'overall_drift_threshold': 0.3,  # % of columns that need to drift to trigger an alert
            'categorical_columns': ['category1', 'category2'],
            'numerical_columns': ['feature1', 'feature2', 'feature3'],
            'datetime_column': 'timestamp',
            'reports_dir': 'monitoring/reports'
        }
        
        with open(config_path, 'w') as f:
            json.dump(sample_monitoring_config, f, indent=4)
    
    # Load monitoring configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create data drift monitor
    logger.info("Creating data drift monitor...")
    monitor = DataDriftMonitor(config)
    
    # Check if reference data exists
    reference_data_path = config.get('reference_data_path')
    if not os.path.exists(reference_data_path):
        logger.error(f"Reference data not found at {reference_data_path}")
        logger.info("Please provide a reference dataset or run the data pipeline first")
        return
    
    # Check if current data exists
    current_data_path = config.get('current_data_path')
    if not os.path.exists(current_data_path):
        logger.error(f"Current data not found at {current_data_path}")
        logger.info("Please provide a current dataset to compare against the reference data")
        return
    
    # Load reference data
    logger.info(f"Loading reference data from {reference_data_path}")
    monitor.load_reference_data(reference_data_path)
    
    # Load current data
    logger.info(f"Loading current data from {current_data_path}")
    file_ext = os.path.splitext(current_data_path)[1].lower()
    if file_ext == '.csv':
        current_data = pd.read_csv(current_data_path)
    elif file_ext == '.parquet':
        current_data = pd.read_parquet(current_data_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return
    
    # Detect data drift
    logger.info("Detecting data drift...")
    results = monitor.detect_drift(current_data)
    
    # Log results
    logger.info(f"Data drift detected: {results['data_drift_detected']}")
    logger.info(f"Drift score: {results['drift_score']:.2f} ({results['columns_drifted']} out of {results['total_columns']} columns)")
    
    if results['data_drift_detected']:
        logger.warning("Significant data drift detected! Model retraining may be necessary.")
        
        # Log drifted columns
        drifted_columns = [col for col, result in results['column_drifts'].items() if result['drift_detected']]
        logger.warning(f"Columns with drift: {drifted_columns}")
    else:
        logger.info("No significant data drift detected.")
    
    # Log report path
    if 'report_path' in results and results['report_path']:
        logger.info(f"Detailed drift report saved to: {results['report_path']}")
    
    return results

if __name__ == "__main__":
    main()