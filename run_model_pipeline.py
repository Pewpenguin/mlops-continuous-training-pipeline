#!/usr/bin/env python
"""
Model Pipeline Runner

This script runs the model pipeline using the configuration file.
It demonstrates how to use the model pipeline module to process data and train models.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from pipelines.data_pipeline import create_data_pipeline
from pipelines.model_pipeline import create_model_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the model pipeline."""
    # Path to the configuration files
    data_config_path = 'data/config/sample_pipeline_config.json'
    model_config_path = 'data/config/sample_model_config.json'
    
    # First run the data pipeline to get processed data
    logger.info(f"Creating data pipeline using configuration: {data_config_path}")
    data_pipeline = create_data_pipeline(data_config_path)
    
    logger.info("Running data pipeline...")
    processed_data, data_metadata = data_pipeline.run()
    
    # Create and run the model pipeline
    logger.info(f"Creating model pipeline using configuration: {model_config_path}")
    
    # Check if model config exists, if not create a sample one
    if not os.path.exists(model_config_path):
        logger.info(f"Model configuration not found. Creating sample configuration at {model_config_path}")
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        
        # Sample model configuration
        sample_model_config = {
            'trainer': {
                'model_type': 'random_forest',
                'task_type': 'classification',  # or 'regression'
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            },
            'evaluator': {
                'task_type': 'classification',  # or 'regression'
                'metrics': ['accuracy', 'precision', 'recall', 'f1']  # or ['mse', 'rmse', 'mae', 'r2'] for regression
            },
            'registry': {
                'experiment_name': 'sample_experiment',
                'model_name': 'sample_model',
                'tracking_uri': None  # Set to MLflow tracking server URI if available
            },
            'test_size': 0.2,
            'val_size': 0.25,
            'random_state': 42,
            'register_model': True,
            'model_stage': 'Staging'  # 'None', 'Staging', or 'Production'
        }
        
        with open(model_config_path, 'w') as f:
            json.dump(sample_model_config, f, indent=4)
    
    # Load model configuration
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create model pipeline
    model_pipeline = create_model_pipeline(model_config)
    
    # Prepare data for model training
    target_column = model_config.get('target_column', 'target')
    if target_column not in processed_data.columns:
        logger.warning(f"Target column '{target_column}' not found in processed data. Using the last column as target.")
        target_column = processed_data.columns[-1]
    
    X = processed_data.drop(columns=[target_column]).values
    y = processed_data[target_column].values
    
    # Run the model pipeline
    logger.info("Running model pipeline...")
    run_name = model_config.get('run_name', 'model_training_run')
    results = model_pipeline.run(X, y, run_name)
    
    # Log model results
    logger.info(f"Model pipeline completed. Model type: {results['model_type']}")
    logger.info(f"Validation metrics: {results['val_metrics']}")
    logger.info(f"Test metrics: {results['test_metrics']}")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"MLflow run ID: {results['mlflow_run_id']}")
    
    if 'model_version' in results:
        logger.info(f"Model registered as version: {results['model_version']}")
    
    return results

if __name__ == "__main__":
    main()