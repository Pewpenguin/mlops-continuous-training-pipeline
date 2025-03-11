"""Main Script

This script demonstrates how to use the data and model pipelines together
to create an end-to-end machine learning workflow.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pipelines.data_pipeline import create_data_pipeline
from pipelines.model_pipeline import create_model_pipeline

def run_ml_pipeline(data_config_path: str, model_config_path: str, target_column: str):
    """Run the complete machine learning pipeline.
    
    Args:
        data_config_path: Path to the data pipeline configuration file
        model_config_path: Path to the model pipeline configuration file
        target_column: Name of the target column in the dataset
    """
    # Create and run data pipeline
    data_pipeline = create_data_pipeline(data_config_path)
    processed_data, data_metadata = data_pipeline.run()
    
    # Prepare features and target
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]
    
    # Create and run model pipeline
    model_pipeline = create_model_pipeline(model_config_path)
    model_metadata = model_pipeline.run(X.values, y.values, 'model_v1')
    
    # Save pipeline metadata
    pipeline_metadata = {
        'data_pipeline': data_metadata,
        'model_pipeline': model_metadata
    }
    
    os.makedirs('metadata', exist_ok=True)
    with open('metadata/pipeline_metadata.json', 'w') as f:
        json.dump(pipeline_metadata, f, indent=4)

if __name__ == "__main__":
    # Create sample data for testing
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/sample_data.csv', index=False)
    
    # Run the pipeline
    run_ml_pipeline(
        data_config_path='data/config/sample_pipeline_config.json',
        model_config_path='models/config/sample_model_config.json',
        target_column='target'
    )
    
    print("ML pipeline completed successfully!")
    print("Check metadata/pipeline_metadata.json for results.")