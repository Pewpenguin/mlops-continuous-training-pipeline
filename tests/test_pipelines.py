import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

from pipelines.data_pipeline import create_data_pipeline
from pipelines.model_pipeline import create_model_pipeline

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def data_config():
    """Create sample data pipeline configuration."""
    return {
        'input_path': 'data/raw/sample_data.csv',
        'output_path': 'data/processed/sample_data.csv',
        'preprocessing_steps': [
            {'type': 'drop_duplicates'},
            {'type': 'encode_categorical', 'columns': ['category']},
            {'type': 'scale_numerical', 'columns': ['feature1', 'feature2']}
        ]
    }

@pytest.fixture
def model_config():
    """Create sample model pipeline configuration."""
    return {
        'model_type': 'random_forest',
        'task_type': 'classification',
        'model_params': {
            'n_estimators': 100,
            'max_depth': 5
        },
        'metrics': ['accuracy', 'precision', 'recall', 'f1']
    }

def test_data_pipeline_creation(data_config):
    """Test data pipeline creation."""
    pipeline = create_data_pipeline(data_config)
    assert pipeline is not None
    assert pipeline.config == data_config

def test_model_pipeline_creation(model_config):
    """Test model pipeline creation."""
    pipeline = create_model_pipeline(model_config)
    assert pipeline is not None
    assert hasattr(pipeline, 'train')
    assert hasattr(pipeline, 'evaluate')
    assert hasattr(pipeline, 'predict')

def test_model_pipeline_execution(sample_data, model_config):
    """Test model pipeline execution."""
    # Split data into features and target
    X = sample_data.drop(['id', 'target'], axis=1)
    y = sample_data['target']
    
    # Encode categorical features
    X = pd.get_dummies(X, columns=['category'])
    
    # Create and run pipeline
    pipeline = create_model_pipeline(model_config)
    pipeline.train(X, y)
    
    # Test predictions
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
    
    # Test evaluation
    metrics = pipeline.evaluate(X, y)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics

def test_end_to_end_pipeline(sample_data, data_config, model_config):
    """Test end-to-end pipeline execution."""
    # Save sample data
    os.makedirs('data/raw', exist_ok=True)
    sample_data.to_csv('data/raw/sample_data.csv', index=False)
    
    # Run data pipeline
    data_pipeline = create_data_pipeline(data_config)
    processed_data, _ = data_pipeline.run()
    
    # Split processed data and encode categorical features
    X = processed_data.drop(['target'], axis=1)
    X = pd.get_dummies(X, columns=['category'])
    y = processed_data['target']
    
    # Run model pipeline
    model_pipeline = create_model_pipeline(model_config)
    model_pipeline.train(X, y)
    
    # Make predictions
    predictions = model_pipeline.predict(X)
    assert len(predictions) == len(y)
    
    # Clean up
    if os.path.exists('data/raw/sample_data.csv'):
        os.remove('data/raw/sample_data.csv')
    if os.path.exists('data/processed/sample_data.csv'):
        os.remove('data/processed/sample_data.csv')