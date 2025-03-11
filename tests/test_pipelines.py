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
        'metrics': ['accuracy', 'precision', 'recall', 'f1'],
        'experiment_name': 'test_experiment',
        'model_name': 'test_model',
        'model_params': {'max_depth': 5, 'n_estimators': 100},
        'preprocessing_steps': [  # Add preprocessing configuration
            {'columns': ['category'], 'type': 'encode_categorical'}
        ]
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
    assert pipeline.config == model_config

def test_data_pipeline_execution(sample_data, data_config, tmp_path):
    """Test data pipeline execution."""
    # Setup temporary paths
    input_path = tmp_path / 'raw'
    input_path.mkdir()
    output_path = tmp_path / 'processed'
    output_path.mkdir()
    
    # Save sample data
    sample_data.to_csv(input_path / 'sample_data.csv', index=False)
    
    # Update config with temporary paths
    data_config['input_path'] = str(input_path / 'sample_data.csv')
    data_config['output_path'] = str(output_path / 'sample_data.csv')
    
    # Create and run pipeline
    pipeline = create_data_pipeline(data_config)
    processed_data, metadata = pipeline.run()
    
    # Verify results
    assert processed_data is not None
    assert isinstance(processed_data, pd.DataFrame)
    assert metadata is not None
    assert isinstance(metadata, dict)
    
    # Check if processed data was saved
    assert os.path.exists(data_config['output_path'])

def test_model_pipeline_execution(sample_data, model_config):
    """Test model pipeline execution."""
    # Prepare data
    X = sample_data.drop(columns=['target', 'id'])
    y = sample_data['target']
    
    # Create and run pipeline
    pipeline = create_model_pipeline(model_config)
    metadata = pipeline.run(X.values, y.values, 'test_run')
    
    # Verify results
    assert metadata is not None
    assert isinstance(metadata, dict)
    assert 'metrics' in metadata
    assert 'model_path' in metadata
    
    # Check if model was saved
    assert os.path.exists(metadata['model_path'])

def test_end_to_end_pipeline(sample_data, data_config, model_config, tmp_path):
    """Test end-to-end pipeline execution."""
    # Setup temporary paths
    input_path = tmp_path / 'raw'
    input_path.mkdir()
    output_path = tmp_path / 'processed'
    output_path.mkdir()
    
    # Save sample data
    sample_data.to_csv(input_path / 'sample_data.csv', index=False)
    
    # Update data config with temporary paths
    data_config['input_path'] = str(input_path / 'sample_data.csv')
    data_config['output_path'] = str(output_path / 'sample_data.csv')
    
    # Run data pipeline
    data_pipeline = create_data_pipeline(data_config)
    processed_data, data_metadata = data_pipeline.run()
    
    # Prepare features and target
    X = processed_data.drop(columns=['target'])
    y = processed_data['target']
    
    # Run model pipeline
    model_pipeline = create_model_pipeline(model_config)
    model_metadata = model_pipeline.run(X.values, y.values, 'test_run')
    
    # Verify results
    assert data_metadata is not None
    assert model_metadata is not None
    assert 'metrics' in model_metadata
    assert 'model_path' in model_metadata
    
    # Check if all artifacts were saved
    assert os.path.exists(data_config['output_path'])
    assert os.path.exists(model_metadata['model_path'])