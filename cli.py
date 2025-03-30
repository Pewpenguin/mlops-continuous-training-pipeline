"""Command Line Interface for MLOps Pipeline

This module provides a unified CLI interface for running all pipeline components,
including data processing, model training, and monitoring tasks.
"""

import click
import json
from pathlib import Path
from pipelines.data_pipeline import create_data_pipeline
from pipelines.model_pipeline import create_model_pipeline
from monitoring.model_monitor import ModelMonitor
from api.model_api import create_api

@click.group()
def cli():
    """MLOps Pipeline CLI - Manage your ML workflows"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def process_data(config_path):
    """Process data using the data pipeline."""
    pipeline = create_data_pipeline(config_path)
    pipeline.run()
    click.echo('Data processing completed successfully!')

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def train_model(config_path):
    """Train a model using the model pipeline."""
    pipeline = create_model_pipeline(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Assuming data is available in the path specified in config
    data_path = config.get('data_path', 'data/processed/train.csv')
    target_column = config.get('target_column')
    
    import pandas as pd
    data = pd.read_csv(data_path)
    results = pipeline.run(data, target_column)
    
    click.echo(f'Model training completed! Results: {results}')

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def monitor_model(config_path):
    """Monitor model performance and data drift."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    monitor = ModelMonitor(config)
    
    # Load reference and current data
    ref_data = pd.read_csv(config['reference_data_path'])
    current_data = pd.read_csv(config['current_data_path'])
    
    # Set reference data
    monitor.set_reference_data(ref_data, config.get('target_column'))
    
    # Detect drift
    drift_detected, drift_metrics = monitor.detect_data_drift(current_data)
    
    click.echo(f'Drift detection completed! Drift detected: {drift_detected}')
    click.echo(f'Drift metrics: {drift_metrics}')

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def serve_model(config_path):
    """Serve the model via REST API."""
    app = create_api(config_path)
    click.echo('Starting model serving API...')
    app.run(host='0.0.0.0', port=8000)

if __name__ == '__main__':
    cli()