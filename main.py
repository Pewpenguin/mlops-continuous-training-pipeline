#!/usr/bin/env python
"""
MLOps Pipeline Main Entry Point

This script serves as the main entry point for the MLOps pipeline.
It provides a command-line interface to run different components of the pipeline.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from pipelines.data_pipeline import create_data_pipeline
from pipelines.model_pipeline import create_model_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the pipeline."""
    directories = [
        'data/raw',
        'data/processed',
        'data/config',
        'models',
        'monitoring/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_data_pipeline(config_path):
    """Run the data pipeline."""
    from run_data_pipeline import main as run_data
    logger.info(f"Running data pipeline with config: {config_path}")
    return run_data()

def run_model_pipeline(config_path):
    """Run the model pipeline."""
    from run_model_pipeline import main as run_model
    logger.info(f"Running model pipeline with config: {config_path}")
    return run_model()

def run_monitoring(config_path):
    """Run the monitoring pipeline."""
    from run_monitoring import main as run_monitor
    logger.info(f"Running monitoring with config: {config_path}")
    return run_monitor()

def start_api(host='0.0.0.0', port=8000):
    """Start the model serving API."""
    import uvicorn
    logger.info(f"Starting model API server on {host}:{port}")
    uvicorn.run("api.model_api:app", host=host, port=port, reload=True)

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
    
    logger.info("ML pipeline completed successfully!")
    logger.info("Check metadata/pipeline_metadata.json for results.")
    
    return pipeline_metadata

def main():
    """Main entry point for the MLOps pipeline."""
    parser = argparse.ArgumentParser(description="MLOps Pipeline CLI")
    parser.add_argument('--setup', action='store_true', help='Setup directories')
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline component to run')
    
    # Data pipeline parser
    data_parser = subparsers.add_parser('data', help='Run data pipeline')
    data_parser.add_argument('--config', default='data/config/sample_pipeline_config.json', 
                            help='Path to data pipeline config')
    
    # Model pipeline parser
    model_parser = subparsers.add_parser('model', help='Run model pipeline')
    model_parser.add_argument('--config', default='data/config/sample_model_config.json', 
                             help='Path to model pipeline config')
    
    # Monitoring parser
    monitor_parser = subparsers.add_parser('monitor', help='Run monitoring')
    monitor_parser.add_argument('--config', default='data/config/sample_monitoring_config.json', 
                               help='Path to monitoring config')
    
    # API parser
    api_parser = subparsers.add_parser('api', help='Start model serving API')
    api_parser.add_argument('--host', default='0.0.0.0', help='API host')
    api_parser.add_argument('--port', type=int, default=8000, help='API port')
    
    # Full pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--data-config', default='data/config/sample_pipeline_config.json', 
                                help='Path to data pipeline config')
    pipeline_parser.add_argument('--model-config', default='data/config/sample_model_config.json', 
                                help='Path to model pipeline config')
    pipeline_parser.add_argument('--target-column', default='target',
                                help='Name of the target column in the dataset')
    
    # Generate sample data parser
    sample_parser = subparsers.add_parser('generate-sample', help='Generate sample data for testing')
    sample_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Setup directories if requested
    if args.setup:
        setup_directories()
    
    # Run the requested command
    if args.command == 'data':
        run_data_pipeline(args.config)
    elif args.command == 'model':
        run_model_pipeline(args.config)
    elif args.command == 'monitor':
        run_monitoring(args.config)
    elif args.command == 'api':
        start_api(args.host, args.port)
    elif args.command == 'pipeline':
        run_ml_pipeline(args.data_config, args.model_config, args.target_column)
    elif args.command == 'generate-sample':
        # Generate synthetic dataset
        os.makedirs('data/raw', exist_ok=True)
        np.random.seed(42)
        n_samples = args.samples
        
        data = {
            'id': range(n_samples),
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('data/raw/sample_data.csv', index=False)
        logger.info(f"Generated {n_samples} sample records at data/raw/sample_data.csv")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
    
    return pipeline_metadata

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
        'category': np.random.choice(['