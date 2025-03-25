"""MLflow Utilities Module

This module provides utility functions for MLflow integration,
including loading environment variables and configuring MLflow.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import mlflow

# Setup logging
logger = logging.getLogger(__name__)

def load_mlflow_config() -> Dict[str, str]:
    """Load MLflow configuration from environment variables.
    
    Returns:
        Dictionary with MLflow configuration
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get MLflow configuration from environment variables
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'default')
    model_registry_stage = os.environ.get('MODEL_REGISTRY_STAGE', 'Staging')
    
    config = {
        'tracking_uri': tracking_uri,
        'experiment_name': experiment_name,
        'model_registry_stage': model_registry_stage
    }
    
    logger.info(f"Loaded MLflow configuration: {config}")
    return config

def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> str:
    """Setup MLflow tracking and experiment.
    
    Args:
        tracking_uri: MLflow tracking URI (overrides environment variable)
        experiment_name: MLflow experiment name (overrides environment variable)
        
    Returns:
        Experiment ID
    """
    # Load configuration from environment variables
    config = load_mlflow_config()
    
    # Override with provided values if any
    if tracking_uri:
        config['tracking_uri'] = tracking_uri
    if experiment_name:
        config['experiment_name'] = experiment_name
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['tracking_uri'])
    logger.info(f"Set MLflow tracking URI to {config['tracking_uri']}")
    
    # Set or create the experiment
    try:
        experiment_id = mlflow.create_experiment(config['experiment_name'])
        logger.info(f"Created new MLflow experiment '{config['experiment_name']}' with ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{config['experiment_name']}' with ID: {experiment_id}")
    
    return experiment_id

def get_latest_model_version(model_name: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the latest version of a model from the MLflow Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        stage: Stage to filter by (None, Staging, Production)
        
    Returns:
        Dictionary with model information or None if not found
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.get_latest_versions(model_name, stages=[stage] if stage else None)
        
        if not versions:
            logger.warning(f"No versions found for model '{model_name}'")
            return None
        
        # Get the latest version
        latest_version = versions[0]
        
        model_info = {
            'name': model_name,
            'version': latest_version.version,
            'stage': latest_version.current_stage,
            'run_id': latest_version.run_id,
            'uri': f"models:/{model_name}/{latest_version.version}"
        }
        
        logger.info(f"Found latest model version: {model_info}")
        return model_info
    except Exception as e:
        logger.error(f"Error getting latest model version: {str(e)}")
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the current MLflow run.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step for the metrics (for iterative algorithms)
    """
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value, step=step)
    logger.info(f"Logged {len(metrics)} metrics to MLflow")


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to the current MLflow run.
    
    Args:
        params: Dictionary of parameter names and values
    """
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
    logger.info(f"Logged {len(params)} parameters to MLflow")


def log_artifacts(artifacts: Dict[str, str]) -> None:
    """Log artifacts to the current MLflow run.
    
    Args:
        artifacts: Dictionary of artifact names and file paths
    """
    for artifact_name, artifact_path in artifacts.items():
        mlflow.log_artifact(artifact_path, artifact_name)
    logger.info(f"Logged {len(artifacts)} artifacts to MLflow")


def transition_model_stage(model_name: str, version: str, stage: str) -> None:
    """Transition a model version to a new stage in the MLflow Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        version: Version of the model to transition
        stage: Target stage (None, Staging, Production, Archived)
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    logger.info(f"Transitioned model '{model_name}' version {version} to stage '{stage}'")


def get_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """Get all versions of a model from the MLflow Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        
    Returns:
        List of dictionaries with model version information
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.get_latest_versions(model_name)
        
        if not versions:
            logger.warning(f"No versions found for model '{model_name}'")
            return []
        
        # Convert to list of dictionaries
        model_versions = [{
            'name': model_name,
            'version': version.version,
            'stage': version.current_stage,
            'run_id': version.run_id,
            'uri': f"models:/{model_name}/{version.version}"
        } for version in versions]
        
        logger.info(f"Found {len(model_versions)} versions for model '{model_name}'")
        return model_versions
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        return []