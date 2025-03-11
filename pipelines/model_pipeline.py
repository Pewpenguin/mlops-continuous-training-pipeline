\"""Model Pipeline Module

This module handles model training, evaluation, and registration.
It includes components for training models, evaluating performance,
and registering models with MLflow.
"""

import os
import pandas as pd
import numpy as np
import mlflow
import joblib
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model trainer.
        
        Args:
            config: Configuration dictionary with model and training parameters
        """
        self.config = config
        self.model_type = config.get('model_type', 'random_forest')
        self.model_params = config.get('model_params', {})
        self.random_state = config.get('random_state', 42)
        self.model = None
        
    def _create_model(self) -> BaseEstimator:
        """Create a model instance based on the configuration.
        
        Returns:
            Initialized model instance
        """
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            task_type = self.config.get('task_type', 'classification')
            
            if task_type == 'classification':
                return RandomForestClassifier(random_state=self.random_state, **self.model_params)
            else:
                return RandomForestRegressor(random_state=self.random_state, **self.model_params)
                
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            task_type = self.config.get('task_type', 'classification')
            
            if task_type == 'classification':
                return GradientBoostingClassifier(random_state=self.random_state, **self.model_params)
            else:
                return GradientBoostingRegressor(random_state=self.random_state, **self.model_params)
                
        elif self.model_type == 'linear':
            from sklearn.linear_model import LogisticRegression, LinearRegression
            task_type = self.config.get('task_type', 'classification')
            
            if task_type == 'classification':
                return LogisticRegression(random_state=self.random_state, **self.model_params)
            else:
                return LinearRegression(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} model with parameters: {self.model_params}")
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return self.model
    
    def save_model(self, path: str) -> str:
        """Save the trained model to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            Path to the saved model file
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{self.model_type}_model.joblib")
        
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path


class ModelEvaluator:
    """Evaluates model performance on validation data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.task_type = config.get('task_type', 'classification')
        self.metrics = config.get('metrics', [])
        
    def evaluate(self, model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evaluate the model on validation data.
        
        Args:
            model: Trained model to evaluate
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        y_pred = model.predict(X_val)
        metrics_results = {}
        
        if self.task_type == 'classification':
            # Classification metrics
            if 'accuracy' in self.metrics or not self.metrics:
                metrics_results['accuracy'] = accuracy_score(y_val, y_pred)
                
            if 'precision' in self.metrics:
                metrics_results['precision'] = precision_score(y_val, y_pred, average='weighted')
                
            if 'recall' in self.metrics:
                metrics_results['recall'] = recall_score(y_val, y_pred, average='weighted')
                
            if 'f1' in self.metrics:
                metrics_results['f1'] = f1_score(y_val, y_pred, average='weighted')
                
        else:
            # Regression metrics
            if 'mse' in self.metrics or not self.metrics:
                metrics_results['mse'] = mean_squared_error(y_val, y_pred)
                
            if 'rmse' in self.metrics:
                metrics_results['rmse'] = np.sqrt(mean_squared_error(y_val, y_pred))
                
            if 'mae' in self.metrics:
                from sklearn.metrics import mean_absolute_error
                metrics_results['mae'] = mean_absolute_error(y_val, y_pred)
                
            if 'r2' in self.metrics:
                from sklearn.metrics import r2_score
                metrics_results['r2'] = r2_score(y_val, y_pred)
        
        logger.info(f"Evaluation results: {metrics_results}")
        return metrics_results


class ModelRegistry:
    """Handles model registration and versioning with MLflow."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model registry.
        
        Args:
            config: Configuration dictionary with registry parameters
        """
        self.config = config
        self.experiment_name = config.get('experiment_name', 'default')
        self.model_name = config.get('model_name', 'default_model')
        self.tracking_uri = config.get('tracking_uri', None)
        
        # Set MLflow tracking URI if provided
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create the experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            
        logger.info(f"Using MLflow experiment '{self.experiment_name}' with ID: {self.experiment_id}")
    
    def log_model(self, model: BaseEstimator, metrics: Dict[str, float], params: Dict[str, Any], 
                 artifacts: Optional[Dict[str, str]] = None) -> str:
        """Log a model to MLflow with its metrics, parameters, and artifacts.
        
        Args:
            model: Trained model to log
            metrics: Evaluation metrics
            params: Model parameters
            artifacts: Optional dictionary of artifact paths to log
            
        Returns:
            Run ID of the logged model
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log additional artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
            
            run_id = run.info.run_id
            logger.info(f"Model logged to MLflow with run ID: {run_id}")
            
            return run_id
    
    def register_model(self, run_id: str, stage: str = "None") -> str:
        """Register a model version in the MLflow Model Registry.
        
        Args:
            run_id: Run ID of the logged model
            stage: Stage to assign to the model version (None, Staging, Production)
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(model_uri, self.model_name)
        version = result.version
        
        if stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
        
        logger.info(f"Model registered as version {version} in stage '{stage}'")
        return version


class ModelPipeline:
    """End-to-end model pipeline combining training, evaluation, and registration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config
        self.trainer = ModelTrainer(config.get('trainer', {}))
        self.evaluator = ModelEvaluator(config.get('evaluator', {}))
        self.registry = ModelRegistry(config.get('registry', {}))
        
        # Data split parameters
        self.test_size = config.get('test_size', 0.2)
        self.val_size = config.get('val_size', 0.25)  # % of test_size
        self.random_state = config.get('random_state', 42)
        
    def run(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Run the full model pipeline.
        
        Args:
            data: Preprocessed data for model training
            target_column: Name of the target column
            
        Returns:
            Dictionary with pipeline results
        """
        # Split data into features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        logger.info(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Train model
        model = self.trainer.train(X_train, y_train)
        
        # Evaluate model
        val_metrics = self.evaluator.evaluate(model, X_val, y_val)
        test_metrics = self.evaluator.evaluate(model, X_test, y_test)
        
        # Save model locally
        model_path = self.trainer.save_model("models")
        
        # Log model to MLflow
        run_id = self.registry.log_model(
            model=model,
            metrics={**val_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}},
            params=self.config.get('trainer', {}).get('model_params', {}),
            artifacts={"model": model_path}
        )
        
        # Register model if specified
        version = None
        if self.config.get('register_model', False):
            stage = self.config.get('model_stage', 'None')
            version = self.registry.register_model(run_id, stage)
        
        # Prepare results
        results = {
            'model_type': self.config.get('trainer', {}).get('model_type', 'unknown'),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_path': model_path,
            'mlflow_run_id': run_id
        }
        
        if version:
            results['model_version'] = version
        
        return results


def create_model_pipeline(config_path: str) -> ModelPipeline:
    """Factory function to create a model pipeline from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configured ModelPipeline instance
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return ModelPipeline(config)


if __name__ == "__main__":
    # Example usage
    import json
    
    # Sample configuration
    sample_config = {
        'trainer': {
            'model_type': 'random_forest',
            'task_type': 'classification',
            'model_params': {
                'n_estimators': 100,
                'max_depth':