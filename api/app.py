"""FastAPI Model Serving Application

This module provides a unified REST API for model inference using FastAPI.
It loads trained models from MLflow registry or local storage and exposes
endpoints for making predictions, checking model health, and listing available models.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path to allow imports from other modules
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.model_pipeline import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model API",
    description="API for serving machine learning models with MLflow integration",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request and response models
class PredictionRequest(BaseModel):
    """Model for prediction request data."""
    features: Union[List[List[float]], Dict[str, Any]] = Field(
        ..., 
        description="Feature values for prediction. Can be a list of feature vectors or a dictionary of feature names and values"
    )
    model_name: Optional[str] = Field(None, description="Name of the model to use for prediction")
    model_version: Optional[str] = Field(None, description="Version of the model to use")

class PredictionResponse(BaseModel):
    """Model for prediction response data."""
    predictions: List[Union[float, int, str]] = Field(..., description="Prediction results")
    prediction_probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities for classification models")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")

class ModelInfo(BaseModel):
    """Model for model information."""
    name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    source: str = Field(..., description="Model source (mlflow or local)")
    creation_time: Optional[str] = Field(None, description="Model creation time")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")

# Model loading utility
def load_model(model_name: Optional[str] = None, model_version: Optional[str] = None):
    """Load a model from the local filesystem or MLflow registry.
    
    Args:
        model_name: Name of the model in MLflow registry
        model_version: Version of the model in MLflow registry
        
    Returns:
        Tuple of (loaded model object, model info dictionary)
    """
    # If model_name and model_version are provided, load from MLflow
    if model_name and model_version:
        try:
            # Set MLflow tracking URI if configured
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get model creation time and metrics if available
            creation_time = None
            metrics = None
            try:
                client = mlflow.tracking.MlflowClient()
                model_details = client.get_model_version(model_name, model_version)
                creation_time = model_details.creation_timestamp
                
                # Get the run that produced this model version
                run_id = model_details.run_id
                if run_id:
                    run = client.get_run(run_id)
                    metrics = run.data.metrics
            except:
                pass
                
            model_info = {
                "name": model_name,
                "version": model_version,
                "source": "mlflow",
                "creation_time": creation_time,
                "metrics": metrics
            }
            
            logger.info(f"Loaded model {model_name} version {model_version} from MLflow registry")
            return model, model_info
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {str(e)}")
            # Fall back to local model if MLflow fails
    
    # Otherwise, load the latest local model
    models_dir = Path("models")
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="No models available")
    
    # First try to find models in the trained directory
    trained_dir = models_dir / "trained"
    if trained_dir.exists():
        model_files = list(trained_dir.glob("*.joblib")) + list(trained_dir.glob("*.pkl"))
        if model_files:
            # Sort by modification time (most recent first)
            latest_model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            try:
                model = joblib.load(latest_model_file)
                model_info = {
                    "name": latest_model_file.stem,
                    "path": str(latest_model_file),
                    "source": "local",
                    "creation_time": latest_model_file.stat().st_mtime
                }
                logger.info(f"Loaded local model from {latest_model_file}")
                return model, model_info
            except Exception as e:
                logger.error(f"Error loading model from {latest_model_file}: {str(e)}")
    
    # If no models in trained directory, look in the root models directory
    model_files = list(models_dir.glob("*_model.joblib")) + list(models_dir.glob("*.pkl"))
    if not model_files:
        raise HTTPException(status_code=404, detail="No models available")
    
    # Sort by modification time (most recent first)
    latest_model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    try:
        model = joblib.load(latest_model_file)
        model_info = {
            "name": latest_model_file.stem.replace("_model", ""),
            "path": str(latest_model_file),
            "source": "local",
            "creation_time": latest_model_file.stat().st_mtime
        }
        logger.info(f"Loaded local model from {latest_model_file}")
        return model, model_info
    except Exception as e:
        logger.error(f"Error loading local model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MLOps Model API",
        "version": "0.1.0",
        "endpoints": ["/predict", "/health", "/models"]
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Try to load a model to verify system health
        model, _ = load_model()
        return {"status": "healthy", "model_available": True}
    except HTTPException as e:
        if e.status_code == 404:
            # No models available is not a system error
            return {"status": "healthy", "model_available": False}
        else:
            return {"status": "unhealthy", "error": str(e.detail)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/models")
async def list_models():
    """List available model versions."""
    models = []
    
    # Check local models
    models_dir = Path("models")
    if models_dir.exists():
        # Check trained directory
        trained_dir = models_dir / "trained"
        if trained_dir.exists():
            local_models = list(trained_dir.glob("*.joblib")) + list(trained_dir.glob("*.pkl"))
            for model_file in local_models:
                models.append({
                    "name": model_file.stem,
                    "version": "latest",
                    "source": "local",
                    "path": str(model_file),
                    "last_modified": model_file.stat().st_mtime
                })
        
        # Check root models directory
        root_models = list(models_dir.glob("*_model.joblib")) + list(models_dir.glob("*.pkl"))
        for model_file in root_models:
            models.append({
                "name": model_file.stem.replace("_model", ""),
                "version": "latest",
                "source": "local",
                "path": str(model_file),
                "last_modified": model_file.stat().st_mtime
            })
    
    # Check MLflow models if MLflow is configured
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            registered_models = client.list_registered_models()
            
            for rm in registered_models:
                model_name = rm.name
                versions = client.get_latest_versions(model_name)
                
                for version in versions:
                    models.append({
                        "name": model_name,
                        "version": version.version,
                        "source": "mlflow",
                        "stage": version.current_stage,
                        "creation_time": version.creation_timestamp
                    })
    except Exception as e:
        logger.warning(f"Could not retrieve MLflow models: {str(e)}")
    
    return {"models": models}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the model."""
    try:
        # Load the model
        model, model_info = load_model(request.model_name, request.model_version)
        
        # Process input features
        if isinstance(request.features, dict):
            # Convert dictionary to DataFrame
            features_df = pd.DataFrame([request.features])
            features = features_df.values
        else:
            # Convert list to numpy array
            features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        # Get prediction probabilities if available (for classification)
        prediction_probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                prediction_probabilities = model.predict_proba(features).tolist()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        return {
            "predictions": predictions,
            "prediction_probabilities": prediction_probabilities,
            "model_info": model_info
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(request: PredictionRequest):
    """Make batch predictions using the model."""
    try:
        # Load the model
        model, model_info = load_model(request.model_name, request.model_version)
        
        # Process input features
        if isinstance(request.features, dict):
            # Convert dictionary to DataFrame
            features_df = pd.DataFrame([request.features])
            features = features_df.values
        else:
            # Convert list to numpy array
            features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        # Get prediction probabilities if available (for classification)
        prediction_probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                prediction_probabilities = model.predict_proba(features).tolist()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        return {
            "predictions": predictions,
            "prediction_probabilities": prediction_probabilities,
            "model_info": model_info,
            "batch_size": len(predictions)
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info(model_name: Optional[str] = None, model_version: Optional[str] = None):
    """Get information about a specific model."""
    try:
        # Load the model to get its info
        _, model_info = load_model(model_name, model_version)
        return {"model_info": model_info}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Run the API with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)