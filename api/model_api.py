"""Model API Module

This module provides a FastAPI application for serving ML models.
It includes endpoints for model inference and health checks.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLOps Model API",
    description="API for serving machine learning models",
    version="0.1.0"
)

# Define request and response models
class PredictionRequest(BaseModel):
    """Model for prediction request data."""
    features: List[List[float]] = Field(..., description="List of feature vectors for prediction")
    model_name: Optional[str] = Field(None, description="Name of the model to use for prediction")
    model_version: Optional[str] = Field(None, description="Version of the model to use")

class PredictionResponse(BaseModel):
    """Model for prediction response data."""
    predictions: List[Union[float, int, str]] = Field(..., description="Prediction results")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")

# Model loading utility
def load_model(model_name: Optional[str] = None, model_version: Optional[str] = None):
    """Load a model from the local filesystem or MLflow registry.
    
    Args:
        model_name: Name of the model in MLflow registry
        model_version: Version of the model in MLflow registry
        
    Returns:
        Loaded model object
    """
    # If model_name and model_version are provided, load from MLflow
    if model_name and model_version:
        try:
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model {model_name} version {model_version} from MLflow registry")
            return model, {"name": model_name, "version": model_version, "source": "mlflow"}
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {str(e)}")
            # Fall back to local model if MLflow fails
    
    # Otherwise, load the latest local model
    models_dir = Path("models")
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="No models available")
    
    # Find the most recent model file
    model_files = list(models_dir.glob("*_model.joblib"))
    if not model_files:
        raise HTTPException(status_code=404, detail="No models available")
    
    # Sort by modification time (most recent first)
    latest_model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    try:
        model = joblib.load(latest_model_file)
        model_info = {
            "name": latest_model_file.stem.replace("_model", ""),
            "path": str(latest_model_file),
            "source": "local"
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
        "endpoints": ["/predict", "/health"]
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

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the model."""
    try:
        # Load the model
        model, model_info = load_model(request.model_name, request.model_version)
        
        # Convert input features to numpy array
        features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        return {
            "predictions": predictions,
            "model_info": model_info
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the API with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_api:app", host="0.0.0.0", port=8000, reload=True)