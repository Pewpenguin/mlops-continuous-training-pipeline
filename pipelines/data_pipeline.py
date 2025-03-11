\"""Data Pipeline Module

This module handles data ingestion, validation, and preprocessing for the ML pipeline.
It includes components for loading data from various sources, validating data quality,
and preprocessing data for model training.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data loading from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data ingestion component.
        
        Args:
            config: Configuration dictionary with data source details
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data'))
        
    def load_data(self) -> pd.DataFrame:
        """Load data from the specified source.
        
        Returns:
            DataFrame containing the loaded data
        """
        source_type = self.config.get('source_type', 'csv')
        source_path = self.config.get('source_path')
        
        if not source_path:
            raise ValueError("Source path must be specified in the configuration")
            
        full_path = self.data_dir / source_path
        
        logger.info(f"Loading data from {full_path}")
        
        if source_type == 'csv':
            return pd.read_csv(full_path)
        elif source_type == 'parquet':
            return pd.read_parquet(full_path)
        elif source_type == 'json':
            return pd.read_json(full_path)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")


class DataValidator:
    """Validates data quality and schema."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize the data validator.
        
        Args:
            schema: Optional schema definition for validation
        """
        self.schema = schema
        
    def validate(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate the data against the schema and quality checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {}
        
        # Basic validation checks
        results['missing_values'] = data.isnull().sum().to_dict()
        results['duplicate_rows'] = data.duplicated().sum()
        
        # Check data types if schema is provided
        if self.schema and 'dtypes' in self.schema:
            type_check = {}
            for col, expected_type in self.schema['dtypes'].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    type_check[col] = {
                        'expected': expected_type,
                        'actual': actual_type,
                        'matches': expected_type == actual_type
                    }
            results['type_check'] = type_check
        
        # Determine if data is valid
        is_valid = True
        if results.get('duplicate_rows', 0) > 0:
            is_valid = False
        
        if 'type_check' in results:
            for col_result in results['type_check'].values():
                if not col_result['matches']:
                    is_valid = False
                    break
        
        logger.info(f"Data validation completed. Valid: {is_valid}")
        return is_valid, results


class DataPreprocessor:
    """Preprocesses data for model training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing steps
        """
        self.config = config
        self.preprocessing_steps = config.get('preprocessing_steps', [])
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the data.
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = data.copy()
        
        for step in self.preprocessing_steps:
            step_type = step.get('type')
            
            if step_type == 'drop_columns':
                columns = step.get('columns', [])
                processed_data = processed_data.drop(columns=columns, errors='ignore')
                
            elif step_type == 'fill_missing':
                columns = step.get('columns', processed_data.columns.tolist())
                method = step.get('method', 'mean')
                
                for col in columns:
                    if col in processed_data.columns:
                        if method == 'mean':
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                        elif method == 'median':
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                        elif method == 'mode':
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
                        elif method == 'constant':
                            processed_data[col] = processed_data[col].fillna(step.get('value', 0))
            
            elif step_type == 'encode_categorical':
                columns = step.get('columns', [])
                method = step.get('method', 'one_hot')
                
                if method == 'one_hot':
                    processed_data = pd.get_dummies(processed_data, columns=columns, drop_first=step.get('drop_first', False))
                # Add more encoding methods as needed
        
        logger.info(f"Data preprocessing completed. Shape: {processed_data.shape}")
        return processed_data


class DataPipeline:
    """End-to-end data pipeline combining ingestion, validation, and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config
        self.ingestion = DataIngestion(config.get('ingestion', {}))
        self.validator = DataValidator(config.get('validation_schema'))
        self.preprocessor = DataPreprocessor(config.get('preprocessing', {}))
        
    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the full data pipeline.
        
        Returns:
            Tuple of (processed_data, pipeline_metadata)
        """
        # Load data
        raw_data = self.ingestion.load_data()
        
        # Validate data
        is_valid, validation_results = self.validator.validate(raw_data)
        
        if not is_valid and not self.config.get('force_continue', False):
            raise ValueError("Data validation failed. Set 'force_continue' to True to proceed anyway.")
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess(raw_data)
        
        # Prepare metadata
        metadata = {
            'raw_shape': raw_data.shape,
            'processed_shape': processed_data.shape,
            'validation_results': validation_results
        }
        
        return processed_data, metadata


def create_data_pipeline(config_path: str) -> DataPipeline:
    """Factory function to create a data pipeline from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configured DataPipeline instance
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return DataPipeline(config)


if __name__ == "__main__":
    # Example usage
    import json
    
    # Sample configuration
    sample_config = {
        'ingestion': {
            'data_dir': 'data',
            'source_type': 'csv',
            'source_path': 'raw/sample_data.csv'
        },
        'validation_schema': {
            'dtypes': {
                'feature1': 'float64',
                'feature2': 'float64',
                'target': 'int64'
            }
        },
        'preprocessing': {
            'preprocessing_steps': [
                {'type': 'drop_columns', 'columns': ['id']},
                {'type': 'fill_missing', 'method': 'mean'},
                {'type': 'encode_categorical', 'columns': ['category'], 'method': 'one_hot'}
            ]
        },
        'force_continue': True
    }
    
    # Save sample config for testing
    os.makedirs('data/config', exist_ok=True)
    with open('data/config/sample_pipeline_config.json', 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    print("Created sample configuration at data/config/sample_pipeline_config.json")
    print("To run the pipeline, use: create_data_pipeline('data/config/sample_pipeline_config.json')")