"""Data Pipeline Module

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
        
        # Map input_path to source_path if provided
        if 'input_path' in config and config['input_path']:
            self.config['source_path'] = str(Path(config['input_path']))
        
    def load_data(self) -> pd.DataFrame:
        """Load data from the specified source.
        
        Returns:
            DataFrame containing the loaded data
        """
        source_type = self.config.get('source_type', 'csv')
        source_path = self.config.get('source_path')
        
        # If source_path is not provided, use input_path
        if not source_path:
            source_path = self.config.get('input_path')
            if source_path:
                # Update the config with the source_path
                self.config['source_path'] = str(Path(source_path))
            
        if isinstance(source_path, list):
            source_path = source_path[0]

        if not source_path or not os.path.exists(source_path):
            raise ValueError(f"Source path must be specified in the configuration and must exist. Path: {source_path}")
        
        logger.info(f"Loading data from {source_path}")
        
        if source_type == 'csv':
            return pd.read_csv(source_path)
        elif source_type == 'parquet':
            return pd.read_parquet(source_path)
        elif source_type == 'json':
            return pd.read_json(source_path)
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
        self.encoders = {}
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the data.
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = data.copy()
        
        # Process each step in order
        for step in self.preprocessing_steps:
            step_type = step.get('type')
            
            # Handle drop duplicates
            if step_type == 'drop_duplicates':
                processed_data = processed_data.drop_duplicates()
                logger.info(f"Dropped duplicate rows. New shape: {processed_data.shape}")
            
            # Handle categorical encoding
            elif step_type == 'encode_categorical':
                columns = step.get('columns', [])
                method = step.get('method', 'label')
                
                for col in columns:
                    if col in processed_data.columns:
                        if method == 'label':
                            # Label encoding (convert to numeric codes)
                            from sklearn.preprocessing import LabelEncoder
                            if col not in self.encoders:
                                self.encoders[col] = LabelEncoder()
                                self.encoders[col].fit(processed_data[col])
                            
                            processed_data[col] = self.encoders[col].transform(processed_data[col])
                            # Ensure the column is converted to float to avoid type issues with models
                            processed_data[col] = processed_data[col].astype('float')
                            logger.info(f"Label encoded column: {col}")
                        elif method == 'label_encode':
                            # Alternative label encoding for compatibility with test cases
                            from sklearn.preprocessing import LabelEncoder
                            if col not in self.encoders:
                                self.encoders[col] = LabelEncoder()
                                self.encoders[col].fit(processed_data[col])
                            
                            processed_data[col] = self.encoders[col].transform(processed_data[col])
                            processed_data[col] = processed_data[col].astype('float')
                            logger.info(f"Label encoded column: {col}")
                        elif method == 'one_hot':
                            # One-hot encoding
                            one_hot = pd.get_dummies(processed_data[col], prefix=col, drop_first=step.get('drop_first', False))
                            processed_data = pd.concat([processed_data.drop(columns=[col]), one_hot], axis=1)
                            logger.info(f"One-hot encoded column: {col}")
            
            # Handle numerical scaling
            elif step_type == 'scale_numerical':
                columns = step.get('columns', [])
                for col in columns:
                    if col in processed_data.columns:
                        # Ensure column is numeric before scaling
                        if pd.api.types.is_numeric_dtype(processed_data[col]):
                            # Standardize to mean=0, std=1
                            processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
                            logger.info(f"Scaled numerical column: {col}")
                        else:
                            logger.warning(f"Column {col} is not numeric and cannot be scaled. Skipping.")
            
            # Handle missing values
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
                        logger.info(f"Filled missing values in column: {col} using {method}")
            
            # Handle dropping columns
            elif step_type == 'drop_columns':
                columns = step.get('columns', [])
                processed_data = processed_data.drop(columns=columns, errors='ignore')
                logger.info(f"Dropped columns: {columns}")
        
        # Additional preprocessing steps are already handled in the main preprocessing loop above
        
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
        
        # Create ingestion config with input_path if provided
        ingestion_config = config.get('ingestion', {})
        if 'input_path' in config and config['input_path']:
            ingestion_config['input_path'] = config['input_path']
            
        self.ingestion = DataIngestion(ingestion_config)
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
        
        # Save processed data if output_path is specified
        output_path = self.config.get('output_path')
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_data.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        
        # Prepare metadata
        metadata = {
            'raw_shape': raw_data.shape,
            'processed_shape': processed_data.shape,
            'validation_results': validation_results
        }
        
        return processed_data, metadata


def create_data_pipeline(config_path: str | dict) -> DataPipeline:
    """Factory function to create a data pipeline from a configuration file or dictionary.
    
    Args:
        config_path: Path to the configuration file or configuration dictionary
        
    Returns:
        Configured DataPipeline instance
    """
    import json
    
    if isinstance(config_path, dict):
        config = config_path
    else:
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