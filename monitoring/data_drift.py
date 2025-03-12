\"""Data Drift Monitoring Module

This module provides functionality for detecting data drift between reference and current datasets.
It helps identify when model retraining might be necessary due to changes in data distribution.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.metric_preset import DataDriftPreset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """Monitors data drift between reference and current datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data drift monitor.
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config
        self.reference_data_path = config.get('reference_data_path')
        self.threshold = config.get('drift_threshold', 0.05)  # Default p-value threshold
        self.categorical_columns = config.get('categorical_columns', [])
        self.numerical_columns = config.get('numerical_columns', [])
        self.datetime_column = config.get('datetime_column')
        self.reports_dir = config.get('reports_dir', 'monitoring/reports')
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load reference data if provided
        self.reference_data = None
        if self.reference_data_path and os.path.exists(self.reference_data_path):
            self.load_reference_data()
    
    def load_reference_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load reference data from file.
        
        Args:
            path: Optional path to reference data file
            
        Returns:
            Reference data as DataFrame
        """
        if path:
            self.reference_data_path = path
        
        if not self.reference_data_path or not os.path.exists(self.reference_data_path):
            raise ValueError(f"Reference data path not found: {self.reference_data_path}")
        
        file_ext = os.path.splitext(self.reference_data_path)[1].lower()
        
        if file_ext == '.csv':
            self.reference_data = pd.read_csv(self.reference_data_path)
        elif file_ext == '.parquet':
            self.reference_data = pd.read_parquet(self.reference_data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loaded reference data from {self.reference_data_path} with shape {self.reference_data.shape}")
        return self.reference_data
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data.
        
        Args:
            current_data: Current data to compare against reference data
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")
        
        # Ensure columns match between reference and current data
        ref_columns = set(self.reference_data.columns)
        curr_columns = set(current_data.columns)
        
        if ref_columns != curr_columns:
            logger.warning(f"Column mismatch between reference and current data. ")
            logger.warning(f"Missing in current: {ref_columns - curr_columns}")
            logger.warning(f"Extra in current: {curr_columns - ref_columns}")
            
            # Use only common columns
            common_columns = list(ref_columns.intersection(curr_columns))
            reference_data = self.reference_data[common_columns]
            current_data = current_data[common_columns]
        else:
            reference_data = self.reference_data
        
        # Initialize results
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_drift_detected': False,
            'column_drifts': {},
            'drift_score': 0.0,
            'columns_drifted': 0,
            'total_columns': len(common_columns)
        }
        
        # Detect drift for each column
        for column in common_columns:
            # Skip datetime column
            if column == self.datetime_column:
                continue
                
            # Check if column is categorical or numerical
            if column in self.categorical_columns or reference_data[column].dtype == 'object':
                # Categorical column - use chi-square test
                drift_result = self._detect_categorical_drift(reference_data[column], current_data[column])
            else:
                # Numerical column - use KS test
                drift_result = self._detect_numerical_drift(reference_data[column], current_data[column])
            
            results['column_drifts'][column] = drift_result
            
            if drift_result['drift_detected']:
                results['columns_drifted'] += 1
        
        # Calculate overall drift score
        if results['total_columns'] > 0:
            results['drift_score'] = results['columns_drifted'] / results['total_columns']
        
        # Determine if data drift is detected
        drift_threshold = self.config.get('overall_drift_threshold', 0.3)  # Default 30% of columns
        results['data_drift_detected'] = results['drift_score'] >= drift_threshold
        
        # Generate Evidently report
        report_path = self._generate_evidently_report(reference_data, current_data)
        results['report_path'] = report_path
        
        logger.info(f"Data drift detection completed. Drift detected: {results['data_drift_detected']}")
        logger.info(f"Drift score: {results['drift_score']:.2f} ({results['columns_drifted']} out of {results['total_columns']} columns)")
        
        return results
    
    def _detect_numerical_drift(self, reference_col: pd.Series, current_col: pd.Series) -> Dict[str, Any]:
        """Detect drift for numerical column using Kolmogorov-Smirnov test.
        
        Args:
            reference_col: Reference data column
            current_col: Current data column
            
        Returns:
            Dictionary with drift detection results for the column
        """
        # Drop NaN values
        reference_col = reference_col.dropna()
        current_col = current_col.dropna()
        
        if len(reference_col) == 0 or len(current_col) == 0:
            return {
                'drift_detected': False,
                'p_value': None,
                'statistic': None,
                'method': 'ks_test',
                'error': 'Insufficient data after dropping NaN values'
            }
        
        try:
            # Perform KS test
            statistic, p_value = ks_2samp(reference_col, current_col)
            
            return {
                'drift_detected': p_value < self.threshold,
                'p_value': p_value,
                'statistic': statistic,
                'method': 'ks_test'
            }
        except Exception as e:
            logger.error(f"Error performing KS test: {str(e)}")
            return {
                'drift_detected': False,
                'p_value': None,
                'statistic': None,
                'method': 'ks_test',
                'error': str(e)
            }
    
    def _detect_categorical_drift(self, reference_col: pd.Series, current_col: pd.Series) -> Dict[str, Any]:
        """Detect drift for categorical column using distribution comparison.
        
        Args:
            reference_col: Reference data column
            current_col: Current data column
            
        Returns:
            Dictionary with drift detection results for the column
        """
        try:
            # Get value counts and normalize
            ref_dist = reference_col.value_counts(normalize=True).to_dict()
            curr_dist = current_col.value_counts(normalize=True).to_dict()
            
            # Get all unique values
            all_values = set(ref_dist.keys()).union(set(curr_dist.keys()))
            
            # Calculate Jensen-Shannon divergence
            js_divergence = 0.0
            for value in all_values:
                p = ref_dist.get(value, 0)
                q = curr_dist.get(value, 0)
                # Add a small epsilon to avoid log(0)
                epsilon = 1e-10
                p = max(p, epsilon)
                q = max(q, epsilon)
                m = (p + q) / 2
                js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
            
            # Determine if drift is detected based on JS divergence
            # JS divergence is between 0 and 1, with 0 meaning identical distributions
            js_threshold = self.config.get('js_divergence_threshold', 0.1)
            drift_detected = js_divergence > js_threshold
            
            return {
                'drift_detected': drift_detected,
                'js_divergence': js_divergence,
                'method': 'js_divergence'
            }
        except Exception as e:
            logger.error(f"Error calculating JS divergence: {str(e)}")
            return {
                'drift_detected': False,
                'js_divergence': None,
                'method': 'js_divergence',
                'error': str(e)
            }
    
    def _generate_evidently_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> str:
        """Generate Evidently data drift report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Path to the generated report
        """
        try:
            # Create Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            # Run the report
            report.run(reference_data=reference_data, current_data=current_data)
            
            # Save the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.reports_dir, f"data_drift_report_{timestamp}.html")
            report.save_html(report_path)
            
            logger.info(f"Evidently report saved to {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating Evidently report: {str(e)}")
            return None


class ModelPerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model performance monitor.
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config
        self.baseline_metrics_path = config.get('baseline_metrics_path')
        self.metrics_threshold = config.get('metrics_threshold', 0.1)  # Default threshold for metric degradation
        self.reports_dir = config.get('reports_dir', 'monitoring/reports')
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load baseline metrics if provided
        self.baseline_metrics = None
        if self.baseline_metrics_path and os.path.exists(self.baseline_metrics_path):
            self.load_baseline_metrics()
    
    def load_baseline_metrics(self, path: Optional[str] = None) -> Dict[str, float]:
        """Load baseline metrics from file.
        
        Args:
            path: Optional path to baseline metrics file
            
        Returns:
            Baseline metrics as dictionary
        """
        if path:
            self.baseline_metrics_path = path
        
        if not self.baseline_metrics_path or not os.path.exists(self.baseline_metrics_path):
            raise ValueError(f"Baseline metrics path not found: {self.baseline_metrics_path}")
        
        with open(self.baseline_metrics_path, 'r') as f:
            self.baseline_metrics = json.load(f)
        
        logger.info(f"Loaded baseline metrics from {self.baseline_metrics_path}")
        return self.baseline_metrics
    
    def detect_performance_degradation(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect model performance degradation.
        
        Args:
            current_metrics: Current model performance metrics
            
        Returns:
            Dictionary with performance degradation results
        """
        if self.baseline_metrics is None:
            raise ValueError("Baseline metrics not loaded. Call load_baseline_metrics() first.")
        
        # Initialize results
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_degradation_detected': False,
            'metric_changes': {},
            'degraded_metrics': 0,
            'total_metrics': len(current_metrics)
        }
        
        # Check each metric for degradation
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.baseline_metrics:
                logger.warning(f"Metric {metric_name} not found in baseline metrics")
                continue
            
            baseline_value = self.baseline_metrics[metric_name]
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = (current_value - baseline_value) / abs(baseline_value)
            else:
                relative_change = float('inf') if current_value != 0 else 0
            
            # Determine if metric has degraded
            # For metrics where higher is better (e.g., accuracy, precision, recall, f1)
            higher_is_better = self.config.get('higher_is_better_metrics', [])
            lower_is_better = self.config.get('lower_is_better_metrics', [])
            
            if metric_name in higher_is_better:
                degrade