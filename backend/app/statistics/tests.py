import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any

class StatisticalTests:
    """
    Implements statistical tests to detect distribution drift and leakage.
    """
    
    @staticmethod
    def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) to measure distribution shift.
        expected: Train data (reference)
        actual: Test data (production/current)
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        if expected.nunique() < buckets:
             # Categorical or low cardinality, use value counts directly if possible, or simple linear
             # For simplicity in this robust calculation, we'll try numeric
             pass

        # Simple binning strategy for numerical features
        try:
            expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
        except ValueError:
            # Fallback for weird ranges or categorical handled as numbers
            return 0.0

        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return psi_value

    @staticmethod
    def kolmogorov_smirnov_test(train_col: pd.Series, test_col: pd.Series) -> Dict[str, Any]:
        """
        Performs KS Test to check if two samples differ significantly.
        """
        statistic, p_value = ks_2samp(train_col, test_col)
        return {
            'ks_statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < 0.05  # Standard threshold
        }
