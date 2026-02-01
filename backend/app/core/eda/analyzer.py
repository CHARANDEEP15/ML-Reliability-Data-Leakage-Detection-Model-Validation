import pandas as pd
import numpy as np
from typing import Dict, List

class EDAAnalyzer:
    """
    Performs Exploratory Data Analysis to identify potential leakage signals.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        self.train = train_df
        self.test = test_df
        self.target_col = target_col
        
    def check_feature_correlation(self) -> pd.DataFrame:
        """
        Calculates correlation of features with target in Train vs Test.
        Large differences might indicate learned leakage or covariate shift.
        """
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c != self.target_col]
        
        correlations = []
        for col in numerical_cols:
            corr_train = self.train[col].corr(self.train[self.target_col])
            # If target exists in test (labeled test set), we can check this. 
            # Often test sets in production don't have targets immediately, 
            # but for leakage detection in evaluation, we usually assume a hold-out set with labels.
            if self.target_col in self.test.columns:
                corr_test = self.test[col].corr(self.test[self.target_col])
            else:
                corr_test = np.nan
                
            correlations.append({
                'feature': col,
                'train_correlation': corr_train,
                'test_correlation': corr_test,
                'abs_diff': abs(corr_train - corr_test) if not pd.isna(corr_test) else 0
            })
            
        return pd.DataFrame(correlations).sort_values(by='train_correlation', key=abs, ascending=False)

    def check_near_perfect_correlation(self, threshold: float = 0.95) -> List[str]:
        """
        Identifies features with suspiciously high correlation to the target.
        """
        corr_df = self.check_feature_correlation()
        suspicious = corr_df[corr_df['train_correlation'].abs() >= threshold]['feature'].tolist()
        return suspicious

    def check_variance(self) -> Dict[str, List[str]]:
        """
        Checks for features that have zero variance in one split but not the other.
        """
        train_std = self.train.std(numeric_only=True)
        test_std = self.test.std(numeric_only=True)
        
        zero_var_train = train_std[train_std == 0].index.tolist()
        zero_var_test = test_std[test_std == 0].index.tolist()
        
        return {
            'zero_variance_train': zero_var_train,
            'zero_variance_test': zero_var_test
        }
