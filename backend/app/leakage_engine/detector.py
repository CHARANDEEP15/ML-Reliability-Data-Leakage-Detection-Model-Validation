import pandas as pd
import numpy as np
from typing import Dict, List, Any
from backend.app.core.eda.analyzer import EDAAnalyzer
from backend.app.statistics.tests import StatisticalTests

class LeakageDetector:
    """
    Orchestrates the detection of data leakage by combining signals from EDA and Statistical Tests.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        # Optimization: Use subsample for heavy statistical tests (KS, PSI, Corr)
        # 5000 rows is statistically sufficient for drift/correlation detection
        MAX_SAMPLES = 5000
        if len(train_df) > MAX_SAMPLES:
            print(f"LeakageDetector: Subsampling to {MAX_SAMPLES} rows for fast detection.")
            self.train_df = train_df.sample(n=MAX_SAMPLES, random_state=42)
            self.test_df = test_df.sample(n=min(len(test_df), MAX_SAMPLES), random_state=42)
        else:
            self.train_df = train_df
            self.test_df = test_df
            
        self.target_col = target_col
        self.eda = EDAAnalyzer(self.train_df, self.test_df, target_col)
        self.stats = StatisticalTests()
        
    def run_detection(self) -> pd.DataFrame:
        """
        Runs full detection suite and returns a risk report per feature.
        """
        # 1. Correlation Checks
        corr_report = self.eda.check_feature_correlation()
        
        # 2. Distribution Drift Checks
        feature_risks = []
        
        for _, row in corr_report.iterrows():
            feature = row['feature']
            train_feat = self.train_df[feature]
            test_feat = self.test_df[feature]
            
            # KS Test
            ks_result = self.stats.kolmogorov_smirnov_test(train_feat, test_feat)
            
            # PSI
            psi = self.stats.calculate_psi(train_feat, test_feat)
            
            # Risk Scoring Logic
            risk_score = 0.0
            reasons = []
            
            # High Correlation in Train (Potential Target Leakage)
            if abs(row['train_correlation']) > 0.95:
                risk_score += 0.5
                reasons.append(f"Near-perfect train correlation ({row['train_correlation']:.2f})")
            elif abs(row['train_correlation']) > 0.8:
                risk_score += 0.2
                reasons.append(f"High train correlation ({row['train_correlation']:.2f})")
                
            # Correlation Drift (Train vs Test)
            # If correlation drops significantly in test, it might be overfitting/leakage
            # If target is not in test, we skip this check or use 0
            if not pd.isna(row['test_correlation']):
                corr_diff = abs(row['train_correlation'] - row['test_correlation'])
                if corr_diff > 0.3:
                    risk_score += 0.3
                    reasons.append(f"Correlation drop in test (Diff: {corr_diff:.2f})")
            
            # Distribution Drift
            if ks_result['drift_detected']:
                risk_score += 0.1
                reasons.append("Distribution drift (KS Test)")
            
            if psi > 0.25:
                # 0.25 is a common threshold for significant drift
                risk_score += 0.2
                reasons.append(f"High PSI ({psi:.2f})")
                
            feature_risks.append({
                'feature': feature,
                'risk_score': min(risk_score, 1.0), # Cap at 1.0
                'train_correlation': row['train_correlation'],
                'drift_p_value': ks_result['p_value'],
                'psi': psi,
                'reasons': "; ".join(reasons)
            })
            
        return pd.DataFrame(feature_risks).sort_values(by='risk_score', ascending=False)
