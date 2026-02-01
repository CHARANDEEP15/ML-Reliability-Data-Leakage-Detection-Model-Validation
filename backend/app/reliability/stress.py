import pandas as pd
import numpy as np
from typing import Dict, List, Any
from backend.app.modeling.validator import ModelValidator

class StressTester:
    """
    Performs 'Trust Stress Testing' by iteratively vetting the model against 
    increasingly aggressive feature removal scenarios.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, problem_type: str = 'classification'):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.problem_type = problem_type
        self.validator = ModelValidator(train_df, test_df, target_col, problem_type)

    def run_stress_test(self, risk_report: pd.DataFrame) -> Dict[str, Any]:
        """
        Iteratively removes top N risky features and measures performance decay.
        """
        if risk_report.empty:
            return {"curve": [], "resilience_score": 1.0}

        # Optimization: Use subsample for stress testing to speed up (max 2000 rows)
        MAX_SAMPLES = 2000
        if len(self.train_df) > MAX_SAMPLES:
            print(f"StressTester: Subsampling data to {MAX_SAMPLES} rows for performance.")
            stress_train = self.train_df.sample(n=MAX_SAMPLES, random_state=42)
            stress_test = self.test_df.sample(n=min(len(self.test_df), MAX_SAMPLES), random_state=42)
        else:
            stress_train = self.train_df
            stress_test = self.test_df
            
        # Use a lightweight validator for the stress loop
        stress_validator = ModelValidator(stress_train, stress_test, self.target_col, self.problem_type)

        # Sort features by risk
        sorted_risk = risk_report.sort_values(by='risk_score', ascending=False)
        risky_features = sorted_risk['feature'].tolist()
        
        # All available features (excluding target)
        all_features = [c for c in stress_train.columns if c != self.target_col]
        
        # Define removal steps (Baseline, Remove Top 1, Top 3, Top 5)
        steps = [0, 1, 3, 5]
        curve = []
        
        baseline_auc = 0.0
        
        for k in steps:
            # Identity features to DROP
            if k > len(risky_features):
                # If we don't have enough risky features to drop, skip (or just drop all)
                continue
                
            features_to_drop = set(risky_features[:k])
            
            # Identity features to KEEP
            remaining_features = [f for f in all_features if f not in features_to_drop]
            
            if not remaining_features:
                 # No features left
                curve.append({
                    "features_removed_count": k,
                    "features_removed_names": list(features_to_drop),
                    "auc": 0.5, # Random guess
                    "percent_of_baseline": 0.0
                })
                continue
            
            try:
                # Reuse validator to train on subset
                metrics = stress_validator._train_evaluate(remaining_features)
                
                score = metrics.get('roc_auc', 0.0)
                # Fallback to accuracy if AUC not present (e.g. regression/single class?) 
                # But problem_type defaults to classification.
                
                if k == 0:
                    baseline_auc = score
                    
                curve.append({
                    "features_removed_count": k,
                    "features_removed_names": list(features_to_drop),
                    "auc": score,
                    "percent_of_baseline": (score / baseline_auc) if baseline_auc > 0 else 0
                })
                
            except Exception as e:
                print(f"Error in stress test step k={k}: {e}")
                curve.append({
                    "features_removed_count": k,
                    "auc": 0.0,
                    "error": str(e)
                })

        # Calculate Resilience Score
        if curve:
            # Average retention of performance
            resilience_score = np.mean([p.get('percent_of_baseline', 0) for p in curve])
        else:
            resilience_score = 0.0
            
        return {
            "curve": curve,
            "resilience_score": resilience_score
        }
