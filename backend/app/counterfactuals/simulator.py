from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class CounterfactualSimulator:
    """
    Simulates optimization/leakage scenarios by removing features and retraining models
    to quantify dependency.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, task_type: str = 'classification'):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.task_type = task_type
        
    def _train_evaluate(self, features: List[str]) -> Dict[str, float]:
        """
        Trains a model on specified features and returns metrics.
        Duplicated from ModelValidator for independence, but shared logic.
        """
        X_train = self.train_df[features].copy()
        y_train = self.train_df[self.target_col]
        X_test = self.test_df[features].copy()
        
        # Robust encoding
        for col in X_train.columns:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
                categories = combined.unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1)
                X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1)
            
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(-1)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)
            except:
                X_train.drop(columns=[col], inplace=True)
                X_test.drop(columns=[col], inplace=True)

        X_test = X_test[features] 
        
        if self.target_col in self.test_df.columns:
            y_test = self.test_df[self.target_col]
        else:
            return {}

        if self.task_type == 'classification':
             model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
             model.fit(X_train, y_train)
             preds = model.predict(X_test)
             probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
             
             return {
                 'accuracy': accuracy_score(y_test, preds),
                 'roc_auc': roc_auc_score(y_test, probs)
             }
        else:
             model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
             model.fit(X_train, y_train)
             preds = model.predict(X_test)
             
             return {
                 'rmse': np.sqrt(mean_squared_error(y_test, preds))
             }

    def run_simulation(self, features_to_simulate: List[str]) -> List[Dict[str, Any]]:
        """
        Runs multiple scenarios:
        1. Baseline (All features)
        2. For each feature in list: Drop feature -> Retrain -> Measure
        """
        all_features = [c for c in self.train_df.columns if c != self.target_col]
        results = []
        
        # 1. Baseline
        print("Running Baseline Simulation...")
        baseline_metrics = self._train_evaluate(all_features)
        baseline_score = baseline_metrics.get('roc_auc') if self.task_type == 'classification' else -baseline_metrics.get('rmse', 0)
        
        results.append({
            "scenario": "Baseline",
            "dropped_feature": None,
            "metrics": baseline_metrics,
            "score_decay": 0.0,
            "dependency_score": 0.0
        })
        
        # 2. Per Feature Drop
        for feat in features_to_simulate:
            if feat not in all_features:
                continue
                
            print(f"Running Simulation: Dropping {feat}...")
            subset_features = [f for f in all_features if f != feat]
            
            metrics = self._train_evaluate(subset_features)
            score = metrics.get('roc_auc') if self.task_type == 'classification' else -metrics.get('rmse', 0) # Negative RMSE for easier comparison
            
            # Decay = Baseline - Current. Positive means Baseline was better (Feature was useful/improving)
            decay = baseline_score - score
            
            # Dependency Score: Normalized decay?? or just raw decay
            dependency = max(0, decay) 
            
            results.append({
                "scenario": f"Without {feat}",
                "dropped_feature": feat,
                "metrics": metrics,
                "score_decay": decay,
                "dependency_score": dependency
            })
            
        return results
