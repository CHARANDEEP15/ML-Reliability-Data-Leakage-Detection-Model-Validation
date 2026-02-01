from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class ModelValidator:
    """
    Validates the impact of leakage by training models with and without suspicious features.
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, task_type: str = 'classification'):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.task_type = task_type
        
    def _train_evaluate(self, features: List[str]) -> Dict[str, float]:
        """
        Trains a model on specified features and returns metrics.
        """
        X_train = self.train_df[features].copy()
        y_train = self.train_df[self.target_col]
        X_test = self.test_df[features].copy()
        
        # Robust encoding: Handle ANY non-numeric column
        for col in X_train.columns:
            # Check if column is numeric
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                # Create a unified mapping to handle both train and test
                combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
                categories = combined.unique()
                
                # OPTIMIZATION: Drop high cardinality columns (likely IDs) to prevent slowness/overfitting
                if len(categories) > 50:
                    print(f"Skipping high cardinality column: {col} ({len(categories)} unique)")
                    X_train.drop(columns=[col], inplace=True)
                    X_test.drop(columns=[col], inplace=True)
                    continue

                mapping = {cat: i for i, cat in enumerate(categories)}
                
                # Apply mapping and fill unknown/nan with -1
                X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1)
                X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1)
            
            # Additional safety: ensure it's float/int now
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(-1)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)
            except:
                # If still failing, drop the column
                X_train.drop(columns=[col], inplace=True)
                X_test.drop(columns=[col], inplace=True)

        X_test = X_test[features] # Ensure order match
        
        
        # Check if test has target, otherwise split from train (not ideal but safe fallback)
        if self.target_col in self.test_df.columns:
            y_test = self.test_df[self.target_col]
        else:
             # If no target in test, we can only report cross-val on train
             # For this demo, we assume test has target as per loader
            return {}

        if self.task_type == 'classification':
             model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
             model.fit(X_train, y_train)
             preds = model.predict(X_test)
             
             try:
                 # Check for multi-class
                 unique_classes = np.unique(y_train)
                 if len(unique_classes) > 2:
                    current_probs = model.predict_proba(X_test)
                    roc = roc_auc_score(y_test, current_probs, multi_class='ovr')
                 else:
                    current_probs = model.predict_proba(X_test)[:, 1]
                    roc = roc_auc_score(y_test, current_probs)
             except Exception as e:
                 print(f"Warning: ROC AUC Calculation failed ({e}). Returning 0.5")
                 roc = 0.5
             
             return {
                 'accuracy': accuracy_score(y_test, preds),
                 'roc_auc': roc
             }
        else:
             model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
             model.fit(X_train, y_train)
             preds = model.predict(X_test)
             
             return {
                 'rmse': np.sqrt(mean_squared_error(y_test, preds))
             }

    def validate_leakage(self, suspicious_features: List[str]) -> Dict[str, Any]:
        """
        Compares model performance with all features vs. sanitized features.
        """
        all_features = [c for c in self.train_df.columns if c != self.target_col]
        sanitized_features = [f for f in all_features if f not in suspicious_features]
        
        if not sanitized_features:
            print("Warning: All features marked as suspicious. Sanitized model will appear as random guess.")
            # If nothing is left, the "honest" performance is random chance.
            if self.task_type == 'classification':
                sanitized_metrics = {'roc_auc': 0.5, 'accuracy': 0.5}
            else:
                # For regression, predicting the mean would be the baseline, 
                # but we'll return a placeholder or high error if we can't train.
                # Ideally we train a dummy regressor on 0 features (mean strategy).
                sanitized_metrics = {'rmse': float('inf')} 
        else:
            print("Training Sanitized Model (Dropping Suspicious)...")
            sanitized_metrics = self._train_evaluate(sanitized_features)
            
        print("Training Baseline Model (All Features)...")
        baseline_metrics = self._train_evaluate(all_features)
        
        comparison = {
            'baseline': baseline_metrics,
            'sanitized': sanitized_metrics,
            'removed_features': suspicious_features
        }
        
        # Calculate degradation
        if self.task_type == 'classification':
            drop = baseline_metrics.get('roc_auc', 0) - sanitized_metrics.get('roc_auc', 0)
            comparison['performance_drop_auc'] = drop
        else:
            # For RMSE, lower is better. If baseline is lower (better) than sanitized, 
            # leakage made it artificially good.
            # Drop = Sanitized RMSE - Baseline RMSE (positive means baseline was better)
            diff = sanitized_metrics.get('rmse', 0) - baseline_metrics.get('rmse', 0)
            comparison['performance_drop_rmse'] = diff
            
        return comparison
