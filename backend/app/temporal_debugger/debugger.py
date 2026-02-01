import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

class TimeTravelDebugger:
    """
    Simulates dataset evolution over time to identify when leakage was introduced.
    """
    
    def __init__(self, df: pd.DataFrame, time_col: str, target_col: str, problem_type: str = 'classification'):
        self.df = df.sort_values(by=time_col)
        self.time_col = time_col
        self.target_col = target_col
        self.problem_type = problem_type
        
    def run_analysis(self, buckets: int = 10) -> List[Dict[str, Any]]:
        """
        Splits data into N time buckets.
        Iteratively trains on [0..t] and evaluates on [t+1].
        Unusually high performance or sudden jumps indicate leakage introduction.
        """
        n = len(self.df)
        bucket_size = n // buckets
        
        timeline = []
        
        # We need at least 2 buckets to train and test
        if n < 50 or buckets < 2:
            return [{"error": "Not enough data for temporal analysis"}]
            
        for i in range(1, buckets):
            split_idx = i * bucket_size
            
            # Expanding Window: Train on all history up to i
            train_df = self.df.iloc[:split_idx].copy()
            # Test on next bucket (simulating "next month")
            test_df = self.df.iloc[split_idx : split_idx + bucket_size].copy()
            
            if len(test_df) == 0:
                break
                
            current_time = train_df[self.time_col].max()
            
            # Simple modeling (Train & Eval)
            metrics = self._train_eval(train_df, test_df)
            
            timeline.append({
                "step": i,
                "timestamp": str(current_time),
                "train_size": len(train_df),
                "metrics": metrics
            })
            
        return timeline
        
    def _train_eval(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        X_train = train_df.drop(columns=[self.target_col, self.time_col])
        y_train = train_df[self.target_col]
        X_test = test_df.drop(columns=[self.target_col, self.time_col])
        y_test = test_df[self.target_col]
        
        # Numeric only for speed/stability in this debugger
        X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
        X_test = X_test.select_dtypes(include=[np.number]).fillna(0)
        
        # Align columns
        cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[cols]
        X_test = X_test[cols]
        
        if self.problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                 probs = model.predict_proba(X_test)[:, 1]
                 # ROC AUC needs at least 2 classes in y_test
                 if len(y_test.unique()) > 1:
                     score = roc_auc_score(y_test, probs)
                 else:
                     score = 0.5 # Default fallback
                 return {"roc_auc": score}
            else:
                 preds = model.predict(X_test)
                 return {"accuracy": accuracy_score(y_test, preds)}
        else:
            model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return {"rmse": np.sqrt(mean_squared_error(y_test, preds))}
