import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
from scipy.spatial.distance import cosine

class FingerprintEngine:
    """
    Generates and stores statistical fingerprints of datasets to identify recurring leakage patterns.
    """
    
    def __init__(self, storage_path: str = "backend/data/fingerprints.json"):
        self.storage_path = storage_path
        
    def generate_fingerprint(self, df: pd.DataFrame, target_col: str, job_id: str) -> Dict[str, Any]:
        """
        Creates a signature based on correlation structure and schema.
        """
        # 1. Schema Signature (Set of columns)
        columns = sorted(list(df.columns))
        
        # 2. Correlation Vector (Top 20 absolute correlations with target)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            corr_matrix = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            top_correlations = corr_matrix.head(20).to_dict()
        else:
            top_correlations = {}
            
        # 3. Distribution Stats (Simple Mean/Std for top 5 correlated features)
        stats = {}
        for feat in list(top_correlations.keys())[:5]:
            stats[feat] = {
                "mean": float(df[feat].mean()),
                "std": float(df[feat].std())
            }
            
        fingerprint = {
            "job_id": job_id,
            "target": target_col,
            "columns": columns,
            "top_correlations": top_correlations,
            "stats": stats,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        return fingerprint

    def save_fingerprint(self, fingerprint: Dict[str, Any]):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    data = []
        else:
            data = []
            
        # Append new
        data.append(fingerprint)
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=4)

    def find_matches(self, current_fp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compares current fingerprint against history.
        """
        if not os.path.exists(self.storage_path):
            return []
            
        with open(self.storage_path, 'r') as f:
            history = json.load(f)
            
        matches = []
        current_cols = set(current_fp['columns'])
        current_corrs = current_fp['top_correlations']
        
        for hist in history:
            if hist['job_id'] == current_fp['job_id']: 
                continue # Skip self
            
            score = 0.0
            reasons = []
            
            # 1. Schema Overlap
            hist_cols = set(hist['columns'])
            intersection = current_cols.intersection(hist_cols)
            union = current_cols.union(hist_cols)
            jaccard = len(intersection) / len(union) if union else 0
            
            if jaccard > 0.8:
                score += 0.4
                reasons.append(f"High schema overlap ({jaccard:.2f})")
            elif jaccard > 0.5:
                score += 0.2
                reasons.append(f"Moderate schema overlap ({jaccard:.2f})")
                
            # 2. Correlation Pattern Match (Leakage Signature)
            # Check if same features have high correlation
            match_count = 0
            for feat, corr in current_corrs.items():
                if feat in hist['top_correlations']:
                    hist_corr = hist['top_correlations'][feat]
                    # If both high (e.g. > 0.8) or similar value
                    if abs(corr - hist_corr) < 0.1:
                        match_count += 1
            
            if match_count >= 3:
                score += 0.5
                reasons.append(f"Similar leakage pattern (Matched {match_count} high-corr features)")
                
            if score > 0.3:
                matches.append({
                    "job_id": hist['job_id'],
                    "similarity_score": min(score, 1.0),
                    "reasons": reasons,
                    "date": hist['created_at']
                })
                
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
