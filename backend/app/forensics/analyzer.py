import pandas as pd
import numpy as np
from typing import Dict, List, Any

class ForensicsAnalyzer:
    """
    Analyzes 'why' a model performs well. Detects if success is due to 
    feature dominance (single feature driving everything) or actual signal.
    """

    def analyze_feature_dominance(self, risk_report: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the Gini coefficient of feature importance/risk.
        Returns dominance metrics and a generated narrative.
        """
        if risk_report.empty:
            return {"dominance_score": 0.0, "narrative": "No features to analyze.", "dominant_features": []}

        # We use 'risk_score' or 'train_correlation' as a proxy for importance if not available
        # Assuming risk_report has 'risk_score' and 'feature'
        scores = risk_report['risk_score'].values
        
        # Calculate Gini Coefficient of risk distribution
        # High Gini = One or two features hold all the risk/importance (Bad)
        # Low Gini = Risk is spread out (Better, but could still be high)
        if len(scores) > 1:
            sorted_scores = np.sort(scores)
            n = len(scores)
            index = np.arange(1, n + 1)
            gini = ((2 * np.sum(index * sorted_scores)) / (n * np.sum(sorted_scores))) - ((n + 1) / n)
        else:
            gini = 1.0 # One feature dominates completely
            
        # Identify dominant features (e.g. top 10% holding 50% of risk)
        total_risk = np.sum(scores)
        risk_report_sorted = risk_report.sort_values(by='risk_score', ascending=False)
        
        cumulative_risk = 0
        dominant_features = []
        for _, row in risk_report_sorted.iterrows():
            cumulative_risk += row['risk_score']
            dominant_features.append(row['feature'])
            if cumulative_risk >= total_risk * 0.5:
                break
                
        dominance_score = float(gini)
        
        # Generate Narrative
        narrative = self._generate_narrative(dominance_score, dominant_features, len(risk_report))
        
        return {
            "dominance_score": dominance_score,
            "dominant_features": dominant_features,
            "narrative": narrative,
            "concentration_warning": dominance_score > 0.6
        }

    def _generate_narrative(self, gini: float, dominant_features: List[str], total_features: int) -> str:
        if gini > 0.7:
            return (f"⚠️ **Suspiciously Simple**: This model is heavily relying on just {len(dominant_features)} features "
                    f"({', '.join(dominant_features[:3])}) to make its predictions. "
                    f"In complex real-world tasks, this usually indicates data leakage (e.g., an ID or future-dated column).")
        elif gini > 0.4:
            return (f"ℹ️ **Concentrated Signal**: The model relies mostly on a small subset of features. "
                    f"Ensure {', '.join(dominant_features[:3])} are valid predictors available at inference time.")
        else:
            return (f"✅ **Distributed Signal**: The model uses a wide range of features ({total_features} total) to predict. "
                    f"This reduces the risk of 'Single Point of Failure' leakage.")
