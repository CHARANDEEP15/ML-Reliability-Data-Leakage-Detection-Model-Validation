from typing import List, Dict, Any

class RiskTranslator:
    """
    Translates technical leakage metrics into business risk and deployment recommendations.
    """
    
    @staticmethod
    def translate_risk(risk_report: Any, 
                      forensics_data: Dict[str, Any] = None, 
                      stress_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyzes the risk report, forensics, and stress tests to generate a 
        Deployment Readiness Score (0-100).
        """
        if hasattr(risk_report, 'to_dict'):
            items = risk_report.to_dict(orient='records')
        else:
            items = risk_report
            
        # 1. Aggregate Risk
        max_risk = 0.0
        high_risk_features = []
        
        for item in items:
            score = item.get('risk_score', 0)
            if score > max_risk:
                max_risk = score
            if score >= 0.4:
                high_risk_features.append(item['feature'])
                
        # 2. Calculate Deployment Readiness Score (0-100)
        # Base Score 100
        readiness_score = 100.0
        penalties = []
        
        # Leakage Penalty
        if max_risk >= 0.8:
            readiness_score -= 50
            penalties.append("Critical Leakage Detected (-50)")
        elif max_risk >= 0.5:
            readiness_score -= 30
            penalties.append("High Leakage Detected (-30)")
        elif max_risk >= 0.3:
            readiness_score -= 10
            penalties.append("Moderate Leakage Risks (-10)")
            
        # Forensics Penalty (Feature Dominance)
        if forensics_data:
            dom_score = forensics_data.get('dominance_score', 0)
            if dom_score > 0.7:
                readiness_score -= 20
                penalties.append("Single Feature Dominance (-20)")
            elif dom_score > 0.5:
                readiness_score -= 10
                penalties.append("Features Highly Concentrated (-10)")
                
        # Stress Test Penalty (Resilience)
        if stress_data:
            resilience = stress_data.get('resilience_score', 1.0)
            if resilience < 0.6:
                readiness_score -= 30
                penalties.append("Failed Stress Test (-30)")
            elif resilience < 0.8:
                readiness_score -= 15
                penalties.append("Weak Stress Resilience (-15)")
                
        # Clamp Score
        readiness_score = max(0.0, min(100.0, readiness_score))
        
        # 3. Determine Final Verdict
        if readiness_score >= 85:
            level = "LOW"
            recommendation = "APPROVED"
            color = "green"
            summary = "Model is robust and ready for deployment."
        elif readiness_score >= 60:
            level = "MEDIUM"
            recommendation = "APPROVED WITH RISK"
            color = "yellow"
            summary = "Model has risks but may be shippable with monitoring. See penalties."
        else:
            level = "HIGH"
            recommendation = "NO-GO"
            color = "red"
            summary = "Deployment blocked. Model is not trusted due to significant leakage or fragility."
            
        # 4. Business Impact Narrative
        if high_risk_features:
            impact_text = f"The model is over-reliant on {len(high_risk_features)} features ({', '.join(high_risk_features[:3])}{'...' if len(high_risk_features)>3 else ''}). " \
                          f"Forensics indicate this is likely artificial performance that will not hold in production."
        else:
            impact_text = "Model logic appears distributed and robust. Expected to perform consistently in production."
            
        return {
            "overall_risk_score": max_risk,
            "readiness_score": int(readiness_score),
            "penalties": penalties,
            "risk_level": level,
            "recommendation": recommendation,
            "ui_color": color,
            "executive_summary": summary,
            "business_impact": impact_text,
            "flagged_feature_count": len(high_risk_features)
        }
