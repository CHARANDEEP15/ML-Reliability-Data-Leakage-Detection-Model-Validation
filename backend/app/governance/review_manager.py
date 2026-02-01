from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class ReviewDecision(BaseModel):
    feature: str
    decision: str # "APPROVE" (keep feature, ignore leakage flag) or "REJECT" (confirm leakage, drop feature)
    justification: str = ""

class ReviewManager:
    """
    Manages human-in-the-loop review process for flagged leakage features.
    """
    
    @staticmethod
    def process_review(current_risk_report: List[Dict[str, Any]], decisions: List[ReviewDecision]) -> Dict[str, Any]:
        """
        Updates the risk report based on human decisions.
        Returns:
            - ignored_features: Features that user CONFIRMED as leakage (REJECTED) -> should be dropped in next run.
            - accepted_features: Features user APPROVED (override flag) -> should be kept even if risky.
        """
        decision_map = {d.feature: d for d in decisions}
        
        ignored_features = [] # To be removed from model
        accepted_features = [] # To be kept despite risk
        
        updated_report = []
        
        for item in current_risk_report:
            feature = item['feature']
            if feature in decision_map:
                dec = decision_map[feature]
                item['human_decision'] = dec.decision
                item['justification'] = dec.justification
                item['reviewed_at'] = datetime.now().isoformat()
                
                if dec.decision == "REJECT":
                    # User confirms it is leakage, so we must remove it
                    ignored_features.append(feature)
                elif dec.decision == "APPROVE":
                    # User says it's fine, so keep it
                    accepted_features.append(feature)
            
            updated_report.append(item)
            
        return {
            "updated_report": updated_report,
            "ignored_features": ignored_features, # Features to DROP
            "accepted_features": accepted_features # Features to KEEP (whitelist)
        }
