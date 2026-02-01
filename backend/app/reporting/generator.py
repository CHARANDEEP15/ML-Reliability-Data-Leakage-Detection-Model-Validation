import pandas as pd
import os
from typing import Dict, Any

class ReportGenerator:
    """
    Generates a Markdown report from the findings.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def generate_report(self, risk_df: pd.DataFrame, model_comparison: Dict[str, Any], 
                       governance_risk: Dict[str, Any] = None,
                       fingerprint_analysis: list = None,
                       temporal_analysis: Dict[str, Any] = None,
                       honest_metrics: Dict[str, Any] = None,
                       output_filename: str = 'leakage_audit_report.md'):
        """
        Creates the markdown report.
        """
        filepath = os.path.join(self.output_dir, output_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Automated Data Leakage Audit Report\n\n")
            
            # 1. Executive Summary (Governance Risk)
            f.write("## 1. Executive Summary\n")
            if governance_risk:
                emoji = "✅" if governance_risk['recommendation'] == 'GO' else "❌" if governance_risk['recommendation'] == 'NO-GO' else "⚠️"
                f.write(f"### {emoji} Recommendation: {governance_risk['recommendation']}\n")
                f.write(f"**Overall Risk Level**: {governance_risk['risk_level']} (Score: {governance_risk['overall_risk_score']:.2f})\n\n")
                f.write(f"**Business Impact**: {governance_risk['business_impact']}\n\n")
                f.write(f">{governance_risk['executive_summary']}\n\n")
            else:
                high_risk = risk_df[risk_df['risk_score'] >= 0.5]
                if not high_risk.empty:
                    f.write(f"⚠️ **CRITICAL LEAKAGE DETECTED**: Found {len(high_risk)} high-risk features.\n\n")
                else:
                    f.write("✅ No critical leakage detected.\n\n")
                
            # 2. Honest Model Performance (Leakage-Aware Scoring)
            f.write("## 2. Model Reliability & Honest Scoring\n")
            baseline = model_comparison.get('baseline', {})
            sanitized = model_comparison.get('sanitized', {})
            
            def safe_fmt(val):
                if isinstance(val, (int, float)):
                    return f"{val:.4f}"
                return str(val)

            f.write("| Metric | Raw / Inflated Score | Honest Score (Est.) | True Performance (Sanitized) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            raw_auc = baseline.get('roc_auc', 'N/A')
            san_auc = sanitized.get('roc_auc', 'N/A')
            honest_auc = honest_metrics.get('roc_auc', 'N/A') if honest_metrics else 'N/A'
            
            f.write(f"| ROC AUC | {safe_fmt(raw_auc)} | **{safe_fmt(honest_auc)}** | {safe_fmt(san_auc)} |\n")
            f.write(f"| Accuracy | {safe_fmt(baseline.get('accuracy', 'N/A'))} | {safe_fmt(honest_metrics.get('accuracy', 'N/A') if honest_metrics else 'N/A')} | {safe_fmt(sanitized.get('accuracy', 'N/A'))} |\n\n")
            
            if honest_metrics:
                f.write("> **Note**: The 'Honest Score' penalizes the raw metrics based on the probability of leakage, giving a more realistic expectation of production performance.\n\n")

            # 3. High Risk Features details
            f.write("## 3. Detected Leaky Features\n")
            if not risk_df.empty:
                f.write(risk_df.head(15).to_markdown(index=False))
            else:
                f.write("No features analyzed.")
            f.write("\n\n")
            
            # 4. Leakage Fingerprints
            if fingerprint_analysis:
                f.write("## 4. Historical Leakage Matches (Fingerprints)\n")
                f.write("This dataset shares characteristics with previously known leakage cases:\n\n")
                for match in fingerprint_analysis[:5]:
                    f.write(f"- **Match Score {match['similarity_score']*100:.0f}%**: Similar to job `{match['job_id']}`\n")
                f.write("\n")
            
            # 5. Temporal Analysis
            if temporal_analysis:
                f.write("## 5. Temporal Leakage Analysis\n")
                f.write("Time-Travel Debugging results:\n")
                if 'timeline' in temporal_analysis and len(temporal_analysis['timeline']) > 0:
                     f.write(f"- Analyzed {len(temporal_analysis['timeline'])} time-steps.\n")
                     f.write("- See Dashboard for full interactive timeline.\n")
                else:
                    f.write("- No temporal anomalies detected or insufficient time data.\n")
                f.write("\n")

            f.write("## 6. Visualizations\n")
            f.write("See `figures/` directory or interactive Dashboard for detailed charts.\n")
            
        return filepath
