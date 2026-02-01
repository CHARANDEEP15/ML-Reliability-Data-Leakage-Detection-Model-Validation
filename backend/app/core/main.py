import argparse
import pandas as pd
import os
import sys

# Ensure the package is in python path
from data_leakage_detection.data.loader import DataLoader
from data_leakage_detection.leakage_detection.detector import LeakageDetector
from data_leakage_detection.modeling.validator import ModelValidator
from data_leakage_detection.visualization.plotter import Visualizer
from data_leakage_detection.reports.generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description='Automated Data Leakage Detection System')
    parser.add_argument('--leakage_type', type=str, default='target_leakage', 
                        choices=['none', 'target_leakage', 'temporal_leakage'],
                        help='Type of leakage to simulate for demo')
    args = parser.parse_args()
    
    print(f"Starting Data Leakage Detection Pipeline (Simulation: {args.leakage_type})...")
    
    # 1. Load Data
    print("Step 1: Loading/Generating Data...")
    df = DataLoader.generate_synthetic_data(n_samples=2000, leakage_type=args.leakage_type)
    
    # Split Data
    # For temporal leakage, we should ideally use time split, but for this demo using random is also fine 
    # to show how it might catch (or fail to catch) things if not split correctly, 
    # OR we can enforce time split if 'temporal' is chosen.
    # Let's stick to random split for consistency in this basic demo unless it's temporal.
    
    split_type = 'time' if args.leakage_type == 'temporal_leakage' else 'random'
    time_col = 'timestamp' if args.leakage_type == 'temporal_leakage' else None
    
    train_df, test_df = DataLoader.split_data(df, target_col='target_class', split_type=split_type, time_col=time_col)
    
    print(f"Train Clean Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    
    # 2. Leakage Detection
    print("Step 2: Scanning for Leakage Signals...")
    detector = LeakageDetector(train_df, test_df, target_col='target_class')
    risk_report = detector.run_detection()
    
    print("\nTop Risk Features:")
    print(risk_report.head())
    
    # 3. Visualization
    print("Step 3: Generating Visualizations...")
    viz = Visualizer(output_dir='data_leakage_detection/reports/figures')
    viz.plot_risk_summary(risk_report)
    
    # Plot top 3 risky features
    for feature in risk_report.head(3)['feature']:
        viz.plot_feature_distribution(train_df, test_df, feature)
        
    # 4. Model Validation
    print("Step 4: Validating with Models...")
    suspicious_features = risk_report[risk_report['risk_score'] >= 0.4]['feature'].tolist()
    
    if suspicious_features:
        print(f"Validating impact of {len(suspicious_features)} suspicious features: {suspicious_features}")
        validator = ModelValidator(train_df, test_df, target_col='target_class')
        validation_results = validator.validate_leakage(suspicious_features)
    else:
        print("No high-risk features found to validate.")
        validation_results = {}

    # 5. Reporting
    print("Step 5: Generating Report...")
    reporter = ReportGenerator(output_dir='data_leakage_detection/reports')
    report_path = reporter.generate_report(risk_report, validation_results)
    
    print(f"\nAudit Complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()
