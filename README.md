# Automated Data Leakage Detection System

## Overview
This system automatically detects, quantifies, and explains data leakage in machine learning datasets. It
acts as an automated auditor that checks for common leakage signals before a model goes to production.

## Why is Data Leakage Dangerous?
Data leakage occurs when information from outside the training dataset is used to create the model. This
leads to **overly optimistic performance** during training/testing but **catastrophic failure** in production.

Common types:
- **Target Leakage**: Features that are proxies for the target (e.g., invoice_paid when predicting
  is_customer_retained).
- **Temporal Leakage**: Future data leaking into the past.
- **Train-Test Contamination**: Overlap between training and testing data.

## Features
- **Automated Detection**: Scans for "too good to be true" correlations and distribution drifts.
- **Risk Scoring**: Assigns a risk score (0-1) to each feature.
- **Model Validation**: Retrains models without suspicious features to measure "performance collapse".
- **Reporting**: Generates a Markdown report and visualizations.

## How to Run

1. **Install Dependencies**:
   `ash
   pip install -r backend/requirements.txt
   `

2. **Run the Audit (Demo Mode)**:

   **Scenario 1: Target Leakage (Default)**
   `ash
   python backend/debug_comprehensive.py --leakage_type target_leakage
   `

   **Scenario 2: Clean Data**
   `ash
   python backend/debug_comprehensive.py --leakage_type none
   `

   **Scenario 3: Temporal Leakage**
   `ash
   python backend/debug_comprehensive.py --leakage_type temporal_leakage
   `

## Project Structure
- ackend/: Python backend code for detection, evaluation, and reports.
- rontend/: Next.js frontend for viewing reports and history.
- data/: Example inputs, fingerprints, and stored reports.

## Extending the System
To use your own dataset, modify the loader in ackend/app/core/main.py (or ackend/utils/data_loader.py) to
load your CSV/Parquet file instead of the built-in synthetic generators.

## Repository
This project is also published as ML-Reliability-Data-Leakage-Detection-Model-Validation on GitHub.
