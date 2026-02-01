from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import pandas as pd
import os
import uuid
import shutil
from backend.app.leakage_engine.detector import LeakageDetector
from backend.app.modeling.validator import ModelValidator
from backend.app.reporting.generator import ReportGenerator
from backend.app.fingerprinting.engine import FingerprintEngine
from backend.app.governance.risk_translator import RiskTranslator
from backend.app.temporal_debugger.debugger import TimeTravelDebugger
from backend.app.modeling.scoring import ScoreAdjuster
from backend.app.forensics.analyzer import ForensicsAnalyzer
from backend.app.reliability.stress import StressTester

import json
import math

def sanitize_floats(data):
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_floats(v) for v in data]
    return data

router = APIRouter()

UPLOAD_DIR = "backend/data/uploads"
REPORT_DIR = "backend/data/reports"
JOBS_FILE = "backend/data/jobs.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def save_jobs():
    try:
        with open(JOBS_FILE, 'w') as f:
            json.dump(jobs, f, default=str) # default=str to handle UUIDs/Dates
    except Exception as e:
        print(f"Error saving jobs: {e}")

def load_jobs():
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, 'r') as f:
                loaded_jobs = json.load(f)
                # Cleanup interrupted jobs
                for jid, job in loaded_jobs.items():
                    if job['status'] == 'processing':
                        job['status'] = 'failed'
                        job['error'] = 'Interrupted by server restart'
                return loaded_jobs
        except Exception as e:
            print(f"Error loading jobs: {e}")
            return {}
    return {}

# Initialize jobs with persistence
jobs = load_jobs()
save_jobs() # Save immediately to handle the cleanup

class AuditConfig(BaseModel):
    target_col: str
    split_strategy: str = "random" # random, time
    time_col: Optional[str] = None
    problem_type: str = "classification"
    auto_convert: bool = False
    # New integration fields
    source_type: str = "file" # file, snowflake, bigquery
    source_query: Optional[str] = None # SQL query for DBs
    connection_config: Optional[dict] = {}

from backend.app.utils.data_loader import DataLoader

def run_audit_task(job_id: str, file_path: str, config: AuditConfig, ignored_features: List[str] = [], accepted_features: List[str] = []):
    try:
        print(f"[{job_id}] STATUS: Starting Audit Task")
        jobs[job_id]["status"] = "processing"
        save_jobs()
        
        # Load Data using Strategy Pattern
        # If file, usage file_path. If DB, use source_query.
        source_id = file_path if config.source_type == 'file' else config.source_query
        
        print(f"[{job_id}] STEP: Loading Data")
        df = DataLoader.load_data(
            source_identifier=source_id, 
            source_type=config.source_type, 
            config=config.connection_config
        )
        print(f"[{job_id}] Data Loaded. Shape: {df.shape}")
        
        df = DataLoader.preprocess_data(df, config.auto_convert)
        
        # Drop confirmed leakage features (REJECTED by user)
        if ignored_features:
            print(f"[{job_id}] Dropping ignored features: {len(ignored_features)}")
            df = df.drop(columns=[f for f in ignored_features if f in df.columns])
            
        # Split Data
        print(f"[{job_id}] STEP: Splitting Data")
        train_df, test_df = DataLoader.split_data(df, config)

        print(f"DEBUG: Train Columns: {train_df.columns.tolist()}")

        # Run Detection
        print(f"[{job_id}] STEP: Running LeakageDetector")
        detector = LeakageDetector(train_df, test_df, config.target_col)
        risk_report = detector.run_detection()
        print(f"DEBUG: Risk Report Top:\\n{risk_report.head()}")
        
        # FEATURE 2: Fingerprinting
        print(f"[{job_id}] STEP: Fingerprinting")
        fp_engine = FingerprintEngine()
        fingerprint = fp_engine.generate_fingerprint(df, config.target_col, job_id)
        fp_engine.save_fingerprint(fingerprint)
        matches = fp_engine.find_matches(fingerprint)
        
        # FEATURE 7: Forensics & Stress Testing (Phase 2)
        print(f"[{job_id}] STEP: Forensics")
        forensics = ForensicsAnalyzer()
        forensics_results = forensics.analyze_feature_dominance(risk_report)
        
        print(f"[{job_id}] STEP: Stress Testing")
        stress_tester = StressTester(train_df, test_df, config.target_col, config.problem_type)
        stress_results = stress_tester.run_stress_test(risk_report)
        print(f"[{job_id}] Stress Testing Complete")

        # FEATURE 4: Business Risk Translation (Updated with Phase 2 inputs)
        business_risk = RiskTranslator.translate_risk(
            risk_report, 
            forensics_data=forensics_results,
            stress_data=stress_results
        )
        
        # Validate Models
        # Filter high risk features that are NOT in accepted list
        suspicious = risk_report[risk_report['risk_score'] >= 0.4]['feature'].tolist()
        
        print(f"[{job_id}] STEP: Model Validation (Suspicious: {len(suspicious)})")
        validator = ModelValidator(train_df, test_df, config.target_col, config.problem_type)
        
        if suspicious:
            val_results = validator.validate_leakage(suspicious)
        else:
            # If no suspicious features, the current state IS the clean/honest state.
            # We train on all available features to get the current performance.
            all_feats = [c for c in train_df.columns if c != config.target_col]
            current_metrics = validator._train_evaluate(all_feats)
            val_results = {
                "baseline": current_metrics,
                "sanitized": current_metrics,
                "removed_features": [],
                "performance_drop_auc": 0.0
            }
        print(f"[{job_id}] Model Validation Complete")
            
        # FEATURE 6: Leakage-Aware Scoring
        # We need a 'raw' model score first. The validator gives baseline vs sanitized.
        # Use baseline metrics as 'current'
        honest_metrics = {}
        if 'baseline' in val_results:
            honest_metrics = ScoreAdjuster.calculate_honest_score(
                val_results['baseline'], 
                business_risk['overall_risk_score'], 
                config.problem_type
            )
            
        # Generate Report
        print(f"[{job_id}] STEP: Generating Report")
        reporter = ReportGenerator(output_dir=REPORT_DIR)
        report_path = reporter.generate_report(
            risk_report, 
            val_results, 
            governance_risk=business_risk,
            fingerprint_analysis=matches,
            honest_metrics=honest_metrics,
            output_filename=f"{job_id}_report.md"
        )
        
        jobs[job_id]["status"] = "completed"
        print(f"[{job_id}] COMPLETED")
        jobs[job_id]["result"] = {
            "risk_summary": risk_report.to_dict(orient="records"),
            "model_impact": val_results,
            "honest_metrics": honest_metrics,
            "fingerprint_analysis": matches,
            "governance_risk": business_risk,
            "forensics_data": forensics_results,
            "stress_data": stress_results,
            "report_url": f"/api/download/{job_id}",
            "ignored_features": ignored_features,
            "accepted_features": accepted_features
        }
        save_jobs()
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        save_jobs()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"file_id": file_id, "filename": file.filename}

@router.post("/audit/{file_id}")
async def start_audit(file_id: str, config: AuditConfig, background_tasks: BackgroundTasks):
    # Find file
    files = os.listdir(UPLOAD_DIR)
    target_file = None
    for f in files:
        if f.startswith(file_id):
            target_file = os.path.join(UPLOAD_DIR, f)
            break
            
    if not target_file:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Pre-check for strict types if NOT auto-converting
    if not config.auto_convert:
        try:
             df_peek = DataLoader.load_file(target_file)
             
             # Identify non-numeric columns (excluding target and time col if specified)
             exclude_cols = [config.target_col]
             if config.time_col: exclude_cols.append(config.time_col)
             
             feature_cols = [c for c in df_peek.columns if c not in exclude_cols]
             non_numeric_cols = df_peek[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
             
             if non_numeric_cols:
                 return {
                     "status": "requires_confirmation", 
                     "message": f"Found {len(non_numeric_cols)} non-numeric columns. Convert to numbers?",
                     "columns": non_numeric_cols
                 }
                 
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "config": config.dict(), "file_path": target_file}
    save_jobs()
    
    background_tasks.add_task(run_audit_task, job_id, target_file, config)
    
    return {"job_id": job_id, "status": "pending"}

@router.post("/audit/db")
async def start_db_audit(config: AuditConfig, background_tasks: BackgroundTasks):
    """
    Starts an audit from a database source (Snowflake, BigQuery).
    No file upload required.
    """
    if config.source_type not in ['snowflake', 'bigquery']:
         raise HTTPException(status_code=400, detail="Use /audit/{file_id} for files.")
         
    if not config.source_query:
        raise HTTPException(status_code=400, detail="source_query is required for DB audits.")

    job_id = str(uuid.uuid4())
    # We pass 'DB' as file_path placeholder since usage depends on source_type
    jobs[job_id] = {"status": "pending", "config": config.dict(), "file_path": "EXTERNAL_DB"}
    save_jobs()
    
    background_tasks.add_task(run_audit_task, job_id, "EXTERNAL_DB", config)
    
    return {"job_id": job_id, "status": "pending"}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return sanitize_floats(jobs[job_id])

@router.get("/jobs")
async def list_jobs():
    """
    Returns a summary list of all jobs.
    """
    summary = []
    for jid, job in jobs.items():
        # Basic info for the list view
        summary.append({
            "job_id": jid,
            "status": job.get("status"),
            "target_col": job.get("config", {}).get("target_col"),
            "source_type": job.get("config", {}).get("source_type", "file"),
            "created_at": job.get("created_at", "N/A"), # If we tracked time
            "error": job.get("error")
        })
    return summary

from fastapi.responses import FileResponse

from backend.app.counterfactuals.simulator import CounterfactualSimulator

class SimulateRequest(BaseModel):
    features: List[str]

@router.post("/audit/{job_id}/simulate")
async def run_simulation(job_id: str, request: SimulateRequest):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Audit must be completed before simulation")
        
    file_path = job.get('file_path')
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Source file not found")
    
    # Config
    config = AuditConfig(**job['config'])

    # Load & Preprocess
    df = DataLoader.load_file(file_path)
    df = DataLoader.preprocess_data(df, config.auto_convert)
         
    # Split
    train_df, test_df = DataLoader.split_data(df, config)
        
    simulator = CounterfactualSimulator(train_df, test_df, config.target_col, config.problem_type)
    results = simulator.run_simulation(request.features)
    
    jobs[job_id]['simulation_result'] = results
    return {"job_id": job_id, "results": results}

from backend.app.governance.review_manager import ReviewManager, ReviewDecision

@router.post("/audit/{job_id}/review")
async def submit_review(job_id: str, decisions: List[ReviewDecision], background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Audit must be completed before review")
        
    # Process decisions
    print(f"[{job_id}] Received decisions: {decisions}")
    current_risks = job['result'].get('risk_summary', [])
    review_summary = ReviewManager.process_review(current_risks, decisions)
    print(f"[{job_id}] Review Summary: dropped={review_summary['ignored_features']}, approved={review_summary['accepted_features']}")
    
    # Re-run audit with confirmed drops
    file_path = job['file_path']
    config = AuditConfig(**job['config'])
    
    jobs[job_id]['status'] = 'processing_review'
    
    background_tasks.add_task(
        run_audit_task, 
        job_id, 
        file_path, 
        config, 
        review_summary['ignored_features'], 
        review_summary['accepted_features']
    )
    
    return {
        "message": "Review submitted. Re-running audit...",
        "job_id": job_id,
        "actions": {
            "dropped": len(review_summary['ignored_features']),
            "approved": len(review_summary['accepted_features'])
        }
    }

@router.get("/audit/{job_id}/temporal")
async def run_temporal_analysis(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job['status'] != 'completed' and job['status'] != 'processing_review':
        # Allow if completed or re-processing (since file exists)
        if job['status'] != 'completed': 
             raise HTTPException(status_code=400, detail="Audit must be completed before temporal analysis")
        
    config = AuditConfig(**job['config'])
    if not config.time_col:
        raise HTTPException(status_code=400, detail="Temporal analysis requires a time column")
        
    file_path = job.get('file_path')
    
    # Load & Preprocess
    df = DataLoader.load_file(file_path)
    df = DataLoader.preprocess_data(df, config.auto_convert)
    
    # Run Debugger
    debugger = TimeTravelDebugger(df, config.time_col, config.target_col, config.problem_type)
    timeline = debugger.run_analysis()
    
    jobs[job_id]['temporal_result'] = timeline
    return {"job_id": job_id, "timeline": timeline}

@router.get("/download/{job_id}")
async def download_report(job_id: str):
    report_path = os.path.join(REPORT_DIR, f"{job_id}_report.md")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path)
