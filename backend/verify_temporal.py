import requests
import pandas as pd
import time
import os
import numpy as np

# 1. Generate Temporal Data
print("Generating temporal test data...")
dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
df = pd.DataFrame({
    'timestamp': dates,
    'feature1': range(200),
    # Leakage introduced after index 100
    'feature_leak': [np.random.rand() if x < 100 else (1 if x % 2 == 0 else 0) for x in range(200)], 
    'target': [1 if x % 2 == 0 else 0 for x in range(200)] 
})
# Make feature_leak perfect predictor in second half
df['feature_leak'] = df.apply(lambda row: row['target'] if row.name >= 100 else row['feature_leak'], axis=1)

df.to_csv('test_temporal.csv', index=False)

BASE_URL = "http://localhost:8000/api"

try:
    # 2. Upload
    print("Uploading file...")
    with open('test_temporal.csv', 'rb') as f:
        resp = requests.post(f"{BASE_URL}/upload", files={'file': f})
    
    if resp.status_code != 200:
        print(f"Upload failed: {resp.text}")
        exit(1)
        
    file_id = resp.json()['file_id']
    print(f"File uploaded. ID: {file_id}")

    # 3. Start Audit
    print("Starting audit with Time Split...")
    payload = {
        "target_col": "target",
        "split_strategy": "time",
        "time_col": "timestamp",
        "problem_type": "classification",
        "auto_convert": True
    }
    resp = requests.post(f"{BASE_URL}/audit/{file_id}", json=payload)
    
    if resp.status_code != 200:
        print(f"Audit start failed: {resp.text}")
        exit(1)
        
    job_id = resp.json()['job_id']
    print(f"Audit started. Job ID: {job_id}")

    # 4. Poll Status
    print("Polling status...")
    for i in range(20):
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
        data = resp.json()
        status = data['status']
        print(f"Status: {status}")
        
        if status in ['completed', 'failed']:
            break
        time.sleep(1)
        
    if status == 'completed':
        print("\n✅ Audit Completed. Testing Temporal Debugger...")
        
        # 5. Get Temporal Analysis
        temp_resp = requests.get(f"{BASE_URL}/audit/{job_id}/temporal")
        if temp_resp.status_code == 200:
            timeline = temp_resp.json()['timeline']
            print(f"✅ Temporal Analysis Success. Steps: {len(timeline)}")
            for step in timeline:
                metrics = step['metrics']
                auc = metrics.get('roc_auc', 0)
                print(f"Step {step['step']} ({step['timestamp']}): AUC = {auc:.2f}")
                
                # Check for leakage detection (AUC shoud jump in second half)
                # indices are roughly 0-200. Buckets=10 -> Size=20.
                # Leak starts at 100 (Step 5).
                
        else:
             print(f"❌ Temporal Analysis Failed: {temp_resp.text}")
             
    else:
        print(f"\n❌ Verification Failed. Final Status: {status}")
        if 'error' in data:
            print(f"Error: {data['error']}")
            
except Exception as e:
    print(f"Exception during test: {e}")
finally:
    if os.path.exists('test_temporal.csv'):
        os.remove('test_temporal.csv')
