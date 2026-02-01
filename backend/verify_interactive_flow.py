import requests
import pandas as pd
import time
import os

# 1. Generate "Dirty" Data with Strings
print("Generating 'dirty' test data with string columns...")
df = pd.DataFrame({
    'feature_num': range(10),
    'feature_text': ['low', 'med', 'high', 'low', 'med', 'high', 'low', 'med', 'high', 'low'], # String column
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})
df.to_csv('test_dirty.csv', index=False)

BASE_URL = "http://localhost:8000/api"

try:
    # 2. Upload
    print("Uploading file...")
    with open('test_dirty.csv', 'rb') as f:
        resp = requests.post(f"{BASE_URL}/upload", files={'file': f})
    
    if resp.status_code != 200:
        print(f"Upload failed: {resp.text}")
        exit(1)
        
    file_id = resp.json()['file_id']
    print(f"File uploaded. ID: {file_id}")

    # 3. Start Audit (Expect Confirmation Request)
    print("Starting audit (Attempt 1 - Expecting Confirmation Request)...")
    payload = {
        "target_col": "target",
        "split_strategy": "random",
        "problem_type": "classification",
        "auto_convert": False
    }
    resp = requests.post(f"{BASE_URL}/audit/{file_id}", json=payload)
    data = resp.json()
    
    if data.get('status') == 'requires_confirmation':
        print("✅ Correctly received 'requires_confirmation' status.")
        print(f"Server Message: {data.get('message')}")
        print(f"Flagged Columns: {data.get('columns')}")
    else:
        print(f"❌ Failed. Expected 'requires_confirmation', got: {data}")
        exit(1)

    # 4. Retry with Auto-Convert
    print("\nRetrying audit with auto_convert=True...")
    payload['auto_convert'] = True
    resp = requests.post(f"{BASE_URL}/audit/{file_id}", json=payload)
    
    if resp.status_code != 200:
        print(f"Retry failed: {resp.text}")
        exit(1)
        
    job_id = resp.json()['job_id']
    print(f"Audit started. Job ID: {job_id}")
    
    # 5. Poll Status
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
        print("\n✅ Validated Interactive Flow Successfully!")
    else:
        print(f"\n❌ Verification Failed. Final Status: {status}")
        if 'error' in data:
            print(f"Error: {data['error']}")

except Exception as e:
    print(f"Exception during test: {e}")
finally:
    if os.path.exists('test_dirty.csv'):
        os.remove('test_dirty.csv')
