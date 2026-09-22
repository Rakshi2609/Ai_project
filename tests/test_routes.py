"""
Automated Route & API Endpoint Verification Suite (BCSE306L DA-1)
Tests:
- Next.js Web Routes (/, /capture, /robot, /analytics)
- FastAPI Inference & Evaluation Endpoints (/api/infer, /api/eval/baselines)
"""

import sys
import json
import urllib.request
import urllib.error

ROUTES_TO_TEST = [
    ("Next.js Master Cockpit", "http://127.0.0.1:3000/"),
    ("Next.js Biometric Capture", "http://127.0.0.1:3000/capture"),
    ("Next.js 3D Cobot Twin", "http://127.0.0.1:3000/robot"),
    ("Next.js Analytics & Retraining", "http://127.0.0.1:3000/analytics"),
    ("FastAPI Scenarios API", "http://127.0.0.1:8000/api/scenarios"),
    ("FastAPI Baselines API", "http://127.0.0.1:8000/api/eval/baselines"),
    ("FastAPI Pretrained Models API", "http://127.0.0.1:8000/api/models/pretrained/status"),
]

def test_endpoints():
    print("=" * 70)
    print(" BCSE306L DA-1: MULTI-ROUTE HEALTH VERIFICATION")
    print("=" * 70)
    passed = 0

    for name, url in ROUTES_TO_TEST:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthChecker/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.getcode()
                if status == 200:
                    print(f"  ✔ [200 OK] {name:<32} -> {url}")
                    passed += 1
                else:
                    print(f"  ✘ [HTTP {status}] {name:<32} -> {url}")
        except Exception as e:
            print(f"  ○ [OFFLINE] {name:<32} -> {url} ({e})")

    # Test POST /api/infer
    try:
        infer_url = "http://127.0.0.1:8000/api/infer"
        data = json.dumps({"task_name": "HealthCheck", "facial_params": {"brow_furrow": 0.2}}).encode("utf-8")
        req = urllib.request.Request(infer_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.getcode() == 200:
                print(f"  ✔ [200 OK] {'FastAPI Multimodal Inference':<32} -> {infer_url}")
                passed += 1
    except Exception as e:
        print(f"  ○ [OFFLINE] {'FastAPI Multimodal Inference':<32} -> {infer_url} ({e})")

    print("=" * 70)
    print(f" Total Passed: {passed}/{len(ROUTES_TO_TEST) + 1}")
    print("=" * 70)
    return passed == (len(ROUTES_TO_TEST) + 1)

if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)
