import requests
import json
import time

BASE = "http://localhost:8000"

def test_async():
    print("Testing asynchronous Bunny CDN video embedding...")
    payload = {
        "fileId": "demo-bunny-video-id",
        "type": "video",
        "callbackUrl": "http://localhost:8000/health"
    }
    try:
        start_time = time.time()
        r = requests.post(f"{BASE}/api/embed-file", json=payload, timeout=10)
        elapsed = time.time() - start_time
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_async()
