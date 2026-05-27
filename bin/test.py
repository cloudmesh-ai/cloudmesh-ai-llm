#!/usr/bin/env python3
import requests
import subprocess

ENDPOINTS = [
    {"name": "white", "host": "192.168.50.100", "port": 18000},
    #{"name": "spark", "host": "192.168.50.102", "port": 18001},
    {"name": "litellm", "host": "localhost", "port": 4000}
]

import os

def get_master_key():
    key_file = os.path.expanduser("~/gemma/server_master_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()
    return "sk-1234" # Fallback if file not found

MASTER_KEY = get_master_key()
AUTH_HEADER = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}

def test_model(target, port, model_id):
    url = f"http://{target}:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0
    }
    
    try:
        # 60s timeout for large models
        response = requests.post(url, headers=AUTH_HEADER, json=payload, timeout=60)
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
        
        if "ok" in content:
            print(f"      Testing {model_id}: \033[92mSUCCESS\033[0m")
        else:
            print(f"      Testing {model_id}: \033[91mFAILED\033[0m (Got: {content})")
    except Exception as e:
        print(f"      Testing {model_id}: \033[91mFAILED\033[0m ({e})")

def main():
    print("=================================================================")
    print("        CLUSTER + LITELLM DIAGNOSTIC (Python v2.2)               ")
    print("=================================================================")

    for ep in ENDPOINTS:
        print(f"\n--- Testing {ep['name']} at {ep['host']}:{ep['port']} ---")
        try:
            resp = requests.get(f"http://{ep['host']}:{ep['port']}/v1/models", headers=AUTH_HEADER, timeout=10)
            models_data = resp.json()
            model_ids = [m["id"] for m in models_data.get("data", []) if "modelperm-" not in m["id"]]
            
            print(f"      Found {len(model_ids)} valid models.")
            for model in model_ids:
                test_model(ep['host'], ep['port'], model)
        except Exception as e:
            print(f"FAILED to reach registry: {e}")

if __name__ == "__main__":
    main()