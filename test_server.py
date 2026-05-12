#!/usr/bin/env python
import argparse
import os
import json
import requests
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test AI Commander server")
    parser.add_argument(
        "port", 
        nargs="?", 
        default="181818", 
        help="Port of the AI Commander server (default: 181818)"
    )
    args = parser.parse_args()
    
    port = args.port
    base_url = f"http://localhost:{port}"
    
    # API Key for authentication
    key_path = Path("~/.config/cloudmesh/llm/server_master_key.txt").expanduser()
    api_key = ""
    if key_path.exists():
        try:
            api_key = key_path.read_text().strip()
        except Exception as e:
            print(f"Error reading API key: {e}")
    
    print(f"Testing AI Commander server at {base_url}...")
    print("-" * 50)
    
    results = {}
    
    # a) Health Check
    print("Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        status = response.status_code
        print(f"{status}", end="")
        if status == 200:
            print(" [OK] Health endpoint responded")
            results['health'] = "OK"
        else:
            print(f" [FAIL] Health endpoint responded with status {status}")
            results['health'] = f"FAIL ({status})"
    except Exception as e:
        print(f" [FAIL] Health endpoint failed: {e}")
        results['health'] = f"FAIL ({e})"
    print("")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    available_models = []
    
    # b) Models Check
    print("Testing Models Endpoint...")
    try:
        response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            
            # Extract model IDs into an array
            if "data" in data and isinstance(data["data"], list):
                available_models = [m.get("id") for m in data["data"] if m.get("id")]
            
            print(" [OK] Models endpoint responded")
            models_list = ", ".join(available_models) if available_models else "None"
            results['models'] = f"OK ({len(available_models)} models found: {models_list})"
        else:
            print(f" [FAIL] Models endpoint failed with status {response.status_code}")
            print(response.text)
            results['models'] = f"FAIL ({response.status_code})"
    except Exception as e:
        print(f" [FAIL] Models endpoint failed: {e}")
        results['models'] = f"FAIL ({e})"
    print("")
    
    # c) Test Query
    print("Testing Chat Completion Query...")
    
    # Use the first model from the array, fallback to default if empty
    model_to_use = available_models[0] if available_models else "google/gemma-4-31B-it"
    if not available_models:
        print(f"[dim]No models found in /v1/models, using default: {model_to_use}[/dim]")
    else:
        print(f"[dim]Using model: {model_to_use}[/dim]")

    payload = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": "Hello, are you running?"}]
    }
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions", 
            headers={**headers, "Content-Type": "application/json"}, 
            json=payload, 
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            print(" [OK] Query endpoint responded")
            
            # Extract human readable response
            try:
                content = data['choices'][0]['message']['content']
                results['query'] = content
            except (KeyError, IndexError):
                results['query'] = "OK (but no content found in response)"
        else:
            print(f" [FAIL] Query endpoint failed with status {response.status_code}")
            print(response.text)
            results['query'] = f"FAIL ({response.status_code})"
    except Exception as e:
        print(f" [FAIL] Query endpoint failed: {e}")
        results['query'] = f"FAIL ({e})"
    
    print("-" * 50)
    print("Test complete.")
    
    print("\n--- Human Readable Summary ---")
    print(f"Health Check: {results.get('health', 'Not tested')}")
    print(f"Models Check: {results.get('models', 'Not tested')}")
    print(f"Query Response: {results.get('query', 'Not tested')}")
    print("-" * 30)

if __name__ == "__main__":
    main()