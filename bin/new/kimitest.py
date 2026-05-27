import subprocess
import json
import os

def test_kimi_api_readable():
    key_path = os.path.expanduser("~/gemma/uva-kimmi-key.txt")
    with open(key_path, "r") as f:
        api_key = f.read().strip()

    # Note: I removed '"stream": False' to let it stream, 
    # so we can demonstrate how to read it.
    curl_command = [
        "curl", "-ks", 
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Host: open-webui.rc.virginia.edu",
        "-H", "Content-Type: application/json",
        "-X", "POST",
        "https://localhost:8080/api/chat/completions",
        "-d", json.dumps({
            "model": "Kimi K2.5",
            "messages": [{"role": "user", "content": "Write a Hello World script."}],
            "stream": True 
        })
    ]

    print("--- Reading Stream ---")
    process = subprocess.Popen(curl_command, stdout=subprocess.PIPE, text=True)
    
    full_response = ""
    
    for line in process.stdout:
        if line.startswith("data: "):
            content = line[len("data: "):].strip()
            if content == "[DONE]":
                break
            try:
                chunk = json.loads(content)
                # Extract the delta content
                delta = chunk['choices'][0]['delta'].get('content', '')
                print(delta, end="", flush=True)
                full_response += delta
            except json.JSONDecodeError:
                continue
                
    print("\n\n--- Done ---")

if __name__ == "__main__":
    test_kimi_api_readable()