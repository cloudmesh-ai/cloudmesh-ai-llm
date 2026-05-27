#!/usr/bin/env python3
import os
import subprocess
import sys

def main():
    # 1. Paths configuration
    config_file = "config.yaml"
    master_key_file = os.path.expanduser("~/gemma/server_master_key.txt")
    openai_key_file = os.path.expanduser("~/gemma/openai_api_key.txt")

    # 2. Validation
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        sys.exit(1)
    
    if not os.path.exists(master_key_file):
        print(f"Error: Master key file not found at {master_key_file}")
        sys.exit(1)

    # 3. Read secrets (using strip to avoid newline/whitespace issues)
    with open(master_key_file, "r") as f:
        master_key = f.read().strip()
    
    openai_api_key = ""
    if os.path.exists(openai_key_file):
        with open(openai_key_file, "r") as f:
            openai_api_key = f.read().strip()

    # 4. Cleanup
    print("Cleaning up existing 'litellm' container...")
    subprocess.run(["docker", "rm", "-f", "litellm"], capture_output=True)

    # 5. Execution
    print(f"Starting LiteLLM with Master Key from {master_key_file}...")
    
    current_dir = os.getcwd()
    
    cmd = [
        "docker", "run", "-d",
        "--name", "litellm",
        "-p", "4000:4000",
        "--add-host", "white:host-gateway",
        "--add-host", "spark:host-gateway",
        "-v", f"{current_dir}/{config_file}:/app/config.yaml",
        "-e", f"OPENAI_API_KEY={openai_api_key}",
        "-e", f"LITELLM_MASTER_KEY={master_key}",
        "ghcr.io/berriai/litellm:main-latest",
        "--config", "/app/config.yaml", 
        "--port", "4000", 
        "--debug"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Success! LiteLLM is running on port 4000.")
        print("Check logs with: docker logs -f litellm")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start container: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()