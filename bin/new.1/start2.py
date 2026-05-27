#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import yaml
import requests
from tabulate import tabulate

def probe_endpoint(api_base, timeout=5):
    """Check if a model endpoint is reachable by querying /v1/models"""
    try:
        # Most vLLM/OpenAI-compatible endpoints expose /v1/models
        response = requests.get(
            f"{api_base}/models", 
            headers={"Authorization": "Bearer sk-dummy"},
            timeout=timeout
        )
        return response.status_code == 200
    except Exception:
        return False

def display_probe_table(config_path):
    """Display table with probe results as first column"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    table_data = []
    model_list = config.get("model_list", [])
    
    print(f"Probing {len(model_list)} endpoints...")
    
    for model in model_list:
        litellm_params = model.get("litellm_params", {})
        api_base = litellm_params.get("api_base", "unknown")
        model_name = model.get("model_name", "unknown")
        
        # Probe the endpoint
        is_reachable = probe_endpoint(api_base)
        status = "✅" if is_reachable else "❌"
        
        # Extract host:port from api_base for display
        # e.g., "http://white:18000/v1" -> "white:18000"
        host_port = api_base.replace("http://", "").replace("/v1", "")
        
        table_data.append([
            status,
            model_name,
            host_port,
            api_base,
            litellm_params.get("model", "unknown")
        ])
    
    headers = ["Probe", "Model Name", "Host:Port", "API Base", "Model ID"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary statistics
    reachable = sum(1 for row in table_data if row[0] == "✅")
    print(f"\nSummary: {reachable}/{len(model_list)} endpoints reachable")

def start_litellm(config_path, master_key_file, openai_key_file):
    """Original start logic"""
    # Validation
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    if not os.path.exists(master_key_file):
        print(f"Error: Master key file not found at {master_key_file}")
        sys.exit(1)

    # Read secrets (using strip to avoid newline/whitespace issues)
    with open(master_key_file, "r") as f:
        master_key = f.read().strip()
    
    openai_api_key = ""
    if os.path.exists(openai_key_file):
        with open(openai_key_file, "r") as f:
            openai_api_key = f.read().strip()

    # Cleanup
    print("Cleaning up existing 'litellm' container...")
    subprocess.run(["docker", "rm", "-f", "litellm"], capture_output=True)

    # Execution
    print(f"Starting LiteLLM with Master Key from {master_key_file}...")
    
    current_dir = os.getcwd()
    
    cmd = [
        "docker", "run", "-d",
        "--name", "litellm",
        "-p", "4000:4000",
        "--add-host", "white:host-gateway",
        "--add-host", "spark:host-gateway",
        "-v", f"{current_dir}/{config_path}:/app/config.yaml",
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

def main():
    parser = argparse.ArgumentParser(description="LiteLLM Proxy Manager")
    parser.add_argument(
        "--probe", 
        action="store_true", 
        help="Probe all model endpoints and display status table"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    master_key_file = os.path.expanduser("~/gemma/server_master_key.txt")
    openai_key_file = os.path.expanduser("~/gemma/openai_api_key.txt")
    
    if args.probe:
        # Execute the probe table function
        display_probe_table(args.config)
    else:
        # Original start functionality
        start_litellm(args.config, master_key_file, openai_key_file)

if __name__ == "__main__":
    main()