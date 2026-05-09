# Copyright 2026 Gregor von Laszewski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import click
import subprocess
import os
import yaml
from pathlib import Path
from rich.console import Console
from cloudmesh.ai.vpn.vpn import Vpn
from cloudmesh.ai.command.launch import AiderLauncher

console = Console()

def run_script(script_name: str, platform: str = "mac"):
    """Run a gemma script from the template directory."""
    script_path = Path(__file__).parent / "template" / platform / script_name
    if not script_path.exists():
        console.print(f"[bold red]Error: Script {script_name} not found at {script_path}.[/bold red]")
        return

    try:
        console.print(f"[blue]Running {script_name} for {platform}...[/blue]")
        subprocess.run(["bash", str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running {script_name}:[/bold red] {e}")

@click.group()
def gemma_group():
    """Gemma management extension."""
    pass

@gemma_group.command(name="start")
@click.option("--platform", default="dgx", type=click.Choice(["dgx", "mac", "uva"]), help="Platform to start on.")
def start_cmd(platform):
    """Start Gemma services."""
    if platform == "dgx":
        console.print("[blue]Checking VPN connection for DGX...[/blue]")
        vpn = Vpn()
        if not vpn.enabled():
            console.msg("VPN is disconnected. Attempting to connect...")
            if not vpn.connect():
                console.error("VPN connection failed. Please connect to the VPN and try again.")
                return
            console.ok("VPN connected successfully!")
        else:
            console.ok("VPN is already active.")
        
        script = "start.sh"
        run_script(script, platform)
        return

    if platform == "uva":
        console.print("[blue]Starting UVA HPC Orchestration...[/blue]")
        
        # 1. VPN Connection
        vpn = Vpn()
        if not vpn.enabled():
            console.msg("VPN is disconnected. Attempting to connect...")
            if not vpn.connect():
                console.error("VPN connection failed. Please connect to the VPN and try again.")
                return
            console.ok("VPN connected successfully!")
        else:
            console.ok("VPN is already active.")

        # 2. Resource Allocation (ijob)
        console.print("[blue]Requesting GPU allocation via ijob...[/blue]")
        ijob_cmd = [
            "ijob", 
            "--partition=bii-gpu", 
            "--reservation=bi_fox_dgx", 
            "--account=bi_dsc_community", 
            "--gpus=a100:4", 
            "--cpus-per-task=32", 
            "--mem=96gb", 
            "--time=03:00:00"
        ]
        
        try:
            # We use shell=True or a wrapper because ijob might be a shell alias/function
            result = subprocess.run(" ".join(ijob_cmd), shell=True, capture_output=True, text=True, check=True)
            output = result.stdout + result.stderr
            
            # Parse node name (e.g., "Nodes udc-an26-1 are ready")
            import re
            match = re.search(r"Nodes\s+([a-zA-Z0-9-]+)\s+are ready", output)
            if not match:
                console.error(f"Could not find allocated node in ijob output:\n{output}")
                return
            
            node_name = match.group(1)
            console.ok(f"Allocated node: {node_name}")
            
            # 3. Dynamic SSH Tunnel
            # Local port 18123 -> Node port 18123
            console.print(f"[blue]Establishing dynamic tunnel to {node_name}...[/blue]")
            tunnel_cmd = f"ssh -L 18123:{node_name}:18123 uva -N"
            subprocess.Popen(tunnel_cmd, shell=True)
            console.ok("Tunnel established in background.")
            
            # 4. Remote Launch
            console.print("[blue]Launching vLLM on allocated node...[/blue]")
            
            # Get keys from environment or config
            hf_token = os.environ.get("HF_TOKEN", "TBD")
            vllm_key = os.environ.get("VLLM_API_KEY", "TBD")
            
            remote_cmd = (
                f"cd /scratch/{os.environ.get('USER', 'user')} && "
                f"module load apptainer && "
                f"apptainer run --nv "
                f"-B /scratch/{os.environ.get('USER', 'user')}/hf_cache:/root/.cache/huggingface "
                f"--env HF_TOKEN='{hf_token}' "
                f"--env VLLM_API_KEY='{vllm_key}' "
                f"vllm_gemma4.sif "
                f"--model google/gemma-4-31B-it "
                f"--tensor-parallel-size 4 "
                f"--gpu-memory-utilization 0.85 "
                f"--max-model-len 131072 "
                f"--enable-prefix-caching "
                f"--load-format safetensors "
                f"--tool-call-parser gemma4 "
                f"--host 0.0.0.0 "
                f"--port 18123"
            )
            
            # Execute on the allocated node
            subprocess.run(f"ssh {node_name} '{remote_cmd}'", shell=True)
            
        except subprocess.CalledProcessError as e:
            console.error(f"HPC Orchestration failed: {e}")
            return

        return

    # Default to mac/gui
    script = "gui.sh"
    run_script(script, platform)

@gemma_group.command(name="gui")
def gui_cmd():
    """Launch the Gemma GUI (Mac)."""
    run_script("gui.sh", "mac")

@gemma_group.command(name="test")
def test_cmd():
    """Run Gemma tests (Mac)."""
    run_script("test.sh", "mac")

@gemma_group.command(name="clean")
def clean_cmd():
    """Clean Gemma environment (Mac)."""
    run_script("clean.sh", "mac")

@gemma_group.command(name="ascii")
def ascii_cmd():
    """Start aider with Gemma 4 using config from aider.yaml."""
    console.print("[blue]Starting aider with Gemma 4...[/blue]")
    
    # Path to the config file
    config_path = Path(__file__).parent / "template" / "aider.yaml"
    
    if not config_path.exists():
        console.print(f"[bold red]Error: Config file not found at {config_path}[/bold red]")
        return

    try:
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate the new hierarchy: cloudmesh -> ai -> aider
        cloudmesh_cfg = config.get("cloudmesh", {})
        ai_cfg = cloudmesh_cfg.get("ai", {})
        aider_config = ai_cfg.get("aider", {})
        
        model = aider_config.get("model")
        key_file_path = aider_config.get("key_file")
        url = aider_config.get("url")

        if not all([model, key_file_path, url]):
            console.print("[bold red]Error: Missing required configuration (model, key_file, or url) in aider.yaml under cloudmesh.ai.aider[/bold red]")
            return

        # Resolve key file path
        key_path = Path(key_file_path).expanduser()
        if not key_path.exists():
            console.print(f"[bold red]Error: Key file not found at {key_path}[/bold red]")
            return

        # Read key and strip whitespace/newlines
        api_key = key_path.read_text().strip()
        
        # Prepare config for AiderLauncher
        # AiderLauncher expects keys like OPENAI_API_KEY and OPENAI_API_BASE
        launcher_config = {
            "model": model,
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": url,
        }
        
        # Use the robust AiderLauncher from cloudmesh-ai-llm
        AiderLauncher().launch(client_config=launcher_config)
        
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing aider.yaml:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

def register(cli):
    cli.add_command(gemma_group, name="gemma")
