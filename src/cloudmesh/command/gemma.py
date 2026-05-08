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
@click.option("--platform", default="dgx", type=click.Choice(["dgx", "mac"]), help="Platform to start on.")
def start_cmd(platform):
    """Start Gemma services."""
    script = "start.sh" if platform == "dgx" else "gui.sh" # Assuming mac start is gui
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
        
        # Set environment variables
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = url
        env["OPENAI_API_KEY"] = api_key
        env["AIDER_MODEL"] = model
        
        # Launch aider
        subprocess.run(["aider", "--model", model], env=env, check=True)
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing aider.yaml:[/bold red] {e}")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching aider:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

def register(cli):
    cli.add_command(gemma_group, name="gemma")
