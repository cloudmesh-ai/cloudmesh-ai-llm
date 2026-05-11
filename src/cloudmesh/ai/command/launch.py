"""
Cloudmesh AI Launcher Extension
===============================

This extension provides tools to launch and manage AI user interfaces 
and other supporting tools using Docker.

Usage:
    cmc launch webui
    cmc launch stop webui
"""

import click
import os
import sys
import subprocess
import yaml
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac

from cloudmesh.ai.command.docker_manager import DockerManager
from cloudmesh.ai.command.webui_launcher import WebUILauncher
from cloudmesh.ai.command.claude_launcher import ClaudeLauncher
from cloudmesh.ai.command.aider_launcher import AiderLauncher
from cloudmesh.ai.command.orchestrator import VLLMOrchestrator, get_vllm_api_key

@click.group()
def launch_group():
    """Launch AI tools and interfaces."""
    pass

@click.group()
def llm_group():
    """Manage vLLM servers."""
    pass

@launch_group.command(name="webui")
def launch_webui():
    """Launch the Open WebUI container."""
    launcher = WebUILauncher()
    launcher.launch()

@launch_group.command(name="claude")
def launch_claude():
    """Launch Claude Code with vLLM backend."""
    launcher = ClaudeLauncher()
    launcher.launch()

@launch_group.command(name="aider")
@click.option("--docker", is_flag=True, help="Run Aider in a Docker container to avoid Python version issues")
@click.option("--force", is_flag=True, help="Force rebuild the container from scratch")
def launch_aider(docker, force):
    """Launch Aider with vLLM backend."""
    launcher = AiderLauncher()
    if docker:
        launcher.launch_docker(force=force)
    else:
        launcher.launch()

@launch_group.group(name="init")
def init_group():
    """Initialize AI server configurations."""
    pass

@init_group.command(name="server")
@click.argument("host")
def init_server(host):
    """Initialize server configuration in llm.yaml from SSH config host."""
    config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Load existing config
    data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            console.error(f"Could not read config file: {e}")
            return

    # Ensure structure exists
    if "cloudmesh" not in data: data["cloudmesh"] = {}
    if "ai" not in data["cloudmesh"]: data["cloudmesh"]["ai"] = {}
    if "server" not in data["cloudmesh"]["ai"]: data["cloudmesh"]["ai"]["server"] = {}
    
    # Determine platform and defaults based on host
    platform = "uva" if "uva" in host.lower() else "dgx" if "dgx" in host.lower() else "default"
    server_name = f"gemma-{host}"
    
    if server_name in data["cloudmesh"]["ai"]["server"]:
        if not console.ynchoice(f"Server {server_name} already exists. Overwrite?", default=False):
            return

    # Determine default directory based on platform
    if platform == "uva":
        default_dir = "/scratch/{user}/cloudmesh/vllm/{port}"
        default_port = 18123
    elif platform == "dgx":
        default_dir = "/raid/{user}/cloudmesh/vllm/{port}"
        default_port = 8000
    else:
        default_dir = "/home/{user}/cloudmesh/vllm/{port}"
        default_port = 8000

    # Default config
    data["cloudmesh"]["ai"]["server"][server_name] = {
        "platform": platform,
        "host": host,
        "user": f"{{~/.ssh/config:{host}.user}}",
        "dir": default_dir,
        "local_port": default_port,
        "remote_port": default_port,
        "model": "google/gemma-4-31B-it"
    }
    
    try:
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        console.ok(f"Initialized server '{server_name}' (platform: {platform}) in {config_path}")
    except Exception as e:
        console.error(f"Failed to save config: {e}")

@launch_group.group(name="config")
def config_group():
    """Manage AI configuration."""
    pass

@config_group.command(name="info")
def config_info():
    """Print the resolved in-memory LLM configuration."""
    try:
        orchestrator = VLLMOrchestrator()
        # YamlDB.yaml() returns the current state of self.data as YAML
        # Since load() resolves variables and merges #load files, this is the final resolved state
        resolved_yaml = orchestrator.db.yaml()
        
        if not resolved_yaml or resolved_yaml == "{}":
            console.warning("The in-memory configuration is empty. This may be due to a loading error or an empty config file.")
        else:
            console.print(banner("In-Memory Resolved Configuration", f"Source: {orchestrator.config_path}\n\n{resolved_yaml}"))
            
    except yaml.YAMLError as e:
        error_msg = f"YAML Syntax Error in configuration file:\n{e}"
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            error_msg += f"\nLocation: Line {mark.line + 1}, Column {mark.column + 1}"
        console.error(error_msg)
    except Exception as e:
        console.error(f"Error retrieving configuration info: {e}")

@llm_group.command(
    name="start", 
    context_settings=dict(allow_intermixed_options=True)
)
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
@click.option("--info", is_flag=True, help="Display server configuration info")
@click.option("--export", is_flag=True, help="Export launch scripts to local directory for customization")
@click.option("--port", type=int, help="Override both local and remote ports")
@click.argument("name")
def start_vllm(name, ui, claude, info, export, port):
    """Full pipeline: Start vLLM server -> Tunnel -> Health Check -> Optional UI."""
    try:
        if info:
            orchestrator = VLLMOrchestrator()
            servers = orchestrator.db.get("cloudmesh.ai.server", {})
            config_path = orchestrator.config_path
            if isinstance(servers, dict) and name in servers:
                config = servers[name]
                info_text = f"Config File: {config_path}\n"
                info_text += "\n".join([f"{k}: {v}" for k, v in config.items()])
                console.print(banner(f"Server Info: {name}", info_text))
            else:
                available = list(servers.keys()) if isinstance(servers, dict) else []
                error_msg = (
                    f"Server '{name}' not found in configuration.\n"
                    f"Available servers: {', '.join(available) if available else 'None'}"
                )
                console.error(error_msg)
                
                try:
                    console.print(banner("Configuration File Location", config_path))
                    with open(config_path, 'r') as f:
                        content = f.read()
                        console.print(banner("Configuration File Contents", content))
                except Exception as e:
                    console.error(f"Could not read configuration file: {e}")
            return

        orchestrator = VLLMOrchestrator()

        if export:
            console.print(f"[blue]Exporting scripts for {name}...[/blue]")
            if orchestrator.export_scripts(name):
                console.ok("Scripts exported successfully. You can now edit them locally.")
            else:
                console.error("Failed to export scripts.")
            return
        
        # Check if the name refers to a client instead of a server
        clients = orchestrator.db.get("cloudmesh.ai.client", {})
        if isinstance(clients, dict) and name in clients:
            client_config = clients[name]
            launcher_name = client_config.get("launcher")
            
            console.print(banner(f"Launching Client: {name}", f"Host: {client_config.get('host')}\nPort: {client_config.get('port')}\nLauncher: {launcher_name}"))
            
            launchers = {
                "webui": WebUILauncher,
                "claude": ClaudeLauncher,
                "aider": AiderLauncher,
            }
            
            launcher_class = launchers.get(launcher_name)
            if launcher_class:
                launcher_class().launch(client_config=client_config)
            else:
                console.error(f"Unsupported or missing launcher '{launcher_name}' for client '{name}'.")
            return

        if orchestrator.prepare_backend(name, port_override=port):
            console.ok(f"Backend {name} is ready!")
            
            if ui:
                console.print("[bold green]Launching WebUI...[/bold green]")
                # For the --ui flag, we use the default WebUI config if available
                clients = orchestrator.db.get("cloudmesh.ai.client", {})
                webui_cfg = clients.get("openwebui", {}) if isinstance(clients, dict) else {}
                WebUILauncher().launch(client_config=webui_cfg)
            elif claude:
                console.print("[bold green]Launching Claude...[/bold green]")
                # For the --claude flag, we use the default Claude config if available
                clients = orchestrator.db.get("cloudmesh.ai.client", {})
                claude_cfg = clients.get("claude", {}) if isinstance(clients, dict) else {}
                ClaudeLauncher().launch(client_config=claude_cfg)
            else:
                # Display a dark green banner with service details
                servers = orchestrator.db.get("cloudmesh.ai.server", {})
                config = servers.get(name, {}) if isinstance(servers, dict) else {}
                model_name = config.get("model", "Unknown Model")
                actual_port = port or config.get("remote_port", 8000)
                
                banner_text = f"Model: {model_name}\nPort: {actual_port}"
                console.banner(label="llm service started", txt=banner_text, color="dark_green")
                
                console.msg("Backend is ready. You can now run 'cmc launch webui' or 'cmc launch claude'.")
                console.print(f"\n[dim]To stop this server, run: cmc llm stop {name} --port {actual_port}[/dim]")
        else:
            console.error("Backend preparation failed.")
    except Exception as e:
        console.error(f"Error orchestrating vLLM launch: {e}")

@llm_group.command(name="stop")
@click.argument("identifier", required=False)
@click.option("--port", type=str, help="Port or partial port (e.g. '123') to identify the job")
def stop_vllm(identifier, port):
    """Stop a vLLM server (UVA HPC specific). Supports JobID, fuzzy port, or config port."""
    try:
        orchestrator = VLLMOrchestrator()
        
        # 1. If identifier is provided, it's either a server name or a fuzzy port/JobID
        if identifier:
            # If it's a number, treat as fuzzy port/JobID
            if identifier.isdigit():
                if orchestrator.stop_uva(port_pattern=identifier):
                    console.ok(f"Successfully stopped server matching {identifier}.")
                    return
            # Otherwise treat as server name
            if orchestrator.stop_uva(server_name=identifier, port_pattern=port):
                console.ok(f"Successfully stopped server {identifier}.")
                return
        
        # 2. If --port is provided but no identifier
        elif port:
            # We need a server name to use stop_uva's config logic, 
            # but if we only have a port, we can use the fuzzy match logic
            if orchestrator.stop_uva(port_pattern=port):
                console.ok(f"Successfully stopped server matching port {port}.")
                return

        # 3. No args: try to stop the last started server from config
        else:
            servers = orchestrator.db.get("cloudmesh.ai.server", {})
            if not servers:
                console.error("No servers configured. Use 'cmc llm start <name>' first.")
                return
            
            # Find the server with a persisted job_id
            last_server = None
            for name, cfg in servers.items():
                if cfg.get("job_id"):
                    last_server = name
                    break
            
            if last_server:
                if orchestrator.stop_uva(server_name=last_server):
                    console.ok(f"Successfully stopped last started server: {last_server}.")
                    return
            
            console.error("No active server found in configuration to stop.")
            
    except Exception as e:
        console.error(f"Error stopping vLLM server: {e}")

@llm_group.command(name="info")
def info_vllm():
    """List all running vLLM servers."""
    try:
        orchestrator = VLLMOrchestrator()
        orchestrator.list_running_servers()
    except Exception as e:
        console.error(f"Error listing servers: {e}")

