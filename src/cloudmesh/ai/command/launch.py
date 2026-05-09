"""
Cloudmesh AI Launcher Extension
===============================

This extension provides tools to launch and manage AI user interfaces 
and other supporting tools using Docker.

Usage:
    cme launch webui
    cme launch stop webui
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

@launch_group.command(name="llm")
@click.argument("name")
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
@click.option("--info", is_flag=True, help="Display server configuration info")
@click.option("--export", is_flag=True, help="Export launch scripts to local directory for customization")
@click.option("--port", type=int, help="Override both local and remote ports")
def launch_vllm(name, ui, claude, info, export, port):
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
                console.msg("Backend is ready. You can now run 'cme launch webui' or 'cme launch claude'.")
        else:
            console.error("Backend preparation failed.")
    except Exception as e:
        console.error(f"Error orchestrating vLLM launch: {e}")

@launch_group.command(name="install")
@click.argument("tool")
def install_tool(tool):
    """Install AI tools (e.g., aider)."""
    if tool == "aider":
        # Aider and its dependencies are most stable on Python 3.10-3.12.
        if not (sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12):
            console.error(f"Aider installation is most stable on Python 3.10-3.12. Current version: {sys.version.split()[0]}")
            console.print("Python 3.13+ is causing 'AttributeError: module 'pkgutil' has no attribute 'ImpImporter'' during installation.")
            console.print("Please use pyenv to install a compatible version: 'pyenv install 3.12.0 && pyenv local 3.12.0'")
            return

        console.print(banner("Installing Aider", "Running 'pip install aider-chat'..."))
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "aider-chat"], check=True)
            console.ok("Aider installed successfully!")
            
            # Check for pandoc dependency
            try:
                subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                console.warning("Pandoc not found! Aider requires pandoc for some file conversions.")
                if os_is_mac():
                    console.print("Please install it using: 'brew install pandoc'")
                else:
                    console.print("Please install pandoc using your system package manager.")
        except subprocess.CalledProcessError as e:
            console.error(f"Failed to install Aider: {e}")
