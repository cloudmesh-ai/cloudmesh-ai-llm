"""
Cloudmesh AI LLM Management Extension
======================================

This extension provides tools to launch, manage, and interact with LLM 
server containers on DGX and UVA hosts using named configurations.

Usage Examples:
-------------------------------------------------------------------------------
1. Set a default server:
   $ cmc llm default server uva-default

2. Start an LLM server using interactive UI:
   $ cmc llm start --ui

3. Start an LLM server from a named config:
   $ cmc llm start uva-default --tunnel

4. Check the status of a server:
   $ cmc llm status uva-default

5. View the last 100 lines of logs:
   $ cmc llm logs uva-default

6. Stop a server gracefully:
   $ cmc llm stop uva-default --tunnel

7. Forcefully kill a server:
   $ cmc llm kill uva-default --tunnel

8. Send a prompt to the running server:
   $ cmc llm prompt "What is the capital of France?"
-------------------------------------------------------------------------------

Usage:
    llm start [NAME] [--tunnel] [--ui]
    llm stop [NAME] [--tunnel]
    llm kill [NAME] [--tunnel]
    llm status [NAME]
    llm logs [NAME]
    llm default server [NAME]
    llm default client [NAME]
    llm tunnel stop [NAME]
    llm configure
    llm prompt [text] [--file <file>]
"""

import click
import os
import subprocess
import requests
import yaml
from pathlib import Path
from rich.padding import Padding
from cloudmesh.ai.common.io import console
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Header, Footer
from yamldb import YamlDB
from cloudmesh.ai.common.remote import RemoteExecutor
from cloudmesh.ai.command.launch import AiderLauncher
from cloudmesh.ai.vllm.server_uva import ServerUVA
from cloudmesh.ai.vllm.server_dgx import ServerDGX
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.vllm.exceptions import VLLMError, VLLMConnectionError, VLLMConfigError, VLLMRuntimeError
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.ijob import IJob
from cloudmesh.ai.command.orchestrator import VLLMOrchestrator, get_default_host, get_server

class RenderVLLMTable:
    """Helper class to render vLLM configurations into a Textual DataTable."""
    
    @staticmethod
    def render(table: DataTable, servers: dict):
        table.cursor_type = "row"
        table.add_columns(
            "Service Name", "Host", "Model", "Image", 
            "TP Size", "GPU Util", "Port", "Account", 
            "Partition", "Reservation", "GRES", "CPUs", "Mem"
        )

        if isinstance(servers, dict):
            for name, config in servers.items():
                if isinstance(config, dict):
                    table.add_row(
                        name,
                        config.get("host", "N/A"),
                        config.get("model", "N/A"),
                        config.get("image", "N/A"),
                        str(config.get("tensor_parallel_size", "N/A")),
                        str(config.get("gpu_memory_utilization", "N/A")),
                        str(config.get("port", "8000")),
                        config.get("account", "N/A"),
                        config.get("partition", "N/A"),
                        config.get("reservation", "N/A"),
                        config.get("gres", "N/A"),
                        str(config.get("cpus", "N/A")),
                        config.get("mem", "N/A"),
                    )
class VLLMServiceSelector(App):
    """Textual App for selecting a vLLM service."""
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, servers):
        super().__init__()
        self.servers = servers
        self.selected_service = None
        self.selected_host = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        RenderVLLMTable.render(table, self.servers)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one(DataTable)
        row_data = table.get_row(event.row_key)
        self.selected_service = row_data[0]
        self.selected_host = row_data[1]
        self.exit()

def select_vllm_service(db, group_filter=None):
    """Interactively select a vLLM service from the available configurations using Textual."""
    servers = db.get("cloudmesh.ai.server", {})
    
    if not servers:
        console.error("No vLLM server configurations found in the config file.")
        return None, None, None

    app = VLLMServiceSelector(servers)
    app.run()
    
    return app.selected_service, None, app.selected_host


@click.group()
def llm_group():
    """LLM management extension."""
    pass

@llm_group.command(name="start")
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
@click.option("--info", is_flag=True, help="Display server configuration info")
@click.option("--export", is_flag=True, help="Export launch scripts to local directory for customization")
@click.option("--port", type=int, help="Override both local and remote ports")
@click.argument("name")
def start(name, ui, claude, info, export, port):
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
                clients = orchestrator.db.get("cloudmesh.ai.client", {})
                webui_cfg = clients.get("openwebui", {}) if isinstance(clients, dict) else {}
                WebUILauncher().launch(client_config=webui_cfg)
            elif claude:
                console.print("[bold green]Launching Claude...[/bold green]")
                clients = orchestrator.db.get("cloudmesh.ai.client", {})
                claude_cfg = clients.get("claude", {}) if isinstance(clients, dict) else {}
                ClaudeLauncher().launch(client_config=claude_cfg)
            else:
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
        import traceback
        console.error(f"Error orchestrating vLLM launch: {e}")
        console.print(traceback.format_exc())

@llm_group.command(name="stop")
@click.argument("identifier", required=False)
@click.option("--port", type=str, help="Port or partial port (e.g. '123') to identify the job")
def stop(identifier, port):
    """Stop a vLLM server (UVA HPC specific). Supports JobID, fuzzy port, or config port."""
    try:
        orchestrator = VLLMOrchestrator()
        
        if identifier:
            if identifier.isdigit():
                if orchestrator.stop_uva(port_pattern=identifier):
                    console.ok(f"Successfully stopped server matching {identifier}.")
                    return
            if orchestrator.stop_uva(server_name=identifier, port_pattern=port):
                console.ok(f"Successfully stopped server {identifier}.")
                return
        elif port:
            if orchestrator.stop_uva(port_pattern=port):
                console.ok(f"Successfully stopped server matching port {port}.")
                return
        else:
            servers = orchestrator.db.get("cloudmesh.ai.server", {})
            if not servers:
                console.error("No servers configured. Use 'cmc llm start <name>' first.")
                return
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

@llm_group.command(name="kill")
@click.argument("name")
@click.option("--tunnel", is_flag=True, help="Close the SSH tunnel after killing")
def kill(name, tunnel):
    """Forcefully kill vLLM server."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        servers = db.get("cloudmesh.ai.server", {})
        target_host = None
        if isinstance(servers, dict):
            target_host = servers.get(name, {}).get("host")
        
        if not target_host:
            target_host = get_default_host()
            if not target_host:
                raise ValueError(f"Could not resolve host for service '{name}' and no default host configured.")
        
        server = get_server(target_host)
        server.kill(name)
        
        if tunnel:
            # Use TunnelManager to stop the tunnel
            config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
            db = YamlDB(filename=config_path)
            server_config = db.get(f"cloudmesh.ai.server.{name}", {})
            port = server_config.get('port', '8000')
            
            success, result = tunnel_manager.stop_tunnel(target_host, port)
            if success:
                console.ok(f"Tunnel closed (PID: {result})")
            else:
                console.warning(f"Could not close tunnel: {result}")
            
        console.ok(f"Successfully killed vLLM server '{name}' on {target_host}")
    except Exception as e:
        console.error(f"Error killing vLLM server: {e}")

@llm_group.command(name="status")
@click.argument("name")
def status(name):
    """Check vLLM server status."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        servers = db.get("cloudmesh.ai.server", {})
        target_host = None
        group = None
        if isinstance(servers, dict):
            target_host = servers.get(name, {}).get("host")
        
        if not target_host:
            target_host = get_default_host()
            if not target_host:
                raise ValueError(f"Could not resolve host for service '{name}' and no default host configured.")
            group = "uva" if ("uva" in target_host.lower() or "rivanna" in target_host.lower()) else "dgx"
        
        config = VLLMConfig(db, group, name)
        client = VLLMClient(config)
        
        status_text = client.get_status()
        
        # Check for tunnel (simplified check: is localhost:port reachable?)
        tunnel_status = "Unknown"
        try:
            import socket
            with socket.create_connection(("127.0.0.1", config.get('port', 8000)), timeout=1):
                tunnel_status = "Active"
        except:
            tunnel_status = "Inactive"

        console.print(f"Server '{name}' on {target_host} status: [bold]{status_text}[/bold]")
        console.print(f"Tunnel status: {tunnel_status}")
    except Exception as e:
        console.error(f"Error checking status: {e}")

@llm_group.command(name="logs")
@click.argument("name")
def logs(name):
    """Retrieve logs for the vLLM server."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        servers = db.get("cloudmesh.ai.server", {})
        target_host = None
        group = None
        if isinstance(servers, dict):
            target_host = servers.get(name, {}).get("host")
        
        if not target_host:
            target_host = get_default_host()
            if not target_host:
                raise ValueError(f"Could not resolve host for service '{name}' and no default host configured.")
            group = "uva" if ("uva" in target_host.lower() or "rivanna" in target_host.lower()) else "dgx"
        
        config = VLLMConfig(db, group, name)
        client = VLLMClient(config)
        
        log_content = client.get_logs()
        console.print(f"\n[bold blue]Logs for {name} on {target_host}:[/bold blue]\n{log_content}")
    except Exception as e:
        console.error(f"Error retrieving logs: {e}")

@llm_group.command(name="info")
def info():
    """List all running vLLM servers."""
    try:
        orchestrator = VLLMOrchestrator()
        orchestrator.list_running_servers()
    except Exception as e:
        console.error(f"Error listing servers: {e}")

@click.group(name="tunnel")
def tunnel_group():
    """Tunnel management commands."""
    pass

@tunnel_group.command(name="stop")
@click.argument("name")
def stop_tunnel(name):
    """Stop the SSH tunnel for a specific server."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        server_config = db.get(f"cloudmesh.ai.server.{name}", {})
        if not server_config:
            raise VLLMConfigError(f"Server '{name}' not found in configuration.")
        
        host = server_config.get("host")
        port = server_config.get("port", "8000")
        if not host:
            raise VLLMConfigError(f"Host not specified for server '{name}'.")
            
        success, result = tunnel_manager.stop_tunnel(host, port)
        if success:
            console.ok(f"Tunnel for {name} stopped (PID: {result})")
        else:
            console.error(result)
    except VLLMError as e:
        console.error(str(e))
    except Exception as e:
        console.error(f"Unexpected error stopping tunnel: {e}")

llm_group.add_command(tunnel_group, name="tunnel")

@llm_group.command(name="default")
@click.argument("type", type=click.Choice(['server', 'client'], case_sensitive=False))
@click.argument("name")
def set_default(type, name):
    """Set the default server or client. Usage: cmc llm default [server|client] [NAME]"""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        db.set(f"cloudmesh.ai.default.{type}", name)
        console.ok(f"Default {type} set to: {name}")
    except Exception as e:
        console.error(f"Error setting default {type}: {e}")

@llm_group.command(name="configure")
def configure():
    """Interactively configure vLLM settings."""
    try:
        # 1. Config file path
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        
        # 2. Current state
        current_server = db.get("cloudmesh.ai.default.server")
        
        console.banner("vLLM Configuration", f"Config File: {config_path}\nCurrent Default Server: [bold]{current_server or 'Not set'}[/bold]")
        
        # 3. List available servers from server config
        servers = db.get("cloudmesh.ai.server", {})
        if servers:
            console.print("\n[bold]Available servers in llm.yaml:[/bold]")
            for name in sorted(servers.keys()):
                marker = "[green]✓[/green]" if name == current_server else " "
                console.print(f" {marker} {name}")
        else:
            console.warning("No servers found in llm.yaml.")

        # 4. Interactive prompt
        console.print("\n")
        prompt_text = f"Enter default server [{current_server}]: " if current_server else "Enter default server: "
        new_server = input(prompt_text).strip()
        
        if new_server and new_server != current_server:
            db.set("cloudmesh.ai.default.server", new_server)
            console.ok(f"Default server updated to: {new_server}")
        elif not new_server and not current_server:
            console.error("A server must be specified.")
        else:
            console.msg("No changes made to configuration.")
            
    except Exception as e:
        console.error(f"Error during configuration: {e}")

@llm_group.command(name="init")
def init():
    """Initialize vLLM server configurations with defaults for DGX and UVA."""
    config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
    
    if os.path.exists(config_path):
        console.warning(f"Configuration file already exists at {config_path}")
        if not console.ynchoice("Do you want to overwrite it with default configurations?", default=False):
            console.msg("Initialization cancelled.")
            return

    if VLLMConfig.reset():
        console.ok("vLLM server configurations initialized successfully!")
        console.banner("Welcome to vLLM Management", 
            "Default configurations for DGX and UVA have been added to your config.\n"
            "You can now use 'cmc llm start [NAME]' to launch a server.")
    else:
        console.error("Error: Could not find default configuration file to initialize from.")

@llm_group.command(name="reset")
def reset():
    """Reset the vLLM server configuration YAML file to the default version."""
    if console.ynchoice("This will overwrite your current vLLM server configurations. Are you sure you want to proceed?", default=False):
        if VLLMConfig.reset():
            console.ok("vLLM server configuration has been reset to defaults.")
        else:
            console.error("Error: Could not find default configuration file to reset from.")
    else:
        console.warning("Reset cancelled.")

@llm_group.command(name="launch")
@click.argument("client")
def launch(client):
    """Launch a specific LLM client (e.g., 'aider')."""
    if client == "aider":
        try:
            # Path to the config file (relative to this file: src/cloudmesh/ai/command/vllm.py)
            # Templates are in src/cloudmesh/ai/command/template/
            config_path = Path(__file__).parent / "template" / "aider.yaml"
            
            if not config_path.exists():
                console.error(f"Config file not found at {config_path}")
                return

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            aider_config = config.get("cloudmesh", {}).get("ai", {}).get("aider", {})
            model = aider_config.get("model")
            key_file_path = aider_config.get("key_file")
            url = aider_config.get("url")

            if not all([model, key_file_path, url]):
                console.error("Missing required configuration (model, key_file, or url) in aider.yaml")
                return

            key_path = Path(key_file_path).expanduser()
            if not key_path.exists():
                console.error(f"Key file not found at {key_path}")
                return

            api_key = key_path.read_text().strip()
            
            launcher_config = {
                "model": model,
                "OPENAI_API_KEY": api_key,
                "OPENAI_API_BASE": url,
            }
            
            console.print("[blue]Launching Aider with Gemma 4...[/blue]")
            AiderLauncher().launch(client_config=launcher_config)
            
        except Exception as e:
            console.error(f"Error launching aider: {e}")
    else:
        console.error(f"Unsupported client '{client}'. Supported clients: aider")

@llm_group.command(name="prompt")
@click.argument("text", required=False)
@click.option("--file", type=click.Path(exists=True), help="Prompt from file")
def prompt(text, file):
    """Send a prompt to the vLLM API."""
    prompt_text = ""
    if file:
        with open(file, "r") as f:
            prompt_text = f.read().strip()
    elif text:
        prompt_text = text
    else:
        console.error("Error: Please provide either a prompt text or a --file")
        return

    try:
        payload = {
            "model": "google/gemma-4-31B-it", # Default model
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7
        }
        response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        console.print(f"\n[bold blue]vLLM Response:[/bold blue]\n{content}")
    except Exception as e:
        console.error(f"Error calling vLLM API: {e}")

def register(cli=None, **kwargs):
    """Register the llm command group. 
    If cli is None, it's being called as the entry point directly by DelegatingCommand.
    """
    if cli is None:
        # DelegatingCommand passes args and standalone_mode via kwargs
        args = kwargs.get('args')
        standalone_mode = kwargs.get('standalone_mode', True)
        try:
            llm_group.main(args=args, standalone_mode=standalone_mode)
        except Exception:
            # Fallback for cases where .main() is not available or fails
            llm_group()
        return
    cli.add_command(llm_group, name="llm")