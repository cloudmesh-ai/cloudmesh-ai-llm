"""
Cloudmesh AI LLM Management Extension
======================================

This extension provides tools to launch, manage, and interact with LLM 
server containers on DGX and UVA hosts using named configurations.

Usage Examples:
-------------------------------------------------------------------------------
1. Set a default host:
   $ cmc llm set-host dgx-node-1

2. Set a default service:
   $ cmc llm default uva-default

3. Start an LLM server using interactive UI:
   $ cmc llm start --ui

4. Start an LLM server from a named config:
   $ cmc llm start uva-default --tunnel

5. Check the status of a server:
   $ cmc llm status uva-default

6. View the last 100 lines of logs:
   $ cmc llm logs uva-default

7. Stop a server gracefully:
   $ cmc llm stop uva-default --tunnel

8. Forcefully kill a server:
   $ cmc llm kill uva-default --tunnel

9. Send a prompt to the running server:
   $ cmc llm prompt "What is the capital of France?"
-------------------------------------------------------------------------------

Usage:
    llm start [NAME] [--tunnel] [--ui]
    llm stop [NAME] [--tunnel]
    llm kill [NAME] [--tunnel]
    llm status [NAME]
    llm logs [NAME]
    llm set-host [HOST]
    llm default [NAME]
    llm configure
    llm prompt [text] [--file <file>]
    llm -h | --help
"""

import click
import os
import subprocess
import requests
from rich.padding import Padding
from cloudmesh.ai.common.io import console
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Header, Footer
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.remote import RemoteExecutor
from cloudmesh.ai.vllm.server_uva import ServerUVA
from cloudmesh.ai.vllm.server_dgx import ServerDGX
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.ijob import IJob


def get_default_host(db=None):
    """Retrieve the default host by resolving the default server name."""
    if db is None:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
    
    # 1. Get the default server name
    default_server_name = db.get("cloudmesh.ai.default.server")
    if default_server_name:
        # 2. Resolve the host for that server
        servers = db.get("cloudmesh.ai.server", {})
        if isinstance(servers, dict):
            host = servers.get(default_server_name, {}).get("host")
            if host:
                return host
    
    # Fallback to legacy config.default_host
    return db.get("config.default_host")

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

def get_server(host):
    """Helper to instantiate the correct server class based on host."""
    if "uva" in host.lower() or "rivanna" in host.lower():
        return ServerUVA(host)
    return ServerDGX(host)

@click.group()
def llm_group():
    """LLM management extension."""
    pass

@llm_group.command(name="start")
@click.argument("name", required=False)
@click.option("--tunnel", is_flag=True, help="Create an SSH tunnel to the server")
@click.option("--ui", is_flag=True, help="Interactively select the vLLM service from a table")
@click.option("--sbatch", is_flag=True, help="Use Slurm sbatch instead of ijob (UVA only)")
def start(name, tunnel, ui, sbatch):
    """Start a vLLM server using a named configuration."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        
        if ui and not name:
            name, group, selected_host = select_vllm_service(db)
            if not name:
                return # select_vllm_service already prints error
        
        if not name:
            # Try to get default service name
            name = db.get("config.default_service")
            if not name:
                raise ValueError("No service name provided. Use 'cmc llm start [NAME]' or 'cmc llm start --ui'.")

        # Resolve host from the service name in the config
        servers = db.get("cloudmesh.ai.server", {})
        target_host = None
        group = None
        if isinstance(servers, dict):
            config = servers.get(name, {})
            target_host = config.get("host")
        
        if not target_host:
            # Fallback to default host if service not found or host not specified in service
            target_host = get_default_host(db)
            if not target_host:
                raise ValueError(f"Could not resolve host for service '{name}' and no default host configured.")
            # Try to guess group based on default host
            group = "uva" if ("uva" in target_host.lower() or "rivanna" in target_host.lower()) else "dgx"
        
        server = get_server(target_host)
        config = VLLMConfig(db, group, name)
        
        start_cmd = server.get_start_command(name)
        
        # Banner 1: Server Details
        details_content = f"Host: {target_host}\nService: {name}\n"
        if config:
            details_content += f"Model: {config.get('model')}\nImage: {config.get('image')}\nPort: {config.get('port', '8000')}\n"
        
        console.print(banner(f"Start vLLM Server: {name}", details_content))
        
        # Banner 2: Execution Plan
        ijob_helper = IJob(db).get(group, name)
        user = ijob_helper.username()
        working_dir = config.get('working_dir', '/scratch/$USER/cloudmesh/run').replace("$USER", user)
        script_path = f"{working_dir}/start_{name}.sh"
        exec_mode = "sbatch" if sbatch else "ijob"
        
        # Determine the detailed execution command (including partition/reservation for UVA ijob)
        exec_cmd_display = exec_mode
        if exec_mode == "ijob" and group == "uva":
            batch_job = VLLMBatchJob(config, script_path)
            try:
                exec_cmd_display = batch_job.generate_ijob_command()
            except ValueError as e:
                raise ValueError(f"Configuration error for interactive startup: {e}")

        execution_plan = (
            f"Step 1. Shell script: {script_path}\n"
            f"Step 2. Upload method: SSH heredoc (cat << 'EOF')\n"
            f"Step 3. Execution mode: {exec_mode}\n"
            f"Step 4. Remote command: {exec_cmd_display} {script_path}"
        )
        console.print(banner("Execution Plan", execution_plan))

        # Detailed Step Panels (indented)
        step1_cmd = f"mkdir -p {working_dir}"
        step2_cmd = f"Write content to {script_path}"
        step4_cmd = f"{exec_cmd_display} {script_path}"

        console.print(Padding(banner('Step 1: Create directory', step1_cmd), (0, 0, 0, 4)))
        console.print(Padding(banner('Step 2: Upload script', step2_cmd), (0, 0, 0, 4)))
        console.print(Padding(banner('Step 3: Execution mode', exec_mode), (0, 0, 0, 4)))
        console.print(Padding(banner('Step 4: Execute remote command', step4_cmd), (0, 0, 0, 4)))

        if exec_mode == "ijob" and group == "uva":
            # Print ijob details in a banner
            ijob_details = (
                f"Allocation: {config.get('allocation') or 'Not specified'}\n"
                f"Partition: {config.get('partition')}\n"
                f"Time: {config.get('time', '24:00:00')}\n"
                f"GRES: {config.get('gres', 'gpu:1')}\n"
                f"Reservation: {config.get('reservation') or 'None'}"
            )
            console.print(banner("ijob Parameters", ijob_details))

            # Print the exact ijob command in a banner
            console.print(banner("ijob Command", f"{exec_cmd_display} {script_path}"))
        
        # Banner 3: Command to be executed
        console.print(banner("Command to be executed", start_cmd))

        raw_sequence = (
            f"RemoteExecutor.execute('mkdir -p {working_dir}')\n"
            f"RemoteExecutor.write_remote_file(content, '{script_path}')\n"
            f"RemoteExecutor.execute('chmod +x {script_path}')\n"
            f"RemoteExecutor.execute('{exec_cmd_display} {script_path}')"
        )
        console.print(f"\n[bold]Remote Execution Sequence:[/bold]\n{raw_sequence}\n")

        if not console.ynchoice(f"Do you want to proceed with starting vLLM server '{name}' on {target_host}?", default=True):
            console.warning("Start cancelled by user.")
            return
        
        with RemoteExecutor(target_host) as executor:
            executor.execute(f"mkdir -p {working_dir}")
            executor.write_remote_file(start_cmd, script_path)
            executor.execute(f"chmod +x {script_path}")
            executor.execute(f"{exec_cmd_display} {script_path}")
        
        server.start(name, sbatch=sbatch)
        console.ok(f"Successfully started vLLM server '{name}' on {target_host}")
        
        if tunnel:
            console.print(f"[blue]Creating SSH tunnel to {target_host}...[/blue]")
            server.tunnel(name)
            console.ok(f"SSH tunnel created: localhost:{config.get('port', '8000')} -> {target_host}")
    except Exception as e:
        console.error(f"Error starting vLLM server: {e}")

@llm_group.command(name="stop")
@click.argument("name")
@click.option("--tunnel", is_flag=True, help="Close the SSH tunnel after stopping")
def stop(name, tunnel):
    """Stop the vLLM server."""
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
        server.stop(name)
        
        if tunnel:
            console.msg(f"Closing tunnel for {name}...")
            
        console.ok(f"Successfully stopped vLLM server '{name}' on {target_host}")
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
            console.msg(f"Closing tunnel for {name}...")
            
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
        
        alive = client.is_alive()
        status_text = "ALIVE" if alive else "DEAD"
        
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
    """Display current vLLM configuration."""
    try:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)
        
        current_host = db.get("config.default_host")
        
        info_content = (
            f"Config File: {config_path}\n"
            f"Default Host: [bold]{current_host or 'Not set'}[/bold]"
        )
        console.print(banner("vLLM Configuration Info", info_content))
        
    except Exception as e:
        console.error(f"Error retrieving configuration info: {e}")

@llm_group.command(name="default")
@click.argument("type")
@click.argument("name")
def set_default(type, name):
    """Set the default server or client. Usage: cmc llm default [server|client] [NAME]"""
    if type not in ["server", "client"]:
        console.error("Invalid type. Use 'server' or 'client'.")
        return
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
        current_host = db.get("config.default_host")
        
        console.print(banner("vLLM Configuration", f"Config File: {config_path}\nCurrent Default Host: [bold]{current_host or 'Not set'}[/bold]"))
        
        # 3. List available hosts from servers config
        servers = db.get("cloudmesh.ai.server", {})
        available_hosts = set()
        if isinstance(servers, dict):
            for config in servers.values():
                if isinstance(config, dict) and config.get("host"):
                    available_hosts.add(config.get("host"))
        
        if available_hosts:
            console.print("\n[bold]Available hosts in vllm_servers.yaml:[/bold]")
            for host in sorted(list(available_hosts)):
                marker = "[green]✓[/green]" if host == current_host else " "
                console.print(f" {marker} {host}")
        else:
            console.warning("No hosts found in vllm_servers.yaml.")

        # 4. Interactive prompt
        console.print("\n")
        prompt_text = f"Enter default host [{current_host}]: " if current_host else "Enter default host: "
        new_host = input(prompt_text).strip()
        
        if new_host and new_host != current_host:
            db.set("config.default_host", new_host)
            console.ok(f"Default host updated to: {new_host}")
        elif not new_host and not current_host:
            console.error("A host must be specified.")
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
        console.print(banner("Welcome to vLLM Management", 
            "Default configurations for DGX and UVA have been added to your config.\n"
            "You can now use 'cmc llm start [NAME]' to launch a server."))
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