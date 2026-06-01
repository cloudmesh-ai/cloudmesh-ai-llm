import click
import requests
import socket
import os
import yaml
import subprocess
from rich.table import Table
from cloudmesh.ai.common import DotDict, banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator, get_vllm_api_key
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.process_manager import process_registry
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.exceptions import VLLMError, VLLMConfigError

@click.command(name="processes")
def processes():
    """List all background processes managed by the orchestrator (tunnels, logs)."""
    try:
        # 1. Get transient processes from registry
        tracked = process_registry.list_processes()
        
        # 2. Get persistent tunnels from manager
        tunnels = tunnel_manager._load_state()
        
        console.print("\n[bold blue]Managed Background Processes[/bold blue]\n")
        
        if not tracked and not tunnels:
            console.msg("No background processes are currently running.")
            return

        if tunnels:
            console.print("[bold]Active Tunnels:[/bold]")
            for key, pid in tunnels.items():
                console.print(f" - {key:<20} PID: {pid}")
            console.print("")

        if tracked:
            console.print("[bold]Active Registry Processes (Transient):[/bold]")
            for proc in tracked:
                status_color = "green" if proc["status"] == "Running" else "dim"
                console.print(f" - {proc['name']:<20} PID: {proc['pid']:<10} Status: [{status_color}]{proc['status']}[/{status_color}]")
        
        console.print("")

    except Exception as e:
        console.error(f"Error listing processes: {e}")

@click.command(name="status")
@click.argument("name", required=False)
@click.pass_context
def status(ctx, name):
    """Check vLLM server status. If NAME is omitted, lists all running servers."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        
        if not name:
            # List all running servers with a rich table
            # We call it with quiet=True to avoid the redundant discovery table
            running = orchestrator.list_running_servers(quiet=True)
            if not running:
                console.msg("No vLLM servers are currently running.")
                return

            table = Table(title="vLLM Server Status", show_header=True, header_style="bold magenta")
            table.add_column("Server", style="cyan")
            table.add_column("Job ID", style="dim")
            table.add_column("Job Name", style="dim")
            table.add_column("Health", justify="center")
            table.add_column("Node", style="dim")
            table.add_column("GPUs", justify="center")
            table.add_column("Port", justify="right")
            table.add_column("Tunnel", justify="center")
            table.add_column("Start", justify="center")
            table.add_column("TTL", justify="center")

            for server_info in running:
                sname = server_info["server"]
                jid = server_info["job_id"]
                jname_slurm = server_info.get("job_name", "Unknown")
                node = server_info["node"]
                gpus = server_info.get("gpus", "Unknown")
                remote_port = server_info["port"]
                local_port = server_info["local_port"]
                start = server_info.get("start", "Unknown")
                ttl = server_info.get("ttl", "Unknown")
                
                # Resolve health via local tunnel (127.0.0.1)
                try:
                    # We check health on 127.0.0.1 using the local_port,
                    # as this is how the user actually accesses the server.
                    client = VLLMClient(orchestrator.config, server_name=sname, port=local_port, host="127.0.0.1", debug=debug)
                    health = client.get_status()
                    health_fmt = {
                        "READY": "🟢 [bold green]READY[/bold green]",
                        "STARTING": "🟡 [bold yellow]STARTING[/bold yellow]",
                        "OFFLINE": "🔴 [bold red]OFFLINE[/bold red]"
                    }.get(health, f"⚪ [bold yellow]{health}[/bold yellow]")
                except Exception:
                    health_fmt = "[bold red]ERROR[/bold red]"

                # Resolve tunnel status
                tunnel_status = "🔴 [bold red]Inactive[/bold red]"
                try:
                    if local_port and str(local_port).isdigit():
                        with socket.create_connection(("127.0.0.1", int(local_port)), timeout=1):
                            tunnel_status = "🟢 [bold green]Active[/bold green]"
                except Exception:
                    pass

                table.add_row(sname, str(jid), jname_slurm, health_fmt, node, str(gpus), f"{local_port}:{remote_port}", tunnel_status, start, ttl)

            console.print("\n")
            console.print(table)
            console.print("")
            return
    
        # Detailed status for a specific server
        server_config = orchestrator.config.get_server(name)
        if not server_config:
            console.error(f"Server '{name}' not found in configuration.")
            return
    
        # Resolve identity for the client
        client = VLLMClient(orchestrator.config, server_name=name, debug=debug)
        status_text = client.get_status()
        
        # Check for tunnel status
        tunnel_status = "Inactive"
        try:
            local_port = server_config.get("local_port") or orchestrator.config.get("port", 8000)
            with socket.create_connection(("127.0.0.1", int(local_port)), timeout=1):
                tunnel_status = "Active"
        except (socket.error, OSError):
            pass

        # Get allocated node from state
        state = orchestrator._load_state()
        node_name = state.get(name, {}).get("node_name", "Unknown")
        
        # Use a small table for detailed view
        detail_table = Table(show_header=False, box=None)
        detail_table.add_row("Host", f"[cyan]{client.host}[/cyan]")
        detail_table.add_row("Node", f"[dim]{node_name}[/dim]")
        detail_table.add_row("Port", f"{client.port}")
        
        health_emoji = "🟢" if status_text == "READY" else "🟡" if status_text == "STARTING" else "🔴"
        health_color = "green" if status_text == "READY" else "yellow" if status_text == "STARTING" else "red"
        detail_table.add_row("Health", f"{health_emoji} [{health_color}][bold]{status_text}[/{health_color}]")
        
        tunnel_emoji = "🟢" if tunnel_status == "Active" else "🔴"
        tunnel_color = "green" if tunnel_status == "Active" else "red"
        detail_table.add_row("Tunnel", f"{tunnel_emoji} [{tunnel_color}]{tunnel_status}[/{tunnel_color}]")

        console.banner(f"Status: {name}")
        console.print(detail_table)
    
    except Exception as e:
        console.error(f"Error checking status: {e}")

@click.command(name="logs")
@click.argument("name")
@click.option("--follow", "-f", is_flag=True, help="Stream logs in real-time")
@click.option("--grep", help="Filter logs by keyword (case-insensitive)")
@click.pass_context
def logs(ctx, name, follow, grep):
    """Retrieve logs for the vLLM server."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        identity = orchestrator.config.resolve_server_identity(name)
        target_host = identity["host"]
        
        client = VLLMClient(orchestrator.config, server_name=name, debug=debug)
        
        if follow:
            process = None
            try:
                process = client.stream_logs(grep=grep)
                process_name = f"logs_{name}"
                process_registry.register(process_name, process)
                
                console.print(f"\n[bold blue]Streaming logs for {name} on {target_host} (Ctrl+C to stop)...[/bold blue]\n")
                if grep:
                    console.print(f"[dim]Filtering by keyword: {grep}[/dim]")
                
                # Read output line by line
                for line in process.stdout:
                    console.print(line, end="")
                    
            except ValueError as e:
                console.error(str(e))
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped streaming logs.[/dim]")
            finally:
                if process:
                    process.stdout.close()
                    process.terminate()
                    process_registry.unregister(f"logs_{name}")
        else:
            log_content = client.get_logs(grep=grep)
            if not log_content:
                console.msg(f"No log entries found matching '{grep}'" if grep else "No logs found.")
            else:
                console.print(f"\n[bold blue]Logs for {name} on {target_host}:[/bold blue]\n{log_content}")
    except Exception as e:
        console.error(f"Error retrieving logs: {e}")

@click.command(name="prompt")
@click.argument("text", required=False)
@click.option("--file", type=click.Path(exists=True), help="Prompt from file")
def prompt(text, file):
    """Send a prompt to the vLLM API using the configured default server."""
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
        orchestrator = VLLMOrchestrator()
        # Resolve default server identity and config
        identity = orchestrator.config.resolve_server_identity("default")
        server_name = identity.get("server_name")
        
        server_config = orchestrator.config.get_server(server_name) if server_name else {}
        
        # Use config values with sensible fallbacks
        model = server_config.get("model", "google/gemma-4-31B-it")
        port = identity.get("port", 8000)
        api_key = get_vllm_api_key(orchestrator.config)
        
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7
        }
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        console.print(f"[dim]Prompting model {model} on port {port}...[/dim]")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        console.print(f"\n[bold blue]vLLM Response:[/bold blue]\n{content}")
    except Exception as e:
        console.error(f"Error calling vLLM API: {e}")

@click.command(name="install")
@click.argument("tool")
def install_tool(tool):
    """Install AI tools (e.g., aider)."""
    if tool == "aider":
        console.print(banner("Installing Aider", "Preparing isolated installation via pipx..."))
        
        # Print the plan
        plan = (
            "The following steps will be performed:\n"
            "1. Verify pipx installation (required for isolation)\n"
            "2. Verify Python 3.12 installation (required for Aider compatibility)\n"
            "3. Install 'aider-chat' using pipx with Python 3.12\n"
            "4. Check for 'pandoc' dependency (recommended for document conversion)"
        )
        console.print(f"\n[blue]{plan}[/blue]\n")

        if not console.ynchoice("Do you want to proceed with the installation?", default=True):
            console.msg("Installation cancelled.")
            return
        
        # Check if pipx is installed
        try:
            subprocess.run(["pipx", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.error("pipx not found. Aider requires pipx for isolated installation to avoid Python version conflicts.")
            if os_is_mac():
                console.print("Please install pipx using: 'brew install pipx && pipx ensurepath'")
            else:
                console.print("Please install pipx using: 'pip install pipx && pipx ensurepath'")
            return

        # Verify Python 3.12 is installed
        try:
            subprocess.run(["python3.12", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.error("Python 3.12 not found. Aider requires Python 3.12 for stability and compatibility.")
            if os_is_mac():
                console.print("Please install Python 3.12 using: 'brew install python@3.12'")
            else:
                console.print("Please install Python 3.12 using your system package manager.")
            return

        try:
            # Install aider-chat using pipx with explicit Python 3.12
            console.print("Installing aider-chat using Python 3.12...")
            subprocess.run(["pipx", "install", "--python", "python3.12", "aider-chat"], check=True)
            console.ok("Aider installed successfully via pipx using Python 3.12!")
            
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
            console.error(f"Failed to install Aider via pipx: {e}")
            console.print("Hint: Ensure pipx is updated ('pipx upgrade') and Python 3.12 is correctly installed in your PATH.")
    else:
        console.error(f"Unsupported tool '{tool}'. Currently only 'aider' is supported for installation.")

@click.group(name="tunnel")
def tunnel_group():
    """Tunnel management commands."""
    pass

@tunnel_group.command(name="stop")
@click.argument("name")
def stop_tunnel(name):
    """Stop the SSH tunnel for a specific server."""
    try:
        config_path = VLLMConfig.DEFAULT_USER_CONFIG_PATH
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                db = DotDict(yaml.safe_load(f) or {})
        else:
            db = DotDict()
        
        # Handle dot-notation lookup
        server_config = None
        if "cloudmesh" in db:
            server_config = db.get("cloudmesh", {}).get("ai", {}).get("server", {}).get(name, {})
        
        if not server_config:
            server_config = {}
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