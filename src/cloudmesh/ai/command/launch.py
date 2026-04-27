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
import subprocess
import textwrap
import urllib.request
import sys
from rich.padding import Padding
from cloudmesh.ai.common.config import Config
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
import time
import yaml
from cloudmesh.ai.command.vllm import get_server, get_default_host
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from yamldb import YamlDB

def get_vllm_api_key(config, keys_path_override=None):
    """Retrieve vLLM API key from main config or a specified keys file."""
    # 1. Try main config
    api_key = config.get("ai.llm.vllm_api_key")
    if api_key:
        return api_key

    # 2. Try keys file (either from config or default)
    keys_path = keys_path_override or config.get("keys") or os.path.expanduser("~/.config/cloudmesh/keys.yaml")
    keys_path = os.path.expanduser(keys_path).replace("$HOME", os.path.expanduser("~"))
    
    if os.path.exists(keys_path):
        try:
            with open(keys_path, "r") as f:
                keys = yaml.safe_load(f)
                if keys and "VLLM_API_KEY" in keys:
                    return keys["VLLM_API_KEY"]
        except Exception:
            pass
    
    return None

class DockerManager:
    """Handles Docker operations and lifecycle management."""

    def check_docker(self):
        """Verify that Docker is running. On macOS, offer to start it."""
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            if os_is_mac():
                if console.ynchoice("Docker is not running. Would you like to start Docker Desktop?", default=True):
                    console.print("Starting Docker Desktop...")
                    subprocess.run(["open", "-a", "Docker"], capture_output=True)
                    return self._wait_for_docker()
            
            console.error("Docker is not running or could not be found.")
            console.print("Please start the Docker Desktop application and try again.")
            return False

    def _wait_for_docker(self):
        """Poll docker info until it becomes available."""
        console.print("Waiting for Docker to start...", end="", flush=True)
        for _ in range(30):  # Wait up to 30 seconds
            try:
                subprocess.run(["docker", "info"], check=True, capture_output=True)
                console.print("\n")
                console.ok("Docker is now available!")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(".", end="", flush=True)
                time.sleep(1)
        
        console.print("\n")
        console.error("Docker failed to start within 30 seconds.")
        return False

    def stop_container(self, container_name: str):
        """Stop and remove a specific Docker container."""
        console.print(f"Stopping existing container {container_name} if it exists...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

    def run_container(self, cmd: str):
        """Execute a docker run command."""
        try:
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Error launching container: {e.stderr.decode()}")
            return False

class WebUILauncher:
    """Handles the lifecycle of the Open WebUI Docker container."""

    def __init__(self):
        self.config = Config()
        self.docker = DockerManager()
        self.container_name = "open-webui"
        self.webui_port = 3000
        self.local_tunnel_port = 8001
        self.image = "ghcr.io/open-webui/open-webui:main"

    def _wait_for_webui(self, timeout=30):
        """Poll the WebUI port until it returns a successful response."""
        url = f"http://localhost:{self.webui_port}"
        console.print(f"[blue]Waiting for WebUI to be ready at {url}...[/blue]", end="")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.getcode() == 200:
                        console.print(" [bold green]Ready![/bold green]")
                        return True
            except Exception:
                print(".", end="", flush=True)
                time.sleep(1)
        
        console.print("\n")
        console.warning("WebUI took too long to respond. Opening browser anyway...")
        return False

    def launch(self, client_config=None):
        """Launch the Open WebUI container."""
        if not self.docker.check_docker():
            return

        self.docker.stop_container(self.container_name)

        # Retrieve keys and config from cloudmesh.ai configuration
        # Use client_config if provided to get the specific keys path
        api_key = get_vllm_api_key(self.config, keys_path_override=client_config.get("keys") if client_config else None)
        webui_name = self.config.get("ai.llm.webui_name", "Cloudmesh AI Portal")
        
        if not api_key:
            console.error("vllm_api_key not found in configuration.")
            console.print("Please set it in ~/.config/cloudmesh/ai/common.yaml or ~/.config/cloudmesh/keys.yaml")
            return

        console.print(banner("Launching Open WebUI", f"Image: {self.image}\nPort: {self.webui_port}"))

        # Construct the docker run command
        # Quote values to prevent shell injection or parsing errors (especially for API keys)
        # We use explicit backslashes for a nicely formatted multi-line command output
        cmd = textwrap.dedent(f"""\
            docker run -d \\
              -p {self.webui_port}:8080 \\
              --add-host=host.docker.internal:host-gateway \\
              -v open-webui:/app/backend/data \\
              -e OPENAI_API_BASE_URL="http://host.docker.internal:{self.local_tunnel_port}/v1" \\
              -e OPENAI_API_KEY="{api_key}" \\
              -e VLLM_API_KEY="{api_key}" \\
              -e HF_TOKEN="{api_key}" \\
              -e WEBUI_NAME="{webui_name}" \\
              --name {self.container_name} \\
              --restart always \\
              {self.image}
        """).strip()

        console.print(f"[blue]Executing command:[/blue]\n{cmd}")

        if self.docker.run_container(cmd):
            success_msg = (
                f"Setup Complete!\n"
                f"1. Ensure your SSH tunnel is running (localhost:{self.local_tunnel_port} -> server).\n"
                f"2. Access the UI at: http://localhost:{self.webui_port}\n"
                f"3. Opening the UI in your default browser in a few seconds..."
            )
            console.print(banner("Success", success_msg))
            
            # Wait for the application to be fully ready before opening the browser
            self._wait_for_webui()
            os.system(f"open http://localhost:{self.webui_port}")

class ClaudeLauncher:
    """Handles the launch of Claude Code with vLLM backend."""

    def __init__(self):
        self.config = Config()

    def launch(self, client_config=None):
        """Launch the claude CLI with required environment variables."""
        api_key = get_vllm_api_key(self.config, keys_path_override=client_config.get("keys") if client_config else None)
        claude_config = self.config.get("ai.llm.claude", {})
        
        base_url = claude_config.get("base_url", "http://127.0.0.1:8001")
        model = claude_config.get("model", "google/gemma-4-31B-it")
        
        if not api_key:
            console.error("vllm_api_key not found in configuration.")
            return

        console.print(banner("Launching Claude Code", f"Backend: {base_url}\nModel: {model}"))

        # Prepare environment variables
        env = os.environ.copy()
        env.update({
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        })

        try:
            # Use subprocess.run without capturing output to allow interactive CLI
            subprocess.run(["claude"], env=env, check=True)
        except FileNotFoundError:
            console.error("'claude' command not found. Please install Claude Code.")
        except subprocess.CalledProcessError as e:
            console.error(f"Claude Code exited with error: {e}")

class AiderLauncher:
    """Handles the launch of Aider with vLLM backend."""

    def __init__(self):
        self.config = Config()

    def launch(self, client_config=None):
        """Launch the aider CLI with required environment variables."""
        # Aider and its dependencies (like numpy) are most stable on Python 3.10-3.12.
        # Versions 3.13+ often fail due to removed modules (e.g., pkgutil.ImpImporter).
        if not (sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12):
            console.error(f"Aider is most stable on Python 3.10 to 3.12. Current version: {sys.version.split()[0]}")
            console.print("Python 3.13+ is currently causing installation failures with dependencies.")
            return

        api_key = get_vllm_api_key(self.config, keys_path_override=client_config.get("keys") if client_config else None)
        aider_config = self.config.get("ai.llm.aider", {})
        
        model = aider_config.get("model", "google/gemma-4-31B-it")
        base_url = aider_config.get("base_url", "http://127.0.0.1:8001/v1")
        
        if not api_key:
            console.error("vllm_api_key not found in configuration.")
            return

        console.print(banner("Launching Aider", f"Backend: {base_url}\nModel: {model}"))
        
        # Prepare environment variables for Aider (OpenAI compatible)
        env = os.environ.copy()
        env.update({
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": base_url,
        })
        
        try:
            # Launch aider with the specified model
            subprocess.run(["aider", "--model", f"openai/{model}"], env=env, check=True)
        except FileNotFoundError:
            console.error("'aider' command not found. Please install Aider.")
        except subprocess.CalledProcessError as e:
            console.error(f"Aider exited with error: {e}")

class VLLMOrchestrator:
    """Orchestrates the full pipeline from server start to client launch."""

    def __init__(self):
        self.config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        self.db = YamlDB(filename=self.config_path)

    def prepare_backend(self, name):
        """Ensure vLLM server is running, tunneled, and healthy."""
        # Resolve host from the service name in the config
        servers = self.db.get("cloudmesh.ai.server", {})
        target_host = None
        if isinstance(servers, dict):
            config = servers.get(name, {})
            target_host = config.get("host")

        if not target_host:
            raise ValueError(f"Could not resolve host for service '{name}' in configuration.")

        server = get_server(target_host)
        config = VLLMConfig(self.db, name)
        client = VLLMClient(config)

        console.print(banner("Orchestrating vLLM Backend", f"Host: {target_host}\nService: {name}"))
        
        # 1. Initial Health Check (Check if already running and tunneled)
        console.print("[blue]Checking if vLLM server is already available...[/blue]")
        if client.is_alive():
            console.ok("vLLM server is already ALIVE and tunneled!")
            return True

        # 2. Try establishing tunnel first (Skip for localhost)
        if target_host not in ["localhost", "127.0.0.1"]:
            console.print("[blue]Establishing SSH tunnel...[/blue]")
            server.tunnel(name)
        else:
            console.print("[blue]Local host detected, skipping SSH tunnel...[/blue]")
        
        if client.is_alive():
            console.ok("vLLM server is now available!")
            return True

        # 3. Start Server (Skip remote start for localhost)
        if target_host not in ["localhost", "127.0.0.1"]:
            console.print("[blue]Starting vLLM server on remote host...[/blue]")
            server.start(name)
        else:
            console.warning("Local host detected. Please ensure the vLLM server is started locally.")
        
        # 4. Final Health Check Poll
        console.print("[blue]Verifying model health...[/blue]")
        for i in range(24):
            if client.is_alive():
                console.ok("vLLM server is now ALIVE and model is loaded!")
                return True
            console.print(f"Waiting for model to load... ({i+1}/24)", end="\r")
            time.sleep(5)
        
        console.error("vLLM server failed to become healthy within 2 minutes.")
        return False

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
def launch_aider():
    """Launch Aider with vLLM backend."""
    launcher = AiderLauncher()
    launcher.launch()

@launch_group.command(name="llm")
@click.argument("name")
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
@click.option("--info", is_flag=True, help="Display server configuration info")
def launch_vllm(name, ui, claude, info):
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

        if orchestrator.prepare_backend(name):
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
        except subprocess.CalledProcessError as e:
            console.error(f"Failed to install Aider: {e}")
    else:
        console.error(f"Installation for tool '{tool}' is not supported. Supported tools: aider")

@launch_group.command(name="stop")
@click.argument("tool")
def stop_tool(tool):
    """Stop a launched AI tool."""
    if tool == "webui":
        docker = DockerManager()
        docker.stop_container("open-webui")
        console.ok(f"Stopped {tool} container.")
    elif tool in ["claude", "aider"]:
        console.print(f"[yellow]{tool.capitalize()} is a foreground process. Please use Ctrl+C to exit.[/yellow]")
    else:
        console.print(f"[red]Unknown tool: {tool}[/red]")

def register(cli=None, **kwargs):
    """Register the launch command group."""
    if cli is None:
        # Handle standalone execution
        args = kwargs.get('args')
        standalone_mode = kwargs.get('standalone_mode', True)
        try:
            launch_group.main(args=args, standalone_mode=standalone_mode)
        except Exception:
            launch_group()
        return
    cli.add_command(launch_group, name="launch")