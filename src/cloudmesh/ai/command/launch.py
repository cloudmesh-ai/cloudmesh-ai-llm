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
from rich.padding import Padding
from cloudmesh.ai.common.config import Config
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
import time
from cloudmesh.ai.command.vllm import get_server, get_default_host
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from yamldb import YamlDB

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

    def launch(self):
        """Launch the Open WebUI container."""
        if not self.docker.check_docker():
            return

        self.docker.stop_container(self.container_name)

        # Retrieve keys and config from cloudmesh.ai configuration
        api_key = self.config.get("ai.llm.vllm_api_key")
        webui_name = self.config.get("ai.llm.webui_name", "Cloudmesh AI Portal")
        
        if not api_key:
            console.error("vllm_api_key not found in configuration.")
            console.print("Please set it in ~/.config/cloudmesh/ai/common.yaml or via AI_AI_LLM_VLLM_API_KEY")
            return

        console.print(banner("Launching Open WebUI", f"Image: {self.image}\nPort: {self.webui_port}"))

        # Construct the docker run command
        cmd = textwrap.dedent(f"""
            docker run -d \
              -p {self.webui_port}:8080 \
              --add-host=host.docker.internal:host-gateway \
              -v open-webui:/app/backend/data \
              -e OPENAI_API_BASE_URL=http://host.docker.internal:{self.local_tunnel_port}/v1 \
              -e OPENAI_API_KEY={api_key} \
              -e VLLM_API_KEY={api_key} \
              -e HF_TOKEN={api_key} \
              -e WEBUI_NAME={webui_name} \
              --name {self.container_name} \
              --restart always \
              {self.image}
        """).strip()

        if self.docker.run_container(cmd):
            success_msg = (
                f"Setup Complete!\n"
                f"1. Ensure your SSH tunnel is running (localhost:{self.local_tunnel_port} -> server).\n"
                f"2. Access the UI at: http://localhost:{self.webui_port}"
            )
            console.print(banner("Success", success_msg))

class ClaudeLauncher:
    """Handles the launch of Claude Code with vLLM backend."""

    def __init__(self):
        self.config = Config()

    def launch(self):
        """Launch the claude CLI with required environment variables."""
        api_key = self.config.get("ai.llm.vllm_api_key")
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

    def launch(self):
        """Launch the aider CLI with required environment variables."""
        api_key = self.config.get("ai.llm.vllm_api_key")
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
        self.config_path = os.path.expanduser("~/.config/cloudmesh/ai/vllm_servers.yaml")
        self.db = YamlDB(filename=self.config_path)

    def prepare_backend(self, name):
        """Ensure vLLM server is running, tunneled, and healthy."""
        target_host = get_default_host()
        if not target_host:
            raise ValueError("No default host configured. Use 'cme vllm set-host [HOST]' first.")

        server = get_server(target_host)
        group = "uva" if ("uva" in target_host.lower() or "rivanna" in target_host.lower()) else "dgx"
        config = VLLMConfig(self.db, group, name)
        client = VLLMClient(config)

        console.print(banner("Orchestrating vLLM Backend", f"Host: {target_host}\nService: {name}"))
        
        # 1. Initial Health Check (Check if already running and tunneled)
        console.print("[blue]Checking if vLLM server is already available...[/blue]")
        if client.is_alive():
            console.ok("vLLM server is already ALIVE and tunneled!")
            return True

        # 2. Try establishing tunnel first (Server might be running, but tunnel is down)
        console.print("[blue]Establishing SSH tunnel...[/blue]")
        server.tunnel(name)
        
        if client.is_alive():
            console.ok("vLLM server is now available via tunnel!")
            return True

        # 3. Start Server (Neither tunnel nor server was available)
        console.print("[blue]Starting vLLM server on remote host...[/blue]")
        server.start(name)
        
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

@launch_group.command(name="vllm")
@click.argument("name")
@click.option("--ui", is_flag=True, help="Launch WebUI after backend is ready")
@click.option("--claude", is_flag=True, help="Launch Claude after backend is ready")
def launch_vllm(name, ui, claude):
    """Full pipeline: Start vLLM server -> Tunnel -> Health Check -> Optional UI."""
    try:
        orchestrator = VLLMOrchestrator()
        if orchestrator.prepare_backend(name):
            console.ok(f"Backend {name} is ready!")
            
            if ui:
                console.print("[bold green]Launching WebUI...[/bold green]")
                WebUILauncher().launch()
            elif claude:
                console.print("[bold green]Launching Claude...[/bold green]")
                ClaudeLauncher().launch()
            else:
                console.msg("Backend is ready. You can now run 'cme launch webui' or 'cme launch claude'.")
        else:
            console.error("Backend preparation failed.")
    except Exception as e:
        console.error(f"Error orchestrating vLLM launch: {e}")

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