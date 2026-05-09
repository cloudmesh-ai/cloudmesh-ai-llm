import os
import subprocess
import textwrap
import urllib.request
import time
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
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
                if console.ynchoice(
                    "Docker is not running. Would you like to start Docker Desktop?",
                    default=True,
                ):
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
        self.docker = DockerManager()
        # Use YamlDB to load the resolved configuration in memory
        self.db = YamlDB(
            filename=os.path.expanduser("~/.config/cloudmesh/llm.yaml"),
            backend=":memory:",
        )
        self.container_name = "open-webui"
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
