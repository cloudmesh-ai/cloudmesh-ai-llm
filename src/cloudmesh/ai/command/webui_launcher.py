import os
import time
import textwrap
import urllib.request
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.command.docker_manager import DockerManager

class WebUILauncher:
    """Handles the lifecycle of the Open WebUI Docker container."""

    def __init__(self):
        self.docker = DockerManager()
        # Use YamlDB to load the resolved configuration in memory
        self.db = YamlDB(filename=os.path.expanduser("~/.config/cloudmesh/llm.yaml"), backend=":memory:")
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
        if not self.docker.check_docker():
            return

        self.docker.stop_container(self.container_name)

        # Use resolved config from YamlDB
        webui_config = self.db.get("cloudmesh.ai.client.openwebui") or self.db.get("cloudmesh.ai.llm.openwebui", {})
        config = {**webui_config, **(client_config or {})}
        
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")
        webui_name = config.get("webui_name", "Cloudmesh AI Portal")
        self.webui_port = config.get("PORT") or config.get("port", 3000)
        
        # Handle API Base for Docker
        base_url = config.get("OPENAI_API_BASE") or config.get("openai_api_base") or config.get("base_url", f"http://host.docker.internal:{self.local_tunnel_port}/v1")
        if base_url:
            base_url = base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

        if not api_key:
            console.error("openai_api_key not found in resolved configuration.")
            return

        console.print(banner("Launching Open WebUI", f"Image: {self.image}\nPort: {self.webui_port}"))

        # Construct the docker run command
        cmd = textwrap.dedent(f"""\
            docker run -d \\
              -p {self.webui_port}:8080 \\
              --add-host=host.docker.internal:host-gateway \\
              -v open-webui:/app/backend/data \\
              -e OPENAI_API_BASE_URL="{base_url}" \\
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