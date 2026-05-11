import os
import subprocess
import textwrap
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.command.docker_manager import DockerManager

class AiderLauncher:
    """Handles the launch of Aider with vLLM backend."""

    def __init__(self):
        self.docker = DockerManager()
        # Use YamlDB to load the resolved configuration in memory
        self.db = YamlDB(filename=os.path.expanduser("~/.config/cloudmesh/llm.yaml"), backend=":memory:")

    def launch(self, client_config=None):
        """Launch the aider CLI with required environment variables."""
        # Use resolved config from YamlDB - check both client and llm paths for compatibility
        aider_config = self.db.get("cloudmesh.ai.client.aider") or self.db.get("cloudmesh.ai.llm.aider", {})
        
        # Merge with client_config if provided
        config = {**aider_config, **(client_config or {})}
        
        # Support both uppercase and lowercase keys
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")
        model = config.get("model", "google/gemma-4-31B-it")
        
        # Resolve base URL: use explicit base_url if provided, otherwise construct from port
        base_url = config.get("OPENAI_API_BASE") or config.get("openai_api_base") or config.get("base_url")
        if not base_url:
            port = config.get("port", 8001)
            base_url = f"http://127.0.0.1:{port}/v1"
        
        if not api_key:
            console.error("openai_api_key not found in resolved configuration.")
            return

        console.print(banner("Launching Aider", f"Backend: {base_url}\nModel: {model}"))
        
        # Prepare environment variables for Aider (OpenAI compatible)
        env = os.environ.copy()
        env.update({
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": base_url,
        })
        
        aider_model = self._get_aider_model(model)
        try:
            # Launch aider with the specified model
            subprocess.run(["aider", "--model", aider_model], env=env, check=True)
        except FileNotFoundError:
            install_guide = (
                "Aider is not installed. Please follow these steps to install it:\n\n"
                "1. Install pipx (if not already installed):\n"
                "   - macOS: brew install pipx && pipx ensurepath\n"
                "   - Linux: pip install pipx && pipx ensurepath\n\n"
                "2. Install Aider using the Cloudmesh AI tool:\n"
                "   cmc llm install aider\n\n"
                "3. (Optional) Install pandoc for better document conversion:\n"
                "   - macOS: brew install pandoc\n"
                "   - Linux: sudo apt-get install pandoc"
            )
            console.error(f"'aider' command not found.\n\n{install_guide}")
        except subprocess.CalledProcessError as e:
            console.error(f"Aider exited with error: {e}")

    def _get_aider_model(self, model):
        """Ensure the model has the 'openai/' prefix for litellm/aider to recognize the provider when using vLLM."""
        # When using vLLM (OpenAI compatible), litellm requires the 'openai/' prefix 
        # even if the model name itself contains 'google/'.
        if model.startswith("openai/"):
            return model
        return f"openai/{model}"

    def launch_docker(self, client_config=None, force=False):
        """Launch Aider inside a Docker container to avoid Python version issues."""
        # Use resolved config from YamlDB - check both client and llm paths for compatibility
        aider_config = self.db.get("cloudmesh.ai.client.aider") or self.db.get("cloudmesh.ai.llm.aider", {})
        
        # Merge with client_config if provided
        config = {**aider_config, **(client_config or {})}
        
        # Support both uppercase and lowercase keys
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")
        model = config.get("model", "google/gemma-4-31B-it")
        base_url = config.get("OPENAI_API_BASE") or config.get("openai_api_base") or config.get("base_url", "http://host.docker.internal:8001/v1")
        
        # In Docker, localhost refers to the container. Replace with host.docker.internal to reach the host.
        if base_url:
            base_url = base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            
        container_name = "cloudmesh-aider"
        
        if not api_key:
            console.error("openai_api_key not found in resolved configuration.")
            return

        # Debug: show a hint of the key being used to help troubleshoot 401 errors
        key_hint = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        console.print(f"[dim]Using API Key: {key_hint}[/dim]")

        # If force is True, remove the existing container to rebuild from scratch
        if force:
            console.print(banner("Force Rebuilding Aider Container", f"Container: {container_name}"))
            self.docker.stop_container(container_name)
            container_exists = False
        else:
            # Check if container already exists
            container_exists = False
            try:
                result = subprocess.run(["docker", "inspect", container_name], capture_output=True, text=True)
                if result.returncode == 0:
                    container_exists = True
            except FileNotFoundError:
                console.error("Docker not found. Please install Docker.")
                return

        aider_model = self._get_aider_model(model)

        if container_exists:
            console.print(banner("Restarting Aider Container", f"Container: {container_name}"))
            # Use 'start -ai' to attach to the existing container interactively
            cmd = f"docker start -ai {container_name}"
        else:
            console.print(banner("Creating Aider Container", f"Model: {aider_model}\nMounting current directory..."))
            # Create container without --rm, give it a name, and install dependencies once
            cmd = textwrap.dedent(f"""\
                docker run -it \\
                  --name {container_name} \\
                  -v "$(pwd):/app" \\
                  -w /app \\
                  --add-host=host.docker.internal:host-gateway \\
                  -e OPENAI_API_KEY="{api_key}" \\
                  -e OPENAI_API_BASE="{base_url}" \\
                  python:3.12-slim \\
                  /bin/bash -c "apt-get update && apt-get install -y pandoc && pip install --quiet --no-cache-dir aider-chat && aider --model {aider_model}"
            """).strip()

        console.print(f"[blue]Executing Docker command:[/blue]\n{cmd}")
        
        try:
            subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            console.error(f"Aider Docker container exited with error: {e}")