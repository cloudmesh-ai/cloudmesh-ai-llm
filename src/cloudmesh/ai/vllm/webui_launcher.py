import os
import time
import textwrap
import urllib.request
import yaml
from cloudmesh.ai.common import banner, DotDict
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.docker_manager import DockerManager
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.vllm.client import VLLMClient

class WebUILauncher:
    """Handles the lifecycle of the Open WebUI Docker container."""

    def __init__(self):
        self.docker = DockerManager()
        # Use VLLMConfig to get the merged configuration (internal defaults + user config)
        self.db = VLLMConfig()
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

    def stop(self):
        """Stop the Open WebUI container."""
        console.print(f"[blue]Stopping Open WebUI container ({self.container_name})...[/blue]")
        self.docker.stop_container(self.container_name)

    def launch(self, client_config=None, port=None):
        """Launch the Open WebUI container."""
        if not self.docker.check_docker():
            return

        # 1. Resolve port and host for connectivity check
        # Priority: provided port -> config local_port -> default local_tunnel_port
        target_port = port
        if not target_port:
            # Try to find the default server's port
            default_server = self.db.get("cloudmesh.ai.default.server")
            servers = self.db.get("cloudmesh.ai.server", {})
            if default_server and isinstance(servers, dict):
                target_port = servers.get(default_server, {}).get("local_port")
            
            if not target_port:
                target_port = self.local_tunnel_port

        # Resolve host for the determined port
        host = None
        servers_by_host = self.db.get("cloudmesh.ai.server", {})
        if isinstance(servers_by_host, dict):
            for host_name, servers in servers_by_host.items():
                if isinstance(servers, dict):
                    for s_name, s_cfg in servers.items():
                        if isinstance(s_cfg, dict) and (s_cfg.get("local_port") == target_port or s_cfg.get("remote_port") == target_port):
                            # Use host from config or the top-level host key
                            host = s_cfg.get("host") or host_name
                            break
                if host: break

        # 2. Ensure Tunnel and Backend are ready
        if host:
            console.print(f"[blue]Verifying tunnel and backend for {host}:{target_port}...[/blue]")
            success, msg = tunnel_manager.start_tunnel(host, target_port)
            if not success and msg != "Tunnel already active":
                console.error(f"Tunnel failed: {msg}")
                return

            # Health check
            client = VLLMClient(self.db, port=target_port)
            if not client.is_alive():
                console.warning(f"Backend at port {target_port} is not responding to health checks. The UI may not work, but proceeding with launch...")
            else:
                console.ok("Tunnel active and backend healthy!")
        else:
            console.warning(f"Could not resolve host for port {target_port}. Skipping tunnel check, but backend may be unreachable.")

        self.stop()

        # Use resolved config from DotDict
        webui_config = self.db.get("cloudmesh.ai.client.openwebui") or self.db.get("cloudmesh.ai.llm.openwebui", {})
        if not isinstance(webui_config, dict):
            webui_config = {}
        config = {**webui_config, **(client_config or {})}
        
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")
        
        # Resolve placeholders like {SERVER_MASTER_KEY} in the API key
        if api_key and "{" in api_key and "}" in api_key:
            from cloudmesh.ai.vllm.orchestrator import get_vllm_api_key
            # Attempt to resolve the key using the orchestrator's logic
            resolved_key = get_vllm_api_key(self.db)
            if resolved_key:
                api_key = resolved_key

        webui_name = config.get("webui_name", "Cloudmesh AI Portal")
        self.webui_port = config.get("PORT") or config.get("port", 3000)
        
        # Handle API Base for Docker
        if target_port:
            base_url = f"http://host.docker.internal:{target_port}/v1"
        else:
            base_url = config.get("OPENAI_API_BASE") or config.get("openai_api_base") or config.get("base_url", f"http://host.docker.internal:{self.local_tunnel_port}/v1")
        if base_url:
            base_url = base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

        # Resolve default model for the UI
        default_model = config.get("model") or config.get("default_model")
        if not default_model:
            # 1. Try to find model by port in server config
            servers_by_host = self.db.get("cloudmesh.ai.server", {})
            if isinstance(servers_by_host, dict):
                for host_name, servers in servers_by_host.items():
                    if isinstance(servers, dict):
                        for s_name, s_cfg in servers.items():
                            if isinstance(s_cfg, dict) and (s_cfg.get("local_port") == port or s_cfg.get("remote_port") == port):
                                default_model = s_cfg.get("model")
                                break
                    if default_model:
                        break
            
            # 2. Fallback: Try to discover model live from the backend
            if not default_model and target_port:
                try:
                    client = VLLMClient(self.db, port=target_port)
                    models = client.get_models()
                    if models and len(models) > 0:
                        # Assume the first model returned is the primary one
                        default_model = models[0].get("id")
                        if default_model:
                            console.print(f"[dim]Discovered model {default_model} from backend at port {target_port}[/dim]")
                except Exception as e:
                    console.print(f"[dim]Live model discovery failed: {e}[/dim]")

        if not api_key:
            console.error("openai_api_key not found in resolved configuration.")
            return

        console.print(banner("Launching Open WebUI", f"Image: {self.image}\nPort: {self.webui_port}"))

        # To avoid secrets appearing in cleartext in 'ps' or 'docker inspect',
        # we use -e KEY (without value). Docker will then pull the value 
        # from the environment of the process that launches it.
        env_vars = {
            "OPENAI_API_BASE_URL": base_url,
            "OPENAI_API_KEY": api_key,
            "VLLM_API_KEY": api_key,
            "HF_TOKEN": api_key,
            "WEBUI_NAME": webui_name,
        }
        if default_model:
            env_vars["DEFAULT_MODELS"] = default_model

        # Create the flags list: -e KEY for each variable
        env_flags = " ".join([f"-e {key}" for key in env_vars.keys()])

        console.print(f"[blue]Launching Open WebUI (securely passing secrets)...[/blue]")
        
        cmd = (
            f"docker run -d -p {self.webui_port}:8080 "
            f"--add-host=host.docker.internal:host-gateway "
            f"-v open-webui:/app/backend/data "
            f"{env_flags} "
            f"--name {self.container_name} "
            f"--restart always "
            f"{self.image}"
        )
        
        console.print("\n[bold yellow]Executing Docker Command (Secrets Hidden):[/bold yellow]")
        console.print(f"[white]{cmd}[/white]\n")
        
        success = self.docker.run_container(cmd, env=env_vars)

        if success:
            # Extract the port from the base_url for the success message
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(base_url)
                actual_port = parsed.port or self.local_tunnel_port
            except Exception:
                actual_port = self.local_tunnel_port

            model_info = f"Model: {default_model}" if default_model else "Model: Not specified"
            success_msg = (
                f"Setup Complete!\n"
                f"1. SSH tunnel and backend verified (localhost:{actual_port} -> server).\n"
                f"2. {model_info}\n"
                f"3. Access the UI at: http://localhost:{self.webui_port}\n"
                f"4. Opening the UI in your default browser in a few seconds..."
            )
            console.print(banner("Success", success_msg))
            
            # Wait for the application to be fully ready before opening the browser
            self._wait_for_webui()
            os.system(f"open http://localhost:{self.webui_port}")