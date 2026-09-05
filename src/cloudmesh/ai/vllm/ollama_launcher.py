# -------------------------------------------------------------------------
# File: cloudmesh/ai/vllm/ollama_launcher.py
# -------------------------------------------------------------------------
import webbrowser
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.tunnel import tunnel_manager

class OllamaLauncher:
    """
    Very small helper that ensures a tunnel is up (if the server is remote)
    and then opens the default Ollama web UI in the local browser.
    """

    def __init__(self):
        self.cfg = VLLMConfig()
        self.webui_port = self.cfg.get("cloudmesh.ai.client.ollama.OPENAI_API_BASE")
        # ``OPENAI_API_BASE`` normally looks like http://localhost:11434/v1.
        # Pull the host/port out of it.
        if self.webui_port:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.webui_port)
            self.host = parsed.hostname or "localhost"
            self.port = parsed.port or 11434
        else:
            self.host = "localhost"
            self.port = 11434

    def launch(self):
        # If the server lives on a remote host we need an SSH tunnel.
        default_server = self.cfg.get("cloudmesh.ai.default.server")
        if default_server:
            # Resolve the host of the default server
            s_cfg = self.cfg.get_server(default_server)
            remote_host = s_cfg.get("host")
            if remote_host and remote_host not in ("localhost", "127.0.0.1"):
                console.print(f"[blue]Creating tunnel to {remote_host}:{self.port} …[/blue]")
                tunnel_manager.start_tunnel(remote_host, self.port)

        url = f"http://localhost:{self.port}"
        console.print(f"[green]Opening Ollama UI at {url} …[/green]")
        webbrowser.open(url)