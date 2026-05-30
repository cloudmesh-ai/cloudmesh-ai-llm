import os
import subprocess
from cloudmesh.ai.vllm.config import VLLMConfig
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console

class ClaudeLauncher:
    """Handles the launch of Claude Code with vLLM backend."""

    def __init__(self):
        # Use VLLMConfig to get the merged configuration (internal defaults + user config)
        self.db = VLLMConfig()

    def _get_claude_model(self, model):
        """Return the model name as is for vLLM backends."""
        return model

    def launch(self, client_config=None, port=None):
        """Launch the claude CLI with required environment variables."""
        from cloudmesh.ai.vllm.tunnel import tunnel_manager
        from cloudmesh.ai.vllm.client import VLLMClient

        # 1. Resolve port and host for connectivity check
        target_port = port
        if not target_port:
            default_server = self.db.get("cloudmesh.ai.default.server")
            servers = self.db.get("cloudmesh.ai.server", {})
            if default_server and isinstance(servers, dict):
                target_port = servers.get(default_server, {}).get("local_port")
            
            if not target_port:
                target_port = 8001

        # Resolve host for the determined port
        host = None
        servers_by_host = self.db.get("cloudmesh.ai.server", {})
        if isinstance(servers_by_host, dict):
            for host_name, servers in servers_by_host.items():
                if isinstance(servers, dict):
                    for s_name, s_cfg in servers.items():
                        if isinstance(s_cfg, dict) and (s_cfg.get("local_port") == target_port or s_cfg.get("remote_port") == target_port):
                            host = s_cfg.get("host")
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
                console.error(f"Backend at port {target_port} is not responding. Please check if the vLLM server is running.")
                return
            console.ok("Tunnel active and backend healthy!")
        else:
            console.warning(f"Could not resolve host for port {target_port}. Skipping tunnel check, but backend may be unreachable.")

        # Use resolved config from YamlDB - check both client and llm paths for compatibility
        claude_config = self.db.get("cloudmesh.ai.client.claude") or self.db.get("cloudmesh.ai.llm.claude", {})
        
        # Merge with client_config if provided
        config = {**claude_config, **(client_config or {})}
        
        # Support both uppercase and lowercase keys
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")

        # Resolve placeholders like {SERVER_MASTER_KEY} in the API key
        if api_key and "{" in api_key and "}" in api_key:
            from cloudmesh.ai.vllm.orchestrator import get_vllm_api_key
            resolved_key = get_vllm_api_key(self.db)
            if resolved_key:
                api_key = resolved_key

        model = config.get("model") or config.get("ANTHROPIC_MODEL", "google/gemma-4-31B-it")
        
        if target_port:
            base_url = f"http://127.0.0.1:{target_port}"
        else:
            base_url = config.get("OPENAI_API_BASE") or config.get("openai_api_base") or config.get("base_url", "http://127.0.0.1:8001")
        
        if not api_key:
            console.error("openai_api_key not found in resolved configuration.")
            return

        # Ensure model is used as defined in config for vLLM compatibility
        resolved_model = self._get_claude_model(model)

        # Claude Code often appends /v1 to the base URL. Remove it if present to avoid /v1/v1
        clean_base_url = base_url.rstrip('/')
        if clean_base_url.endswith('/v1'):
            clean_base_url = clean_base_url[:-3]

        console.print(banner("Launching Claude Code", f"Backend: {clean_base_url}\nModel: {resolved_model}"))

        # Prepare environment variables
        env = os.environ.copy()
        env.update({
            "ANTHROPIC_AUTH_TOKEN": config.get("ANTHROPIC_AUTH_TOKEN") or config.get("anthropic_auth_token", api_key),
            "ANTHROPIC_BASE_URL": clean_base_url,
            "ANTHROPIC_MODEL": resolved_model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": config.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") or config.get("anthropic_default_haiku_model", resolved_model),
            "ANTHROPIC_DEFAULT_SONNET_MODEL": config.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or config.get("anthropic_default_sonnet_model", resolved_model),
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": str(config.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", 1)),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": str(config.get("CLAUDE_CODE_ATTRIBUTION_HEADER", 0)),
            # These environment variables are used to satisfy the API requirement for auto tool choice
            "CLAUDE_CODE_ENABLE_AUTO_TOOL_CHOICE": "true",
            "CLAUDE_CODE_TOOL_CALL_PARSER": "openai",
        })

        try:
            # Use subprocess.run without capturing output to allow interactive CLI
            subprocess.run(["claude"], env=env, check=True)
        except FileNotFoundError:
            console.error("'claude' command not found. Please install Claude Code.")
        except subprocess.CalledProcessError as e:
            console.error(f"Claude Code exited with error: {e}")