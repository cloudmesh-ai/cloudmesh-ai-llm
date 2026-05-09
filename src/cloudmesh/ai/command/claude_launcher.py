import os
import subprocess
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console

class ClaudeLauncher:
    """Handles the launch of Claude Code with vLLM backend."""

    def __init__(self):
        # Use YamlDB to load the resolved configuration in memory
        self.db = YamlDB(filename=os.path.expanduser("~/.config/cloudmesh/llm.yaml"), backend=":memory:")

    def _get_claude_model(self, model):
        """Return the model name as is for vLLM backends."""
        return model

    def launch(self, client_config=None):
        """Launch the claude CLI with required environment variables."""
        # Use resolved config from YamlDB - check both client and llm paths for compatibility
        claude_config = self.db.get("cloudmesh.ai.client.claude") or self.db.get("cloudmesh.ai.llm.claude", {})
        
        # Merge with client_config if provided
        config = {**claude_config, **(client_config or {})}
        
        # Support both uppercase and lowercase keys
        api_key = config.get("OPENAI_API_KEY") or config.get("openai_api_key")
        model = config.get("model") or config.get("ANTHROPIC_MODEL", "google/gemma-4-31B-it")
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
        })

        try:
            # Use subprocess.run without capturing output to allow interactive CLI
            subprocess.run(["claude"], env=env, check=True)
        except FileNotFoundError:
            console.error("'claude' command not found. Please install Claude Code.")
        except subprocess.CalledProcessError as e:
            console.error(f"Claude Code exited with error: {e}")