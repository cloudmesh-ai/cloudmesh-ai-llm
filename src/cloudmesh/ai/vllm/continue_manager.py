import os
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.util import backup_name
from cloudmesh.ai.vllm.config import VLLMConfig

class ContinueManager:
    def __init__(self):
        self.config_path = Path.home() / ".continue" / "config.yaml"

    def verify_installation(self):
        """Verify that the Continue extension is installed in VS Code."""
        extensions_dir = Path(os.path.expanduser("~/.vscode/extensions"))
        if not extensions_dir.exists():
            return False
        # Check for the Continue extension directory
        for ext in extensions_dir.iterdir():
            if "continue" in ext.name.lower():
                return True
        return False

    def get_config(self) -> Dict[str, Any]:
        """Read the Continue config.yaml file."""
        if not self.config_path.exists():
            console.warning(f"Continue config not found at {self.config_path}")
            return {}
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            console.error(f"Error reading Continue config: {e}")
            return {}

    def probe(self) -> Dict[str, Any]:
        """Probe the current GUI settings."""
        return self.get_config()

    def set_config(self, config: Dict[str, Any]):
        """Write the Continue config.yaml file with a backup."""
        # Create incremental backup
        backup_path = backup_name(self.config_path)
        try:
            with open(self.config_path, "r") as f:
                original_content = f.read()
            with open(backup_path, "w") as f:
                f.write(original_content)
            console.print(f"Backup created: {backup_path}")
        except Exception as e:
            console.warning(f"Could not create backup: {e}")

        try:
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            console.error(f"Error writing Continue config: {e}")
            raise e

    def propose_model_update(self, profile: str, plan: Optional[str] = None, act: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
        """Propose updates to the models list based on a profile."""
        config = self.get_config()
        if not config:
            raise ValueError("Continue config is empty or missing.")

        # Resolve profile from llm.yaml
        profile_data = VLLMConfig().get_server(profile)
        if not profile_data:
            raise ValueError(f"Profile '{profile}' not found in llm.yaml")

        model_id = profile_data.get("model")
        api_base = profile_data.get("apiBase")

        if not model_id or not api_base:
            raise ValueError(f"Profile '{profile}' is missing model or apiBase")

        # Use overrides if provided. 
        # In Continue, we primarily update the main chat/edit model.
        # We'll use 'act' as the primary driver, falling back to 'plan' or profile model.
        target_model = act or plan or model_id

        models = config.get("models", [])
        changed = False
        
        # Find the primary model (the one with 'chat' role)
        primary_model_idx = -1
        for i, m in enumerate(models):
            if "chat" in m.get("roles", []):
                primary_model_idx = i
                break

        if primary_model_idx != -1:
            # Update existing primary model
            if models[primary_model_idx].get("model") != target_model or \
               models[primary_model_idx].get("apiBase") != api_base:
                models[primary_model_idx]["model"] = target_model
                models[primary_model_idx]["apiBase"] = api_base
                changed = True
        else:
            # No primary model found, add a new one
            new_model = {
                "name": f"Cloudmesh - {profile}",
                "provider": "vllm",
                "model": target_model,
                "apiBase": api_base,
                "roles": ["chat", "edit", "apply", "summarize"]
            }
            models.append(new_model)
            changed = True

        config["models"] = models
        return config, changed