import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.util import backup_name, yn_choice
from cloudmesh.ai.vllm.config import VLLMConfig

class ClineManager:
    """Manager for Cline configuration files (~/.cline/data)."""

    def __init__(self):
        self.data_dir = Path(os.path.expanduser("~/.cline/data"))
        self.global_state_path = self.data_dir / "globalState.json"
        self.secrets_path = self.data_dir / "secrets.json"
        self.config = VLLMConfig()

    def verify_installation(self):
        """Verify that the Cline extension is installed in VS Code."""
        extensions_dir = Path(os.path.expanduser("~/.vscode/extensions"))
        if not extensions_dir.exists():
            return False
        # Check for the Cline extension directory (saoudrizwan.claude-dev or similar)
        for ext in extensions_dir.iterdir():
            if "claude-dev" in ext.name.lower() or "cline" in ext.name.lower():
                return True
        return False

    def _read_json(self, path: Path) -> Dict[str, Any]:
        """Reads a JSON file and returns its content."""
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            console.error(f"Failed to read {path}: {e}")
            return {}

    def _write_json(self, path: Path, data: Dict[str, Any]):
        """Writes data to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_global_state(self) -> Dict[str, Any]:
        """Retrieve current global state."""
        return self._read_json(self.global_state_path)

    def set_global_state(self, data: Dict[str, Any], backup: bool = True):
        """Save global state with optional incremental backup."""
        if backup and self.global_state_path.exists():
            bak = backup_name(self.global_state_path)
            import shutil
            shutil.copy2(self.global_state_path, bak)
            console.msg(f"Created backup: {bak}")
        
        self._write_json(self.global_state_path, data)

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key."""
        secrets = self._read_json(self.secrets_path)
        return secrets.get(key)

    def get_all_secrets(self) -> Dict[str, Any]:
        """Retrieve all secrets."""
        return self._read_json(self.secrets_path)

    def set_secret(self, key: str, value: str):
        """Set a secret value."""
        secrets = self._read_json(self.secrets_path)
        secrets[key] = value
        self._write_json(self.secrets_path, secrets)

    def probe(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Probe the GUI settings. If state is provided, probe that state instead of global state."""
        if state is None:
            state = self.get_global_state()
        
        # Model info is nested in these objects
        act_info = state.get("actModeOpenAiModelInfo", {})
        plan_info = state.get("planModeOpenAiModelInfo", {})
        
        def format_cost(info, key):
            val = info.get(key)
            if val is None or val == 0 or val == "0":
                return "Free"
            return str(val)

        return {
            "API Provider": state.get("apiProvider"),
            "Base URL": state.get("openAiBaseUrl"),
            "Different Models for Plan/Act": state.get("useDifferentModelsForPlanAndAct"),
            "Plan Model": state.get("planModeOpenAiModelId"),
            "Plan Context": plan_info.get("contextWindow"),
            "Plan Input": format_cost(plan_info, "inputPrice"),
            "Plan Output": format_cost(plan_info, "outputPrice"),
            "Act Model": state.get("actModeOpenAiModelId"),
            "Act Context": act_info.get("contextWindow"),
            "Act Input": format_cost(act_info, "inputPrice"),
            "Act Output": format_cost(act_info, "outputPrice"),
        }

    def resolve_profile(self, profile_name: str) -> Tuple[Dict[str, Any], str]:
        """
        Resolves model and base URL from llm.yaml profile.
        Returns (profile_cfg, base_url).
        """
        # Use VLLMConfig to get the merged config for the profile
        profile_cfg = self.config.get_server(profile_name)
        if not profile_cfg:
            raise ValueError(f"Profile '{profile_name}' not found in llm.yaml")

        if not profile_cfg.get("model"):
            raise ValueError(f"Profile '{profile_name}' does not specify a model")

        # Base URL is typically http://localhost:{port}/v1 if tunneled, 
        # but we follow the standard vLLM OpenAI compatible path.
        port = profile_cfg.get("remote_port", "8000")
        base_url = f"http://localhost:{port}/v1"
        
        return profile_cfg, base_url

    def propose_model_update(self, profile_name: str, plan_model: Optional[str] = None, act_model: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Proposes updates to plan and act models based on a profile and optional overrides.
        Returns (current_state, proposed_state, changed).
        """
        profile_cfg, base_url = self.resolve_profile(profile_name)
        profile_model = profile_cfg.get("model")
        
        # Logic: If --plan/--act not specified, use profile model
        target_plan = plan_model if plan_model else profile_model
        target_act = act_model if act_model else profile_model

        current_state = self.get_global_state()
        proposed_state = current_state.copy()
        
        current_plan = current_state.get("planModeOpenAiModelId")
        current_act = current_state.get("actModeOpenAiModelId")
        current_url = current_state.get("openAiBaseUrl")

        # Resolve target context window from profile
        target_context_window = profile_cfg.get("context_window") or profile_cfg.get("max_model_len", 65536)
        
        current_plan_info = current_state.get("planModeOpenAiModelInfo", {})
        current_act_info = current_state.get("actModeOpenAiModelInfo", {})
        current_plan_ctx = current_plan_info.get("contextWindow")
        current_act_ctx = current_act_info.get("contextWindow")

        changed = False
        if current_plan != target_plan:
            proposed_state["planModeOpenAiModelId"] = target_plan
            changed = True
        if current_act != target_act:
            proposed_state["actModeOpenAiModelId"] = target_act
            changed = True
        if current_url != base_url:
            proposed_state["openAiBaseUrl"] = base_url
            changed = True
        if current_plan_ctx != target_context_window or current_act_ctx != target_context_window:
            changed = True

        if changed:
            # Update ModelInfo objects to match the new model's context window and pricing
            model_info = {
                "maxTokens": -1,
                "contextWindow": target_context_window,
                "supportsImages": True,
                "supportsPromptCache": False,
                "inputPrice": 0,
                "outputPrice": 0,
                "temperature": 0,
                "isR1FormatRequired": False
            }
            
            proposed_state["planModeOpenAiModelInfo"] = model_info.copy()
            proposed_state["actModeOpenAiModelInfo"] = model_info.copy()
            
        return current_state, proposed_state, changed
