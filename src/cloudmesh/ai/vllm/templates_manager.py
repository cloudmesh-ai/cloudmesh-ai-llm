import os
import yaml
from cloudmesh.ai.common import DotDict
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.config import VLLMConfig

class TemplatesManager:
    """Handles listing and applying vLLM configuration templates."""

    def __init__(self):
        # Templates are stored in the 'templates' subdirectory of the vllm package
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.config_path = VLLMConfig.DEFAULT_USER_CONFIG_PATH

    def list_templates(self):
        """List all available YAML templates in the templates directory."""
        if not os.path.exists(self.templates_dir):
            console.error(f"Templates directory not found at {self.templates_dir}")
            return []
        
        templates = [f[:-5] for f in os.listdir(self.templates_dir) if f.endswith(".yaml")]
        return sorted(templates)

    def apply_template(self, template_name):
        """Apply a specific template to the user's llm.yaml configuration."""
        template_path = os.path.join(self.templates_dir, f"{template_name}.yaml")
        
        if not os.path.exists(template_path):
            console.error(f"Template '{template_name}' not found.")
            return False

        try:
            # Load template
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f) or {}

            # Load existing config
            existing_data = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    existing_data = yaml.safe_load(f) or {}

            # Deep merge template into existing data
            merged_data = self._deep_merge(existing_data, template_data)

            # Save back to config file
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(merged_data, f, default_flow_style=False)
            
            console.print(f"[bold green]Successfully applied template '{template_name}' to {self.config_path}[/bold green]")
            return True

        except Exception as e:
            console.error(f"Failed to apply template '{template_name}': {e}")
            return False

    def _deep_merge(self, base, update):
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

def get_templates_manager():
    """Return a singleton instance of TemplatesManager."""
    if not hasattr(get_templates_manager, "_instance"):
        get_templates_manager._instance = TemplatesManager()
    return get_templates_manager._instance