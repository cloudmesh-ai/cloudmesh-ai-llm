import os


class VLLMConfig:
    """Helper class to manage vLLM service configurations."""

    def __init__(self, servers, name):
        self.servers = servers
        self.name = name
        
        # 1. Load the user's instance configuration
        user_data = {}
        
        # If servers is a YamlDB instance, we use its get method
        if hasattr(servers, 'get') and not isinstance(servers, dict):
            # Try direct lookup (handles flat keys and YamlDB dot-notation)
            user_data = servers.get(f"cloudmesh.ai.server.{name}", {})
            
            # If not found and name contains dots, try manual nested traversal
            if not user_data and "." in name:
                try:
                    base_servers = servers.get("cloudmesh.ai.server", {})
                    if isinstance(base_servers, dict):
                        current = base_servers
                        for part in name.split("."):
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        if current and isinstance(current, dict):
                            user_data = current
                except Exception:
                    pass
        elif isinstance(servers, dict):
            # Direct dictionary lookup
            user_data = servers.get(name, {})
            
            # Manual nested traversal for dot-notation (e.g., uva.qwen)
            if not user_data and "." in name:
                current = servers
                for part in name.split("."):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if current and isinstance(current, dict):
                    user_data = current
        
        # DEBUG: Print lookup results to console
        print(f"DEBUG: VLLMConfig lookup for {name} -> user_data: {user_data}")
        
        # Fallback: if 'uva.qwen' still not found, try 'qwen-uva' or 'uva.qwen-uva'
        if not user_data and "." in name:
            parts = name.split(".")
            if len(parts) == 2:
                host, model = parts
                # Try both 'qwen-uva' and 'uva.qwen-uva'
                fallbacks = [f"{model}-{host}", f"{host}.{model}-{host}"]
                for fb in fallbacks:
                    if hasattr(servers, 'get') and not isinstance(servers, dict):
                        # For YamlDB, try both flat and nested paths
                        res = servers.get(f"cloudmesh.ai.server.{fb}", {})
                        if not res and "." in fb:
                            # Manual nested traversal for the fallback
                            base = servers.get("cloudmesh.ai.server", {})
                            if isinstance(base, dict):
                                curr = base
                                for p in fb.split("."):
                                    curr = curr.get(p) if isinstance(curr, dict) else None
                                res = curr if isinstance(curr, dict) else {}
                        user_data = res
                    elif isinstance(servers, dict):
                        user_data = servers.get(fb, {})
                        if not user_data and "." in fb:
                            curr = servers
                            for p in fb.split("."):
                                curr = curr.get(p) if isinstance(curr, dict) else None
                            user_data = curr if isinstance(curr, dict) else {}
                    
                    if user_data:
                        self.name = fb
                        break
        
        # 2. Check for a template reference
        template_name = user_data.get("template")
        if template_name:
            # Load the internal template catalog
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            template_path = os.path.join(root_dir, "config", "llm.yaml")
            
            from yamldb import YamlDB
            template_db = YamlDB(filename=template_path)
            
            # Try direct lookup (e.g., uva.qwen-uva or uva_qwen-uva)
            template_data = template_db.get(f"cloudmesh.ai.server.{template_name}", {})
            
            # Fallback: if template_name is 'uva_qwen-uva', try 'uva.qwen-uva' and 'uva.qwen'
            if not template_data and "_" in template_name:
                # Replace first underscore with dot to try nested lookup (e.g., uva.qwen-uva)
                nested_template_name = template_name.replace("_", ".", 1)
                template_data = template_db.get(f"cloudmesh.ai.server.{nested_template_name}", {})
                
                # Further fallback: remove '-uva' suffix from the model part (e.g., uva.qwen-uva -> uva.qwen)
                if not template_data and "-uva" in nested_template_name:
                    stripped_template_name = nested_template_name.replace("-uva", "")
                    template_data = template_db.get(f"cloudmesh.ai.server.{stripped_template_name}", {})
            
            print(f"DEBUG: VLLMConfig template lookup for {template_name} -> data: {template_data}")
            
            # Merge: Template < User Instance
            if isinstance(template_data, dict):
                merged_data = template_data.copy()
                merged_data.update(user_data)
                self._data = merged_data
            else:
                self._data = user_data
        else:
            self._data = user_data

    def get(self, key, default=None):
        """Retrieve a configuration value with an optional default."""
        return self._data.get(key, default)

    def set(self, key, value):
        """Set a configuration value."""
        self._data[key] = value

    @property
    def data(self):
        """Return the raw configuration dictionary."""
        return self._data

    @property
    def group(self):
        """Return the platform group (e.g., 'uva' or 'dgx') based on the host."""
        host = self.get("host", "")
        if host == "uva":
            return "uva"
        if host == "dgx":
            return "dgx"
        return "default"

    def __repr__(self):
        return f"VLLMConfig(name={self.name})"

    @staticmethod
    def reset():
        """Initialize the combined vLLM config file with defaults."""
        try:
            # Go up 4 levels from src/cloudmesh/ai/vllm/ to reach project root
            root_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            config_dir = os.path.join(root_dir, "config")
            source_path = os.path.join(config_dir, "llm.yaml")
            if not os.path.exists(source_path):
                source_path = os.path.join(config_dir, "llm.yaml")

            dest_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")

            if os.path.exists(source_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Initialize with the new unified structure using templates
                combined_content = (
                    "#load ~/.config/cloudmesh/keys.yaml\n"
                    "cloudmesh:\n"
                    "  ai:\n"
                    "    default:\n"
                    "      server: gemma-uva\n"
                    "      client: openwebui\n"
                    "    server:\n"
                    "      gemma-uva:\n"
                    "        template: uva.gemma-uva\n"
                    "        host: uva\n"
                    "    client:\n"
                    "      openwebui:\n"
                    "        host: localhost\n"
                    "        port: 3000\n"
                    '        openai_api_key: "{SERVER_MASTER_KEY}"\n'
                )

                with open(dest_path, "w") as dst:
                    dst.write(combined_content)
                return True
            return False
        except Exception:
            return False
