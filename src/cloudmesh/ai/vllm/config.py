import os

class VLLMConfig:
    """Helper class to manage vLLM service configurations."""
    def __init__(self, db, group, name):
        self.db = db
        self.group = group
        self.name = name
        # The combined YAML structure is: cloudmesh -> ai -> server -> name
        self._data = self.db.get(f"cloudmesh.ai.server.{name}", {})

    def get(self, key, default=None):
        """Retrieve a configuration value with an optional default."""
        return self._data.get(key, default)

    @property
    def data(self):
        """Return the raw configuration dictionary."""
        return self._data

    def __repr__(self):
        return f"VLLMConfig(group={self.group}, name={self.name})"

    @staticmethod
    def reset():
        """Initialize the combined vLLM config file with defaults."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            source_path = os.path.join(current_dir, "vllm_servers.yaml")
            if not os.path.exists(source_path):
                source_path = os.path.join(current_dir, "vllm_servers_example.yaml")
                
            dest_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
            
            if os.path.exists(source_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Initialize with the new unified structure
                combined_content = (
                    "cloudmesh:\n"
                    "  ai:\n"
                    "    default:\n"
                    "      server: gemma-4-31b\n"
                    "      client: default-client\n"
                    "    server:\n"
                    "      gemma-4-31b:\n"
                    "        host: dgx-node-1\n"
                    "        model: google/gemma-4-31B-it\n"
                    "        image: vllm-gemma4:latest\n"
                    "        port: 8000\n"
                    "        tensor_parallel_size: 4\n"
                    "        gpu_memory_utilization: 0.90\n"
                    "    client: {}\n"
                )
                
                with open(dest_path, "w") as dst:
                    dst.write(combined_content)
                return True
            return False
        except Exception:
            return False
