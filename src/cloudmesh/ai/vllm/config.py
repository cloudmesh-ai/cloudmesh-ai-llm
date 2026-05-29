import os
import yaml
import sys
import re
from cloudmesh.ai.common import DotDict
from cloudmesh.ai.common.ssh.ssh_config import SSHConfig

# Optional import for python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

"""VLLMConfig provides a centralized way to manage vLLM configurations.

It loads configurations from internal and user-defined YAML files, merges them,
and allows for dynamic expansion of placeholders and external references (e.g., SSH configs).
The class inherits from DotDict to provide attribute-style access to configuration keys.

Example:
    >>> # Initialize global config
    >>> config = VLLMConfig()
    >>> # Access configuration values directly
    >>> print(config.cloudmesh.ai.server.uva.gemma.remote_port)
    >>> # Merge runtime overrides from a YAML string
    >>> overrides = 'cloudmesh:\n  ai:\n    server:\n      uva:\n        gemma:\n          remote_port: 8081'
    >>> config.merge(yaml.safe_load(overrides))
    >>> # Expand external references and placeholders
    >>> expanded = config.expand_external_references()
    >>> print(expanded.cloudmesh.ai.server.uva.gemma.user)
    >>> # Expand a specific variable
    >>> print(config.expand_var("{user}", "myuser"))

Sample YAML Structure (~/.config/cloudmesh/llm.yaml):
    cloudmesh:
      ai:
        server:
          uva:
            gemma:
              remote_port: 8000
              user: "myuser"
              ssh_command: "ssh {user}@{host}"
            llama3:
              remote_port: 8001maybe we just need a property dict in 
              user: "myuser"
"""

class VLLMConfig(DotDict):
    """Simplified helper class to manage vLLM configurations.
    
    Inherits from DotDict to allow direct attribute access to configuration.
    """

    DEFAULT_USER_CONFIG_PATH = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
    _global_cache = None

    def __init__(self, db=None, user_config_path=None):
        """Initializes the VLLMConfig.

        Args:
            db (DotDict, optional): An optional pre-loaded configuration database. 
                Defaults to None, in which case it loads from YAML files.
            user_config_path (str, optional): Path to the user configuration YAML file.
                Defaults to DEFAULT_USER_CONFIG_PATH.
        """
        self.user_config_path = user_config_path or self.DEFAULT_USER_CONFIG_PATH
        self._config = self._get_global_config(db)
        
        # Initialize DotDict with the full merged configuration
        super().__init__(self._config)

    def _get_global_config(self, db):
        """Handles loading and caching of the global configuration.

        Args:
            db (DotDict, optional): A pre-loaded configuration database.

        Returns:
            DotDict: The merged global configuration database.
        """
        if db is not None:
            return db
        
        if VLLMConfig._global_cache is None:
            global_data = self._load_merged_config()
            VLLMConfig._global_cache = DotDict(global_data)
            
        return VLLMConfig._global_cache

    def _load_merged_config(self):
        """Loads internal and user configurations and merges them.

        Returns:
            dict: The merged global configuration.
        """
        # 1. Load the main internal config file from the code
        internal_path = os.path.join(os.path.dirname(__file__), "configuration", "llm.yaml")
        with open(internal_path, "r") as f:
            global_config = yaml.safe_load(f) or {}

        # 2. Load the local config file from the filesystem
        user_path = self.user_config_path
        if os.path.exists(user_path):
            with open(user_path, "r") as f:
                user_data = yaml.safe_load(f) or {}
            
            # 3. Deep merge user data into the global config
            global_config = DotDict(global_config)
            global_config.merge(user_data)
            return global_config.to_dict()
        
        return global_config

    @property
    def yaml(self):
        """Returns the YAML representation of the global configuration.

        Returns:
            str: The YAML string of the global configuration.
        """
        return self._config.yaml

    def _resolve_external_reference(self, ref: str) -> str:
        """Resolves a single external reference in the format 'path:key'.

        Args:
            ref (str): The reference string to resolve (e.g., '~/.ssh/config:uva.User').

        Returns:
            str: The resolved value, or the original reference wrapped in braces if resolution fails.
        """
        if ":" not in ref:
            return f"{{{ref}}}"
        
        path_part, lookup_key = ref.split(":", 1)
        full_path = os.path.expanduser(path_part)
        
        if not os.path.exists(full_path):
            return f"{{{ref}}}"
        
        try:
            with open(full_path, "r") as f:
                lines = f.readlines()
            
            # Use SSHConfig for robust parsing if it's an SSH config or section.attr lookup
            if "." in lookup_key or "ssh" in path_part.lower():
                try:
                    ssh_cfg = SSHConfig(filename=full_path)
                    if ssh_cfg.conf and "." in lookup_key:
                        section, attr = lookup_key.split(".", 1)
                        # Try specific helper methods first
                        if attr.lower() == "user":
                            val = ssh_cfg.username(section)
                        elif attr.lower() == "hostname":
                            val = ssh_cfg.hostname(section)
                        else:
                            val = ssh_cfg.conf.get(section, attr)
                        
                        if val:
                            return val
                except Exception:
                    pass

            # Fallback to simple key-value lookup or manual section parsing
            if "." in lookup_key:
                section, attr = lookup_key.split(".", 1)
                in_section = False
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith(f"host {section.lower()}"):
                        in_section = True
                        continue
                    if in_section:
                        if line.lower().startswith("host "):
                            break
                        if line.lower().startswith(f"{attr.lower()}"):
                            parts = re.split(r"[\s=]+", line, maxsplit=1)
                            if len(parts) > 1:
                                return parts[1].strip()
            else:
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith(lookup_key.lower()):
                        parts = re.split(r"[\s=]+", line, maxsplit=1)
                        if len(parts) > 1:
                            return parts[1].strip()
            
            return f"{{{ref}}}"
        except Exception:
            return f"{{{ref}}}"

    def expand_external_references(self, target=None):
        """Expands placeholders including external file references {~path:key} recursively.

        Args:
            target (dict|DotDict, optional): The configuration section to expand. 
                If None, the current configuration is used.

        Returns:
            DotDict: The configuration with all placeholders and external references resolved.
        """
        # 1. Use standard DotDict expansion for {key}
        # We use self._config as the source of truth for the expansion
        expanded = self._config.expand(target)
        
        if not isinstance(expanded, (dict, DotDict)):
            return expanded

        # 2. Resolve external references {~path:key} recursively
        result = {}
        for k, v in expanded.items():
            if isinstance(v, (dict, DotDict)):
                # Recursively expand nested dictionaries
                result[k] = self.expand_external_references(v)
            elif isinstance(v, str) and "{" in v and "}" in v:
                # Find all {path:key} patterns
                pattern = r"\{([^}]+)\}"
                
                def replace_match(match):
                    return self._resolve_external_reference(match.group(1))
                
                result[k] = re.sub(pattern, replace_match, v)
            else:
                result[k] = v
                
        return DotDict(result)

    @property
    def yaml_data(self):
        """Returns the YAML representation of this server's expanded configuration.

        Returns:
            str: The YAML string of the expanded server configuration.
        """
        return DotDict.yaml.fget(self)

    def resolve_server_identity(self, name):
        """Resolves the core identity (host, user, port) for a specific server.

        Args:
            name (str): The server name (e.g., 'uva.gemma').

        Returns:
            dict: A dictionary containing 'host', 'user', and 'port'.
        """
        server_config = self.get_server(name)
        if not server_config:
            return {"host": None, "user": None, "port": 8000}

        # Resolve user: server-specific -> global config -> system login
        user = server_config.get("user")
        if not user:
            user = self.get("user") or self.get("cloudmesh.ai.user") or os.getlogin()

        # Resolve port: server-specific 'remote_port' or 'port' -> global 'port' -> default 8000
        port = server_config.get("remote_port") or server_config.get("port") or self.get("port") or 8000

        # Resolve host: server-specific -> fallback to the host part of the name if available
        host = server_config.get("host")
        if not host and "." in name:
            host = name.split(".")[0]

        return {
            "host": host,
            "user": user,
            "port": int(port),
        }

    def get_server(self, name):
        """Returns the configuration for a specific server.

        Args:
            name (str): The server name in dot notation (e.g., 'uva.gemma').

        Returns:
            DotDict: The configuration for the specified server, or None if not found.
        """
        # Try direct dot-notation lookup first
        server_config = self.get(f"cloudmesh.ai.server.{name}")
        if server_config:
            return server_config

        # Fallback: Manual traversal to ensure we find the server config
        try:
            current = self
            for part in ["cloudmesh", "ai", "server"] + name.split("."):
                current = current[part]
            return current
        except (KeyError, TypeError):
            return None

    def smart_get(self, key, default=None):
        """Retrieves a value from the configuration using a smart lookup.

        If the key starts with 'cloudmesh', it is treated as a full path.
        If it does not, the method attempts to find the value by trying common 
        prefixes (e.g., 'cloudmesh.ai.', 'cloudmesh.ai.server.') or by searching 
        the configuration structure.

        Args:
            key (str): The configuration key to look up.
            default (Any, optional): The value to return if the key is not found.

        Returns:
            Any: The value found in the configuration, or the default value.

        Example:
            >>> config = VLLMConfig()
            >>> # Full path lookup
            >>> config.smart_get("cloudmesh.ai.server.uva.gemma.port")
            >>> # Smart lookup (automatically finds under cloudmesh.ai.server)
            >>> config.smart_get("uva.gemma.port")
        """
        if key.startswith("cloudmesh"):
            try:
                return self[key]
            except KeyError:
                return default

        # Try common prefixes
        prefixes = ["cloudmesh.ai.server.", "cloudmesh.ai."]
        for prefix in prefixes:
            try:
                return self[prefix + key]
            except KeyError:
                continue

        # Fallback: Use DotDict's recursive search
        return super().smart_get(key, default)

    def resolve_path(self, path_key: str, default_pattern: str = None) -> str:
        """Resolves a path from config, replacing {user} and {port} placeholders.

        Args:
            path_key (str): The key in the configuration containing the path pattern.
            default_pattern (str, optional): A fallback pattern if the key is not found.

        Returns:
            str: The resolved path with placeholders replaced.
        """
        path = self.smart_get(path_key)
        if not path:
            path = default_pattern
        
        if not path:
            return ""

        user = self.get("user", "")
        port = str(self.get("remote_port", 8000))
        
        return path.replace("{user}", user).replace("{port}", port)

    def merge(self, d=None, yaml_file=None):
        """Merges the content from a dictionary or a YAML file into the current configuration.
        
        This method performs a deep merge, meaning that if both the current 
        configuration and the provided data contain the same key and 
        both values are dictionaries, they will be merged recursively rather 
        than the original being overwritten.
        
        Args:
            d (dict, optional): The dictionary containing configuration updates to merge.
            yaml_file (str, optional): Path to a YAML file containing configuration updates to merge.
        
        Example:
            >>> config = VLLMConfig()
            >>> # Merge from a dictionary
            >>> config.merge(d={'cloudmesh': {'ai': {'server': {'uva': {'gemma': {'port': 8080}}}}}})
            >>> # Merge from a YAML file
            >>> config.merge(yaml_file="overrides.yaml")
        """
        # 1. Handle YAML file merge
        if yaml_file:
            try:
                with open(yaml_file, "r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                    self.merge(d=yaml_data)
            except Exception as e:
                print(f"Error loading YAML file {yaml_file}: {e}")

        # 2. Handle dictionary merge
        if d is not None:
            super().merge(d)

    def expand_var(self, var: str, value: any) -> str:
        """Replaces the placeholder in 'var' with 'value'.

        Args:
            var (str): The placeholder (e.g., '{user}').
            value (Any): The value to replace it with.

        Returns:
            str: The value as a string if 'var' is a placeholder, otherwise 'var'.
        """
        if isinstance(var, str) and var.startswith("{") and var.endswith("}"):
            return str(value)
        return var

    @property
    def properties(self):
        """Returns the configuration as a regular Python dictionary.

        Returns:
            dict: The configuration converted from DotDict to a regular dict.
        """
        return self.to_dict()

    def save(self):
        """Persists the current configuration back to the user's YAML file.

        Returns:
            bool: True if the save operation was successful.
        """
        user_path = self.user_config_path
        
        # Save the entire configuration
        with open(user_path, "w") as f:
            yaml.dump(self.properties, f, default_flow_style=False)
        
        return True

    @classmethod
    def load_env_file(cls, env_file: str = None, override: bool = True) -> dict:
        """Load environment variables from a .env file.
        
        Searches multiple locations in order:
        1. The specified env_file path (if provided)
        2. .env in the current directory
        3. ~/.config/cloudmesh/.env
        
        Args:
            env_file (str, optional): Path to the .env file. If None, searches default locations.
            override (bool): Whether to override existing environment variables. Defaults to True.
        
        Returns:
            dict: Dictionary of loaded environment variables.
        
        Example:
            >>> # Load from default .env file
            >>> env_vars = VLLMConfig.load_env_file()
            >>> # Load from specific file
            >>> env_vars = VLLMConfig.load_env_file("/path/to/.env.production")
        """
        if not DOTENV_AVAILABLE:
            raise ImportError(
                "python-dotenv is required for .env file support. "
                "Install it with: pip install python-dotenv"
            )
        
        # Determine which file to load
        if env_file:
            # Explicit path provided - use it or fail
            env_path = env_file
            if not os.path.exists(env_path):
                raise FileNotFoundError(
                    f"Specified env file not found: {env_path}"
                )
        else:
            # Search default locations in order of priority
            search_paths = [
                ".env",  # Current directory
                os.path.expanduser("~/.config/cloudmesh/.env"),  # User config directory
            ]
            
            env_path = None
            for path in search_paths:
                if os.path.exists(path):
                    env_path = path
                    break
            
            if env_path is None:
                searched = ", ".join([f"'{p}'" for p in search_paths])
                raise FileNotFoundError(
                    f"No env file found. Searched: {searched}"
                )
        
        # Load into a temporary dict to not pollute os.environ if not desired
        loaded = load_dotenv(env_path, override=override)
        
        # Return all env vars that were in the file
        with open(env_path, 'r') as f:
            env_vars = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    # Remove quotes if present
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    env_vars[key] = value
            return env_vars

    def merge_env_vars(self, env_vars: dict = None, env_file: str = None):
        """Merge environment variables into the configuration.
        
        This allows environment variables to override or extend the YAML configuration.
        
        Args:
            env_vars (dict, optional): Dictionary of environment variables to merge.
            env_file (str, optional): Path to .env file to load and merge.
        
        Example:
            >>> config = VLLMConfig()
            >>> config.merge_env_vars()  # Loads from default .env
            >>> config.merge_env_vars(env_file=".env.local")
        """
        if env_file:
            try:
                env_vars = self.load_env_file(env_file)
            except FileNotFoundError:
                # Specified file not found - propagate error
                raise
        elif env_vars is None:
            try:
                env_vars = self.load_env_file()
            except FileNotFoundError:
                # No env file found in any default location - that's ok, just return
                return
        
        if not env_vars:
            return
        
        # Define mapping of env var names to config paths
        env_to_config = {
            'CLOUDMESH_AI_API_KEY': 'cloudmesh.ai.api_key',
            'CLOUDMESH_AI_HOST': 'cloudmesh.ai.host',
            'CLOUDMESH_AI_PORT': 'cloudmesh.ai.port',
            'CLOUDMESH_AI_USER': 'cloudmesh.ai.user',
            'VLLM_MODEL': 'cloudmesh.ai.model',
            'VLLM_GPU_MEMORY_UTILIZATION': 'cloudmesh.ai.gpu_memory_utilization',
            'VLLM_MAX_MODEL_LEN': 'cloudmesh.ai.max_model_len',
            'VLLM_TENSOR_PARALLEL_SIZE': 'cloudmesh.ai.tensor_parallel_size',
            'VLLM_DTYPE': 'cloudmesh.ai.dtype',
            'VLLM_API_KEY': 'cloudmesh.ai.api_key',
        }
        
        # Auto-detect env vars starting with CLOUDMESH_ or VLLM_
        for key, value in env_vars.items():
            # Check predefined mappings
            if key in env_to_config:
                self._set_nested_path(env_to_config[key], value)
            # Auto-convert CLOUDMESH_AI_SERVER__UVA__GEMMA__PORT to cloudmesh.ai.server.uva.gemma.port
            elif key.startswith(('CLOUDMESH_', 'VLLM_')):
                config_path = self._env_var_to_config_path(key)
                self._set_nested_path(config_path, value)

    def _env_var_to_config_path(self, env_var: str) -> str:
        """Convert an environment variable name to a config path.
        
        Args:
            env_var (str): Environment variable name (e.g., 'CLOUDMESH_AI_SERVER__UVA__GEMMA__PORT')
        
        Returns:
            str: Config path (e.g., 'cloudmesh.ai.server.uva.gemma.port')
        """
        # Remove prefix
        if env_var.startswith('CLOUDMESH_'):
            base = env_var[len('CLOUDMESH_'):]
        elif env_var.startswith('VLLM_'):
            base = env_var[len('VLLM_'):]
            base = f'AI_{base}'  # Add AI prefix for VLLM vars
        else:
            base = env_var
        
        # Replace double underscores with dots (for nesting)
        # Replace single underscores with dots
        path = base.lower().replace('__', '.').replace('_', '.')
        
        # Ensure it starts with cloudmesh
        if not path.startswith('cloudmesh'):
            path = 'cloudmesh.' + path
        
        return path

    def _set_nested_path(self, path: str, value: any):
        """Set a value at a nested path in the configuration.
        
        Args:
            path (str): Dot-separated path (e.g., 'cloudmesh.ai.server.port')
            value: The value to set
        """
        keys = path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try to convert value to appropriate type
        current[keys[-1]] = self._convert_env_value(value)

    def _convert_env_value(self, value: str):
        """Convert an environment variable string to the appropriate Python type.
        
        Args:
            value (str): The string value from environment variable.
        
        Returns:
            The converted value (int, float, bool, or string).
        """
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Return as string
        return value

    def generate_env_file(self, server_name: str = None) -> str:
        """Generate .env file content from the current configuration.
        
        Args:
            server_name (str, optional): Server name to include in the generated file (e.g., 'uva.gemma')
        
        Returns:
            str: The generated .env file content.
        
        Example:
            >>> config = VLLMConfig()
            >>> env_content = config.generate_env_file('uva.gemma')
            >>> with open('.env', 'w') as f:
            ...     f.write(env_content)
        """
        lines = ["# cloudmesh-ai-llm Environment Configuration", "# Auto-generated from YAML config", ""]
        
        # Core settings
        if server_name:
            server_config = self.get_server(server_name)
            if server_config:
                lines.append(f"# Server: {server_name}")
                lines.extend(self._dict_to_env_vars(server_config, prefix=''))
                lines.append("")
        
        # User-specific paths
        lines.append("# User Configuration")
        lines.append(f"CLOUDMESH_USER_CONFIG_PATH={self.user_config_path}")
        lines.append("")
        
        # SSH settings if available
        ssh_config = self.get('ssh')
        if ssh_config:
            lines.append("# SSH Configuration")
            lines.append(f"SSH_HOST={ssh_config.get('host', '')}")
            lines.append(f"SSH_USER={ssh_config.get('user', '')}")
            lines.append(f"SSH_KEY_PATH={ssh_config.get('key_path', '')}")
            lines.append("")
        
        # Docker settings
        docker_config = self.get('docker')
        if docker_config:
            lines.append("# Docker Configuration")
            if isinstance(docker_config, dict):
                lines.extend(self._dict_to_env_vars(docker_config, prefix='DOCKER'))
            lines.append("")
        
        return '\n'.join(lines)

    def _dict_to_env_vars(self, config: dict, prefix: str = '') -> list:
        """Convert a configuration dictionary to env var lines.
        
        Args:
            config (dict): Configuration dictionary
            prefix (str): Prefix for the env var names
        
        Returns:
            list: List of environment variable assignment strings
        """
        lines = []
        
        for key, value in config.items():
            env_key = f"{prefix}{prefix and '_' or ''}{key.upper()}" if prefix else key.upper()
            
            if isinstance(value, dict):
                lines.extend(self._dict_to_env_vars(value, env_key))
            elif isinstance(value, (list, tuple)):
                # Join lists with commas
                lines.append(f"{env_key}={','.join(str(v) for v in value)}")
            else:
                lines.append(f"{env_key}={value}")
        
        return lines
