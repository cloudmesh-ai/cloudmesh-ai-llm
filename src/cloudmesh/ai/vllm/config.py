import os
import yaml
import sys
import re
from cloudmesh.ai.common import DotDict
from cloudmesh.ai.common.ssh.ssh_config import SSHConfig

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