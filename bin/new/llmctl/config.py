"""Configuration loading and validation for llmctl."""

from pathlib import Path
from typing import Any, Optional

import yaml

from llmctl.utils import ConfigurationError, logger


def load_models_config(config_path: Path = Path("models.yaml")) -> list[Optional[dict[str, Any]]]:
    """Load model configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        List of model configurations, with None for separators.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ConfigurationError: If the YAML is malformed or validation fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}") from e
    
    if not isinstance(data, dict) or "models" not in data:
        raise ConfigurationError("Configuration must contain a 'models' key")
    
    menu: list[Optional[dict[str, Any]]] = []
    for i, item in enumerate(data.get("models", [])):
        if not isinstance(item, dict):
            raise ConfigurationError(f"Model at index {i}: must be a dictionary")
            
        if item.get("separator"):
            menu.append(None)
        else:
            # Validate before processing
            _validate_model_config(item, i)
            
            menu.append({
                "tested": item.get("tested", False),
                "name": item["name"],
                "host": item.get("host", "white"),
                "port": item.get("port", 18000),
                "model_id": item["model_id"],
                "quant": item.get("quant"),
                "memory": item.get("memory", 0.90),
                "max_len": item.get("max_len", 4096),
                "extra": item.get("extra", ""),
                "desc": item.get("desc", ""),
                "litellm": item.get("litellm", {}),
            })
    return menu


def _validate_model_config(item: dict[str, Any], index: int) -> None:
    """Validate a single model configuration entry.
    
    Args:
        item: The model configuration dictionary.
        index: The index in the models list (for error messages).
        
    Raises:
        ConfigurationError: If required fields are missing or values are invalid.
    """
    if item.get("separator"):
        return
        
    required_fields = ["name", "model_id"]
    for field in required_fields:
        if field not in item:
            raise ConfigurationError(
                f"Model at index {index}: missing required field '{field}'"
            )
    
    # Validate port range
    port = item.get("port", 18000)
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        raise ConfigurationError(
            f"Model '{item.get('name', 'unknown')}': "
            f"invalid port {port} (must be 1024-65535)"
        )
    
    # Validate memory utilization
    memory = item.get("memory", 0.90)
    if not isinstance(memory, (int, float)) or not (0.0 < memory <= 1.0):
        raise ConfigurationError(
            f"Model '{item.get('name', 'unknown')}': "
            f"invalid memory value {memory} (must be 0.0 < memory <= 1.0)"
        )
    
    # Validate max_len is positive
    max_len = item.get("max_len", 4096)
    if not isinstance(max_len, int) or max_len <= 0:
        raise ConfigurationError(
            f"Model '{item.get('name', 'unknown')}': "
            f"invalid max_len {max_len} (must be positive integer)"
        )


def load_litellm_config(config_path: Path = Path("config.yaml")) -> dict[str, Any]:
    """Load LiteLLM proxy configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        The LiteLLM configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ConfigurationError: If the YAML is malformed or validation fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML in {config_path}: {e}") from e
    
    return config


def validate_litellm_config(config: dict[str, Any], config_path: str = "config.yaml") -> None:
    """Validate the LiteLLM configuration structure.
    
    Args:
        config: The loaded configuration dictionary.
        config_path: Path to the config file (for error messages).
        
    Raises:
        ConfigurationError: If configuration is invalid.
    """
    if not isinstance(config, dict):
        raise ConfigurationError(f"Config file {config_path} must contain a YAML dictionary")
    
    if "model_list" not in config:
        raise ConfigurationError(f"Config file {config_path} must contain 'model_list' key")
    
    model_list = config.get("model_list", [])
    if not isinstance(model_list, list):
        raise ConfigurationError("'model_list' must be a list")
    
    for i, model in enumerate(model_list):
        if not isinstance(model, dict):
            raise ConfigurationError(f"Model at index {i} must be a dictionary")
        
        if "model_name" not in model:
            raise ConfigurationError(f"Model at index {i}: missing 'model_name' field")
        
        litellm_params = model.get("litellm_params", {})
        if not isinstance(litellm_params, dict):
            raise ConfigurationError(f"Model {model['model_name']}: 'litellm_params' must be a dictionary")
        
        if "api_base" not in litellm_params:
            logger.warning(f"Model {model['model_name']}: missing 'api_base' in litellm_params")