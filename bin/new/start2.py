#!/usr/bin/env python3
"""LiteLLM Proxy Manager - Manages LiteLLM proxy container and probes endpoints."""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from llmctl.utils import get_master_key, get_hf_token, TokenError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration file is missing or malformed."""

    pass


class ProbeError(Exception):
    """Raised when endpoint probing fails."""

    pass


class StartError(Exception):
    """Raised when LiteLLM container fails to start."""

    pass


def probe_endpoint(
    api_base: str, 
    timeout: int = 5, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> bool:
    """Check if a model endpoint is reachable by querying /v1/models.
    
    Args:
        api_base: The base API URL to probe.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        
    Returns:
        True if endpoint is reachable, False otherwise.
    """
    url = f"{api_base}/models"
    headers = {"Authorization": "Bearer sk-dummy"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
            )
            return response.status_code == 200
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout probing {api_base} (attempt {attempt + 1}/{max_retries})")
            
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error probing {api_base}: {e}")
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error probing {api_base}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return False


def validate_config(config: dict[str, Any], config_path: str) -> None:
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


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the LiteLLM configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        The loaded configuration dictionary.
        
    Raises:
        ConfigurationError: If file cannot be read or is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML in {config_path}: {e}") from e
    except IOError as e:
        raise ConfigurationError(f"Failed to read {config_path}: {e}") from e
    
    validate_config(config, config_path)
    return config


def display_probe_table(config_path: str) -> None:
    """Display table with probe results as first column.
    
    Args:
        config_path: Path to the LiteLLM configuration file.
        
    Raises:
        ConfigurationError: If config cannot be loaded.
        ProbeError: If probing fails in an unexpected way.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.error("tabulate module not found. Install with: pip install tabulate")
        raise ConfigurationError("tabulate is required for table display")

    try:
        config = load_config(config_path)
    except ConfigurationError:
        raise
    
    table_data: list[list[str]] = []
    model_list = config.get("model_list", [])
    
    logger.info(f"Probing {len(model_list)} endpoints...")
    
    for model in model_list:
        litellm_params = model.get("litellm_params", {})
        api_base = litellm_params.get("api_base", "unknown")
        model_name = model.get("model_name", "unknown")
        
        # Probe the endpoint
        is_reachable = probe_endpoint(api_base)
        status = "✅" if is_reachable else "❌"
        
        # Extract host:port from api_base for display
        # e.g., "http://white:18000/v1" -> "white:18000"
        host_port = api_base.replace("http://", "").replace("/v1", "")
        
        table_data.append([
            status,
            model_name,
            host_port,
            api_base,
            litellm_params.get("model", "unknown")
        ])
    
    headers = ["Probe", "Model Name", "Host:Port", "API Base", "Model ID"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary statistics
    reachable = sum(1 for row in table_data if row[0] == "✅")
    total = len(model_list)
    print(f"\nSummary: {reachable}/{total} endpoints reachable")
    
    if reachable < total:
        logger.warning(f"{total - reachable} endpoint(s) are not reachable")




def start_litellm(
    config_path: str, 
    openai_key_file: Optional[str] = None
) -> None:
    """Start the LiteLLM proxy container.
    
    Args:
        config_path: Path to the LiteLLM configuration file.
        openai_key_file: Optional path to OpenAI API key file.
        
    Raises:
        StartError: If container fails to start.
        ConfigurationError: If required files are missing.
    """
    # Validate config file before starting
    try:
        load_config(config_path)
    except ConfigurationError:
        raise
    
    # Get master key from environment/utils
    try:
        master_key = get_master_key()
    except TokenError as e:
        raise ConfigurationError(f"Master key error: {e}")
    
    # Read OpenAI key (optional) - keeping this as file based for now if specified, 
    # but could be moved to .env as well.
    openai_api_key = ""
    if openai_key_file:
        try:
            # We can still use a helper for specific files if needed, 
            # but for consistency we check env first
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key and Path(openai_key_file).expanduser().exists():
                openai_api_key = Path(openai_key_file).expanduser().read_text().strip()
        except Exception as e:
            logger.warning(f"Could not read OpenAI key: {e}")
    
    # Cleanup existing container
    logger.info("Cleaning up existing 'litellm' container...")
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", "litellm"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.debug("Removed existing litellm container")
    except FileNotFoundError:
        raise StartError("Docker command not found. Is Docker installed?")
    except subprocess.SubprocessError as e:
        logger.warning(f"Failed to cleanup existing container: {e}")
    
    # Build and execute docker command
    logger.info("Starting LiteLLM with Master Key from environment/config...")
    
    current_dir = os.getcwd()
    
    cmd = [
        "docker", "run", "-d",
        "--name", "litellm",
        "-p", "4000:4000",
        "--add-host", "white:host-gateway",
        "--add-host", "spark:host-gateway",
        "-v", f"{current_dir}/{config_path}:/app/config.yaml",
        "-e", f"OPENAI_API_KEY={openai_api_key}",
        "-e", f"LITELLM_MASTER_KEY={master_key}",
        "ghcr.io/berriai/litellm:main-latest",
        "--config", "/app/config.yaml", 
        "--port", "4000", 
        "--debug"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Success! LiteLLM is running on port 4000.")
        logger.info("Check logs with: docker logs -f litellm")
        
        # Verify container is actually running
        time.sleep(1)
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=litellm", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if "litellm" not in result.stdout:
            raise StartError("Container started but is not running. Check logs.")
            
    except subprocess.CalledProcessError as e:
        raise StartError(f"Failed to start container: {e}") from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LiteLLM Proxy Manager - Manage LiteLLM proxy container"
    )
    parser.add_argument(
        "--probe", 
        action="store_true", 
        help="Probe all model endpoints and display status table"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    openai_key_file = os.path.expanduser("~/gemma/openai_api_key.txt")
    
    try:
        if args.probe:
            display_probe_table(args.config)
        else:
            start_litellm(args.config, openai_key_file)
            return 0
            
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except StartError as e:
        logger.error(f"Start failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())