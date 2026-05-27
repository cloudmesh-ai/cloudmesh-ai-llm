#!/usr/bin/env python3
"""LLM Model Launcher - Manages vLLM containers via SSH."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration file or values."""

    pass


class LaunchError(Exception):
    """Raised when model launch fails."""

    pass


class TokenError(Exception):
    """Raised when token file cannot be read."""

    pass


def set_terminal_title(title: str) -> None:
    """Set the terminal window title.
    
    Args:
        title: The title to display in the terminal window.
    """
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def get_hf_token() -> str:
    """Read HuggingFace token from file with error handling.
    
    Returns:
        The HF token string, or empty string if file not found.
        
    Raises:
        TokenError: If the token file exists but cannot be read.
    """
    token_path = Path.home() / "gemma" / "HF_token.txt"
    try:
        if token_path.exists():
            return token_path.read_text().strip()
        logger.warning(f"Token file not found: {token_path}")
        return ""
    except PermissionError as e:
        raise TokenError(f"Permission denied reading token file: {e}") from e
    except IOError as e:
        raise TokenError(f"IO error reading token file: {e}") from e


def validate_model_config(item: dict[str, Any], index: int) -> None:
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


def load_menu(config_path: Path = Path("models.yaml")) -> list[Optional[dict[str, Any]]]:
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
            validate_model_config(item, i)
            
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
                "desc": item.get("desc", "")
            })
    return menu


def check_ssh_connectivity(host: str, timeout: int = 10) -> bool:
    """Check if SSH connection to host is possible.
    
    Args:
        host: The hostname to connect to.
        timeout: Connection timeout in seconds.
        
    Returns:
        True if SSH connection succeeds, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "echo", "success"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "success" in result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"SSH connection to {host} timed out")
        return False
    except FileNotFoundError:
        logger.error("SSH command not found. Is SSH installed?")
        return False
    except subprocess.SubprocessError as e:
        logger.warning(f"SSH connectivity check failed: {e}")
        return False


def launch_model(config: dict[str, Any]) -> None:
    """Launch a vLLM container on the specified host via SSH.
    
    Args:
        config: The model configuration dictionary.
        
    Raises:
        LaunchError: If the launch command fails.
    """
    port = config.get("port", 18000)
    host = config.get("host", "white")
    model_id = config.get("model_id")
    max_len = config.get("max_len", 4096)
    memory = config.get("memory", 0.90)
    extra = config.get("extra", "")
    quant = config.get("quant")
    name = config.get("name", "Unknown Model")

    set_terminal_title(f"{host.upper()} | {name}")
    
    # Pre-flight check
    logger.info(f"Checking SSH connectivity to {host}...")
    if not check_ssh_connectivity(host):
        raise LaunchError(f"Cannot establish SSH connection to {host}")

    docker_base = textwrap.dedent(f"""
        docker rm -f vllm-server 2>/dev/null || true;
        docker run -it --rm \\
          --name vllm-server \\
          --gpus all \\
          --ipc=host \\
          --ulimit memlock=-1 \\
          --ulimit stack=67108864 \\
          -p {port}:{port} \\
          -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \\
          -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    """).strip()

    image = (
        "nvcr.io/nvidia/vllm:26.03-py3"
        if host == "spark"
        else "vllm/vllm-openai:latest"
    )

    server_cmd = (
        "python3 -m vllm.entrypoints.openai.api_server" if host == "spark" else ""
    )

    cmd_parts = [
        docker_base,
        image,
        server_cmd,
        f"--host 0.0.0.0 --port {port}",
        f'--model "{model_id}"',
        f"--max-model-len {max_len}",
        f"--gpu-memory-utilization {memory}",
        extra,
    ]

    if quant:
        cmd_parts.append(f"--quantization {quant}")

    cmd = " ".join(part for part in cmd_parts if part)

    full_cmd = (
        f"docker stop vllm-server 2>/dev/null || true; "
        f"docker rm vllm-server 2>/dev/null || true; "
        f"fuser -k {port}/tcp 2>/dev/null || true; "
        f"sleep 2; {cmd}"
    )

    logger.info(f"Launching {name} on {host} (Port: {port})...")
    logger.debug(f"Docker command: {cmd[:200]}...")

    token = get_hf_token()
    ssh_cmd = f"ssh -t {host} 'export HF_TOKEN={token}; {full_cmd}'"
    
    try:
        result = subprocess.run(ssh_cmd, shell=True)
        if result.returncode != 0:
            raise LaunchError(f"SSH command failed with return code {result.returncode}")
    except subprocess.SubprocessError as e:
        raise LaunchError(f"Failed to execute launch command: {e}") from e


def show_menu(menu: list[Optional[dict[str, Any]]]) -> None:
    """Display the model selection menu.
    
    Args:
        menu: List of model configurations, with None for separators.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.error("tabulate module not found. Install with: pip install tabulate")
        # Fallback to simple print
        for i, cfg in enumerate(menu):
            if cfg is None:
                print("-" * 40)
            else:
                status = "✅" if cfg["tested"] else "❌"
                print(f"{i}: [{status}] {cfg['name']} ({cfg['host']}:{cfg['port']})")
        return

    table_data = []
    for i, cfg in enumerate(menu):
        if cfg is None:
            table_data.append(["", "", "", "", "", "", "", "", ""])
            continue

        table_data.append(
            [
                i,
                "✅" if cfg["tested"] else "❌",
                cfg["name"],
                cfg["host"].upper(),
                cfg["port"],
                cfg["quant"] or "-",
                cfg["memory"],
                cfg["max_len"],
                cfg["desc"],
            ]
        )

    headers = ["ID", "Tested", "Name", "Host", "Port", "Quant", "Mem", "Ctx", "Desc"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


def get_user_choice(menu: list[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
    """Get and validate user selection from the menu.
    
    Args:
        menu: List of model configurations.
        
    Returns:
        Selected configuration, or None if user quits.
    """
    while True:
        choice = input("\nEnter ID to launch (or 'q' to quit): ").strip()

        if choice.lower() == "q":
            return None

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice)
        if idx < 0 or idx >= len(menu):
            print(f"Invalid ID. Must be between 0 and {len(menu) - 1}.")
            continue
            
        if menu[idx] is None:
            print("Invalid selection (separator).")
            continue
            
        return menu[idx]


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Load configuration with error handling
    try:
        menu = load_menu()
    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.info("Please ensure models.yaml exists in the current directory.")
        return 1
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return 1

    if not menu:
        logger.error("No models defined in configuration.")
        return 1

    try:
        show_menu(menu)
        selected_config = get_user_choice(menu)
        
        if selected_config is None:
            logger.info("Exiting.")
            return 0

        launch_model(selected_config)
        return 0
        
    except LaunchError as e:
        logger.error(f"Launch failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())