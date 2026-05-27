"""Launch command - Interactive model launcher (replaces llm2.py)."""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

from llmctl.config import load_models_config
from llmctl.utils import (
    ConfigurationError,
    LaunchError,
    check_ssh_connectivity,
    get_hf_token,
    logger,
    set_terminal_title,
)


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


def run_launch() -> int:
    """Run the launch command.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Load configuration with error handling
    try:
        menu = load_models_config()
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





def handle(args: argparse.Namespace) -> int:
    """Handle launch subcommand.
    
    Args:
        args: The parsed arguments.
        
    Returns:
        Exit code.
    """
    return run_launch()


def main() -> int:
    """Legacy entry point for standalone usage."""
    return run_launch()
