"""Proxy command - LiteLLM proxy management (replaces start2.py)."""

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from llmctl.config import load_litellm_config, validate_litellm_config
from llmctl.utils import (
    ConfigurationError,
    DockerError,
    get_master_key,
    logger,
    read_secret_file,
)


def probe_endpoint(
    api_base: str, 
    timeout: int = 5, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> tuple[bool, list[str]]:
    """Check if a model endpoint is reachable by querying /v1/models.
    
    Args:
        api_base: The base API URL to probe.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        
    Returns:
        Tuple of (is_reachable, list_of_model_ids).
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
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "unknown") for m in models]
                return True, model_ids
            return False, []
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout probing {api_base} (attempt {attempt + 1}/{max_retries})")
            
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error probing {api_base}: {e}")
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error probing {api_base}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return False, []


def format_models_found(models: list[str], max_length: int = 40) -> str:
    """Format model list for display, truncating if too long.
    
    Args:
        models: List of model IDs.
        max_length: Maximum string length before truncation.
        
    Returns:
        Formatted string of model names.
    """
    if not models:
        return "N/A"
    
    combined = ", ".join(models)
    if len(combined) > max_length:
        return combined[:max_length - 3] + "..."
    return combined


def display_probe_table(config_path: str, max_workers: int = 10) -> int:
    """Display table with probe results.
    
    Args:
        config_path: Path to the LiteLLM configuration file.
        max_workers: Maximum number of parallel probe workers.
        
    Returns:
        Exit code (0 if all reachable, 1 otherwise).
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.error("tabulate module not found. Install with: pip install tabulate")
        return 1

    try:
        config = load_litellm_config(Path(config_path))
        validate_litellm_config(config, config_path)
    except (FileNotFoundError, ConfigurationError) as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    model_list = config.get("model_list", [])
    logger.info(f"Probing {len(model_list)} endpoints with {max_workers} workers...")
    
    # Prepare probe tasks with index to maintain order
    probe_tasks = []
    for idx, model in enumerate(model_list):
        litellm_params = model.get("litellm_params", {})
        api_base = litellm_params.get("api_base", "unknown")
        model_name = model.get("model_name", "unknown")
        model_id = litellm_params.get("model", "unknown")
        
        probe_tasks.append((idx, api_base, model_name, model_id))
    
    # Run probes in parallel with progress bar
    results: list[tuple[int, bool, list[str], str, str, str]] = []
    
    def probe_task_wrapper(task: tuple) -> tuple:
        idx, api_base, model_name, model_id = task
        is_reachable, models_found = probe_endpoint(api_base)
        return (idx, is_reachable, models_found, api_base, model_name, model_id)
    
    # Try to use tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(probe_task_wrapper, task): task 
            for task in probe_tasks
        }
        
        if use_tqdm:
            # Use tqdm progress bar
            with tqdm(total=len(probe_tasks), desc="Probing endpoints") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        else:
            # Fallback without tqdm
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Build table data
    table_data: list[list[str]] = []
    for idx, is_reachable, models_found, api_base, model_name, model_id in results:
        status = "✅" if is_reachable else "❌"
        
        # Extract host:port from api_base for display
        host_port = api_base.replace("http://", "").replace("/v1", "")
        
        models_display = format_models_found(models_found)
        
        table_data.append([
            status,
            model_name,
            host_port,
            api_base,
            model_id,
            models_display
        ])
    
    headers = ["Probe", "Model Name", "Host:Port", "API Base", "Model ID", "Models Found"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary statistics
    reachable = sum(1 for row in table_data if row[0] == "✅")
    total = len(model_list)
    print(f"\nSummary: {reachable}/{total} endpoints reachable")
    
    if reachable < total:
        logger.warning(f"{total - reachable} endpoint(s) are not reachable")
        return 1
    return 0


def start_proxy(config_path: str = "config.yaml") -> int:
    """Start the LiteLLM proxy container.
    
    Args:
        config_path: Path to the LiteLLM configuration file.
        
    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Validate config file before starting
    try:
        config = load_litellm_config(Path(config_path))
        validate_litellm_config(config, config_path)
    except (FileNotFoundError, ConfigurationError) as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Read master key (required)
    try:
        master_key = get_master_key()
    except Exception as e:
        logger.error(f"Failed to read master key: {e}")
        return 1
    
    # Read OpenAI key (optional)
    openai_api_key = ""
    openai_key_file = Path.home() / "gemma" / "openai_api_key.txt"
    if openai_key_file.exists():
        try:
            openai_api_key = read_secret_file(openai_key_file)
        except Exception as e:
            logger.warning(f"Could not read OpenAI key file: {e}")
    
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
        logger.error("Docker command not found. Is Docker installed?")
        return 1
    except subprocess.SubprocessError as e:
        logger.warning(f"Failed to cleanup existing container: {e}")
    
    # Build and execute docker command
    logger.info(f"Starting LiteLLM proxy...")
    
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
            logger.error("Container started but is not running. Check logs with: docker logs litellm")
            return 1
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start container: {e}")
        return 1
    
    return 0


def stop_proxy() -> int:
    """Stop the LiteLLM proxy container.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        subprocess.run(["docker", "stop", "litellm"], capture_output=True)
        subprocess.run(["docker", "rm", "litellm"], capture_output=True)
        logger.info("LiteLLM proxy stopped")
        return 0
    except FileNotFoundError:
        logger.error("Docker command not found. Is Docker installed?")
        return 1


def show_logs(follow: bool = False) -> int:
    """Show LiteLLM proxy logs.
    
    Args:
        follow: Whether to follow logs in real-time.
        
    Returns:
        Exit code.
    """
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append("litellm")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except FileNotFoundError:
        logger.error("Docker command not found. Is Docker installed?")
        return 1




def add_subparser(parser: Any) -> None:
    """Add the proxy subcommand parser.
    
    Args:
        parser: The ArgumentParser to add subcommands to.
    """
    proxy_subparsers = parser.add_subparsers(dest="proxy_command", help="Proxy commands")
    
    # start subcommand
    start_parser = proxy_subparsers.add_parser("start", help="Start LiteLLM proxy")
    start_parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    # stop subcommand
    proxy_subparsers.add_parser("stop", help="Stop LiteLLM proxy")
    
    # probe subcommand
    probe_parser = proxy_subparsers.add_parser("probe", help="Probe all model endpoints")
    probe_parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    probe_parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum number of parallel probe workers (default: 10)"
    )
    
    # logs subcommand
    logs_parser = proxy_subparsers.add_parser("logs", help="Show proxy logs")
    logs_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Follow log output"
    )


def handle(args: argparse.Namespace) -> int:
    """Handle proxy subcommand.
    
    Args:
        args: The parsed arguments.
        
    Returns:
        Exit code.
    """
    if args.proxy_command == "start" or args.proxy_command is None:
        return start_proxy(getattr(args, "config", "config.yaml"))
    elif args.proxy_command == "stop":
        return stop_proxy()
    elif args.proxy_command == "probe":
        return display_probe_table(
            getattr(args, "config", "config.yaml"),
            getattr(args, "workers", 10)
        )
    elif args.proxy_command == "logs":
        return show_logs(getattr(args, "follow", False))
    else:
        logger.error(f"Unknown proxy command: {args.proxy_command}")
        return 1
