"""Shared utilities for llmctl commands."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from standard config path
ENV_PATH = Path.home() / ".config" / "cloudmesh" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration file or values."""


class LaunchError(Exception):
    """Raised when model launch fails."""


class TokenError(Exception):
    """Raised when token file cannot be read."""


class SSHError(Exception):
    """Raised when SSH operation fails."""


class DockerError(Exception):
    """Raised when Docker operation fails."""


def read_secret_file(file_path: str | Path, warn_permissions: bool = True) -> str:
    """Read a secret from a file with proper error handling.
    
    Args:
        file_path: Path to the secret file.
        warn_permissions: Whether to warn if file is world-readable.
        
    Returns:
        The secret content, stripped of whitespace.
        
    Raises:
        TokenError: If file cannot be read.
    """
    path = Path(file_path).expanduser()
    
    if not path.exists():
        raise TokenError(f"Secret file not found: {file_path}")
    
    if not path.is_file():
        raise TokenError(f"Secret path is not a file: {file_path}")
    
    try:
        # Check file permissions (should not be world-readable)
        if warn_permissions:
            stat = path.stat()
            mode = stat.st_mode
            if mode & 0o044:  # Check if readable by group or others
                logger.warning(
                    f"Secret file {file_path} is readable by others. "
                    f"Consider running: chmod 600 {file_path}"
                )
        
        return path.read_text().strip()
    except PermissionError as e:
        raise TokenError(f"Permission denied reading {file_path}: {e}") from e
    except IOError as e:
        raise TokenError(f"Failed to read {file_path}: {e}") from e


def get_hf_token() -> str:
    """Get HuggingFace token from environment.
    
    Returns:
        The HF token string, or empty string if not set.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not found in environment or .env file")
    return token or ""


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


def run_ssh_command(
    host: str, 
    command: str, 
    tty: bool = False,
    capture: bool = False,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """Run a command on a remote host via SSH.
    
    Args:
        host: The remote hostname.
        command: The command to run.
        tty: Whether to allocate a TTY (-t flag).
        capture: Whether to capture stdout/stderr.
        timeout: Timeout in seconds.
        
    Returns:
        CompletedProcess instance.
        
    Raises:
        SSHError: If SSH command fails.
    """
    ssh_cmd = ["ssh"]
    if tty:
        ssh_cmd.append("-t")
    ssh_cmd.extend([host, command])
    
    try:
        return subprocess.run(
            ssh_cmd,
            capture_output=capture,
            text=capture,
            timeout=timeout,
            check=False,
        )
    except subprocess.SubprocessError as e:
        raise SSHError(f"SSH command failed: {e}") from e


def run_docker_command(
    args: list[str],
    capture: bool = False,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """Run a docker command.
    
    Args:
        args: Docker command arguments.
        capture: Whether to capture stdout/stderr.
        timeout: Timeout in seconds.
        
    Returns:
        CompletedProcess instance.
        
    Raises:
        DockerError: If Docker command fails.
    """
    cmd = ["docker"] + args
    
    try:
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=capture,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        raise DockerError("Docker command not found. Is Docker installed?")
    except subprocess.SubprocessError as e:
        raise DockerError(f"Docker command failed: {e}") from e


def get_master_key() -> str:
    """Get LiteLLM master key from environment.
    
    Returns:
        The master key string.
        
    Raises:
        TokenError: If key is not set.
    """
    key = os.getenv("LITELLM_MASTER_KEY")
    if not key:
        raise TokenError("LITELLM_MASTER_KEY not found in environment or .env file")
    return key


def get_uva_kimi_key() -> str:
    """Get UVA Kimi key from environment.
    
    Returns:
        The API key string.
        
    Raises:
        TokenError: If key is not set.
    """
    key = os.getenv("UVA_KIMI_KEY")
    if not key:
        raise TokenError("UVA_KIMI_KEY not found in environment or .env file")
    return key


def set_terminal_title(title: str) -> None:
    """Set the terminal window title.
    
    Args:
        title: The title to display in the terminal window.
    """
    import sys
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()