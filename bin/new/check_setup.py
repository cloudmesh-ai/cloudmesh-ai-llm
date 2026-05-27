#!/usr/bin/env python3
"""Setup Validation Script for Cloudmesh AI LLM.

Checks all prerequisites and configuration to ensure the environment
is properly configured for running LLM models.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{BOLD}{text}{RESET}")
    print("-" * 60)


def print_result(status: str, message: str, details: str = "") -> None:
    """Print a check result with status indicator."""
    print(f"  {status} {message}")
    if details:
        print(f"      {details}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.8+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check required Python packages are installed."""
    required = [
        ("pyyaml", "yaml"),
        ("tabulate", "tabulate"),
        ("requests", "requests"),
        ("flask", "flask"),
        ("urllib3", "urllib3"),
    ]
    results = []
    
    for package_name, import_name in required:
        try:
            __import__(import_name)
            results.append((package_name, True, "installed"))
        except ImportError:
            results.append((package_name, False, "not installed"))
    
    return results


def check_docker() -> Tuple[bool, str]:
    """Check Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "Docker is running"
        else:
            return False, "Docker daemon not responding"
    except FileNotFoundError:
        return False, "Docker not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"


def check_ssh_connection(host: str) -> Tuple[bool, str]:
    """Check SSH connection to a host."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Try to get Docker version on remote host
            docker_result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", host, "docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if docker_result.returncode == 0:
                return True, f"SSH OK, Docker {docker_result.stdout.strip()}"
            return True, "SSH OK, Docker status unknown"
        return False, f"SSH failed: {result.stderr.strip()[:50]}"
    except FileNotFoundError:
        return False, "SSH not installed"
    except subprocess.TimeoutExpired:
        return False, "SSH connection timed out"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def check_token_file(filename: str) -> Tuple[bool, str]:
    """Check a token file exists and has proper permissions."""
    path = Path.home() / "gemma" / filename
    
    if not path.exists():
        return False, f"File not found: {path}"
    
    # Check permissions
    stat = path.stat()
    perms = stat.st_mode & 0o777
    if perms & 0o044:  # Readable by group or others
        return False, f"File is world-readable (perms: {oct(perms)})"
    
    # Check content
    content = path.read_text().strip()
    if not content:
        return False, "File is empty"
    if content in ["your_huggingface_token", "your_secure_random_key", "your_uva_kimi_key"]:
        return False, "Contains placeholder text"
    
    return True, f"OK (perms: {oct(perms)})"


def check_config_files() -> List[Tuple[str, bool, str]]:
    """Check configuration files are valid YAML."""
    files = ["config.yaml", "models.yaml"]
    results = []
    
    for filename in files:
        path = Path(filename)
        if not path.exists():
            results.append((filename, False, "File not found"))
            continue
            
        try:
            with open(path, "r") as f:
                yaml.safe_load(f)
            results.append((filename, True, "Valid YAML"))
        except yaml.YAMLError as e:
            results.append((filename, False, f"Invalid YAML: {e}"))
    
    return results


def check_uva_tunnel() -> Tuple[bool, str]:
    """Check if UVA SSH tunnel is active."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 8080))
        sock.close()
        if result == 0:
            return True, "Port 8080 is active"
        return False, "Port 8080 not responding (tunnel may not be running)"
    except Exception as e:
        return False, f"Error checking port: {e}"


def check_hosts_file() -> Tuple[bool, str]:
    """Check if white and spark are resolvable."""
    import socket
    hosts = ["white", "spark"]
    results = []
    
    for host in hosts:
        try:
            socket.gethostbyname(host)
            results.append(f"{host}: resolvable")
        except socket.gaierror:
            results.append(f"{host}: NOT resolvable")
    
    all_ok = all("resolvable" in r for r in results)
    return all_ok, ", ".join(results)


def main() -> int:
    """Run all checks and display results."""
    print(f"{BOLD}Cloudmesh AI LLM - Setup Validation{RESET}")
    print("=" * 60)
    
    all_passed = True
    
    # Python Version
    print_header("Python Environment")
    passed, info = check_python_version()
    print_result(CHECK if passed else CROSS, "Python Version", info)
    all_passed &= passed
    
    # Dependencies
    print_header("Python Dependencies")
    for pkg, installed, info in check_dependencies():
        status = CHECK if installed else CROSS
        print_result(status, pkg, info)
        all_passed &= installed
    
    # Docker
    print_header("Docker")
    passed, info = check_docker()
    print_result(CHECK if passed else CROSS, "Docker Status", info)
    all_passed &= passed
    
    # Host Resolution
    print_header("Host Resolution")
    passed, info = check_hosts_file()
    print_result(CHECK if passed else WARN, "Hostnames (white, spark)", info)
    # Don't fail on host resolution - might use SSH config
    
    # SSH Connections
    print_header("SSH Connectivity")
    for host in ["white", "spark"]:
        passed, info = check_ssh_connection(host)
        print_result(CHECK if passed else CROSS, f"SSH to {host}", info)
        all_passed &= passed
    
    # Credential Files
    print_header("Credential Files")
    
    # Check gemma directory exists
    gemma_dir = Path.home() / "gemma"
    if not gemma_dir.exists():
        print_result(CROSS, "~/gemma directory", "Directory does not exist")
        all_passed = False
    else:
        print_result(CHECK, "~/gemma directory", "Exists")
        
        for filename in ["HF_token.txt", "server_master_key.txt", "uva-kimmi-key.txt"]:
            passed, info = check_token_file(filename)
            status = CHECK if passed else CROSS
            print_result(status, f"  {filename}", info)
            all_passed &= passed
    
    # Config Files
    print_header("Configuration Files")
    for filename, valid, info in check_config_files():
        status = CHECK if valid else CROSS
        print_result(status, filename, info)
        all_passed &= valid
    
    # UVA Tunnel (optional)
    print_header("UVA SSH Tunnel (Optional)")
    passed, info = check_uva_tunnel()
    status = CHECK if passed else WARN
    print_result(status, "Tunnel Status", info)
    # Don't fail if tunnel is down - it's optional
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All required checks passed!{RESET}")
        print("\nYou can now run:")
        print("  make start    # Start the LiteLLM proxy")
        print("  python llm2.py  # Launch a model directly")
        return 0
    else:
        print(f"{RED}{BOLD}✗ Some checks failed. Please fix the issues above.{RESET}")
        print(f"\n{YELLOW}Installation:{RESET}")
        print("  pip install -r requirements.txt")
        print(f"\n{YELLOW}Setup credentials:{RESET}")
        print("  mkdir -p ~/gemma && chmod 700 ~/gemma")
        print("  # Add your tokens to ~/gemma/*.txt files")
        return 1


if __name__ == "__main__":
    sys.exit(main())