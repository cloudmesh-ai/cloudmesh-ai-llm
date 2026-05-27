"""Tunnel command - SSH tunnel management for UVA Kimi access."""

import argparse
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from llmctl.utils import logger


def is_tunnel_active(host: str = "localhost", port: int = 8080, timeout: float = 2.0) -> bool:
    """Check if SSH tunnel is active by testing port connectivity.
    
    Args:
        host: The host to check (default: localhost).
        port: The port to check (default: 8080).
        timeout: Connection timeout in seconds.
        
    Returns:
        True if tunnel appears to be active, False otherwise.
    """
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_tunnel(
    local_port: int = 8080,
    remote_host: str = "open-webui.rc.virginia.edu",
    remote_port: int = 443,
    ssh_host: str = "uva",
    background: bool = False,
) -> int:
    """Start SSH tunnel for UVA Kimi access.
    
    Args:
        local_port: Local port to bind.
        remote_host: Remote host to tunnel to.
        remote_port: Remote port to tunnel to.
        ssh_host: SSH config host alias.
        background: Whether to run in background (not supported yet).
        
    Returns:
        Exit code (0 for success, 1 for error).
    """
    if is_tunnel_active(port=local_port):
        logger.info(f"SSH tunnel already active on port {local_port}")
        return 0
    
    logger.info(f"Starting SSH tunnel to {remote_host}:{remote_port} via {ssh_host}...")
    logger.info(f"Local endpoint: localhost:{local_port}")
    logger.info("Press Ctrl+C to stop the tunnel")
    
    # Build SSH command
    cmd = [
        "ssh",
        "-L", f"{local_port}:{remote_host}:{remote_port}",
        "-N",  # Don't execute remote command
        ssh_host
    ]
    
    try:
        if background:
            # Start in background (using subprocess.Popen)
            logger.info("Starting tunnel in background...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait a moment to check if it started successfully
            time.sleep(1)
            if process.poll() is not None:
                logger.error("Tunnel process exited unexpectedly")
                return 1
            logger.info(f"Tunnel started in background (PID: {process.pid})")
            logger.info(f"To stop: kill {process.pid}")
            return 0
        else:
            # Run in foreground
            def signal_handler(signum, frame):
                logger.info("\nReceived interrupt signal, stopping tunnel...")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            result = subprocess.run(cmd)
            return result.returncode
            
    except FileNotFoundError:
        logger.error("SSH command not found. Is SSH installed?")
        return 1
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to start tunnel: {e}")
        return 1


def stop_tunnel(port: int = 8080) -> int:
    """Stop SSH tunnel by finding and killing the process.
    
    Args:
        port: The local port the tunnel is using.
        
    Returns:
        Exit code (0 if stopped or not running, 1 for error).
    """
    logger.info(f"Looking for SSH tunnel on port {port}...")
    
    # Try to find and kill SSH process with the specific port forwarding
    try:
        # Use lsof to find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                pid = pid.strip()
                if pid:
                    logger.info(f"Killing process {pid}...")
                    subprocess.run(["kill", pid], capture_output=True)
            time.sleep(0.5)
            
            # Verify it's stopped
            if not is_tunnel_active(port=port):
                logger.info("SSH tunnel stopped")
                return 0
            else:
                # Force kill if still running
                for pid in pids:
                    pid = pid.strip()
                    if pid:
                        logger.info(f"Force killing process {pid}...")
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                
                if not is_tunnel_active(port=port):
                    logger.info("SSH tunnel stopped")
                    return 0
                else:
                    logger.error("Failed to stop tunnel")
                    return 1
        else:
            logger.info("No active tunnel found")
            return 0
            
    except FileNotFoundError:
        logger.error("lsof command not found")
        return 1
    except subprocess.SubprocessError as e:
        logger.error(f"Error stopping tunnel: {e}")
        return 1


def check_tunnel_status() -> int:
    """Check and display tunnel status.
    
    Returns:
        Exit code (0 if active, 1 if inactive).
    """
    if is_tunnel_active():
        logger.info("✓ SSH tunnel is active (port 8080)")
        logger.info("  You can access UVA Kimi at localhost:8080")
        return 0
    else:
        logger.info("✗ SSH tunnel is not active")
        logger.info("  Run 'llmctl tunnel start' to start the tunnel")
        return 1


def add_subparser(parser: Any) -> None:
    """Add the tunnel subcommand parser.
    
    Args:
        parser: The ArgumentParser to add subcommands to.
    """
    tunnel_subparsers = parser.add_subparsers(dest="tunnel_command", help="Tunnel commands")
    
    # start subcommand
    start_parser = tunnel_subparsers.add_parser("start", help="Start SSH tunnel for UVA Kimi")
    start_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Local port to bind (default: 8080)"
    )
    start_parser.add_argument(
        "--background", "-b",
        action="store_true",
        help="Run in background (not interactive)"
    )
    
    # stop subcommand
    tunnel_subparsers.add_parser("stop", help="Stop SSH tunnel")
    
    # status subcommand
    tunnel_subparsers.add_parser("status", help="Check tunnel status")


def handle(args: argparse.Namespace) -> int:
    """Handle tunnel subcommand.
    
    Args:
        args: The parsed arguments.
        
    Returns:
        Exit code.
    """
    if args.tunnel_command == "start" or args.tunnel_command is None:
        return start_tunnel(
            local_port=getattr(args, "port", 8080),
            background=getattr(args, "background", False),
        )
    elif args.tunnel_command == "stop":
        return stop_tunnel()
    elif args.tunnel_command == "status":
        return check_tunnel_status()
    else:
        logger.error(f"Unknown tunnel command: {args.tunnel_command}")
        return 1