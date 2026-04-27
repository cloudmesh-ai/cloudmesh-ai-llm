import os
import json
import subprocess
import signal
from cloudmesh.ai.common.io import console

class TunnelManager:
    """Manages SSH tunnels for vLLM servers, tracking PIDs for cleanup."""
    
    def __init__(self):
        self.state_file = os.path.expanduser("~/.config/cloudmesh/ai/tunnels.json")
        self._ensure_state_file()

    def _ensure_state_file(self):
        if not os.path.exists(self.state_file):
            self._save_state({})

    def _load_state(self):
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self, state):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def is_tunnel_active(self, host, port):
        """Check if a tunnel for the given host and port is active."""
        state = self._load_state()
        key = f"{host}:{port}"
        if key in state:
            pid = state[key]
            try:
                # Signal 0 checks if the process exists
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False
        return False

    def start_tunnel(self, host, port):
        """Start an SSH tunnel and track its PID."""
        if self.is_tunnel_active(host, port):
            return False, "Tunnel already active"

        tunnel_cmd = ["ssh", "-L", f"{port}:localhost:{port}", host, "-N"]
        try:
            process = subprocess.Popen(
                tunnel_cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            state = self._load_state()
            state[f"{host}:{port}"] = process.pid
            self._save_state(state)
            return True, process.pid
        except Exception as e:
            return False, str(e)

    def stop_tunnel(self, host, port):
        """Stop a specific tunnel using its tracked PID."""
        state = self._load_state()
        key = f"{host}:{port}"
        if key not in state:
            return False, "No active tunnel found for this host/port"

        pid = state[key]
        try:
            os.kill(pid, signal.SIGTERM)
            # Clean up state
            del state[key]
            self._save_state(state)
            return True, pid
        except (OSError, ProcessLookupError):
            # Process already gone, just clean up state
            del state[key]
            self._save_state(state)
            return True, pid

    def cleanup_orphans(self):
        """Remove entries from state file for processes that are no longer running."""
        state = self._load_state()
        new_state = {}
        for key, pid in state.items():
            try:
                os.kill(pid, 0)
                new_state[key] = pid
            except (OSError, ProcessLookupError):
                pass
        self._save_state(new_state)

tunnel_manager = TunnelManager()