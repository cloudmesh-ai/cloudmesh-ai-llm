import requests
from cloudmesh.ai.common.remote import RemoteExecutor
from cloudmesh.ai.common.io import console

class VLLMClient:
    """Client to interact with a running vLLM server."""
    def __init__(self, config, server_name=None, debug: bool = False):
        self.config = config
        self.debug = debug
        
        if server_name and hasattr(config, 'resolve_server_identity'):
            identity = config.resolve_server_identity(server_name)
            self.host = identity['host']
            self.port = identity['port']
            self.user = identity['user']
        else:
            self.host = config.get("host")
            self.port = config.get("port")
            self.user = config.get("user")

        self.api_url = f"http://{self.host}:{self.port}/health"

    def _log_request(self, method: str, url: str, data: dict = None):
        """Print a raw curl-like command if debug is enabled."""
        if not self.debug:
            return
        
        cmd = f"curl -X {method} {url}"
        if data:
            import json
            json_data = json.dumps(data)
            cmd += f" -H 'Content-Type: application/json' -d '{json_data}'"
        
        console.print(f"[dim]DEBUG (API): {cmd}[/dim]")

    def get_status(self):
        """
        Check the vLLM server status.
        Returns: 'OFFLINE', 'STARTING', or 'READY'.
        """
        try:
            # 1. Check basic health endpoint
            self._log_request("GET", self.api_url)
            response = requests.get(self.api_url, timeout=5)
            if response.status_code != 200:
                return "STARTING"
            
            # 2. Verify that the model API is responsive
            models_url = f"http://{self.host}:{self.port}/v1/models"
            self._log_request("GET", models_url)
            models_response = requests.get(models_url, timeout=5)
            if models_response.status_code == 200:
                return "READY"
            
            return "STARTING"
        except requests.exceptions.ConnectionError:
            return "OFFLINE"
        except Exception:
            return "OFFLINE"

    def is_alive(self):
        """Backward compatibility: check if the server is READY."""
        return self.get_status() == "READY"

    def get_logs(self, lines=100, grep=None):
        """Retrieve the last N lines of logs from the remote server, optionally filtered by grep."""
        log_file = self.config.get("log_file")
        if not log_file:
            return "No log file configured."

        cmd = f"tail -n {lines} {log_file}"
        if grep:
            cmd += f" | grep -i '{grep}'"

        try:
            with RemoteExecutor(self.host) as ssh:
                status, stdout, stderr = ssh.execute(cmd)
                if status != 0:
                    # grep returns non-zero if no matches are found, which isn't necessarily an error
                    if grep and status == 1:
                        return ""
                    return f"Error retrieving logs (status {status}): {stderr}"
                return stdout
        except Exception as e:
            return f"Error retrieving logs: {str(e)}"

    def stream_logs(self, grep=None):
        """
        Start a log streaming process from the remote server.
        Returns:
            subprocess.Popen: The process object for streaming logs.
        """
        log_file = self.config.get("log_file")
        if not log_file:
            raise ValueError("No log file configured.")

        # Use stdbuf to disable buffering for real-time output
        cmd = f"ssh {self.host} 'stdbuf -oL tail -f {log_file}"
        if grep:
            cmd += f" | grep --line-buffered -i '{grep}'"
        cmd += "'"
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Line buffered
        )
        return process

    def __repr__(self):
        return f"VLLMClient(host={self.host}, port={self.port})"