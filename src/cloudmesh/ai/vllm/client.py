import requests
import subprocess
from cloudmesh.ai.common.remote import RemoteExecutor
from cloudmesh.ai.common.io import console

class VLLMClient:
    """Client to interact with a running vLLM server."""
    def __init__(self, config, server_name=None, port=None, host=None, debug: bool = False):
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

        # Overrides for dynamic port/host (e.g. from --port flag)
        if host:
            self.host = host
        if port:
            self.port = port
        
        # Default to localhost if we have a port but no host
        if self.port and not self.host:
            self.host = "localhost"

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
        Check the vLLM server status by probing multiple endpoints.
        Returns: 'OFFLINE', 'STARTING', or 'READY'.
        """
        api_key = self.config.get("ai.llm.vllm_api_key") or self.config.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Use 127.0.0.1 and bypass any system proxies (HTTP_PROXY, etc.)
        host = "127.0.0.1" if self.host in ["localhost", "127.0.0.1"] else self.host
        base_url = f"http://{host}:{self.port}"
        
        # Disable proxies for local requests
        session = requests.Session()
        session.trust_env = False 

        try:
            # 1. Primary Check: /v1/models OR /health
            # In vLLM, if either of these return 200, the engine is ready for traffic.
            for path in ["/v1/models", "/health"]:
                try:
                    self._log_request("GET", f"{base_url}{path}")
                    res = session.get(f"{base_url}{path}", headers=headers, timeout=3)
                    if res.status_code == 200:
                        return "READY"
                    if res.status_code in [401, 403]:
                        # Alive but requires authentication
                        return "READY"
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    continue

            # 2. Secondary Check: /metrics
            # If ONLY metrics is up, the process is running but the engine might still be loading.
            try:
                self._log_request("GET", f"{base_url}/metrics")
                res = session.get(f"{base_url}/metrics", headers=headers, timeout=3)
                if res.status_code == 200:
                    return "STARTING"
            except:
                pass
            
            return "OFFLINE"

        except Exception as e:
            if self.debug:
                console.print(f"[dim]Health check unexpected error: {e}[/dim]")
            return "OFFLINE"

    def is_alive(self):
        """Backward compatibility: check if the server is READY."""
        return self.get_status() == "READY"

    def get_models(self):
        """
        Retrieve the list of models available on the vLLM server.
        Returns: A list of model dictionaries.
        """
        api_key = self.config.get("ai.llm.vllm_api_key") or self.config.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            models_url = f"http://{self.host}:{self.port}/v1/models"
            self._log_request("GET", models_url)
            response = requests.get(models_url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except Exception as e:
            console.print(f"[dim]Error retrieving models: {e}[/dim]")
            return []

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

    def get_metrics(self):
        """
        Retrieve Prometheus metrics from the vLLM server.
        Returns a dictionary of parsed vLLM metrics.
        """
        api_key = self.config.get("ai.llm.vllm_api_key") or self.config.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            metrics_url = f"http://{self.host}:{self.port}/metrics"
            self._log_request("GET", metrics_url)
            response = requests.get(metrics_url, headers=headers, timeout=5)
            if response.status_code != 200:
                return {}

            metrics = {}
            for line in response.text.splitlines():
                # Ignore comments and empty lines
                if line.startswith("#") or not line.strip():
                    continue
                
                # Prometheus format: metric_name{labels} value
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                name_part = parts[0]
                value = parts[1]
                
                # Strip labels for simple lookup
                name = name_part.split('{')[0]
                
                try:
                    metrics[name] = float(value)
                except ValueError:
                    continue
            
            return metrics
        except Exception as e:
            console.print(f"[dim]Error retrieving metrics: {e}[/dim]")
            return {}

    def __repr__(self):
        return f"VLLMClient(host={self.host}, port={self.port})"
