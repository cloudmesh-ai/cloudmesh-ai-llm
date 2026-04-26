import requests
from cloudmesh.ai.common.remote import RemoteExecutor

class VLLMClient:
    """Client to interact with a running vLLM server."""
    def __init__(self, config):
        self.config = config
        self.host = config.get("host")
        self.port = config.get("port")
        self.api_url = f"http://{self.host}:{self.port}/health"

    def is_alive(self):
        """Check if the vLLM server is responding to health checks."""
        try:
            # Check basic health endpoint
            response = requests.get(self.api_url, timeout=5)
            if response.status_code != 200:
                return False
            
            # Also verify that the model API is responsive
            models_url = f"http://{self.host}:{self.port}/v1/models"
            models_response = requests.get(models_url, timeout=5)
            return models_response.status_code == 200
        except Exception:
            return False

    def get_logs(self, lines=100):
        """Retrieve the last N lines of logs from the remote server."""
        log_file = self.config.get("log_file")
        if not log_file:
            return "No log file configured."

        cmd = f"tail -n {lines} {log_file}"
        try:
            with RemoteExecutor(self.host) as ssh:
                status, stdout, stderr = ssh.execute(cmd)
                if status != 0:
                    return f"Error retrieving logs (status {status}): {stderr}"
                return stdout
        except Exception as e:
            return f"Error retrieving logs: {str(e)}"

    def __repr__(self):
        return f"VLLMClient(host={self.host}, port={self.port})"