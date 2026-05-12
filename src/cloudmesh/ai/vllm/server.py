from abc import ABC, abstractmethod
import subprocess
import logging
import os
import yaml
import time
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.start_script import VLLMStartScript
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.common import DotDict

class Server(ABC):
    """
    Abstract base class for vLLM server implementations.
    """

    def __init__(self, host: str, db=None):
        self.host = host
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if db:
            self.db = db
        else:
            # Use a standard path for the vLLM server configurations
            config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.db = DotDict(yaml.safe_load(f) or {})
            else:
                self.db = DotDict()


    def _get_config(self, name: str) -> VLLMConfig:
        """Retrieve configuration for a specific server name using VLLMConfig for merging."""
        config = VLLMConfig()
        if not config:
            raise ValueError(f"Server configuration for '{name}' not found in YAML database under cloudmesh.ai.server.")
        return config

    def get_start_command(self, name: str, required_fields: list) -> str:
        """Return the command used to start the vLLM server."""
        config = self._get_config(name)
        self._validate_config(config, required_fields)
        return VLLMStartScript(config).generate()

    def start(self, name: str, sbatch: bool = False) -> None:
        """
        Start the vLLM server using the configuration named 'name'.
        """
        config = self._get_config(name)
        working_dir = config.get('working_dir', '/scratch/$USER/cloudmesh/run')
        script_path = f"{working_dir}/start_{name}.sh"
        
        cmd_content = VLLMStartScript(config).generate()
        self._upload_script(cmd_content, script_path)
        
        batch_job = VLLMBatchJob(config, script_path)
        
        if sbatch:
            slurm_script_path = f"{working_dir}/submit_{name}.slurm"
            slurm_content = batch_job.generate_sbatch_content(working_dir)
            self._upload_script(slurm_content, slurm_script_path)
            exec_cmd = batch_job.get_execution_command("sbatch", slurm_script_path)
            mode = "sbatch"
        else:
            exec_cmd = self._get_direct_exec_cmd(name, script_path)
            mode = "direct/ijob"
        
        result = self._run_remote(exec_cmd)
        if result.returncode == 0:
            self.logger.info(f"Started vLLM server '{name}' on {self.host} using {mode} with script {script_path}")
        else:
            raise RuntimeError(f"Failed to start vLLM server via {mode}: {result.stderr}")

    def stop(self, name: str) -> None:
        """Stop the vLLM server gracefully."""
        self._send_stop_signal(name)
        
        # Wait a bit and check if it's still running
        time.sleep(5)
        if self.status(name) == "Running":
            self.logger.info(f"Server {name} still running after stop, sending kill")
            self.kill(name)

    def kill(self, name: str) -> None:
        """Forcefully kill the vLLM server."""
        self._send_kill_signal(name)

    def status(self, name: str) -> str:
        """Return the current status of the vLLM server."""
        config = self._get_config(name)
        port = config.get('port', '8000')
        
        # 1. Check if process is running
        if not self._check_process_running(name):
            return "Stopped"
        
        # 2. Check API health via curl
        health_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health"
        health_result = self._run_remote(health_cmd)
        
        if health_result.stdout.strip() == "200":
            return "Running"
        
        return "Starting/Unhealthy"

    def tunnel(self, name: str) -> None:
        """Create a tunnel to the vLLM server."""
        config = self._get_config(name)
        port = config.get('port', '8000')
        
        success, result = tunnel_manager.start_tunnel(self.host, port)
        if success:
            self.logger.info(f"Tunnel created: localhost:{port} -> {self.host}:{port} (PID: {result})")
        else:
            self.logger.warning(f"Tunnel not created: {result}")

    def get_logs(self, name: str) -> str:
        """Retrieve logs for the vLLM server."""
        cmd = self._get_log_command(name)
        result = self._run_remote(cmd)
        return result.stdout

    @abstractmethod
    def _get_direct_exec_cmd(self, name: str, script_path: str) -> str:
        """Return the command to execute the script directly (non-sbatch)."""
        pass

    @abstractmethod
    def _send_stop_signal(self, name: str) -> None:
        """Send a graceful stop signal to the server."""
        pass

    @abstractmethod
    def _send_kill_signal(self, name: str) -> None:
        """Send a forceful kill signal to the server."""
        pass

    @abstractmethod
    def _check_process_running(self, name: str) -> bool:
        """Check if the server process is running."""
        pass

    @abstractmethod
    def _get_log_command(self, name: str) -> str:
        """Return the command to retrieve logs."""
        pass

    def _run_remote(self, cmd: str) -> subprocess.CompletedProcess:
        """
        Execute a command on the remote host via SSH.
        """
        ssh_cmd = ["ssh", self.host, cmd]
        return subprocess.run(ssh_cmd, capture_output=True, text=True)

    def _upload_script(self, content: str, remote_path: str):
        """
        Upload a script to the remote host.
        """
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(remote_path)
        self._run_remote(f"mkdir -p {dir_path}")

        # Use ssh to write the file
        # We use a heredoc to write the content to the remote file
        # We escape single quotes in the content to avoid breaking the ssh command
        escaped_content = content.replace("'", "'\\''")
        upload_cmd = f"cat << 'EOF' > {remote_path}\n{content}\nEOF"
        
        # Since the content can be large, we use a different approach for uploading
        # to avoid shell argument limits. We'll use a temporary file and scp or 
        # just use ssh with a heredoc if it's reasonably sized.
        # For vLLM start commands, they are small.
        
        ssh_cmd = ["ssh", self.host, f"cat << 'EOF' > {remote_path}\n{content}\nEOF"]
        subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        
        # Make the script executable
        self._run_remote(f"chmod +x {remote_path}")

    def _validate_config(self, config: dict, required_fields: list):
        """
        Validate that the configuration contains all required fields.
        """
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")

