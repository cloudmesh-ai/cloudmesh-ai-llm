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

    def __init__(self, host: str, db=None, launch_mode: str = "remote", debug: bool = False):
        self.host = host
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.launch_mode = launch_mode  # "local" or "remote"
        
        if db:
            # If db is a callable (mock), wrap it in a DotDict proxy to avoid TypeError
            if callable(db):
                class DbProxy(DotDict):
                    def __getitem__(self, key):
                        return db(key)
                    def get(self, key, default=None):
                        try:
                            return db(key)
                        except Exception:
                            return default
                self.db = DbProxy()
            else:
                self.db = db
        else:
            # Initialize db using VLLMConfig to ensure consistent structure and merging
            config_manager = VLLMConfig()
            self.db = config_manager._config


    def _get_config(self, name: str):
        """Retrieve configuration for a specific server name using VLLMConfig for merging."""
        # Now that self.db is guaranteed to be a mapping (via DbProxy if it was callable),
        # we can use VLLMConfig consistently.
        try:
            config_manager = VLLMConfig(db=self.db)
            server_config = config_manager.get_server(name)
            if server_config:
                return server_config
        except Exception as e:
            self.logger.debug(f"Configuration lookup failed for {name}: {e}")
            
        raise ValueError(f"Server configuration for '{name}' not found in configuration database.")

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
        
        result = self._execute(exec_cmd)
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
        health_result = self._execute(health_cmd)
        
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
        result = self._execute(cmd)
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

    def _execute(self, cmd: str) -> subprocess.CompletedProcess:
        """
        Execute command - either locally or via SSH based on launch_mode.
        """
        if self.debug:
            if self.launch_mode == "local":
                console.print(f"[dim]DEBUG (SSH/CMD): {cmd}[/dim]")
            else:
                console.print(f"[dim]DEBUG (SSH/CMD): ssh {self.host} '{cmd}'[/dim]")

        if self.launch_mode == "local":
            self.logger.debug(f"[LOCAL] {cmd}")
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            self.logger.debug(f"[REMOTE:{self.host}] {cmd}")
            ssh_cmd = ["ssh", self.host, cmd]
            return subprocess.run(ssh_cmd, capture_output=True, text=True)

    def _upload_script(self, content: str, path: str):
        """
        Upload/write script locally or remotely based on launch_mode.
        """
        if self.launch_mode == "local":
            # Write file directly to local filesystem
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            os.chmod(path, 0o755)
            self.logger.debug(f"[LOCAL] Script written to {path}")
        else:
            # Upload via SSH to remote host
            dir_path = os.path.dirname(path)
            self._execute(f"mkdir -p {dir_path}")
            
            ssh_cmd = ["ssh", self.host, f"cat << 'EOF' > {path}\n{content}\nEOF"]
            if self.debug:
                console.print(f"[dim]DEBUG (SSH/CMD): {' '.join(ssh_cmd)}[/dim]")
            subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            self._execute(f"chmod +x {path}")
            self.logger.debug(f"[REMOTE:{self.host}] Script uploaded to {path}")

    def upload_env_file(self, local_env_path: str, remote_env_path: str = None) -> bool:
        """
        Upload a local .env file to the remote server.
        
        Args:
            local_env_path (str): Path to local .env file
            remote_env_path (str, optional): Path on remote server. 
                If None, uses ~/.cloudmesh/.env
        
        Returns:
            bool: True if upload successful
        """
        if not os.path.exists(local_env_path):
            self.logger.error(f"Local env file not found: {local_env_path}")
            return False
        
        if remote_env_path is None:
            remote_env_path = f"~/.cloudmesh/.env"
        
        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_env_path)
        self._execute(f"mkdir -p {remote_dir}")
        
        if self.launch_mode == "local":
            # Local copy
            try:
                os.makedirs(remote_dir, exist_ok=True)
                import shutil
                shutil.copy2(local_env_path, remote_env_path)
                self.logger.info(f"[LOCAL] Env file copied to {remote_env_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to copy env file locally: {e}")
                return False
        else:
            # Remote upload via scp or SSH
            try:
                # Method 1: Using scp if available
                scp_result = subprocess.run(
                    ["scp", local_env_path, f"{self.host}:{remote_env_path}"],
                    capture_output=True, text=True
                )
                
                if scp_result.returncode == 0:
                    self.logger.info(f"[REMOTE:{self.host}] Env file uploaded to {remote_env_path}")
                    return True
                else:
                    # Method 2: Fallback to SSH with cat
                    with open(local_env_path, 'r') as f:
                        content = f.read()
                    
                    ssh_cmd = ["ssh", self.host, f"cat << 'ENV_EOF' > {remote_env_path}\n{content}\nENV_EOF"]
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.logger.info(f"[REMOTE:{self.host}] Env file uploaded to {remote_env_path}")
                        return True
                    else:
                        self.logger.error(f"Failed to upload env file: {result.stderr}")
                        return False
                        
            except Exception as e:
                self.logger.error(f"Failed to upload env file: {e}")
                return False

    def run_with_env(self, command: str, env_path: str = None) -> subprocess.CompletedProcess:
        """
        Execute a command with environment variables sourced from a file.
        
        Args:
            command (str): The command to execute
            env_path (str, optional): Path to env file. If None, uses ~/.cloudmesh/.env
        
        Returns:
            subprocess.CompletedProcess: The result of the command execution
        """
        if env_path is None:
            env_path = "~/.cloudmesh/.env"
        
        # Source the env file and run the command
        # Using bash -c to ensure the env vars are available in the subprocess
        full_command = f"bash -c 'source {env_path} && {command}'"
        
        return self._execute(full_command)

    def deploy_with_env(self, name: str, local_env_path: str = None, sbatch: bool = False) -> None:
        """
        Deploy server with environment variables from a .env file.
        
        This combines upload_env_file with the normal start() process.
        
        Args:
            name (str): Server configuration name
            local_env_path (str, optional): Path to local .env file
            sbatch (bool): Whether to use sbatch
        """
        # Upload env file if provided
        if local_env_path and os.path.exists(local_env_path):
            remote_env_path = f"~/.cloudmesh/.env_{name}"
            self.upload_env_file(local_env_path, remote_env_path)
            
            # Merge env vars into config before starting
            config = self._get_config(name)
            config.merge_env_vars(env_file=local_env_path)
        
        # Start the server with the updated config
        self.start(name, sbatch=sbatch)

    def _validate_config(self, config: dict, required_fields: list):
        """
        Validate that the configuration contains all required fields.
        """
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")

