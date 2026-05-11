from abc import ABC, abstractmethod
import subprocess
import logging
import os
import yaml

class Server(ABC):
    """
    Abstract base class for vLLM server implementations.
    """

    def __init__(self, host: str):
        self.host = host
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_examples_if_missing(self, db, config_path):
        """Load example configurations if the main config file is empty or does not exist."""
        if not db.data:
            self.logger.info(f"Config file {config_path} is empty or not found. Loading examples...")
            example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm.yaml")
            if os.path.exists(example_path):
                db.load(filename=example_path)
                db.save(filename=config_path)
            else:
                self.logger.warning(f"Example config file {example_path} not found.")

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

    @abstractmethod
    def get_start_command(self, name: str) -> str:
        """
        Return the command used to start the vLLM server.
        """
        pass

    @abstractmethod
    def start(self, name: str, sbatch: bool = False) -> None:
        """
        Start the vLLM server.
        """
        pass

    @abstractmethod
    def stop(self, name: str) -> None:
        """
        Stop the vLLM server.
        """
        pass

    @abstractmethod
    def kill(self, name: str) -> None:
        """
        Forcefully kill the vLLM server.
        """
        pass

    @abstractmethod
    def status(self, name: str) -> str:
        """
        Return the current status of the vLLM server.
        """
        pass

    @abstractmethod
    def tunnel(self, name: str) -> None:
        """
        Create a tunnel to the vLLM server.
        """
        pass

    @abstractmethod
    def get_logs(self, name: str) -> str:
        """
        Retrieve logs for the vLLM server.
        """
        pass
