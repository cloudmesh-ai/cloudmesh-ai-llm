from cloudmesh.ai.vllm.server import Server
from yamldb import YamlDB
import os
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.start_script import VLLMStartScript
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.tunnel import tunnel_manager

class ServerDGX(Server):
    """
    vLLM server implementation for DGX.
    """

    def __init__(self, host: str, db=None):
        super().__init__(host)
        if db:
            self.db = db
        else:
            # Use a standard path for the vLLM server configurations
            config_path = os.path.expanduser("~/.config/cloudmesh/ai/vllm_servers.yaml")
            self.db = YamlDB(filename=config_path)
            self._load_examples_if_missing(self.db, config_path)

    def _get_config(self, name: str) -> dict:
        """Retrieve configuration for a specific server name from the YAML DB."""
        # Navigate the hierarchy: cloudmesh -> ai -> dgx -> [name]
        config = self.db.get("cloudmesh.ai.dgx." + name)
        if not config:
            raise ValueError(f"Server configuration for '{name}' not found in YAML database under cloudmesh.ai.dgx.")
        return config

    def get_start_command(self, name: str) -> str:
        """Return the command used to start the vLLM server on DGX."""
        config_dict = self._get_config(name)
        self._validate_config(config_dict, ['image', 'model'])
        
        # Use VLLMConfig for the start script generator
        config = VLLMConfig(self.db, "dgx", name)
        return VLLMStartScript(config).generate()

    def _get_container_name(self, name: str) -> str:
        """Return a consistent container name for the server."""
        return f"vllm-dgx-{name}"

    def start(self, name: str, sbatch: bool = False) -> None:
        """
        Start the vLLM server on DGX using the configuration named 'name'.
        """
        config_dict = self._get_config(name)
        working_dir = config_dict.get('working_dir', '/scratch/$USER/cloudmesh/run')
        script_path = f"{working_dir}/start_{name}.sh"
        
        # Use VLLMConfig for the new helper classes
        config = VLLMConfig(self.db, "dgx", name)
        
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
            # DGX typically doesn't use ijob, but we use the batch_job helper for consistency
            # If ijob is not supported on DGX, this will still generate a command
            # but we can also just run the script directly as before.
            # To maintain previous behavior of just running the script:
            exec_cmd = f"{script_path}"
            mode = "direct"
        
        result = self._run_remote(exec_cmd)
        if result.returncode == 0:
            container_name = self._get_container_name(name)
            self.logger.info(f"Started vLLM server '{name}' on {self.host} using {mode} with script {script_path} in container {container_name}")
        else:
            raise RuntimeError(f"Failed to start vLLM server via {mode}: {result.stderr}")

    def stop(self, name: str) -> None:
        """
        Stop the vLLM server on DGX gracefully.
        """
        container_name = self._get_container_name(name)
        cmd = f"docker stop {container_name}"
        self._run_remote(cmd)
        
        # Wait a bit and check if it's still running
        import time
        time.sleep(5)
        if self.status(name) == "Running":
            self.logger.info(f"Server {name} still running after stop, sending kill")
            self.kill(name)

    def kill(self, name: str) -> None:
        """
        Forcefully kill the vLLM server on DGX.
        """
        container_name = self._get_container_name(name)
        cmd = f"docker rm -f {container_name}"
        self._run_remote(cmd)

    def status(self, name: str) -> str:
        """
        Return the status of the vLLM server on DGX.
        """
        config = self._get_config(name)
        port = config.get('port', '8000')
        container_name = self._get_container_name(name)
        
        # 1. Check if container is running
        proc_cmd = f"docker ps -f name={container_name} --format '{{{{.Status}}}}'"
        proc_result = self._run_remote(proc_cmd)
        
        if not proc_result.stdout.strip():
            return "Stopped"
        
        # 2. Check API health via curl
        health_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health"
        health_result = self._run_remote(health_cmd)
        
        if health_result.stdout.strip() == "200":
            return "Running"
        
        return "Starting/Unhealthy"

    def tunnel(self, name: str) -> None:
        """
        Create a tunnel to the vLLM server on DGX.
        """
        config = self._get_config(name)
        port = config.get('port', '8000')
        
        success, result = tunnel_manager.start_tunnel(self.host, port)
        if success:
            self.logger.info(f"Tunnel created: localhost:{port} -> {self.host}:{port} (PID: {result})")
        else:
            self.logger.warning(f"Tunnel not created: {result}")

    def get_logs(self, name: str) -> str:
        """
        Retrieve logs for the vLLM server.
        """
        container_name = self._get_container_name(name)
        result = self._run_remote(f"docker logs --tail 100 {container_name}")
        return result.stdout
