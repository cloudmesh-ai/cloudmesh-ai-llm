from cloudmesh.ai.vllm.server import Server
from yamldb import YamlDB
import os
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.start_script import VLLMStartScript
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.tunnel import tunnel_manager

class ServerUVA(Server):
    """
    vLLM server implementation for UVA.
    """

    def __init__(self, host: str):
        super().__init__(host)
        # Use a standard path for the vLLM server configurations
        config_path = os.path.expanduser("~/.config/cloudmesh/ai/vllm_servers.yaml")
        self.db = YamlDB(filename=config_path)

    def _get_config(self, name: str) -> dict:
        """Retrieve configuration for a specific server name from the YAML DB."""
        # Navigate the hierarchy: cloudmesh -> ai -> uva -> [name]
        config = self.db.get("cloudmesh.ai.uva." + name)
        if not config:
            raise ValueError(f"Server configuration for '{name}' not found in YAML database under cloudmesh.ai.uva.")
        return config

    def get_start_command(self, name: str) -> str:
        """Return the command used to start the vLLM server on UVA."""
        config_dict = self._get_config(name)
        self._validate_config(config_dict, ['account', 'partition', 'image', 'model'])
        
        # Use VLLMConfig for the start script generator
        config = VLLMConfig(self.db, "uva", name)
        return VLLMStartScript(config).generate()

    def start(self, name: str, sbatch: bool = False) -> None:
        """
        Start the vLLM server on UVA using the configuration named 'name'.
        """
        config_dict = self._get_config(name)
        working_dir = config_dict.get('working_dir', '/scratch/$USER/cloudmesh/run')
        script_path = f"{working_dir}/start_{name}.sh"
        
        # Use VLLMConfig for the new helper classes
        config = VLLMConfig(self.db, "uva", name)
        
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
            exec_cmd = batch_job.get_execution_command("ijob")
            mode = "ijob"
        
        result = self._run_remote(exec_cmd)
        if result.returncode == 0:
            self.logger.info(f"Started vLLM server '{name}' on {self.host} using {mode} with script {script_path}")
        else:
            raise RuntimeError(f"Failed to start vLLM server via {mode}: {result.stderr}")

    def stop(self, name: str) -> None:
        """
        Stop the vLLM server on UVA gracefully.
        """
        # Find PID by searching for the model name in the process list
        config = self._get_config(name)
        model = config['model']
        
        # Try SIGTERM first
        cmd = f"pgrep -f '{model}' | xargs kill -15"
        self._run_remote(cmd)
        
        # Wait a bit and check if it's still running
        import time
        time.sleep(5)
        if self.status(name) == "Running":
            self.logger.info(f"Server {name} still running after SIGTERM, sending SIGKILL")
            self.kill(name)

    def kill(self, name: str) -> None:
        """
        Forcefully kill the vLLM server on UVA.
        """
        config = self._get_config(name)
        model = config['model']
        cmd = f"pgrep -f '{model}' | xargs kill -9"
        self._run_remote(cmd)

    def status(self, name: str) -> str:
        """
        Return the status of the vLLM server on UVA.
        """
        config = self._get_config(name)
        port = config.get('port', '8000')
        
        # 1. Check if process is running
        proc_cmd = f"pgrep -f '{config['model']}'"
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
        Create a tunnel to the vLLM server on UVA.
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
        log_file = f"~/vllm_logs/{name}.log"
        result = self._run_remote(f"tail -n 100 {log_file}")
        return result.stdout
