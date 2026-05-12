from cloudmesh.ai.vllm.server import Server
import os
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.start_script import VLLMStartScript
from cloudmesh.ai.vllm.batch_job import VLLMBatchJob
from cloudmesh.ai.vllm.tunnel import tunnel_manager

class ServerUVA(Server):
    """
    vLLM server implementation for UVA.
    """

    def __init__(self, host: str, db=None):
        super().__init__(host, db)

    def get_start_command(self, name: str) -> str:
        """Return the command used to start the vLLM server on UVA."""
        return super().get_start_command(name, ['account', 'partition', 'image', 'model'])

    def start(self, name: str, sbatch: bool = False) -> None:
        """Start the vLLM server on UVA."""
        super().start(name, sbatch)

    def _send_stop_signal(self, name: str) -> None:
        config = self._get_config(name)
        model = config['model']
        self._run_remote(f"pgrep -f '{model}' | xargs kill -15")

    def _send_kill_signal(self, name: str) -> None:
        config = self._get_config(name)
        model = config['model']
        self._run_remote(f"pgrep -f '{model}' | xargs kill -9")

    def _check_process_running(self, name: str) -> bool:
        config = self._get_config(name)
        proc_cmd = f"pgrep -f '{config['model']}'"
        proc_result = self._run_remote(proc_cmd)
        return bool(proc_result.stdout.strip())


    def _get_log_command(self, name: str) -> str:
        return f"tail -n 100 ~/vllm_logs/{name}.log"

    def _get_direct_exec_cmd(self, name: str, script_path: str) -> str:
        config = self._get_config(name)
        batch_job = VLLMBatchJob(config, script_path)
        return batch_job.get_execution_command("ijob")
