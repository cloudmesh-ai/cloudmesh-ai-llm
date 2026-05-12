from cloudmesh.ai.vllm.server import Server
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
        super().__init__(host, db)

    def get_start_command(self, name: str) -> str:
        """Return the command used to start the vLLM server on DGX."""
        return super().get_start_command(name, ['image', 'model'])

    def _get_container_name(self, name: str) -> str:
        """Return a consistent container name for the server."""
        return f"vllm-dgx-{name}"

    def start(self, name: str, sbatch: bool = False) -> None:
        """Start the vLLM server on DGX."""
        super().start(name, sbatch)

    def _send_stop_signal(self, name: str) -> None:
        container_name = self._get_container_name(name)
        self._run_remote(f"docker stop {container_name}")

    def _send_kill_signal(self, name: str) -> None:
        container_name = self._get_container_name(name)
        self._run_remote(f"docker rm -f {container_name}")

    def _check_process_running(self, name: str) -> bool:
        container_name = self._get_container_name(name)
        proc_cmd = f"docker ps -f name={container_name} --format '{{{{.Status}}}}'"
        proc_result = self._run_remote(proc_cmd)
        return bool(proc_result.stdout.strip())


    def _get_log_command(self, name: str) -> str:
        container_name = self._get_container_name(name)
        return f"docker logs --tail 100 {container_name}"

    def _get_direct_exec_cmd(self, name: str, script_path: str) -> str:
        return script_path
