# -------------------------------------------------------------------------
# File: cloudmesh/ai/vllm/server_ollama.py
# -------------------------------------------------------------------------
import os
from cloudmesh.ai.vllm.server import Server
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.docker_manager import DockerManager
from cloudmesh.ai.vllm.tunnel import tunnel_manager

class OllamaServer(Server):
    """
    Server implementation for Ollama.

    * launch_mode = "docker" → run the official Ollama container.
    * launch_mode = "local"  → run the locally‑installed ``ollama`` binary.

    All other Server‑API methods (status, stop, logs, tunnel, …) are provided
    so the existing ``cmc llm`` commands work without changes.
    """

    def __init__(self, host: str, db=None, launch_mode: str = "remote"):
        super().__init__(host, db, launch_mode=launch_mode)
        self.docker = DockerManager()

    # -----------------------------------------------------------------
    # Helper: decide how to start the server
    # -----------------------------------------------------------------
    def _run_ollama(self) -> bool:
        """Start Ollama according to the configured ``launch_mode``."""
        cfg = self.server_config

        # -----------------------------------------------------------------
        # 1️⃣  Docker mode
        # -----------------------------------------------------------------
        if cfg.get("launch_mode") == "docker":
            image   = cfg.get("image", "ollama/ollama:latest")
            name    = f"ollama-{cfg.get('name', 'default')}"
            port    = cfg.get("remote_port", 11434)
            cache   = cfg.get("cache_dir", "/tmp/ollama")
            model   = cfg.get("model", "")

            # Build the docker command – we **never** embed the model name in the
            # command line; we pass it via an environment variable so the secret
            # never appears in ``ps``.
            env_flags = "-e OLLAMA_MODEL"               # variable name only
            docker_cmd = (
                f"docker run -d --rm "
                f"-p {port}:11434 "
                f"-v {cache}:/root/.ollama "
                f"{env_flags} "
                f"--name {name} "
                f"{image}"
            )

            console.print("[blue]Launching Ollama Docker container…[/blue]")
            # Pass the real environment variable out‑of‑band so Docker picks it up
            env = {"OLLAMA_MODEL": model}
            if not self.docker.run_container(docker_cmd, env=env):
                raise RuntimeError("Failed to start Ollama container")
            return True

        # -----------------------------------------------------------------
        # 2️⃣  Native (local) mode
        # -----------------------------------------------------------------
        elif cfg.get("launch_mode") == "local":
            # If the user supplied a custom script, use it; otherwise fall back
            # to the simple “ollama serve --model …” command.
            script = cfg.get("script")
            if script:
                # Write a temporary script file and execute it
                tmp = "/tmp/ollama_start.sh"
                with open(tmp, "w") as f:
                    f.write(script)
                os.chmod(tmp, 0o755)
                cmd = f"bash {tmp}"
            else:
                model = cfg.get("model", "")
                cmd = f"ollama serve --model {model}"
            console.print("[blue]Starting native Ollama binary…[/blue]")
            # ``self._execute`` knows whether we are on a remote host or not.
            self._execute(cmd)
            return True

        else:
            raise ValueError(f"Unsupported launch_mode {cfg.get('launch_mode')} for Ollama")

    # -----------------------------------------------------------------
    # Abstract methods required by ``Server``
    # -----------------------------------------------------------------
    def _get_direct_exec_cmd(self, name: str, script_path: str) -> str:
        """
        For Ollama we never need an ijob script – the server is started
        directly (Docker or binary).  ``script_path`` is only used when a
        custom user script was supplied; otherwise we just return a no‑op.
        """
        cfg = self.server_config
        if cfg.get("script"):
            return f"bash {script_path}"
        # In Docker mode the server is already running after ``_run_ollama``.
        # In native mode the command is executed directly, so we return an empty
        # string – the orchestrator will skip the extra exec step.
        return ""

    def start(self, name: str, sbatch: bool = False) -> None:
        """Public entry point used by the orchestrator."""
        # Resolve the concrete configuration for the name (already stored in
        # ``self.server_config`` by the orchestrator).
        console.banner(f"Starting Ollama server '{name}' (mode={self.server_config.get('launch_mode')})")
        self._run_ollama()

    def _send_stop_signal(self, name: str) -> None:
        cfg = self.server_config
        if cfg.get("launch_mode") == "docker":
            self.docker.stop_container(f"ollama-{cfg.get('name','default')}")
        else:
            # ``ollama stop`` gracefully shuts down the running daemon.
            self._execute("ollama stop")

    def _send_kill_signal(self, name: str) -> None:
        cfg = self.server_config
        if cfg.get("launch_mode") == "docker":
            self.docker.stop_container(f"ollama-{cfg.get('name','default')}")
        else:
            # Force kill any lingering ``ollama serve`` processes.
            self._execute("pkill -9 ollama")

    def _check_process_running(self, name: str) -> bool:
        cfg = self.server_config
        if cfg.get("launch_mode") == "docker":
            out = self._execute(
                f"docker ps -f name=ollama-{cfg.get('name','default')} --format '{{{{.Status}}}}'"
            )
            return bool(out.stdout.strip())
        else:
            out = self._execute("pgrep -f 'ollama serve'")
            return bool(out.stdout.strip())

    def _get_log_command(self, name: str) -> str:
        cfg = self.server_config
        if cfg.get("launch_mode") == "docker":
            return f"docker logs --tail 100 ollama-{cfg.get('name','default')}"
        else:
            # Ollama writes logs to ``~/.ollama/logs``; fall back to tail.
            return "tail -n 100 ~/.ollama/logs/ollama.log || echo 'no log file'"