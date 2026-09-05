# -------------------------------------------------------------------------
# File: cloudmesh/ai/vllm/base_orchestrator.py
# -------------------------------------------------------------------------
import os
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from cloudmesh.ai.common import DotDict, banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.tunnel import tunnel_manager
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.squeue import SQueue
from cloudmesh.ai.common.stopwatch import StopWatch


class BaseOrchestrator(ABC):
    """
    Shared orchestration logic for any LLM‑backend (vLLM, Ollama, …).

    Sub‑classes only need to implement the *backend‑specific* primitives:
      * ``_resolve_image`` – how to obtain the container/image path.
      * ``_build_start_command`` – the exact command that will launch the service.
      * ``_health_check`` – a callable that returns ``True`` when the service
        is ready.
      * ``_default_port`` – the port the backend listens on when the user does
        not override it.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.config = VLLMConfig()
        self.state_path = os.path.expanduser("~/.config/cloudmesh/llm_state.json")
        self.logger = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------------------
    # 1️⃣  Generic helpers – unchanged from the original orchestrator
    # -----------------------------------------------------------------
    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                import json
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception as e:
                console.warning(f"Could not read state file: {e}")
        return {}

    def _save_state(self, state: dict) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        import json
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # -----------------------------------------------------------------
    # 2️⃣  Backend‑specific abstract methods (must be overridden)
    # -----------------------------------------------------------------
    @abstractmethod
    def _resolve_image(self, cfg: DotDict) -> str:
        """Return the Docker image (or Apptainer image) to use."""
        ...

    @abstractmethod
    def _build_start_command(self, cfg: DotDict) -> str:
        """Return the exact shell command that starts the service."""
        ...

    @abstractmethod
    def _health_check(self, host: str, port: int) -> bool:
        """Return ``True`` when the backend answers on its health endpoint."""
        ...

    @property
    @abstractmethod
    def _default_port(self) -> int:
        """Port the backend uses by default (e.g. 8000 for vLLM, 11434 for Ollama)."""
        ...

    # -----------------------------------------------------------------
    # 3️⃣  Shared orchestration flow (unchanged for both back‑ends)
    # -----------------------------------------------------------------
    def _expand_and_prepare(self, name: str, port_override: Optional[int]) -> Tuple[DotDict, str]:
        """
        Resolve the server configuration, expand placeholders, inject ``name`` and ``port``.
        Returns ``(expanded_config, target_host)``.
        """
        # 1️⃣  Pull the raw server block (may be nested under a host)
        raw_cfg = self.config.get_server(name)
        if not raw_cfg:
            raise ValueError(f"Server '{name}' not found in config")

        # 2️⃣  Resolve external references (SSH look‑ups, $HOME, etc.)
        expanded = self.config.expand_external_references(raw_cfg)

        # 3️⃣  Force the logical name into the dict – many helpers (e.g. VLLMStartScript)
        #     expect a ``name`` key.
        expanded["name"] = name

        # 4️⃣  Apply an optional port override (covers both local & remote ports)
        if port_override:
            expanded["remote_port"] = port_override
            expanded["local_port"] = port_override
        else:
            # Ensure a numeric port is present – fall back to the backend default.
            expanded.setdefault("remote_port", self._default_port)
            expanded.setdefault("local_port", expanded["remote_port"])

        # 5️⃣  Resolve the *host* we will SSH to (defaults to the entry’s ``host``)
        target_host = expanded.get("host", "localhost")
        return expanded, target_host

    def _ensure_tunnel(self, host: str, cfg: DotDict) -> None:
        """
        If the target is remote, start an SSH tunnel from ``local_port`` → ``remote_port``.
        The optional ``tunnel`` field in the config can override the default command.
        """
        if host in ("localhost", "127.0.0.1"):
            return  # no tunnel needed

        local = cfg["local_port"]
        remote = cfg["remote_port"]
        custom = cfg.get("tunnel")
        success, msg = tunnel_manager.start_tunnel(host, local, dest_host="127.0.0.1",
                                                   custom_command=custom)
        if not success and msg != "Tunnel already active":
            raise RuntimeError(f"Failed to create tunnel: {msg}")

    # -----------------------------------------------------------------
    # 4️⃣  Public API – identical to the original ``VLLMOrchestrator`` interface
    # -----------------------------------------------------------------
    def prepare_backend(self, name: str, port_override: Optional[int] = None) -> bool:
        """
        Full pipeline:
          * resolve & expand configuration,
          * launch the backend (Docker / native),
          * create an SSH tunnel (if needed),
          * wait for the health endpoint to become ready.
        Returns ``True`` on success.
        """
        StopWatch.start("backend_startup")

        # -------------------------------------------------------------
        # Resolve configuration & inject helpers
        # -------------------------------------------------------------
        cfg, target_host = self._expand_and_prepare(name, port_override)

        # -------------------------------------------------------------
        # Build the actual command that will start the service
        # -------------------------------------------------------------
        start_cmd = self._build_start_command(cfg)

        # -------------------------------------------------------------
        # Launch the backend (Docker, sbatch, local binary, …)
        # -------------------------------------------------------------
        console.banner(f"Launching {self.__class__.__name__} backend")
        console.print(f"[blue]Host:[/blue] {target_host}")
        console.print(f"[blue]Command:[/blue] {start_cmd}")

        # The generic ``Server`` abstraction already knows how to execute a
        # command on a remote host (or locally).  We reuse that helper.
        from cloudmesh.ai.vllm.server import Server   # base class provides ``_execute``
        dummy_server = Server(target_host, launch_mode="remote")
        dummy_server._execute(start_cmd)

        # -------------------------------------------------------------
        # SSH tunnel (if remote)
        # -------------------------------------------------------------
        self._ensure_tunnel(target_host, cfg)

        # -------------------------------------------------------------
        # Health‑check loop
        # -------------------------------------------------------------
        max_tries = 60
        with console.status("[bold blue]Waiting for the backend to become ready…[/bold blue]"):
            for i in range(max_tries):
                if self._health_check(cfg.get("host", "localhost"), cfg["local_port"]):
                    console.ok("Backend is READY")
                    StopWatch.stop("backend_startup")
                    console.print(f"[dim]Total startup time: {StopWatch.get('backend_startup'):.2f}s[/dim]")
                    return True
                time.sleep(5)

        console.error("Backend failed to become ready within the timeout")
        return False

    # -----------------------------------------------------------------
    # 5️⃣  Convenience helpers (stop / kill / status) – delegate to the
    #      concrete ``Server`` implementation (UVA, DGX, Ollama, …)
    # -----------------------------------------------------------------
    def _get_server_instance(self, name: str) -> Server:
        """
        Resolve the concrete server class using the existing ``get_server`` helper.
        This method lives in ``cloudmesh.ai.vllm.orchestrator`` and knows about
        the Ollama dispatch, so we import it lazily here to avoid circular imports.
        """
        from cloudmesh.ai.vllm.orchestrator import get_server
        cfg = self.config.get_server(name)
        host = cfg.get("host", "localhost")
        return get_server(host, server_name=name)

    def stop(self, name: str) -> None:
        srv = self._get_server_instance(name)
        srv.stop(name)

    def kill(self, name: str) -> None:
        srv = self._get_server_instance(name)
        srv.kill(name)

    def status(self, name: str) -> None:
        srv = self._get_server_instance(name)
        console.print(f"[bold]Status for {name}:[/bold] {srv.status(name)}")