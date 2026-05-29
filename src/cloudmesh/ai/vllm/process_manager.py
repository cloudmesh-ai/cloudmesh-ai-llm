import subprocess
import signal
import os
import atexit
from cloudmesh.ai.common.io import console

class ProcessRegistry:
    """
    Registry to track active subprocesses and ensure they are terminated 
    on application exit or error.
    """
    def __init__(self):
        self._processes = {}

    def register(self, name: str, process: subprocess.Popen):
        """Register a process for tracking."""
        self._processes[name] = process
        console.debug(f"Registered process {name} (PID: {process.pid})")

    def unregister(self, name: str):
        """Unregister a process."""
        if name in self._processes:
            del self._processes[name]

    def list_processes(self):
        """Return a list of currently tracked processes."""
        return [{"name": name, "pid": proc.pid, "status": "Running" if proc.poll() is None else "Finished"} 
                for name, proc in self._processes.items()]

    def terminate_all(self):
        """Terminate all tracked processes."""
        for name, process in list(self._processes.items()):
            try:
                if process.poll() is None:
                    console.debug(f"Terminating process {name} (PID: {process.pid})")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                self.unregister(name)
            except Exception as e:
                console.error(f"Error terminating process {name}: {e}")

# Singleton instance
process_registry = ProcessRegistry()

# Ensure all processes are killed on exit
atexit.register(process_registry.terminate_all)