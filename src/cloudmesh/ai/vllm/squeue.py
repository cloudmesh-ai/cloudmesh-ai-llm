import subprocess
import json

class SQueue:
    """Wrapper for Slurm squeue to provide consistent JSON-like output."""

    def __init__(self, host="uva"):
        self.host = host

    def get_jobs(self):
        """Retrieve running jobs, attempting JSON first then falling back to text."""
        try:
            # 1. Try JSON
            cmd = f"ssh {self.host} 'squeue --json'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            
            if "serializer_required" in result.stderr or "could not find plugin" in result.stderr:
                # Fallback to text parsing
                return self._get_jobs_text()
            
            # Other errors
            return []
        except Exception:
            return self._get_jobs_text()

    def _get_jobs_text(self):
        """Fallback: Use pipe-separated format to avoid quoting issues over SSH."""
        try:
            # Use a simple pipe-separated format: job_id|job_name|state|node_list
            fmt = '%i|%j|%T|%N'
            cmd = f"ssh {self.host} 'squeue --noheader --format=\"{fmt}\"'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return []

            jobs = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) < 4:
                    continue
                
                jobs.append({
                    "job_id": parts[0],
                    "name": parts[1],
                    "state": parts[2],
                    "nodes": [{"name": parts[3] if parts[3] else "Unknown"}]
                })
            return jobs
        except Exception:
            return []

    def cancel(self, job_id):
        """Cancel a Slurm job by its ID."""
        try:
            cmd = f"ssh {self.host} 'scancel {job_id}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
