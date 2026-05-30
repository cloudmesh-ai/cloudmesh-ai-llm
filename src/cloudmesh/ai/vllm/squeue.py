import subprocess
import json

class SQueue:
    """Wrapper for Slurm squeue to provide consistent JSON-like output."""

    def __init__(self, host="uva"):
        self.host = host

    def get_jobs(self):
        """Retrieve running jobs for the current user, prioritizing the fast text format."""
        # We prioritize text format because JSON is often slow or missing on Slurm clusters
        try:
            text_jobs = self._get_jobs_text()
            if text_jobs:
                return text_jobs
        except Exception:
            pass

        try:
            # Try JSON as a fallback (though it's usually slower)
            cmd = f"ssh {self.host} 'squeue --me --json'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result and result.returncode == 0:
                data = json.loads(result.stdout)
                jobs_list = data.get("jobs", [])
                normalized_jobs = []
                for job in jobs_list:
                    resources = job.get("job_resources", {})
                    nodes_data = resources.get("nodes", {})
                    allocation = nodes_data.get("allocation", [])
                    normalized_job = job.copy()
                    normalized_job["nodes"] = allocation
                    normalized_jobs.append(normalized_job)
                return normalized_jobs
        except Exception:
            pass
        
        return []

    def _get_jobs_text(self):
        """Fastest method: Use the user-recommended squeue format for high performance."""
        try:
            # User recommended fast command: squeue -a -u "$USER" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
            # Use -u $USER instead of --me for maximum compatibility across Slurm versions
            fmt = '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'
            cmd = f"ssh {self.host} 'squeue --noheader -u $USER -o \"{fmt}\"'"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []
 
            jobs = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                # Split by whitespace
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                # Mapping based on format:
                # %i (0): JobID, %P (1): Partition, %j (2): Name, %u (3): User, %t (4): State, %M (5): Time, %D (6): Nodes, %R (7): NodeList
                job_id = parts[0]
                name = parts[2]
                state = parts[4]
                # NodeList is the last part if it exists
                node_list = parts[7] if len(parts) > 7 else "Unknown"
                
                jobs.append({
                    "job_id": job_id,
                    "name": name,
                    "state": state,
                    "nodes": [{"name": node_list if node_list else "Unknown"}]
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
