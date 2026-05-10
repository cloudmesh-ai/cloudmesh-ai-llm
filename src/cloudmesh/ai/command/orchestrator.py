import os
import re
import shutil
import subprocess
import time
import yaml
from pathlib import Path
from yamldb import YamlDB
from cloudmesh.ai.common import banner
from cloudmesh.ai.common.io import console
from cloudmesh.ai.command.vllm import get_server
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vpn.vpn import Vpn

def get_vllm_api_key(config, keys_path_override=None, lookup_key=None):
    """Retrieve vLLM API key from main config or a specified keys file.
    
    If lookup_key is provided, it is used as the key to find the secret in the keys file.
    Otherwise, it defaults to looking for 'VLLM_API_KEY'.
    """
    # 1. Try main config (specific lookup key first, then default)
    if lookup_key:
        api_key = config.get(f"ai.llm.{lookup_key}")
        if api_key:
            return api_key
            
    api_key = config.get("ai.llm.vllm_api_key")
    if api_key:
        return api_key

    # 2. Try keys file (either from config or default)
    keys_path = keys_path_override or config.get("keys") or os.path.expanduser("~/.config/cloudmesh/keys.yaml")
    keys_path = os.path.expanduser(keys_path).replace("$HOME", os.path.expanduser("~"))
    
    if os.path.exists(keys_path):
        try:
            with open(keys_path, "r") as f:
                keys = yaml.safe_load(f)
                if not keys:
                    return None
                
                # Use the provided lookup_key (e.g., 'SERVER_MASTER_KEY') or default to 'VLLM_API_KEY'
                target_key = lookup_key or "VLLM_API_KEY"
                if target_key in keys:
                    return keys[target_key]
        except Exception:
            pass
    
    return None

class VLLMOrchestrator:
    """Orchestrates the full pipeline from server start to client launch."""

    def __init__(self):
        self.config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        # Use :memory: backend to avoid writing back changes to the config file
        self.db = YamlDB(filename=self.config_path, backend=":memory:")
        self.template_dir = Path(__file__).parent / "templates"

    def _kill_port_process(self, port: int):
        """Kill any process currently binding to the specified local port."""
        try:
            # Use lsof to find the PID of the process using the port
            cmd = f"lsof -t -iTCP:{port} -sTCP:LISTEN"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            pid = result.stdout.strip()
            if pid:
                # PID can be a list of PIDs separated by newlines
                for p in pid.split('\n'):
                    if p:
                        subprocess.run(f"kill -9 {p}", shell=True)
                        console.print(f"[dim]Killed existing process {p} on port {port}[/dim]")
        except Exception as e:
            console.warning(f"Could not kill process on port {port}: {e}")

    def _get_remote_user(self, host: str) -> str:
        """Resolve the remote username from ~/.ssh/config for a given host."""
        ssh_config = os.path.expanduser("~/.ssh/config")
        if os.path.exists(ssh_config):
            try:
                with open(ssh_config, "r") as f:
                    content = f.read()
                    # Look for the Host block that matches the target host
                    # This is a simple parser; it looks for 'Host <host>' followed by 'User <user>'
                    pattern = rf"Host\s+.*{re.escape(host)}.*?\n(.*?)\n\s*Host"
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if not match:
                        # Try matching the last block if no subsequent Host is found
                        pattern = rf"Host\s+.*{re.escape(host)}.*?\n(.*)"
                        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    
                    if match:
                        block = match.group(1)
                        user_match = re.search(r"^\s*User\s+(\S+)", block, re.MULTILINE | re.IGNORECASE)
                        if user_match:
                            return user_match.group(1)
            except Exception:
                pass
        return os.environ.get('USER', 'user')

    def export_scripts(self, server_name: str, destination: str = "."):
        """Export launch scripts to a local directory for customization."""
        servers = self.db.get("cloudmesh.ai.server", {})
        if not isinstance(servers, dict) or server_name not in servers:
            console.error(f"Server {server_name} not found in config.")
            return False
        
        platform = servers[server_name].get("platform", "default")
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        script_map = {"dgx": "start_dgx.sh", "uva": "start_uva.sh"}
        if platform in script_map:
            script_name = script_map[platform]
            src = self.template_dir / script_name
            dst = dest_path / script_name
            if src.exists():
                shutil.copy(src, dst)
                console.ok(f"Exported {script_name} to {dst}")
            else:
                console.error(f"Template {script_name} not found.")
                return False
        else:
            console.warning(f"No exportable script for platform {platform}")
            return False

        # Export the resolved config as yaml
        config_file = dest_path / f"{server_name}_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(servers[server_name], f)
        console.ok(f"Exported config to {config_file}")
        return True

    def launch_dgx(self, server_name: str, port_override: int = None):
        """DGX specific launch: Upload and run script on remote host."""
        servers = self.db.get("cloudmesh.ai.server", {})
        config = servers.get(server_name, {})
        target_host = config.get("host")
        
        # Use explicit user from config if provided, otherwise resolve from ssh config
        remote_user = config.get("user")
        # Support placeholders: {~/.ssh.config.host.user} or {~/.ssh/config:host.user}
        is_placeholder = (
            not remote_user or 
            remote_user == f"{{~/.ssh.config.{target_host}.user}}" or 
            remote_user == f"{{~/.ssh/config:{target_host}.user}}"
        )
        if is_placeholder:
            remote_user = self._get_remote_user(target_host)
            
        remote_port = port_override or config.get("remote_port", 8000)
        remote_dir = config.get("dir", f"/raid/{remote_user}/cloudmesh/vllm_{remote_port}")
        if remote_dir:
            if "{user}" in remote_dir:
                remote_dir = remote_dir.replace("{user}", remote_user)
            if "{port}" in remote_dir:
                remote_dir = remote_dir.replace("{port}", str(remote_port))
        
        script_name = "start_dgx.sh"
        
        # Handle port override
        remote_port = port_override or config.get("remote_port", 8000)
        # We need to pass the port to the script. 
        # Since start_dgx.sh uses environment variables or defaults, 
        # we will pass it as an environment variable to the bash command.
        local_script = Path(".").joinpath(script_name)
        script_path = local_script if local_script.exists() else self.template_dir / script_name
        
        console.print(f"[blue]Deploying {script_name} to {target_host}:{remote_dir}...[/blue]")
        try:
            # Create remote directory and upload script
            subprocess.run(f"ssh {target_host} 'mkdir -p {remote_dir}'", shell=True, check=True)
            subprocess.run(f"scp {script_path} {target_host}:{remote_dir}/{script_name}", shell=True, check=True)
            
            # Execute script in that directory with port override
            remote_cmd = f"cd {remote_dir} && PORT={remote_port} bash {script_name}"
            console.print(f"[blue]Starting vLLM server on {target_host}...[/blue]")
            console.print(f"[dim]Executing: ssh {target_host} '{remote_cmd}'[/dim]")
            subprocess.run(f"ssh {target_host} '{remote_cmd}'", shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"DGX launch failed: {e}")
            return False

    def stop_uva(self, server_name: str, port_pattern: str = None):
        """Cancel the Slurm job for a UVA vLLM server. Supports fuzzy port matching."""
        servers = self.db.get("cloudmesh.ai.server", {})
        config = servers.get(server_name, {})
        if not config:
            console.error(f"Server {server_name} not found in config.")
            return False

        # If a pattern is provided, we search for any job matching vllm_*pattern*
        # Otherwise, we use the exact port from config
        if port_pattern:
            console.print(f"[blue]Searching for vLLM jobs matching pattern 'vllm_*{port_pattern}*'...[/blue]")
            # List all jobs with name starting with vllm_, then grep for the pattern
            find_job_cmd = f"ssh uva 'squeue -h -o %i,%.10n | grep vllm_ | grep {port_pattern}'"
        else:
            remote_port = config.get("remote_port", 8000)
            job_name = f"vllm_{remote_port}"
            console.print(f"[blue]Stopping vLLM server {server_name} (exact port {remote_port})...[/blue]")
            find_job_cmd = f"ssh uva 'squeue -h -o %i -n {job_name}'"

        try:
            result = subprocess.run(find_job_cmd, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            if not output:
                console.warning(f"No running jobs found matching the criteria.")
                return False
            
            # Extract Job IDs (first column)
            job_ids = []
            for line in output.split('\n'):
                if line:
                    job_id = line.split()[0]
                    job_ids.append(job_id)
            
            for id in job_ids:
                console.print(f"[dim]Cancelling job {id}...[/dim]")
                subprocess.run(f"ssh uva 'scancel {id}'", shell=True, check=True)
            
            console.ok(f"Successfully cancelled {len(job_ids)} job(s).")
            return True
            
            if not job_id:
                console.warning(f"No running job found with name {job_name}.")
                return False
            
            # If multiple jobs are found, split by newline and cancel all
            for id in job_id.split('\n'):
                if id:
                    console.print(f"[dim]Cancelling job {id}...[/dim]")
                    subprocess.run(f"ssh uva 'scancel {id}'", shell=True, check=True)
            
            console.ok(f"Successfully cancelled job(s) for {job_name}.")
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Failed to stop UVA server: {e}")
            return False

    def launch_uva(self, server_name: str, port_override: int = None):
        """UVA HPC specific launch: Deployment -> ijob -> Tunnel -> Apptainer."""
        console.print("[bold red]DEBUG: Executing updated launch_uva logic...[/bold red]")
        servers = self.db.get("cloudmesh.ai.server", {})
        config = servers.get(server_name, {})
        
        # Use explicit user from config if provided, otherwise resolve from ssh config
        remote_user = config.get("user")
        is_placeholder = (
            not remote_user or 
            remote_user == "{~/.ssh.config.uva.user}" or 
            remote_user == "{~/.ssh/config:uva.user}"
        )
        if is_placeholder:
            remote_user = self._get_remote_user("uva")
            
        remote_port = port_override or config.get("remote_port", 8000)
        remote_dir = config.get("dir", f"/scratch/{remote_user}/cloudmesh/llm_{remote_port}")
        if remote_dir:
            if "{user}" in remote_dir:
                remote_dir = remote_dir.replace("{user}", remote_user)
            if "{port}" in remote_dir:
                remote_dir = remote_dir.replace("{port}", str(remote_port))

        try:
            # 1. Deployment (MUST happen before ijob)
            script_name = "start_uva.sh"
            script_path = self.template_dir / script_name
            
            console.print(f"[blue]Deploying {script_name} from template to uva:{remote_dir}/{script_name}...[/blue]")
            subprocess.run(f"ssh uva 'mkdir -p {remote_dir}'", shell=True, check=True)
            # Use absolute path for scp to avoid any ambiguity
            scp_cmd = f"scp {script_path} uva:{remote_dir}/{script_name}"
            console.print(f"[dim]Executing: {scp_cmd}[/dim]")
            subprocess.run(scp_cmd, shell=True, check=True)
            console.ok(f"Successfully uploaded {script_name} to {remote_dir}")

            # 2. Allocation
            console.print("[blue]Requesting GPU allocation via sbatch...[/blue]")
            
            # Resolve image path: from config or default to /scratch/{user}/vllm_gemma4.sif
            vllm_image = config.get("image")
            if not vllm_image or not vllm_image.startswith("/"):
                vllm_image = f"/scratch/{remote_user}/{vllm_image or 'vllm_gemma4.sif'}"

            sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=vllm_{remote_port}
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bi_dsc_community
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --output={remote_dir}/sbatch.out
#SBATCH --error={remote_dir}/sbatch.err

cd {remote_dir}
PORT={remote_port} VLLM_IMAGE="{vllm_image}" bash {script_name}
"""
            sbatch_file = f"{remote_dir}/submit.sh"
            subprocess.run(f"ssh uva 'echo \"{sbatch_script}\" > {sbatch_file}'", shell=True, check=True)
            
            submit_cmd = f"ssh uva 'sbatch {sbatch_file}'"
            console.print(f"[dim]Executing: {submit_cmd}[/dim]")
            result = subprocess.run(submit_cmd, shell=True, capture_output=True, text=True, check=True)
            
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                console.error(f"Could not find Job ID in sbatch output: {result.stdout}")
                return False
            job_id = match.group(1)
            console.ok(f"Submitted job {job_id}. Waiting for allocation...")
            
            process = subprocess.Popen(["sleep", "10000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Poll squeue to find the allocated node
            node_name = None
            for i in range(60): # Wait up to 5 minutes
                check_node_cmd = f"ssh uva 'squeue -j {job_id} -h -o %N'"
                node_res = subprocess.run(check_node_cmd, shell=True, capture_output=True, text=True)
                node_out = node_res.stdout.strip()
                if node_out and node_out != "NLIST":
                    # node_out might be "udc-an26-1" or "udc-an26-1,udc-an26-2"
                    node_name = node_out.split(',')[0]
                    break
                time.sleep(5)
            
            if not node_name:
                console.error(f"Job {job_id} did not allocate a node within 5 minutes.")
                process.terminate()
                return False
            
            if not node_name:
                console.error("Could not determine the allocated node name from output.")
                process.terminate()
                return False
                
            console.ok(f"Allocated node: {node_name}")
            
            # The server is already started by the 'ijob' command we injected.
            # We do NOT use separate SSH calls to the compute node to avoid password prompts,
            # as the laptop cannot authenticate directly to the node.
            
            console.ok(f"vLLM server is now starting on {node_name} via ijob allocation.")
            # Return both node_name and the process so we can continue streaming logs
            return node_name, process
        except subprocess.CalledProcessError as e:
            console.error(f"UVA launch failed: {e}")
            return False
        except Exception as e:
            console.error(f"Unexpected error during UVA launch: {e}")
            return False

    def prepare_backend(self, name, port_override: int = None):
        """Ensure vLLM server is running, tunneled, and healthy."""
        # Bypass YamlDB :memory: backend and load directly from disk to ensure fresh config
        try:
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f) or {}
                servers = full_config.get("cloudmesh", {}).get("ai", {}).get("server", {})
        except Exception as e:
            raise ValueError(f"Could not read configuration file {self.config_path}: {e}")

        if not isinstance(servers, dict) or name not in servers:
            available = list(servers.keys()) if isinstance(servers, dict) else []
            available_str = ", ".join(available) if available else "None"
            raise ValueError(
                f"Could not resolve configuration for service '{name}'.\n"
                f"Available servers in config: {available_str}\n"
                f"If no servers are listed, run 'cmc launch init server <host>' first."
            )

        config = servers[name]
        target_host = config.get("host")
        platform = config.get("platform", "default")

        if not target_host:
            raise ValueError(f"Host not specified for service '{name}' in configuration.")

        server = get_server(target_host)
        vllm_config = VLLMConfig(self.db, name)
        client = VLLMClient(vllm_config)

        console.print(banner("Orchestrating vLLM Backend", f"Host: {target_host}\nService: {name}\nPlatform: {platform}"))
        
        # 0. VPN Check
        if target_host not in ["localhost", "127.0.0.1"]:
            console.print("[blue]Checking VPN connection...[/blue]")
            vpn = Vpn()
            if not vpn.enabled():
                console.msg("VPN is disconnected. Attempting to connect...")
                if not vpn.connect():
                    console.error("VPN connection failed. Please connect to the VPN and try again.")
                    return False
                console.ok("VPN connected successfully!")
            else:
                console.ok("VPN is already active.")

        # 1. Initial Health Check
        console.print("[blue]Checking if vLLM server is already available...[/blue]")
        if client.is_alive():
            console.ok("vLLM server is already ALIVE and tunneled!")
            return True

        # 2. Platform-specific Launch
        process = None
        if platform == "uva":
            result = self.launch_uva(name, port_override=port_override)
            if not result:
                return False
            node_name, process = result
        elif platform == "dgx":
            if not self.launch_dgx(name, port_override=port_override):
                return False
        else:
            # Default flow: Tunnel then Start
            if target_host not in ["localhost", "127.0.0.1"]:
                console.print("[blue]Establishing SSH tunnel...[/blue]")
                server.tunnel(name)
                
                if client.is_alive():
                    console.ok("vLLM server is now available!")
                    return True

                console.print("[blue]Starting vLLM server on remote host...[/blue]")
                server.start(name)
            else:
                console.warning("Local host detected. Please ensure the vLLM server is started locally.")
        
        # 3. Final Health Check Poll (Remote check before tunneling)
        if platform == "uva":
            console.print(f"[blue]Verifying model health on {node_name}...[/blue]")
            remote_port = port_override or config.get("remote_port", 8000)
            
            # Resolve remote_dir for log checking
            remote_user = config.get("user") or self._get_remote_user(target_host)
            remote_dir = config.get("dir", f"/scratch/{remote_user}")
            if remote_dir:
                remote_dir = remote_dir.replace("{user}", remote_user).replace("{port}", str(remote_port))

            # Set process output to non-blocking so we can stream logs while polling health
            if process:
                try:
                    import os
                    os.set_blocking(process.stdout.fileno(), False)
                except Exception:
                    pass

            # Retrieve API key for health check
            api_key = get_vllm_api_key(config)
            auth_header = f'-H "Authorization: Bearer {api_key}"' if api_key else ""

            for i in range(120):
                # 1. Stream any available logs from the vLLM process
                if process:
                    try:
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            print(line, end="")
                    except Exception:
                        pass

                # 2. Check if port is open AND application startup is complete in logs.
                # We run the check via the login node to avoid direct SSH authentication issues.
                
                 # Check port
                 # Use direct nc from login node to compute node to avoid nested SSH authentication issues
                port_check_cmd = f"ssh uva \"nc -z {node_name} {remote_port}\""
                port_open = subprocess.run(port_check_cmd, shell=True, capture_output=True).returncode == 0
                
                # Check health endpoint via curl with API key if available
                health_check_cmd = f"ssh uva \"curl -s -f {auth_header} http://{node_name}:{remote_port}/health\""
                app_ready = subprocess.run(health_check_cmd, shell=True, capture_output=True).returncode == 0
                
                if port_open and app_ready:
                    console.ok("\n vLLM server is now ALIVE and Application startup complete!")
                    
                    # NOW establish the tunnel after health check
                    local_port = port_override or config.get("local_port", 8000)
                    
                    # Clean up any existing process using the local port to avoid "Address already in use"
                    self._kill_port_process(local_port)
                    
                    console.print(f"[blue]Establishing dynamic tunnel to {node_name}...[/blue]")
                    tunnel_cmd = f"ssh -L {local_port}:{node_name}:{remote_port} uva -N"
                    subprocess.Popen(tunnel_cmd, shell=True)
                    console.ok("Tunnel established in background.")
                    return True
                
                console.print(f"Waiting for model to load on remote... ({i+1}/120)", end="\r")
                time.sleep(5)
            
            if process:
                process.terminate()
            console.error("\n vLLM server failed to become healthy on remote node within 10 minutes.")
            return False
        else:
            # Existing health check for other platforms (which already have tunnels or are local)
            console.print("[blue]Verifying model health...[/blue]")
            for i in range(120):
                if client.is_alive():
                    console.ok("vLLM server is now ALIVE and model is loaded!")
                    return True
                console.print(f"Waiting for model to load... ({i+1}/120)", end="\r")
                time.sleep(5)
            
            console.error("vLLM server failed to become healthy within 10 minutes.")
            return False