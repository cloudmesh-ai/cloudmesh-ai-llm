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
from cloudmesh.ai.common.stopwatch import StopWatch
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.vllm.server_uva import ServerUVA
from cloudmesh.ai.vllm.server_dgx import ServerDGX
from cloudmesh.ai.vllm.squeue import SQueue
from cloudmesh.ai.vpn.vpn import Vpn


def get_vllm_api_key(config, keys_path_override=None, lookup_key=None):
    """Retrieve vLLM API key from main config, specific key files, or a keys yaml file.

    If lookup_key is provided, it is used to find the secret.
    It checks:
    1. Main config
    2. Specific text files in ~/.config/cloudmesh/llm/
    3. keys.yaml
    """
    # 1. Try main config
    if lookup_key:
        api_key = config.get(f"ai.llm.{lookup_key}")
        if api_key:
            return api_key

    api_key = config.get("ai.llm.vllm_api_key")
    if api_key:
        return api_key

    # 2. Try specific key files in ~/.config/cloudmesh/llm/
    # Mapping lookup keys to filenames
    key_file_map = {
        "SERVER_MASTER_KEY": "server_master_key.txt",
        "VLLM_API_KEY": "server_master_key.txt",
        "HF_TOKEN": "HF_token.txt",
    }
    
    target_key = lookup_key or "VLLM_API_KEY"
    filename = key_file_map.get(target_key)
    
    if filename:
        key_file_path = os.path.expanduser(f"~/.config/cloudmesh/llm/{filename}")
        if os.path.exists(key_file_path):
            try:
                with open(key_file_path, "r") as f:
                    return f.read().strip()
            except Exception:
                pass

    # 3. Try keys yaml file (fallback)
    keys_path = (
        keys_path_override
        or config.get("keys")
        or os.path.expanduser("~/.config/cloudmesh/keys.yaml")
    )
    keys_path = os.path.expanduser(keys_path).replace("$HOME", os.path.expanduser("~"))

    if os.path.exists(keys_path):
        try:
            with open(keys_path, "r") as f:
                keys = yaml.safe_load(f)
                if keys and target_key in keys:
                    return keys[target_key]
        except Exception:
            pass

    return None


def get_default_host(db=None):
    """Retrieve the default host by resolving the default server name."""
    if db is None:
        config_path = os.path.expanduser("~/.config/cloudmesh/llm.yaml")
        db = YamlDB(filename=config_path)

    # 1. Get the default server name
    default_server_name = db.get("cloudmesh.ai.default.server")

    # 2. Get all configured servers
    servers = db.get("cloudmesh.ai.server", {})
    if not isinstance(servers, dict) or not servers:
        return None

    # 3. If no explicit default is set, use the first server in the list
    if not default_server_name:
        default_server_name = next(iter(servers))

    # 4. Resolve the host for the determined server
    host = servers.get(default_server_name, {}).get("host")
    if host:
        return host

    return None


def get_server(host, db=None):
    """Helper to instantiate the correct server class based on host."""
    host_str = str(host)
    if "uva" in host_str.lower() or "rivanna" in host_str.lower():
        return ServerUVA(host_str, db=db)
    return ServerDGX(host_str, db=db)




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
                for p in pid.split("\n"):
                    if p:
                        subprocess.run(f"kill -9 {p}", shell=True)
                        console.print(
                            f"[dim]Killed existing process {p} on port {port}[/dim]"
                        )
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
                        user_match = re.search(
                            r"^\s*User\s+(\S+)", block, re.MULTILINE | re.IGNORECASE
                        )
                        if user_match:
                            return user_match.group(1)
            except Exception:
                pass
        return os.environ.get("USER", "user")

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
        config = VLLMConfig(self.db, server_name).data
        target_host = config.get("host")

        # Use explicit user from config if provided, otherwise resolve from ssh config
        remote_user = config.get("user")
        # Support placeholders: {~/.ssh.config.host.user} or {~/.ssh/config:host.user}
        is_placeholder = (
            not remote_user
            or remote_user == f"{{~/.ssh.config.{target_host}.user}}"
            or remote_user == f"{{~/.ssh/config:{target_host}.user}}"
        )
        if is_placeholder:
            remote_user = self._get_remote_user(target_host)

        remote_port = port_override or config.get("remote_port", 8000)
        remote_dir = config.get(
            "dir", f"/raid/{remote_user}/cloudmesh/vllm_{remote_port}"
        )
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
        script_path = (
            local_script if local_script.exists() else self.template_dir / script_name
        )

        console.print(
            f"[blue]Deploying {script_name} to {target_host}:{remote_dir}...[/blue]"
        )
        try:
            # Create remote directory and upload script
            subprocess.run(
                f"ssh {target_host} 'mkdir -p {remote_dir}'", shell=True, check=True
            )
            subprocess.run(
                f"scp {script_path} {target_host}:{remote_dir}/{script_name}",
                shell=True,
                check=True,
            )

            # Execute script in that directory with port override
            remote_cmd = f"cd {remote_dir} && PORT={remote_port} bash {script_name}"
            console.print(f"[blue]Starting vLLM server on {target_host}...[/blue]")
            console.print(f"[dim]Executing: ssh {target_host} '{remote_cmd}'[/dim]")
            subprocess.run(f"ssh {target_host} '{remote_cmd}'", shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"DGX launch failed: {e}")
            return False

    def stop_uva(
        self, server_name: str = None, port_pattern: str = None, job_id: str = None
    ):
        """Cancel the Slurm job for a UVA vLLM server. Supports JobID, fuzzy port, or config port."""
        if job_id:
            console.print(f"[blue]Stopping job {job_id}...[/blue]")
            if SQueue().cancel(job_id):
                console.ok(f"Successfully cancelled job {job_id}.")
                return True
            else:
                console.error(f"Failed to cancel job {job_id}.")
                return False

        if port_pattern:
            console.print(
                f"[blue]Searching for vLLM jobs matching port pattern '{port_pattern}'...[/blue]"
            )
        elif server_name:
            servers = self.db.get("cloudmesh.ai.server", {})
            config = servers.get(server_name, {})
            if not config:
                console.error(f"Server {server_name} not found in config.")
                return False

            # Try persisted job_id first
            persisted_job = config.get("job_id")
            if persisted_job:
                console.print(
                    f"[blue]Stopping persisted job {persisted_job} for {server_name}...[/blue]"
                )
                if SQueue().cancel(persisted_job):
                    console.ok(f"Successfully cancelled job {persisted_job}.")
                    return True
                else:
                    console.error(f"Failed to cancel persisted job {persisted_job}.")
                    # Fall through to search by name if persisted ID fails

            remote_port = config.get("remote_port", 8000)
            console.print(
                f"[blue]Searching for jobs associated with {server_name} (port {remote_port})...[/blue]"
            )
        else:
            console.error("No server name, port pattern, or job ID provided.")
            return False

        try:
            # Use the SQueue wrapper for robust data retrieval
            # Use the host from the server config if available, otherwise default to 'uva'
            host = "uva"
            if server_name:
                servers_config = self.db.get("cloudmesh.ai.server", {})
                if isinstance(servers_config, dict):
                    host = servers_config.get(server_name, {}).get("host", "uva")
            
            jobs_data = SQueue(host=host).get_jobs()

            # DEBUG: Show all jobs found by SQueue to diagnose matching issues
            if jobs_data:
                console.print(
                    f"[dim]SQueue found {len(jobs_data)} total jobs. Scanning for matches...[/dim]"
                )
            else:
                console.print("[dim]SQueue returned no jobs.[/dim]")

            job_ids = []
            servers_config = self.db.get("cloudmesh.ai.server", {})
            for job in jobs_data:
                jid = job.get("job_id")
                jname = job.get("name", "")

                if port_pattern and port_pattern in jname:
                    console.print(f"[dim]  Match found: job {jid} (name: {jname})[/dim]")
                    job_ids.append(jid)
                elif server_name:
                    server_cfg = (
                        servers_config.get(server_name, {})
                        if isinstance(servers_config, dict)
                        else {}
                    )
                    remote_port = server_cfg.get("remote_port", 8000)
                    # Match if job name is exactly vllm_{port} or contains it
                    if jname == f"vllm_{remote_port}" or f"vllm_{remote_port}" in jname:
                        console.print(f"[dim]  Match found: job {jid} (name: {jname})[/dim]")
                        job_ids.append(jid)

            if not job_ids:
                console.warning(f"No running jobs found matching the criteria.")
                return False

            if not job_ids:
                console.warning(f"No running jobs found matching the criteria.")
                return False

            for id in job_ids:
                console.print(f"[dim]Cancelling job {id}...[/dim]")
                SQueue(host=host).cancel(id)

            console.ok(f"Successfully cancelled {len(job_ids)} job(s).")
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Failed to stop UVA server: {e}")
            return False

    def list_running_servers(self):
        """List all running vLLM servers on UVA."""
        console.banner("Running vLLM Servers")
        try:
            # Use the SQueue wrapper for robust data retrieval
            jobs_data = SQueue().get_jobs()

            if not jobs_data:
                console.warning("No running vLLM servers found on UVA.")
                return []

            servers_config = self.db.get("cloudmesh.ai.server", {})
            running_jobs = []

            for job in jobs_data:
                job_id = job.get("job_id")
                node = (
                    job.get("nodes", [{}])[0].get("name", "Unknown")
                    if isinstance(job.get("nodes"), list)
                    else "Unknown"
                )
                job_name = job.get("name", "")

                if "vllm_" not in job_name:
                    continue

                # Try to find which configured server this job belongs to
                server_name = "Unknown"
                port = "Unknown"
                if isinstance(servers_config, dict):
                    for name, cfg in servers_config.items():
                        if (
                            cfg.get("job_id") == job_id
                            or f"vllm_{cfg.get('remote_port')}" == job_name
                        ):
                            server_name = name
                            port = cfg.get("remote_port", "Unknown")
                            break

                running_jobs.append(
                    {
                        "server": server_name,
                        "job_id": job_id,
                        "node": node,
                        "port": port,
                    }
                )

            if not running_jobs:
                console.warning("No vLLM servers found in the running jobs list.")
                return []

            # Print as a table
            headers = ["Server", "Job ID", "Node", "Port"]
            data = [
                [j["server"], j["job_id"], j["node"], j["port"]] for j in running_jobs
            ]
            console.print_table(headers, data)
            return running_jobs

        except Exception as e:
            console.error(f"Failed to list running servers: {e}")
            return []

    def launch_uva(self, server_name: str, port_override: int = None):
        """UVA HPC specific launch: Deployment -> ijob -> Tunnel -> Apptainer."""
        console.print(
            "[bold red]DEBUG: Executing updated launch_uva logic...[/bold red]"
        )
        config = VLLMConfig(self.db, server_name).data

        # Use explicit user from config if provided, otherwise resolve from ssh config
        remote_user = config.get("user")
        is_placeholder = (
            not remote_user
            or remote_user == "{~/.ssh.config.uva.user}"
            or remote_user == "{~/.ssh/config:uva.user}"
        )
        if is_placeholder:
            remote_user = self._get_remote_user("uva")

        remote_port = port_override or config.get("remote_port", 8000)
        remote_dir = config.get(
            "dir", f"/scratch/{remote_user}/cloudmesh/llm_{remote_port}"
        )
        if remote_dir:
            if "{user}" in remote_dir:
                remote_dir = remote_dir.replace("{user}", remote_user)
            if "{port}" in remote_dir:
                remote_dir = remote_dir.replace("{port}", str(remote_port))

        try:
            # 1. Deployment (MUST happen before ijob)
            console.banner("Deployment")
            script_name = "start_uva.sh"
            script_path = self.template_dir / script_name

            console.print(
                f"[blue]Deploying {script_name} from template to uva:{remote_dir}/{script_name}...[/blue]"
            )
            subprocess.run(f"ssh uva 'mkdir -p {remote_dir}'", shell=True, check=True)
            # Use absolute path for scp to avoid any ambiguity
            scp_cmd = f"scp {script_path} uva:{remote_dir}/{script_name}"
            console.print(f"[dim]Executing: {scp_cmd}[/dim]")
            subprocess.run(scp_cmd, shell=True, check=True)
            console.ok(f"Successfully uploaded {script_name} to {remote_dir}")

            # 2. Allocation
            console.banner("Allocation")
            console.print("[blue]Requesting GPU allocation via sbatch...[/blue]")

            # Resolve image path: from config or default to /scratch/{user}/vllm_gemma4.sif
            vllm_image = config.get("image")
            if not vllm_image or not vllm_image.startswith("/"):
                vllm_image = f"/scratch/{remote_user}/{vllm_image or 'vllm_gemma4.sif'}"

            # Get email from config or use default
            email = config.get("email", "laszewski@gmail.com")

            sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=vllm_{remote_port}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bi_dsc_community
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --output={remote_dir}/vllm_{remote_port}.out
#SBATCH --error={remote_dir}/vllm_{remote_port}.err

cd {remote_dir}
PORT={remote_port} VLLM_IMAGE="{vllm_image}" bash {script_name}
"""
            sbatch_file = f"{remote_dir}/submit.sh"
            subprocess.run(
                f"ssh uva 'echo \"{sbatch_script}\" > {sbatch_file}'",
                shell=True,
                check=True,
            )

            submit_cmd = f"ssh uva 'sbatch {sbatch_file}'"
            console.print(f"[dim]Executing: {submit_cmd}[/dim]")
            result = subprocess.run(
                submit_cmd, shell=True, capture_output=True, text=True, check=True
            )

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                console.error(
                    f"Could not find Job ID in sbatch output: {result.stdout}"
                )
                return False
            job_id = match.group(1)
            console.ok(f"Submitted job {job_id}. Waiting for allocation...")

            # Start a background process to tail the remote logs in real-time
            # Use stdbuf -oL to force line-buffering and tail -F to handle file creation/rotation
            tail_cmd = f'ssh uva "stdbuf -oL tail -F {remote_dir}/vllm_{remote_port}.out {remote_dir}/vllm_{remote_port}.err"'
            process = subprocess.Popen(
                tail_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )

            # Poll squeue to find the allocated node
            node_name = None
            for i in range(60):  # Wait up to 5 minutes
                check_node_cmd = f"ssh uva 'squeue -j {job_id} -h -o %N'"
                node_res = subprocess.run(
                    check_node_cmd, shell=True, capture_output=True, text=True
                )
                node_out = node_res.stdout.strip()
                if node_out and node_out != "NLIST":
                    # node_out might be "udc-an26-1" or "udc-an26-1,udc-an26-2"
                    node_name = node_out.split(",")[0]
                    break
                time.sleep(5)

            if not node_name:
                console.error(f"Job {job_id} did not allocate a node within 5 minutes.")
                process.terminate()
                return False

            if not node_name:
                console.error(
                    "Could not determine the allocated node name from output."
                )
                process.terminate()
                return False

            console.ok(f"Allocated node: {node_name}")

            # Persist job and node info to config
            servers = self.db.get("cloudmesh.ai.server", {})
            if isinstance(servers, dict) and server_name in servers:
                servers[server_name]["job_id"] = job_id
                servers[server_name]["node_name"] = node_name
                self.db.set("cloudmesh.ai.server", servers)

            console.banner("Execution")
            # The server is already started by the 'ijob' command we injected.
            # We do NOT use separate SSH calls to the compute node to avoid password prompts,
            # as the laptop cannot authenticate directly to the node.

            console.ok(
                f"vLLM server is now starting on {node_name} via Slurm allocation."
            )
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
        StopWatch.start("vllm_startup")

        # Bypass YamlDB :memory: backend and load directly from disk to ensure fresh config
        try:
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f) or {}
                servers = (
                    full_config.get("cloudmesh", {}).get("ai", {}).get("server", {})
                )
        except Exception as e:
            raise ValueError(
                f"Could not read configuration file {self.config_path}: {e}"
            )

        # Use VLLMConfig to check if the server exists (handles both flat and nested keys)
        # Pass the raw servers dictionary for more reliable nested lookup
        vllm_config = VLLMConfig(servers, name)
        if not vllm_config.data:
            available = list(servers.keys()) if isinstance(servers, dict) else []
            available_str = ", ".join(available) if available else "None"
            raise ValueError(
                f"Could not resolve configuration for service '{name}'.\n"
                f"Available servers in config: {available_str}\n"
                f"If no servers are listed, run 'cmc launch init server <host>' first."
            )

        config = vllm_config.data
        target_host = config.get("host")
        platform = config.get("platform", "default")

        if not target_host:
            raise ValueError(
                f"Host not specified for service '{name}' in configuration."
            )

        server = get_server(target_host)
        client = VLLMClient(vllm_config)

        console.print(
            banner(
                "Orchestrating vLLM Backend",
                f"Host: {target_host}\nService: {name}\nPlatform: {platform}",
            )
        )

        # 0. VPN Check
        if target_host not in ["localhost", "127.0.0.1"]:
            with StopWatch.timer("vpn_check"):
                console.banner("VPN Connection")
                console.print("[blue]Checking VPN connection...[/blue]")
                vpn = Vpn()
                if not vpn.enabled():
                    console.msg("VPN is disconnected. Attempting to connect...")
                    if not vpn.connect():
                        console.error(
                            "VPN connection failed. Please connect to the VPN and try again."
                        )
                        return False
                    console.ok("VPN connected successfully!")
                else:
                    console.ok("VPN is already active.")
            console.print(
                f"[dim]VPN check took: {StopWatch.get('vpn_check'):.2f}s[/dim]"
            )

        # 1. Initial Health Check
        console.banner("Initial Health Check")
        console.print("[blue]Checking if vLLM server is already available...[/blue]")
        if client.is_alive():
            console.ok("vLLM server is already ALIVE and tunneled!")
            StopWatch.stop("vllm_startup")
            console.print(
                f"[dim]Total startup time: {StopWatch.get('vllm_startup'):.2f}s[/dim]"
            )
            StopWatch.benchmark()
            return True

        # 2. Platform-specific Launch
        with StopWatch.timer("platform_launch"):
            console.banner("Platform Launch")
            process = None
            launch_mode = config.get("launch_mode", "ijob")
            
            if launch_mode == "sbatch":
                if target_host == "uva":
                    result = self.launch_uva(name, port_override=port_override)
                    if not result:
                        return False
                    node_name, process = result
                elif target_host == "dgx":
                    if not self.launch_dgx(name, port_override=port_override):
                        return False
                else:
                    console.error(f"sbatch launch mode requested but host {target_host} is not supported for sbatch.")
                    return False
            else:
                # Default flow: Tunnel then Start (ijob)
                if target_host not in ["localhost", "127.0.0.1"]:
                    console.print("[blue]Establishing SSH tunnel...[/blue]")
                    server.tunnel(name)

                    if client.is_alive():
                        console.ok("vLLM server is now available!")
                        StopWatch.stop("vllm_startup")
                        console.print(
                            f"[dim]Total startup time: {StopWatch.get('vllm_startup'):.2f}s[/dim]"
                        )
                        StopWatch.benchmark()
                        return True

                    console.print("[blue]Starting vLLM server on remote host...[/blue]")
                    server.start(name)
                else:
                    console.warning(
                        "Local host detected. Please ensure the vLLM server is started locally."
                    )
        console.print(
            f"[dim]Platform launch took: {StopWatch.get('platform_launch'):.2f}s[/dim]"
        )

        # 3. Final Health Check Poll (Remote check before tunneling)
        if target_host == "uva" or platform == "uva":
            remote_port = port_override or config.get("remote_port", 8000)
            console.banner(f"Waiting for server startup on {node_name}:{remote_port}")
            console.print(f"[blue]Verifying model health on {node_name}...[/blue]")
            remote_port = port_override or config.get("remote_port", 8000)

            # Resolve remote_dir for log checking
            remote_user = config.get("user") or self._get_remote_user(target_host)
            remote_dir = config.get("dir", f"/scratch/{remote_user}")
            if remote_dir:
                remote_dir = remote_dir.replace("{user}", remote_user).replace(
                    "{port}", str(remote_port)
                )

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

            # Get Job ID for log checking
            job_id = config.get("job_id")
            # If not in config, we might need to find it from squeue
            if not job_id:
                try:
                    job_res = subprocess.run(f'ssh uva "squeue -u {remote_user} -h -o %i"', shell=True, capture_output=True, text=True)
                    job_id = job_res.stdout.strip().split('\n')[0] if job_res.stdout.strip() else None
                except Exception:
                    job_id = None

            success = False
            for i in range(120):
                # 1. Stream any available logs from the vLLM process and check for success
                if process:
                    try:
                        while True:
                            line = process.stdout.readline()
                            print(line)
                            if not line:
                                break
                            print(line, end="")
                            # Real-time detection: check if the success message is in the streamed line
                            if any(pattern in line for pattern in ["vLLM is up and running on 0.0.0.0", "Application startup complete"]):
                                app_ready = True
                    except Exception:
                        pass

                # 2. Check if port is open AND application startup is complete in logs.
                # We run the check via the login node to avoid direct SSH authentication issues.

                # Check port
                # Use direct nc from login node to compute node to avoid nested SSH authentication issues
                port_check_cmd = f'ssh uva "nc -z {node_name} {remote_port}"'
                print(port_check_cmd)
                port_res = subprocess.run(port_check_cmd, shell=True, capture_output=True)
                port_open = port_res.returncode == 0

                # Check health endpoint via curl with API key if available
                # Use -f to ensure non-2xx responses return a non-zero exit code
                health_check_cmd = f'ssh uva "curl -s -f {auth_header} http://{node_name}:{remote_port}/health"'
                print(health_check_cmd)
                console.print(f"[dim]Executing health check: {health_check_cmd}[/dim]")
                health_res = subprocess.run(health_check_cmd, shell=True, capture_output=True, text=True)
                
                # Consider it ready if the curl command succeeded (HTTP 200)
                app_ready = health_res.returncode == 0

                # 3. Check Slurm output files for the success message
                if not app_ready:
                    # Search both .out and .err files for the success message
                    # We check for multiple possible success patterns
                    log_patterns = [
                        "vLLM is up and running on 0.0.0.0",
                        "Application startup complete"
                    ]
                    # Create a regex pattern that matches any of the success strings
                    pattern_regex = "|".join(log_patterns)
                    log_check_cmd = f'ssh uva "grep -E -q \'{pattern_regex}\' {remote_dir}/vllm_{remote_port}.out {remote_dir}/vllm_{remote_port}.err 2>/dev/null"'
                    
                    print(log_check_cmd)

                    log_res = subprocess.run(log_check_cmd, shell=True)
                    if log_res.returncode == 0:
                        app_ready = True

                if i % 10 == 0: # Log every 50s to avoid flooding
                    debug_msg = f"[dim]Debug: port_open={port_open}, app_ready={app_ready} (Response: {health_res.stdout.strip()[:50]})[/dim]"
                    console.print(debug_msg)

                if port_open and app_ready:
                    success = True
                    break

                console.print(
                    f"A- Waiting for model to load on remote... ({i+1}/120)"
                )
                time.sleep(5)

            if success:
                # Establish tunnel only after the server is confirmed healthy
                local_port = port_override or config.get("local_port", 8000)
                
                # Check if tunnel is already running to avoid duplicates
                port_in_use = False
                try:
                    check_port = subprocess.run(f"lsof -t -iTCP:{local_port} -sTCP:LISTEN", shell=True, capture_output=True, text=True)
                    if check_port.stdout.strip():
                        port_in_use = True
                except Exception:
                    pass
                
                if not port_in_use:
                    console.print(f"[blue]Server is healthy. Establishing tunnel to {node_name}...[/blue]")
                    self._kill_port_process(local_port)
                    tunnel_cmd = f"ssh -L {local_port}:{node_name}:{remote_port} uva -N"
                    subprocess.Popen(tunnel_cmd, shell=True)
                    console.ok("Tunnel established in background.")

                console.ok(
                    "\n vLLM server is now ALIVE and Application startup complete!"
                )

                StopWatch.stop("vllm_startup")
                console.print(
                    f"[bold green]Total startup time: {StopWatch.get('vllm_startup'):.2f}s[/bold green]"
                )
                StopWatch.benchmark()
                if process:
                    process.terminate()
                return True
            else:
                if process:
                    process.terminate()
                console.error(
                    "\n vLLM server failed to become healthy on remote node within 10 minutes."
                )
                return False
        else:
            # Existing health check for other platforms (which already have tunnels or are local)
            console.print("[blue]Verifying model health...[/blue]")
            for i in range(120):
                if client.is_alive():
                    console.ok("vLLM server is now ALIVE and model is loaded!")
                    return True
                console.print(f"B - Waiting for model to load... ({i+1}/120)", end="\r")
                time.sleep(5)

            console.error("vLLM server failed to become healthy within 10 minutes.")
            return False
