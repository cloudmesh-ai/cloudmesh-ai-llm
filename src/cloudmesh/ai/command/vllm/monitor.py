import click
import subprocess
import time
import socket
from pathlib import Path
from rich.live import Live
from rich.table import Table
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
from cloudmesh.ai.vllm.orchestrator import VLLMOrchestrator
from cloudmesh.ai.vllm.client import VLLMClient
from cloudmesh.ai.command.vllm.ui import LLMMonitorApp

def get_grafana_url(job_id, node):
    """Construct the UVA Grafana monitoring URL for a specific job."""
    if not job_id or not node:
        return None
    # Template provided by user
    return f"https://grafana.pods.uvarc.io/d/HRLkiLS7k/single-job-stats-input-jobid?orgId=1&theme=light&from=now-3h&to=now&var-node={node}&var-JobID={job_id}"

@click.group(name="monitor")
def monitor_group():
    """Monitoring tools for vLLM servers."""
    pass

@monitor_group.command(name="slurm")
@click.argument("name", required=False)
@click.option("--port", type=int, help="Find the job by its local port")
@click.pass_context
def monitor_slurm(ctx, name, port):
    """Open the Grafana monitoring dashboard for the running vLLM Slurm job."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        
        # Determine which server to monitor
        state = orchestrator._load_state()
        target_server = None

        # 1. Try identifying by port
        if port:
            for s_name, s_state in state.items():
                # Use string comparison to avoid type mismatch (int vs str)
                if s_state and str(s_state.get("local_port")) == str(port):
                    target_server = s_name
                    break
            
            # If not found in state, try dynamic resolution via Slurm
            if not target_server:
                console.print(f"[dim]Port {port} not found in state, attempting dynamic resolution via Slurm...[/dim]")
                resolved = orchestrator.resolve_job_by_port(port)
                if resolved:
                    # Create a synthetic server name and add it to a temporary state
                    target_server = f"port-{port}"
                    state[target_server] = {
                        "job_id": resolved.get("job_id"),
                        "node_name": resolved.get("node_name"),
                        "local_port": port
                    }
                    console.ok(f"Dynamically resolved Job: {resolved.get('job_id')} on Node: {resolved.get('node_name')}")
                else:
                    console.error(f"No running server found using local port {port} in state or Slurm.")
                    return

        # 2. Try identifying by name
        if not target_server and name:
            target_server = name

        # 3. Fallback to default or automatic discovery
        if not target_server:
            default_server = orchestrator.config.get("cloudmesh.ai.default.server")
            running_servers = [s_name for s_name, s_state in state.items() if s_state and "job_id" in s_state]
            
            if not running_servers:
                console.error("No server specified, no port provided, no default server configured, and no running servers found in state.")
                return

            if default_server and default_server in running_servers:
                target_server = default_server
            elif len(running_servers) == 1:
                target_server = running_servers[0]
            else:
                # List all running vLLM jobs on the system, regardless of persisted state
                from cloudmesh.ai.vllm.squeue import SQueue
                sq = SQueue()
                all_jobs = sq.get_jobs()
                
                if not all_jobs:
                    console.error("No running vLLM jobs found on Slurm.")
                    return

                # Map job_id to server_name from state for better display
                job_to_server = {}
                for s_name, s_state in state.items():
                    if s_state and "job_id" in s_state:
                        job_to_server[str(s_state["job_id"])] = s_name

                # We only list jobs that look like vLLM jobs
                vllm_jobs = [j for j in all_jobs if "vllm" in j.get("name", "").lower()]
                
                if not vllm_jobs:
                    console.error("No running jobs with 'vllm' in the name found on Slurm.")
                    return

                if len(vllm_jobs) == 1:
                    # Use server name from state if available, otherwise generate one
                    job = vllm_jobs[0]
                    jid = str(job.get("job_id"))
                    target_server = job_to_server.get(jid, f"vllm-job-{jid}")
                    
                    # If it's a synthetic name, we must ensure it's in state so the 
                    # subsequent URL logic can find the job_id and node.
                    if target_server not in state:
                        nodes = job.get("nodes", [])
                        node_name = "Unknown"
                        if nodes and isinstance(nodes, list) and isinstance(nodes[0], dict):
                            node_name = nodes[0].get("name", "Unknown")
                            if "," in node_name:
                                node_name = node_name.split(",")[0]
                        
                        state[target_server] = {
                            "job_id": jid,
                            "node_name": node_name
                        }
                else:
                    console.print("\n[bold blue]Multiple running vLLM servers found on Slurm:[/bold blue]")
                    for idx, job in enumerate(vllm_jobs, 1):
                        jid = str(job.get("job_id"))
                        s_name = job_to_server.get(jid, f"vllm-job-{jid}")
                        
                        # Resolve details
                        job_name = job.get("name", "Unknown")
                        
                        # Resolve Port from job name or state
                        s_state = state.get(s_name) if s_name in state else {}
                        port = s_state.get("local_port") if s_state else None
                        if not port:
                            import re
                            match = re.search(r"_(\d+)$", job_name)
                            port = match.group(1) if match else "Unknown"
                        else:
                            port = port or "Unknown"
                        
                        # Resolve Node
                        nodes = job.get("nodes", [])
                        node_name = "Unknown"
                        if nodes and isinstance(nodes, list) and isinstance(nodes[0], dict):
                            node_name = nodes[0].get("name", "Unknown")
                            if "," in node_name:
                                node_name = node_name.split(",")[0]
                        
                        console.print(f"{idx}. {s_name} (Job Name: {job_name}, Port: {port}, Job: {jid}, Node: {node_name})")
                    
                    console.print("\n[dim]Please specify a server name or use --port to monitor a specific job.[/dim]")
                    return

        # Retrieve persisted state for the server
        server_state = state.get(target_server)
        
        if not server_state or "job_id" not in server_state:
            console.error(f"No active job found for server '{target_server}'. Please start the server first.")
            return

        job_id = server_state.get("job_id")
        node = server_state.get("node_name")
        
        url = get_grafana_url(job_id, node)
        if url:
            console.ok(f"Opening Grafana monitoring for {target_server} (Job: {job_id}, Node: {node})...")
            if os_is_mac():
                subprocess.run(["open", url])
            else:
                import webbrowser
                webbrowser.open(url)
        else:
            console.error("Could not construct Grafana URL.")
            
    except Exception as e:
        console.error(f"Error opening Slurm monitor: {e}")

def is_port_in_use(port):
    """Check if a port is currently in use on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port):
    """Find the next available port starting from start_port."""
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

@monitor_group.command(name="stack")
@click.option("--port", type=int, help="Local port of the vLLM server to scrape (autodiscovered if not provided)")
@click.option("--grafana-port", type=int, default=3000, help="Port to expose Grafana on the host")
@click.option("--prometheus-port", type=int, default=9090, help="Port to expose Prometheus on the host")
@click.pass_context
def monitor_stack(ctx, port, grafana_port, prometheus_port):
    """Launch a real observability stack (Prometheus & Grafana) using Docker."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        
        # 1. Port Autodiscovery
        final_port = port
        if not final_port:
            console.print("[dim]Attempting to autodiscover vLLM server port...[/dim]")
            state = orchestrator._load_state()
            
            # Collect all potential ports to check
            potential_ports = []
            
            # 1. Try default server from config
            default_server = orchestrator.config.get("cloudmesh.ai.default.server")
            if default_server and default_server in state:
                p = state[default_server].get("local_port")
                if p: potential_ports.append(p)

            # 2. Try all running servers in state
            for s_name, s_state in state.items():
                if s_state and "job_id" in s_state:
                    p = s_state.get("local_port")
                    if p: potential_ports.append(p)

            # 3. Try to find any vllm job on Slurm
            from cloudmesh.ai.vllm.squeue import SQueue
            sq = SQueue()
            jobs = sq.get_jobs()
            vllm_jobs = [j for j in jobs if "vllm" in j.get("name", "").lower()]
            for job in vllm_jobs:
                import re
                job_name = job.get("name", "")
                match = re.search(r"_(\d+)$", job_name)
                if match:
                    potential_ports.append(int(match.group(1)))

            # Remove duplicates and sort
            potential_ports = sorted(list(set(potential_ports)))
            
            # Test each potential port to see which one is actually exporting metrics
            import requests
            for p in potential_ports:
                try:
                    resp = requests.get(f"http://localhost:{p}/metrics", timeout=1)
                    if resp.status_code == 200 and "vllm" in resp.text.lower():
                        final_port = p
                        console.ok(f"Autodiscovered active vLLM server on port {final_port}")
                        break
                except:
                    continue

            if not final_port:
                # Last ditch effort: just take the first potential port if we found any
                if potential_ports:
                    final_port = potential_ports[0]
                    console.print(f"[dim]Could not verify metrics, but trying first candidate port {final_port}...[/dim]")

        if not final_port:
            console.error("Could not autodiscover vLLM server port. Please specify it using --port.")
            return
        
        if final_port != port:
            port = final_port

        # Pre-flight check: Verify vLLM metrics are available on the host
        console.print(f"[dim]Verifying vLLM metrics at http://localhost:{port}/metrics...[/dim]")
        try:
            import requests
            resp = requests.get(f"http://localhost:{port}/metrics", timeout=2)
            if resp.status_code != 200 or "vllm" not in resp.text.lower():
                console.error(f"No vLLM metrics found at port {port}. Please ensure vLLM server is running and tunneled.")
                return
            console.ok("vLLM metrics are reachable on the host.")
        except Exception as e:
            console.error(f"Cannot reach vLLM metrics at port {port}: {e}")
            console.error("Please ensure the vLLM server is running and the tunnel is active.")
            return

        # Automatic Port Discovery
        # We only auto-discover if the user didn't explicitly provide a port (i.e., it's the default)
        final_grafana_port = grafana_port
        if grafana_port == 3000 and is_port_in_use(3000):
            final_grafana_port = find_available_port(3000)
            console.warning(f"Port 3000 is in use. Automatically using port {final_grafana_port} for Grafana.")
            
        final_prom_port = prometheus_port
        if prometheus_port == 9090 and is_port_in_use(9090):
            final_prom_port = find_available_port(9090)
            console.warning(f"Port 9090 is in use. Automatically using port {final_prom_port} for Prometheus.")

        stack_dir = Path.home() / ".cloudmesh" / "ai" / "monitoring"
        stack_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create prometheus.yml
        prom_config = f"""
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets:
          - 'host.docker.internal:{port}'
"""
        with open(stack_dir / "prometheus.yml", "w") as f:
            f.write(prom_config)

        # 2. Create Grafana provisioning for data source
        provisioning_dir = stack_dir / "provisioning"
        ds_dir = provisioning_dir / "datasources"
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_config = """
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      httpMethod: POST
      timeInterval: "5s"
"""
        with open(ds_dir / "datasource.yml", "w") as f:
            f.write(ds_config)

        # 3. Create Grafana provisioning for dashboards
        # We separate the provider configuration from the actual dashboard JSON files
        # for a more robust Grafana provisioning.
        dash_prov_dir = provisioning_dir / "dashboards"
        dash_prov_dir.mkdir(parents=True, exist_ok=True)
        
        dash_json_dir = stack_dir / "dashboards"
        dash_json_dir.mkdir(parents=True, exist_ok=True)

        dash_prov_config = """
apiVersion: 1
providers:
  - name: 'vLLM'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
"""
        with open(dash_prov_dir / "vllm_provider.yml", "w") as f:
            f.write(dash_prov_config)

        # Create a basic vLLM dashboard JSON
        vllm_dash_json = """
{
  "annotations": { "list": [] },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "links": [],
  "liveNow": true,
  "panels": [
    {
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "id": 1,
      "title": "Throughput (Tokens/sec)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "irate(vllm_generation_tokens_total[1m]) or irate(vllm:generation_tokens_total[1m])",
          "legendFormat": "Gen Tokens/s"
        },
        {
          "expr": "irate(vllm_prompt_tokens_total[1m]) or irate(vllm:prompt_tokens_total[1m])",
          "legendFormat": "Prompt Tokens/s"
        }
      ]
    },
    {
      "gridPos": { "h": 6, "w": 6, "x": 0, "y": 0 },
      "id": 10,
      "title": "vLLM Status",
      "type": "stat",
      "targets": [
        {
          "expr": "up{job=\\"vllm\\"}",
          "legendFormat": "Status"
        }
      ],
      "options": {
        "colorMode": "background",
        "justifyMode": "center",
        "textMode": "value"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [
            { "type": "value", "options": { "0": { "text": "DOWN", "color": "red" }, "1": { "text": "UP", "color": "green" } } }
          ]
        }
      }
    },
    {
      "gridPos": { "h": 6, "w": 6, "x": 6, "y": 0 },
      "id": 4,
      "title": "GPU Cache Usage",
      "type": "gauge",
      "targets": [
        {
          "expr": "max({__name__=~\\".*kv_cache_usage.*|.*cache_usage_perc.*|.*cache_percentage.*|.*gpu_cache.*\\"}) * 100",
          "legendFormat": "Usage"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "min": 0,
          "max": 100,
          "unit": "percent",
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "yellow", "value": 70 },
              { "color": "red", "value": 90 }
            ]
          }
        }
      }
    },
    {
      "gridPos": { "h": 6, "w": 12, "x": 12, "y": 0 },
      "id": 1,
      "title": "Throughput (Tokens/sec)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "sum(rate({__name__=~\\".*prompt_tokens_total.*\\"}[1m]))",
          "legendFormat": "Prompt Tokens"
        },
        {
          "expr": "sum(rate({__name__=~\\".*generation_tokens_total.*\\"}[1m]))",
          "legendFormat": "Gen Tokens"
        }
      ]
    },
    {
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 6 },
      "id": 3,
      "title": "Latency (Avg ms)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate({__name__=~\\".*time_to_first_token_seconds_sum.*\\"}[1m]) / rate({__name__=~\\".*time_to_first_token_seconds_count.*\\"}[1m]) * 1000",
          "legendFormat": "TTFT (Avg)"
        },
        {
          "expr": "rate({__name__=~\\".*time_per_output_token_seconds_sum.*|.*inter_token_latency_seconds_sum.*\\"}[1m]) / rate({__name__=~\\".*time_per_output_token_seconds_count.*|.*inter_token_latency_seconds_count.*\\"}[1m]) * 1000",
          "legendFormat": "ITL (Avg)"
        }
      ]
    },
    {
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 6 },
      "id": 2,
      "title": "Scheduler (Request Counts)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "max({__name__=~\\".*num_requests_running.*|.*running_requests.*\\"})",
          "legendFormat": "Running"
        },
        {
          "expr": "max({__name__=~\\".*num_requests_waiting.*|.*waiting_requests.*\\"})",
          "legendFormat": "Waiting"
        },
        {
          "expr": "max({__name__=~\\".*num_requests_swapped.*|.*swapped_requests.*\\"})",
          "legendFormat": "Swapped"
        }
      ]
    },
    {
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 14 },
      "id": 6,
      "title": "Token Length (Avg per Request)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate({__name__=~\\".*request_prompt_tokens_sum.*\\"}[1m]) / rate({__name__=~\\".*request_prompt_tokens_count.*\\"}[1m])",
          "legendFormat": "Prompt Length"
        },
        {
          "expr": "rate({__name__=~\\".*request_generation_tokens_sum.*\\"}[1m]) / rate({__name__=~\\".*request_generation_tokens_count.*\\"}[1m])",
          "legendFormat": "Gen Length"
        }
      ]
    },
    {
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 14 },
      "id": 5,
      "title": "E2E Latency (P95 s)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate({__name__=~\\".*e2e_request_latency_seconds_bucket.*\\"}[1m])) by (le))",
          "legendFormat": "P95 E2E"
        }
      ]
    },
    {
      "gridPos": { "h": 10, "w": 24, "x": 0, "y": 22 },
      "id": 7,
      "title": "Raw Metrics (For Discovery)",
      "type": "table",
      "targets": [
        {
          "expr": "{job=\\"vllm\\"}",
          "format": "table",
          "instant": true
        }
      ]
    }
  ],
  "schemaVersion": 38,
  "style": "dark",
  "tags": [],
  "templating": { "list": [] },
  "time": { "from": "now-15m", "to": "now" },
  "timepicker": {
    "refresh_intervals": [ "5s", "10s", "30s", "1m" ]
  },
  "refresh": "5s",
  "timezone": "",
  "title": "vLLM Server Metrics",
  "uid": "vllm-metrics"
}
"""
        with open(dash_json_dir / "vllm.json", "w") as f:
            f.write(vllm_dash_json)
            
        # 4. Create docker-compose.yml
        # We always include host.docker.internal mapping for reliable host access
        compose_content = f"""
services:
  prometheus:
    image: prom/prometheus
    container_name: cmc-vllm-prometheus
    ports:
      - "{final_prom_port}:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    extra_hosts:
      - "host.docker.internal:host-gateway"

  grafana:
    image: grafana/grafana
    container_name: cmc-vllm-grafana
    ports:
      - "{final_grafana_port}:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/vllm.json
    volumes:
      - ./provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./provisioning/dashboards:/etc/grafana/provisioning/dashboards
      - ./dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    extra_hosts:
      - "host.docker.internal:host-gateway"
"""
        with open(stack_dir / "docker-compose.yml", "w") as f:
            f.write(compose_content)
            
        console.print(f"[blue]Launching observability stack for port {port}...[/blue]")
        
        # 3. Run docker compose
        # Ensure a completely clean state by taking down any existing stack AND removing volumes
        subprocess.run(
            ["docker", "compose", "-f", str(stack_dir / "docker-compose.yml"), "down", "-v"], 
            capture_output=True, text=True
        )
        result = subprocess.run(
            ["docker", "compose", "-f", str(stack_dir / "docker-compose.yml"), "up", "-d"], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            if "port is already allocated" in result.stderr:
                console.error(f"Port conflict detected despite auto-discovery. Please specify ports manually.")
            else:
                console.error(f"Docker Compose error: {result.stderr}")
            return
        
        console.ok("Observability stack is running!")
        
        # --- DIAGNOSTIC VERIFICATION ---
        console.print("\n[bold blue]Verifying stack health...[/bold blue]")
        
        # Wait for services to start
        time.sleep(15)
        
        try:
            # 1. Check Grafana API for the dashboard
            console.print("[dim]Checking Grafana dashboard registration...[/dim]")
            api_check = subprocess.run(
                ["docker", "exec", "cmc-vllm-grafana", "sh", "-c", 
                 "curl -s http://localhost:3000/api/search?query=vllm || wget -qO- http://localhost:3000/api/search?query=vllm"],
                capture_output=True, text=True
            )
            if "vllm-metrics" in api_check.stdout:
                console.ok("Dashboard registered in Grafana.")
            else:
                console.error("Dashboard NOT found in Grafana API.")

            # 2. Check Prometheus Targets
            console.print("[dim]Checking Prometheus scrape targets...[/dim]")
            prom_check = subprocess.run(
                ["docker", "exec", "cmc-vllm-prometheus", "sh", "-c", 
                 "wget -qO- http://localhost:9090/api/v1/targets"],
                capture_output=True, text=True
            )
            import json
            try:
                targets_data = json.loads(prom_check.stdout)
                vllm_target = next((t for t in targets_data.get("data", {}).get("activeTargets", []) 
                                  if "vllm" in t.get("scrapePool", "")), None)
                
                if vllm_target:
                    health = vllm_target.get("health", "unknown")
                    last_error = vllm_target.get("lastError", "None")
                    if health == "up":
                        console.ok("Prometheus is successfully scraping vLLM metrics!")
                    else:
                        console.error(f"Prometheus scrape target is {health.upper()}. Error: {last_error}")
                        console.warning(f"Target URL: {vllm_target.get('scrapeUrl')}")
                else:
                    console.error("vLLM target not found in Prometheus.")
            except Exception as json_e:
                console.warning(f"Could not parse Prometheus targets: {prom_check.stdout[:200]}")
                
        except Exception as diag_e:
            console.warning(f"Diagnostics failed: {diag_e}")

        console.print("\n[bold]Next Steps:[/bold]")
        console.print(f"1. Open Grafana: [underline]http://localhost:{final_grafana_port}[/underline]")
        console.print(f"2. Open Prometheus: [underline]http://localhost:{final_prom_port}[/underline] (Check Targets page)")
        
        # Open browser directly to the vLLM dashboard
        url = f"http://localhost:{final_grafana_port}/d/vllm-metrics"
        if os_is_mac():
            subprocess.run(["open", url])
        else:
            import webbrowser
            webbrowser.open(url)

    except Exception as e:
        console.error(f"Error launching observability stack: {e}")

        console.print("\n[bold]Next Steps:[/bold]")
        console.print(f"1. Open Grafana: [underline]http://localhost:{final_grafana_port}[/underline] (Auto-login enabled)")
        console.print(f"2. Data Source and vLLM Dashboard have been automatically configured.")
        console.print(f"3. Check the 'vLLM Server Metrics' dashboard to see real-time charts.")
        
        # Open browser directly to the vLLM dashboard
        url = f"http://localhost:{final_grafana_port}/d/vllm-metrics"
        if os_is_mac():
            subprocess.run(["open", url])
        else:
            import webbrowser
            webbrowser.open(url)

    except Exception as e:
        console.error(f"Error launching observability stack: {e}")

@monitor_group.command(name="stop-stack")
def stop_stack():
    """Stop and remove the observability stack (Prometheus & Grafana)."""
    try:
        stack_dir = Path.home() / ".cloudmesh" / "ai" / "monitoring"
        if not (stack_dir / "docker-compose.yml").exists():
            console.error("No observability stack found to stop.")
            return
            
        subprocess.run(["docker", "compose", "-f", str(stack_dir / "docker-compose.yml"), "down"], check=True)
        console.ok("Observability stack stopped and removed.")
    except Exception as e:
        console.error(f"Error stopping observability stack: {e}")

@monitor_group.command(name="llm")
@click.argument("name", required=False)
@click.option("--port", type=int, help="Find the job by its local port")
@click.option("--interval", default=1, help="Refresh interval in seconds")
@click.option("--chart", is_flag=True, help="Display a visual trend chart of throughput")
@click.option("--page", is_flag=True, help="Launch a real-time dashboard UI")
@click.option("--grafana", is_flag=True, help="Open the Grafana monitoring dashboard")
@click.option("--job-id", help="Manually specify the Slurm Job ID for Grafana")
@click.option("--node", help="Manually specify the Slurm node name for Grafana")
@click.pass_context
def monitor_llm(ctx, name, port, interval, chart, page, grafana, job_id, node):
    """Monitor the vLLM server performance (API metrics)."""
    try:
        debug = ctx.obj.get("debug", False)
        orchestrator = VLLMOrchestrator(debug=debug)
        
        # Determine server to monitor
        state = orchestrator._load_state()
        target_server = None

        # Handle Grafana early to avoid any side effects like starting services
        if grafana:
            # 1. Try identifying by port for Grafana
            if port:
                # Try state first
                for s_name, s_state in state.items():
                    if s_state and s_state.get("local_port") == port:
                        target_server = s_name
                        break
                
                if not target_server:
                    resolved = orchestrator.resolve_job_by_port(port)
                    if resolved:
                        target_server = f"port-{port}"
                        state[target_server] = {
                            "job_id": resolved.get("job_id"),
                            "node_name": resolved.get("node_name"),
                            "local_port": port
                        }
            
            # 2. Try identifying by name for Grafana
            elif name:
                target_server = name

            # 3. Fallback for Grafana
            else:
                target_server = orchestrator.config.get("cloudmesh.ai.default.server")
                if not target_server:
                    running = [s for s, st in state.items() if st and "job_id" in st]
                    if running:
                        target_server = running[0]

            # Resolve Grafana details
            # Prioritize Dynamic Slurm Resolution over Persisted State to avoid stale Job IDs
            if target_server:
                # Try to find the current running job for this server in Slurm
                from cloudmesh.ai.vllm.squeue import SQueue
                sq = SQueue()
                all_jobs = sq.get_jobs()
                
                # Match by state (job_id) or by expected job name
                found_job = None
                s_state = state.get(target_server, {})
                persisted_jid = s_state.get("job_id")
                
                # We also try to guess the job name based on the server config
                server_cfg = orchestrator.config.get_server(target_server)
                remote_port = server_cfg.get("remote_port", 8000) if server_cfg else 8000
                expected_name = orchestrator.get_job_name(server_cfg or orchestrator.config, remote_port)

                for job in all_jobs:
                    jid = str(job.get("job_id"))
                    jname = job.get("name", "")
                    if jid == str(persisted_jid) or expected_name in jname or (jname.startswith("vllm") and jname.endswith(f"_{remote_port}")):
                        found_job = job
                        break
                
                if found_job:
                    job_id = job_id or found_job.get("job_id")
                    nodes = found_job.get("nodes", [])
                    if nodes and isinstance(nodes, list) and isinstance(nodes[0], dict):
                        node = node or nodes[0].get("name", "Unknown").split(",")[0]
            
            # Fallback to persisted state if dynamic resolution didn't work
            if not (job_id and node) and target_server:
                job_id = job_id or state.get(target_server, {}).get("job_id")
                node = node or state.get(target_server, {}).get("node_name")

            # Dynamic resolution by port as final fallback
            if not (job_id and node) and port:
                resolved = orchestrator.resolve_job_by_port(port)
                if resolved:
                    job_id = job_id or resolved.get("job_id")
                    node = node or resolved.get("node_name")

            if job_id and node:
                url = get_grafana_url(job_id, node)
                if url:
                    console.ok(f"Opening Grafana monitoring for {target_server or 'server'} (Job: {job_id}, Node: {node})...")
                    if os_is_mac():
                        subprocess.run(["open", url])
                    else:
                        import webbrowser
                        webbrowser.open(url)
                    return
                else:
                    console.error("Could not construct Grafana URL.")
                    return
            else:
                console.error("Could not resolve Job ID and Node for Grafana. Please provide them via --job-id and --node.")
                return

        # 1. Try identifying by port
        if port:
            for s_name, s_state in state.items():
                if s_state and s_state.get("local_port") == port:
                    target_server = s_name
                    break
            
            # Even if not found in state, we can try to connect to this port directly
            if not target_server:
                console.print(f"[dim]Port {port} not found in state, attempting direct connection...[/dim]")
                # Use localhost explicitly for direct port connection
                client = VLLMClient(orchestrator.config, port=port, host="localhost", debug=debug)
                if client.get_metrics():
                    # We found a monitorable server on this port, proceed using this client
                    target_server = f"port-{port}"
                else:
                    console.error(f"No running server providing metrics found using local port {port}.")
                    return
            else:
                # Found in state, use the configured client. 
                # We pass host="localhost" because we use a tunnel for monitoring.
                client = VLLMClient(orchestrator.config, server_name=target_server, host="localhost", debug=debug)

        # 2. Try identifying by name
        elif name:
            target_server = name
            client = VLLMClient(orchestrator.config, server_name=target_server, host="localhost", debug=debug)

        # 3. Fallback to default or automatic discovery
        else:
            target_server = orchestrator.config.get("cloudmesh.ai.default.server")
            if not target_server:
                running = [s for s, st in state.items() if st and "job_id" in st]
                if not running:
                    console.error("No running server found to monitor.")
                    return
                target_server = running[0]
            
            client = VLLMClient(orchestrator.config, server_name=target_server, host="localhost", debug=debug)

        # Verify we can actually fetch metrics before starting the loop
        if not client.get_metrics():
            console.error(f"Server '{target_server}' is not providing metrics or is unreachable. Ensure the server is running and tunneled.")
            return

        if page:
            # Launch the Textual Dashboard
            app = LLMMonitorApp(client, target_server)
            app.run()
            return

        console.print(f"[bold blue]Monitoring vLLM metrics for {target_server}... (Ctrl+C to stop)[/bold blue]")
        
        prev_tokens = 0
        prev_time = time.time()
        tps_history = []
        MAX_HISTORY = 40

        def get_sparkline(data):
            if not data: return ""
            # Use visible bars starting from the lowest block for better visibility at 0
            chars = "▂▃▄▅▆▇█" 
            minimum = min(data)
            maximum = max(data)
            diff = maximum - minimum
            
            if diff == 0:
                # If all values are 0, use the lowest bar. 
                # If all are constant non-zero, use a proportional bar.
                if maximum == 0:
                    return chars[0] * len(data)
                # Proportional height for constant non-zero values
                # Use a simple log-like scale or just a mid-point
                idx = min(int(maximum / 100), len(chars) - 1) if maximum < 1000 else len(chars) - 1
                return chars[idx] * len(data)
            
            line = ""
            for val in data:
                # Scale value between 0 and len(chars)-1
                idx = int(((val - minimum) / diff) * (len(chars) - 1))
                line += chars[idx]
            return line

        with Live(console=console, refresh_per_second=1) as live:
            while True:
                metrics = client.get_metrics()
                if not metrics:
                    live.update(console.print("[red]Error fetching metrics...[/red]"))
                    time.sleep(interval)
                    continue

                # Extract key metrics
                prompt_tokens = metrics.get("vllm:prompt_tokens_total", 0)
                gen_tokens = metrics.get("vllm:generation_tokens_total", 0)
                running_reqs = metrics.get("vllm:num_requests_running", 0)
                swapped_reqs = metrics.get("vllm:num_requests_swapped", 0)
                cache_usage = metrics.get("vllm:gpu_cache_usage_perc", 0) * 100
                
                # Calculate throughput
                now = time.time()
                dt = now - prev_time
                total_tokens = prompt_tokens + gen_tokens
                tps = (total_tokens - prev_tokens) / dt if prev_tokens > 0 else 0
                prev_tokens = total_tokens
                prev_time = now

                # Update history for chart
                tps_history.append(tps)
                if len(tps_history) > MAX_HISTORY:
                    tps_history.pop(0)

                # Build Table
                table = Table(title=f"vLLM Metrics: {target_server}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("Total Prompt Tokens", f"{int(prompt_tokens):,}")
                table.add_row("Total Gen Tokens", f"{int(gen_tokens):,}")
                table.add_row("Total Tokens", f"{int(total_tokens):,}")
                table.add_row("Throughput", f"{tps:.2f} tokens/sec")
                table.add_row("Running Requests", f"{int(running_reqs)}")
                table.add_row("Swapped Requests", f"{int(swapped_reqs)}")
                table.add_row("GPU KV Cache Usage", f"{cache_usage:.2f}%")

                if chart:
                    spark = get_sparkline(tps_history)
                    table.add_row("Throughput Trend", f"[green]{spark}[/green]")

                live.update(table)
                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[dim]Monitoring stopped.[/dim]")
    except Exception as e:
        console.error(f"Error monitoring LLM: {e}")