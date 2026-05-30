# Monitoring Your vLLM Services

Once your vLLM server is running, you can monitor its health and performance using a variety of tools provided by the `cmc llm` command suite.

## 1. Infrastructure & Resource Monitoring (UVA Slurm)

If you are running your model on UVA infrastructure, you can track GPU utilization, memory usage, and CPU load via the UVA Grafana dashboards.

### Open Grafana Dashboard
The `monitor slurm` command automatically resolves your current Job ID and Node name and opens the correct dashboard in your browser.

```bash
# Monitor the default server
cmc llm monitor slurm

# Monitor a specific named server
cmc llm monitor slurm my-server-name

# Find the job based on the local port you are using
cmc llm monitor slurm --port 8001
```

**What to look for in Grafana:**
- **GPU Utilization**: Ensure the GPU is active during inference.
- **VRAM Usage**: Monitor for "Out of Memory" (OOM) patterns.
- **Node Health**: Check for CPU spikes or network bottlenecks on the allocated node.

---

## 2. Service Connectivity & Health

To verify that your server is reachable, the SSH tunnel is functioning, and the model is fully loaded, use the `status` command.

```bash
# List all running servers and their health in a summary table
cmc llm status

# Check detailed status for a specific server
cmc llm status my-server-name
```

### Server Status Summary (Rich Summary Table)
When run without arguments, `cmc llm status` provides a high-visibility **Rich Summary Table** of all active vLLM instances. This table uses color-coding and structured formatting to allow for instant health assessment of your LLM fleet.

**Table Columns & Visual Indicators:**
- **Server**: The name of the configured server.
- **Health**: The current operational state of the vLLM engine, highlighted with color-coded status indicators:
  - 🟢 **READY**: The model is fully loaded into VRAM and is serving requests.
  - 🟡 **STARTING**: The process is running, but the model is still loading or initializing.
  - 🔴 **OFFLINE**: The server is unreachable or the process has crashed.
- **Node**: The specific GPU compute node (e.g., `gpu-node-01`) where the model is currently allocated. This helps quickly identify which physical hardware is being utilized.
- **Port**: The remote port the server is listening on.
- **Tunnel**: Connectivity status between your local machine and the remote node:
  - 🟢 **Active**: Local port is successfully mapped.
  - 🔴 **Inactive**: Tunnel is down; you cannot reach the API locally.

### Detailed Server View
Providing a server name gives a focused view of that specific instance, including the host address and allocated node name.

---

## 3. Real-time Engine Logs

For deep debugging or monitoring the model's loading sequence, you can stream the server logs directly from the remote host.

```bash
# Stream logs in real-time
cmc llm logs my-server-name --follow

# Search for specific errors or keywords in the logs
cmc llm logs my-server-name --grep "Error"
```

**Common log events to watch for:**
- `vLLM engine started`: Confirms the model is fully loaded into VRAM.
- `HTTP request received`: Confirms the server is processing incoming prompts.

---

## 4. Full Observability Stack (Prometheus & Grafana)

For professional-grade monitoring with historical data and complex charts, you can launch a dedicated observability stack. This involves running Prometheus to collect metrics from vLLM and Grafana to visualize them.

### Launch the Stack
The `monitor stack` command provides a fully automated observability setup. It autodiscovers your vLLM server port, configures the Prometheus scraper, and provisions the Grafana data source and dashboards automatically using Docker Compose.

```bash
# Launch stack (autodetects active vLLM port)
cmc llm monitor stack

# Launch stack for a specific tunneled port
cmc llm monitor stack --port 18222
```

**What happens automatically:**
- **Port Discovery**: The tool searches for active vLLM metrics endpoints on your host.
- **Zero-Config Setup**: Prometheus is configured to scrape your server, and the "vLLM Server Metrics" dashboard is automatically provisioned.
- **Auto-Login**: Grafana is launched with anonymous admin access enabled.
- **Verification**: The tool verifies that the dashboard is correctly registered in the Grafana API before opening your browser.

**Accessing the Dashboard:**
The tool will automatically open your browser to the metrics dashboard. If you need to access it manually, use:
`http://localhost:3000/d/vllm-metrics` (Note: Port may vary if 3000 is in use).

### Stop the Stack
```bash
cmc llm monitor stop-stack
```

---

## 5. Lightweight API Monitoring

If you don't need a full stack, you can monitor real-time vLLM engine metrics directly in your terminal. For a detailed breakdown of the metrics being tracked, see the [vLLM Engine Metrics API Reference](metrics-api.md).

```bash
# Monitor the default server
cmc llm monitor llm

# Monitor a specific server
cmc llm monitor llm my-server-name

# Monitor by local port (even if not in configuration)
cmc llm monitor llm --port 18222

# Launch a real-time dashboard UI
cmc llm monitor llm --page

# Change refresh interval (default is 1s)
cmc llm monitor llm --interval 2
```

**Metrics Tracked:**
- **Total Tokens**: Combined count of prompt and generation tokens.
- **Throughput**: Real-time tokens per second (TPS).
- **Running Requests**: Number of requests currently being processed by the GPU.
- **Swapped Requests**: Requests currently swapped to CPU memory (indicates VRAM pressure).
- **GPU KV Cache Usage**: Percentage of the allocated KV cache currently in use.
