# Cloudmesh AI LLM Orchestrator

The Cloudmesh AI LLM Orchestrator provides a unified interface to launch, manage, and connect to Large Language Model (LLM) backends across diverse computing environments, including local machines, DGX clusters, and UVA HPC.

## Features

- **Unified Launch**: A single command to handle the entire pipeline from infrastructure allocation to health checks.
- **Platform-Aware Routing**: Specialized workflows for different platforms (UVA, DGX, Default).
- **Automated HPC Pipeline**: For UVA, it automates VPN connection, `ijob` GPU allocation, dynamic SSH tunneling, and Apptainer execution.
- **Hybrid Script-Driven Approach**: Uses shell scripts for the actual vLLM launch, allowing maximum flexibility for GPU flags and mount points.
- **Local Export & Customization**: Export launch scripts locally, modify them, and the orchestrator will use your customized versions.

---

## Configuration

Servers are defined in `~/.config/cloudmesh/llm.yaml`. The key field is `platform`, which determines the launch strategy.

### 1. UVA HPC Example
For UVA, the orchestrator handles the `ijob` allocation and dynamic tunneling.

```yaml
cloudmesh:
  ai:
    server:
      gemma-uva:
        platform: uva
        host: uva
        user: "{~/.ssh/config:uva.user}" # Optional: resolves from ~/.ssh/config
        dir: "/scratch/{user}/cloudmesh/vllm/{port}" # Optional: {user} and {port} are replaced at runtime. Defaults to /scratch/${USER}/cloudmesh/vllm_{port}
        local_port: 18123
        remote_port: 18123
        model: "google/gemma-4-31B-it"
```

### 2. DGX Example
For DGX, the orchestrator ensures VPN connectivity and executes the launch script.

```yaml
cloudmesh:
  ai:
    server:
      gemma-dgx:
        platform: dgx
        host: dgx
        user: "{~/.ssh/config:dgx.user}" # Optional: resolves from ~/.ssh/config
        dir: "/raid/{user}/cloudmesh/vllm/{port}" # Optional: {user} and {port} are replaced at runtime. Defaults to /raid/${USER}/cloudmesh/vllm_{port}
        local_port: 8000
        remote_port: 8000
        model: "google/gemma-4-31B-it"
```

---

## Workflows

### The UVA Pipeline
When you run `cmc launch llm gemma-uva`, the following happens:
1. **VPN**: Checks and connects to the UVA VPN.
2. **Allocation**: Requests a GPU allocation via `ijob` (e.g., 4x A100).
3. **Node Capture**: Identifies the allocated compute node.
4. **Deployment**: Uploads `start_uva.sh` to the node.
5. **Execution**: Runs the script via Apptainer to start the vLLM server.
6. **Health Check**: Verifies the server is alive on the remote node via SSH.
7. **Dynamic Tunnel**: Establishes an SSH tunnel from your local port to the allocated node in a separate background process once the server is ready.

### The DGX Pipeline
When you run `cmc launch llm gemma-dgx`:
1. **VPN**: Ensures the VPN is active.
2. **Execution**: Runs the `start_dgx.sh` script (typically using Docker).
3. **Health Check**: Polls the API until the model is fully loaded.

---

## Customizing Launch Scripts (The Export Feature)

If you need to change vLLM arguments (e.g., `--gpu-memory-utilization` or `--max-model-len`), you don't need to change the Python code.

### 1. Export the scripts
```bash
cmc launch llm gemma-uva --export
```
This will save `start_uva.sh` and `gemma-uva_config.yaml` to your current directory.

### 2. Edit the script
Open `start_uva.sh` in your editor and modify the vLLM flags:
```bash
# Example change in start_uva.sh
apptainer run --nv \
  ... \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768 \
  ...
```

### 3. Launch
Run the launch command again. The orchestrator will detect the local `start_uva.sh` and upload **your modified version** to the remote machine instead of the default template.
```bash
cmc launch llm gemma-uva
```

---

## Command Reference

| Command | Description |
| :--- | :--- |
| `cmc launch llm <name>` | Full pipeline: VPN $\rightarrow$ Launch $\rightarrow$ Tunnel $\rightarrow$ Health Check. |
| `cmc launch llm <name> --export` | Exports the launch scripts and config to the current directory for editing. |
| `cmc launch llm <name> --ui` | Launches the backend and then automatically starts the Open WebUI. |
| `cmc launch llm <name> --claude` | Launches the backend and then starts Claude Code. |
| `cmc launch config info` | Displays the resolved in-memory configuration. |

---

## Appendix: Manual Command Line Workflow

If you want to launch a model on a specific port (e.g., `18124`) without modifying your permanent configuration file, follow these steps:

### 1. Initialize Configuration
Initialize the server configuration in your `llm.yaml` file using a host from your SSH config (e.g., `uva` or `dgx`):
```bash
cmc launch init server uva
```

You can verify the configuration was added correctly by viewing the file:
```bash
cat ~/.config/cloudmesh/llm.yaml
```

### 2. Launch with Port Override
Run the launch command with the `--port` flag. This will override both the local and remote ports to `18124` and create a unique remote directory for this instance.
```bash
cmc launch llm gemma-uva --port 18124
```

### 3. What happens under the hood:
- **Allocation**: The orchestrator requests a GPU node via `ijob`.
- **Deployment**: It creates a directory like `/scratch/{user}/cloudmesh/vllm_18124` on the allocated node.
- **Execution**: It starts the vLLM server on remote port `18124`.
- **Health Check**: It polls the remote node until port `18124` is open.
- **Tunneling**: It establishes a background SSH tunnel: `localhost:18124` $\rightarrow$ `node:18124`.

### 4. Verify and Connect
Once the command reports "Backend is ready!", you can verify the tunnel is active:
```bash
curl http://localhost:18124/v1/models
```
You can then point your UI or Claude Code to `http://localhost:18124/v1`.
