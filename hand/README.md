# vLLM Launch Guide (UVA HPC)

This directory contains the essential scripts and instructions for launching the vLLM backend on the UVA HPC cluster and connecting to it locally.

## Components
- `start_uva.sh`: The shell script executed on the compute node to launch the vLLM server using Apptainer.

## Launch Sequence

### Step 1: Start the Backend (Terminal 1)
Run the orchestrator from your local machine. This command handles VPN connection, submits the job to Slurm, waits for the GPU allocation, verifies the model is loaded, and establishes the SSH tunnel.

```bash
cme launch llm <server_name>
```
*Replace `<server_name>` with the name configured in your `llm.yaml` (e.g., `gemma-uva`).*

**What happens here:**
1. Connects to VPN.
2. Deploys `start_uva.sh` to the remote scratch directory.
3. Submits an `sbatch` job.
4. Polls `squeue` for the allocated compute node.
5. Monitors `sbatch.err` for "Application startup complete."
6. Opens a local SSH tunnel to the compute node.

### Step 2: Start the Client (Terminal 2)
Once Terminal 1 reports that the backend is ready, open a new terminal to launch your preferred interface.

**For Open WebUI:**
```bash
cme launch webui
```

**For Claude Code:**
```bash
cme launch claude
```

**For Aider:**
```bash
cme launch aider
```

## Troubleshooting
- **Image Not Found**: Ensure the `.sif` image is located at `/scratch/{user}/vllm_gemma4.sif` or specify the `image` path in your `llm.yaml` server configuration.
- **VPN Issues**: The orchestrator attempts to connect to the VPN automatically, but you can verify your connection manually if it fails.
- **Logs**: You can check the remote logs on the UVA cluster at:
  `/scratch/{user}/cloudmesh/vllm_{port}/sbatch.err`