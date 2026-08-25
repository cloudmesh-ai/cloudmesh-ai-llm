# Troubleshooting Guide

This guide provides solutions to the most common issues encountered when deploying and monitoring vLLM servers using the Cloudmesh AI LLM Orchestrator.

### Quick Health Check: `cmc llm status`

Before diving into specific errors, the first step in troubleshooting should always be to check the overall health of your services. The `cmc llm status` command provides a high-level summary of all active instances, their health, and their node allocations.

  cmc llm status

**Example Output:**

| Server | Health | Node | Port | Tunnel |
| :--- | :--- | :--- | :--- | :--- |
| `uva.gemma` | 🟢 READY | `gpu-node-01` | 8000 | 🟢 Active |
| `uva.gemma2` | 🟡 STARTING | `gpu-node-05` | 8001 | 🟢 Active |
| `my-test` | 🔴 OFFLINE | `gpu-node-12` | 8002 | 🔴 Inactive |

*   **🟢 READY**: The service is fully operational.
*   **🟡 STARTING**: The model is still loading into VRAM; please wait.
*   **🔴 OFFLINE**: The process has crashed or is unreachable.

---

## 1. Observability Stack (Prometheus & Grafana)

### "Dashboard not found" or "404" when opening Grafana

If you can access the Grafana UI but the "vLLM Server Metrics" dashboard is missing:

*  **Cause**: This is typically caused by stale provisioning state in the Grafana internal database or a mismatch between the provider config and the JSON file location.
*  **Solution**: Stop the stack and remove all associated Docker volumes to ensure a clean state:

        cmc llm monitor stop-stack
        # If problems persist, manually prune volumes:
        docker volume prune

  Then, relaunch using 
  
      cmc llm monitor stack
      
  The tool now performs automated API verification to ensure the dashboard is registered.

### "No data" in the dashboard panels

If the dashboard loads but the charts are empty:

*  **Check Metrics Endpoint**: Verify that vLLM is actually exporting metrics. Run:
  
        curl http://localhost:<your-port>/metrics
  
  You should see a long list of text starting with `# HELP`.

*  **Network Resolution**: On macOS, Docker containers must use `host.docker.internal` to reach the host. The orchestrator configures this automatically, but ensure your Docker Desktop version is up to date.

*  **Prometheus Target**: Check the Prometheus UI (`http://localhost:9090`) &rarr; **Status** &rarr; **Targets**. If the vLLM target is `DOWN`, the issue is network connectivity between the container and the host.

### Grafana UI is unreachable (Connection Refused)

*   **Port Conflict**: Another service may be using port 3000.
*   **Solution**: The `monitor stack` command automatically detects port conflicts and will suggest an alternative port (e.g., 3001). Check the terminal output for the actual port being used.

---

## 2. Connectivity & Tunneling

### "Failed to establish SSH tunnel"

*   **VPN Check**: Ensure you are connected to the UVA or DGX VPN.
*   **SSH Key**: Verify that your SSH key is added to the `ssh-agent`:

        ssh-add -l

*   **Host Verification**: Try connecting to the remote node manually via SSH to ensure there are no fingerprint prompts blocking the automated tunnel.

### "Port already in use"

If you are running multiple models or have a stale tunnel:

*   **Solution**: Use the `--port` flag to specify a unique local and remote port:
  
        cmc llm start my-server --port 18222
  
---

## 3. vLLM Engine & Model Errors

### Out of Memory (OOM) during loading

If the logs show `torch.cuda.OutOfMemoryError`, try the following:

*   **Reduce Memory Utilization**: In your `llm.yaml`, lower the `gpu_memory_utilization` (e.g., from `0.9` to `0.7`).
*   **Quantization**: Use a quantized version of the model (e.g., AWQ or GPTQ) to reduce VRAM footprint.
*   **Max Model Length**: Reduce `--max-model-len` in your launch script to save KV cache space.

### "Model not found" or HuggingFace 401/403

*   **HF Token**: Ensure your `HF_TOKEN` is correctly set in your environment or the launch script.
*   **Model Path**: Verify that the model name matches the HuggingFace repository exactly.

---

## 4. Infrastructure Issues (UVA/DGX)

### Manual Recovery Workflow (Power User)

If the automated `cmc llm start` pipeline fails, you can manually recover your service by following these steps:

1.  **Manually Request Allocation**:
    Request a GPU node via the cluster scheduler (e.g., UVA Rivanna):

        ssh -tt uva "/opt/rci/bin/ijob -A bii_dsc_community -p bii-gpu --gres=gpu:a100:4"
    
2.  **Manual Launch**:
    Once on the node, navigate to your scratch directory and run your launch script:
    
        cd /scratch/${USER}
        ./gemma.sh
    
3.  **Manual Tunneling**:
    From your **local machine**, establish the SSH tunnel using the allocated `<node-id>`:
    
        ssh -L 8000:<node-id>:8000 ${USER}@rivanna.itc.virginia.edu
    
4.  **Sync State**:
    Once manually running, you can use `cmc llm status` to verify the endpoint is reachable.

### Slurm Allocation Timeout

If the `start` command hangs during "Waiting for allocation":

*   **Queue Status**: Check the Slurm queue manually using 

        squeue -u <your-user>
        
*   **Request Size**: If the cluster is full, try requesting fewer GPUs or a different partition.

### VPN Disconnection

If the server was running but suddenly becomes unreachable:

*   **Re-connect VPN**: Re-establish your institutional VPN connection.
*   **Restart Tunnel**: You don't need to restart the whole server; just run the `start` command again to re-establish the tunnel.
