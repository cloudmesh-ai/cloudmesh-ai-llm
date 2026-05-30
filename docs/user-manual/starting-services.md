# How to Start Your AI Models

## Quick Start: Launch a Model

If you are a UVA user and want to get a model running immediately, use the following command:

``` bash
# Start the standard Gemma model
cmc llm start uva.gemma
```

------------------------------------------------------------------------

## Before You Start

To ensure a smooth launch, please verify the following:

- **VPN Connection**: If you are accessing remote GPU clusters (like UVA), ensure you are connected to the institutional VPN.
- **SSH Access**: You should have SSH access to the target GPU nodes. We recommend using an SSH key for password-less login.
- **Configuration**: Most users start with the provided defaults. If you need to customize your server, your settings are stored in `~/.config/cloudmesh/llm.yaml`.

## Getting Started

This guide will help you launch Large Language Models (LLMs) on your GPU hardware, from initial setup to a fully running service.

Cloudmesh AI uses the `cmc llm start` command to handle the "heavy lifting." Instead of you having to manually manage remote servers and complex networking, the tool automatically:

### Configuration Reference

Your servers and clients are defined in `~/.config/cloudmesh/llm.yaml`. This file tells the orchestrator which platform to use and how to connect.

#### Server Configuration Example
Below is an example of how to define a server for the UVA HPC and a local DGX node.

```yaml
cloudmesh:
  ai:
    server:
      uva.gemma:
        platform: uva
        host: uva
        user: "your_uva_username"
        local_port: 8000
        remote_port: 8000
        model: "google/gemma-4-31B-it"
        tensor_parallel_size: 4
        gpu_memory_utilization: 0.90
      
      dgx.llama:
        platform: dgx
        host: dgx-node-01
        user: "your_dgx_username"
        local_port: 8001
        remote_port: 8001
        model: "meta-llama/Llama-3-70B"
        tensor_parallel_size: 4
```

#### Client Configuration Example
You can also define AI clients (like Aider or Open WebUI) so they automatically connect to your running backends.

```yaml
cloudmesh:
  ai:
    client:
      aider:
        OPENAI_API_BASE: http://localhost:8000/v1
        OPENAI_API_KEY: "your-server-key"
        model: "google/gemma-4-31B-it"
        launcher: aider
      openwebui:
        OPENAI_API_BASE: http://localhost:8000/v1
        OPENAI_API_KEY: "your-server-key"
        port: 3000
        launcher: webui
```

---

### Customizing Your Deployment (The Export Feature)

If you need to change vLLM arguments (e.g., `--gpu-memory-utilization` or `--max-model-len`) that are not covered by the standard YAML config, you can customize the launch scripts directly.

1.  **Export the scripts**:
    ```bash
    cmc llm start uva.gemma --export
    ```
    This saves the `start_uva.sh` and configuration files to your current local directory.

2.  **Edit the script**:
    Open `start_uva.sh` and modify the vLLM flags (e.g., change `--max-model-len 16384` to `32768`).

3.  **Launch**:
    Run the start command again. The orchestrator will detect your local modified script and upload it to the remote node instead of using the default template.
    ```bash
    cmc llm start uva.gemma
    ```

---

1.  **Connects**: It sets up a secure "tunnel" so your laptop can talk to the remote GPU as if it were running locally.
2.  **Deploys**: It uploads the necessary start scripts to the GPU node.
3.  **Manages**: It handles the remote process lifecycles (starting and stopping).
4.  **Verifies**: It polls the model's health to tell you exactly when it's ready for use.

**Visual Feedback:** During the launch process, you will see dynamic visual loading indicators (such as spinners and progress bars) in your terminal, providing real-time feedback on the deployment and initialization stages.

### Which model should I use: Gemma or Gemma 2?

We provide two main versions of the Gemma model on the UVA infrastructure:

- **`uva.gemma`**: The standard version. It is typically faster and requires fewer resources, making it ideal for simpler tasks or quick testing.
- **`uva.gemma2`**: The next-generation version. It offers significantly better reasoning capabilities, higher accuracy, and improved performance on complex tasks. **Choose this if you need the highest quality responses.**

To launch your chosen model, run:

``` bash
# For the standard version:
cmc llm start uva.gemma

# For the high-performance version:
cmc llm start uva.gemma2
```

### Configuration Templates

To avoid manual editing of `~/.config/cloudmesh/llm.yaml`, you can use pre-defined configuration templates for popular models.

**1. List available templates:**

``` bash
cmc llm template
```

**2. Apply a template:**

``` bash
cmc llm template gemma
```

This command merges the template settings into your configuration file while preserving your existing user-specific settings (like SSH keys or custom ports).

The `start` command will:

1.  Resolve the server identity from your configuration.
2.  Establish an SSH tunnel to the remote GPU node.
3.  Launch the vLLM server using the pre-defined template.
4.  Poll the health endpoint until the model is ready.

**Validation Workflow:** The tool now prioritizes `cmc llm status` as the primary validation mechanism. It automatically checks the health of the instance and ensures the node is correctly allocated before marking the service as "Ready."

## Hardware-Specific Deployment

Depending on your available hardware, you may need specific configurations to optimize performance.

### RTX 3090 (Consumer GPUs)

For deployments on single RTX 3090 cards, focus on memory utilization and quantization to avoid Out-Of-Memory (OOM) errors.

- **Memory Optimization**: Set `gpu_memory_utilization` to `0.9` or lower in your `llm.yaml`.

- **Quantization**: Use AWQ or GPTQ quantized models to fit larger models into 24GB VRAM.

- **Launch Command**:

  ``` bash
  cmc llm start my-rtx3090-server
  ```

### NVIDIA Spark (Cluster/HPC)

When deploying on NVIDIA Spark clusters or high-performance computing environments:

- **Resource Allocation**: Ensure your `allocation.yaml` correctly specifies the number of GPUs.

- **Launch Mode**: Use `sbatch` or `ijob` as configured in the server identity to handle job scheduling.

- **Launch Command**:

  ``` bash
  cmc llm start spark-cluster-node
  ```

## Validation & Testing

Once the service is reported as "Ready", you should verify the API is functioning correctly.

### Quick Testing with Curl

Use these commands from your local machine (where the tunnel is active).

**1. Verify Models Endpoint** Check if the vLLM server is listing the loaded model:

``` bash
curl http://localhost:8000/v1/models
```

**2. Send a Completion Request** Test a simple prompt to ensure the model generates text:

``` bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2b-it",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```


------------------------------------------------------------------------

## Next Steps: Monitoring Your Service

Once your model is running and validated, you should monitor its resource usage and API health to ensure stability.

For detailed instructions on using Grafana dashboards, checking service status, and streaming logs, please refer to the [Monitoring Guide](monitoring.md).