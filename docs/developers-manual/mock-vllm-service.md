# Mock vLLM Server

The Mock vLLM server is a lightweight FastAPI implementation that mimics the behavior of a real vLLM API server. It is designed to provide a reliable way to test the end-to-end deployment pipeline without requiring expensive GPU resources or enduring the long startup times associated with loading large LLMs.

## Why a Mock Service?

Deploying a vLLM server on HPC clusters involves several complex steps:

- **Slurm Allocation**: Requesting GPUs and CPU resources.
- **Deployment**: Uploading scripts and images to remote scratch space.
- **Execution**: Running the server inside an Apptainer/Singularity container.
- **Health Monitoring**: Polling logs for specific "startup complete" strings.
- **Tunneling**: Establishing SSH tunnels from the compute node to the local machine.

The mock service allows developers to verify that this entire **Orchestration Lifecycle** is working correctly. If the mock server can be launched, detected as healthy, and tunneled, then the infrastructure is ready for a real model.

## Features

- **API Compatibility**: Implements `/v1/models` and `/v1/chat/completions` to satisfy basic client requests.
- **Orchestrator Integration**: Uses the same `VLLMOrchestrator` as production servers, ensuring the deployment logic is identical.
- **Log Mimicry**: Produces the exact log signatures (e.g., `"Application startup complete"`) that the orchestrator uses for health checks.
- **Rapid Feedback**: Starts nearly instantaneously compared to real models.

## How to Use

### 1. Configuration
Ensure you have a server defined in your `llm.yaml` that is intended for mocking. The orchestrator will use the defined `image` (e.g., `mock.sif`) to launch the service.

### 2. Launching the Mock Server
Use the `mock start` command followed by the name of the server configuration:

```bash
cmc mock start <server_name>
```

Example:
```bash
cmc mock start gemma-mock
```

### 3. Overriding Ports
If you need to use a specific port for testing, you can provide a port override:

```bash
cmc mock start <server_name> --port 18000
```

### 4. Verifying the Server
Once launched, you can check the status of the mock server using the standard server list command:

```bash
cmc server list
```

You should see the mock server marked as `🟢 READY` with an `🟢 Active` tunnel.

### 5. Testing the API
You can verify the mock server is responding by calling the API locally:

```bash
curl http://localhost:<local_port>/v1/models
```

## Summary of Lifecycle
When you run `cmc mock start`, the following happens:

1. **Prepare**: The orchestrator resolves configuration and prepares the remote directory.
2. **Deploy**: The `mock_server.py` and `start_uva.sh` are uploaded to the HPC cluster.
3. **Allocate**: An `sbatch` job is submitted to Slurm.
4. **Poll**: The orchestrator monitors the `.out` and `.err` logs for the "Application startup complete" message.
5. **Tunnel**: An SSH tunnel is established from the allocated compute node to your local machine.
6. **Ready**: The service is available for API calls.

## Example YAML Configuration

```yaml
cloudmesh:
  ai:
    server:
      uva:
        mock:
          image: "/scratch/{user}/mock/mock.sif"
          launch_mode: "sbatch"
          account: "bii_dsc_community"
          partition: "bii-gpu"
          reservation: "bi_fox_dgx"
          cpus: 2
          gpus: 1
          mem: "32G"
          host: "uva"
          user: '{~/.ssh/config:uva.user}'
          remote_dir: "/scratch/{user}/cloudmesh/mock"
          copy:
            src: "cloudmesh-ai-llm/src/cloudmesh/ai/mock/mock_server.py"
            dst: "{remote_dir}/mock_server.py"
          tunnel: |
            lsof -ti:{local_port} | xargs kill -9
            ssh -L {local_port}:udc-an26-1:{remote_port} uva -N
          submit_script: |
            #!/bin/bash
            #SBATCH --job-name=vllm_mock_{remote_port}
            #SBATCH --partition={partition}
            #SBATCH --reservation={reservation}
            #SBATCH --account={account}
            #SBATCH --gres=gpu:a100:{gpus}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --mem={mem}
            #SBATCH --time=03:00:00
            #SBATCH --output={remote_dir}/vllm_mock_{remote_port}.out
            #SBATCH --error={remote_dir}/vllm_mock_{remote_port}.err
            
            cd {remote_dir}
            ls -la
            bash start_uva.sh
          script: |
            module load apptainer
            apptainer run --nv \
              -B {remote_dir}:/app \
              -W /app \
              {image} \
              --port {remote_port}
```

The location of the mock service is in 

cloudmesh-ai-llm/src/cloudmesh