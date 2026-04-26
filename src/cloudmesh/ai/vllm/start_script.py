from cloudmesh.ai.vllm.config import VLLMConfig
import textwrap

class VLLMStartScript:
    """
    Manages the creation of the shell script used to start a vLLM server.
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.name = config.name

    def generate(self, use_nohup: bool = False) -> str:
        """
        Generate the content of the start script.
        
        Args:
            use_nohup: If True, wraps the execution in nohup to ensure it continues 
                       running after the shell session ends.
        """
        port = self.config.get('port', '8000')
        log_dir = "~/vllm_logs"
        log_file = f"{log_dir}/{self.name}.log"

        # Base command for vLLM
        # Note: We use apptainer for UVA/Rivanna and docker/direct for DGX.
        # The specific runner is determined by the server implementation, 
        # but the script structure remains similar.
        
        # We'll use a placeholder for the runner which will be filled by the Server class
        # or we can detect it from the config group.
        runner = "apptainer run --nv" if self.config.group == "uva" else "python3 -m vllm.entrypoints.openai.api_server"
        
        # If it's DGX, the command might be different. Let's make it flexible.
        # For now, we'll implement the logic based on the current ServerUVA/ServerDGX.
        
        if self.config.group == "uva":
            cmd_body = (
                f"apptainer run --nv {self.config.get('image')} \\\n"
                f"    --model {self.config.get('model')} \\\n"
                f"    --tensor-parallel-size {self.config.get('tensor_parallel_size', 1)} \\\n"
                f"    --gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.90)} \\\n"
                f"    --max-model-len {self.config.get('max_model_len', 16384)} \\\n"
                f"    --port {port} \\\n"
                f"    --cache-dir {self.config.get('cache_dir', '/scratch/$USER/hf_cache')} \\\n"
                f"    --key-dir {self.config.get('key_dir', '$HOME/gemma')}"
            )
        elif self.config.group == "dgx":
            # DGX implementation using Docker
            container_name = f"vllm-dgx-{self.name}"
            hf_token_path = self.config.get('hf_token_path', '$HOME/.config/cloudmesh/ai/keys/hugging.txt')
            api_key_path = self.config.get('api_key_path', '$HOME/.config/cloudmesh/ai/keys/server_master_key.txt')
            
            # Configurable flags with defaults
            prefix_caching = "--enable-prefix-caching" if self.config.get('enable_prefix_caching', True) else ""
            load_format = f"--load-format {self.config.get('load_format', 'safetensors')}"
            auto_tool_choice = "--enable-auto-tool-choice" if self.config.get('enable_auto_tool_choice', True) else ""
            tool_parser = f"--tool-call-parser {self.config.get('tool_call_parser', 'gemma4')}"
            
            # Handle optional extra arguments
            extra_args = self.config.get('extra_args', '')
            
            cmd_body = (
                f"HF_TOKEN=$(cat {hf_token_path})\n"
                f"VLLM_API_KEY=$(cat {api_key_path})\n"
                f"docker run -d --name {container_name} --gpus all \\\n"
                f"    --shm-size 16gb \\\n"
                f"    -v {self.config.get('cache_dir', '$HOME/.cache/huggingface')}:/root/.cache/huggingface \\\n"
                f"    -p 127.0.0.1:{port}:8000 \\\n"
                f"    -e HF_TOKEN=\"${{HF_TOKEN}}\" \\\n"
                f"    -e VLLM_API_KEY=\"${{VLLM_API_KEY}}\" \\\n"
                f"    -e NVIDIA_VISIBLE_DEVICES={self.config.get('gpu_ids', '0,1,2,3')} \\\n"
                f"    {self.config.get('image')} \\\n"
                f"    --model {self.config.get('model')} \\\n"
                f"    --tensor-parallel-size {self.config.get('tensor_parallel_size', 4)} \\\n"
                f"    --gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.95)} \\\n"
                f"    --max-model-len {self.config.get('max_model_len', 131072)} \\\n"
                f"    {prefix_caching} \\\n"
                f"    {load_format} \\\n"
                f"    {auto_tool_choice} \\\n"
                f"    {tool_parser} \\\n"
                f"    {extra_args}"
            )
        else:
            # Fallback
            cmd_body = (
                f"python3 -m vllm.entrypoints.openai.api_server \\\n"
                f"    --model {self.config.get('model')} \\\n"
                f"    --tensor-parallel-size {self.config.get('tensor_parallel_size', 1)} \\\n"
                f"    --gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.90)} \\\n"
                f"    --port {port}"
            )

        if use_nohup:
            execution_cmd = f"nohup {cmd_body} > {log_file} 2>&1 &"
        else:
            execution_cmd = f"{cmd_body} > {log_file} 2>&1"

        script = f"""
        #!/bin/bash
        mkdir -p {log_dir}
        {execution_cmd}
        """
        return textwrap.dedent(script).strip()