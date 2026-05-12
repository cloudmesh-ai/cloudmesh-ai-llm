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
        """
        port = self.config.get('port', '8000')
        log_dir = "~/vllm_logs"
        log_file = f"{log_dir}/{self.name}.log"

        if self.config.group == "uva":
            # Build arguments list for UVA
            args = [
                f"--model {self.config.get('model')}",
                f"--tensor-parallel-size {self.config.get('tensor_parallel_size', 1)}",
                f"--gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.90)}",
                f"--max-model-len {self.config.get('max_model_len', 16384)}",
            ]
            
            # Add conditional max_num_batched_tokens
            max_batched = self.config.get('max_num_batched_tokens')
            if max_batched:
                args.append(f"--max-num-batched-tokens {max_batched}")
            
            # Add extra_args from config
            extra_args = self.config.get('extra_args', '')
            if extra_args:
                for line in extra_args.splitlines():
                    if line.strip():
                        args.append(line.strip())
            
            args.append(f"--port {port}")
            args.append(f"--cache-dir {self.config.get('cache_dir', '/scratch/$USER/hf_cache')}")
            args.append(f"--key-dir {self.config.get('key_dir', '$HOME/gemma')}")
            
            # Format as shell command
            args_str = "\n".join([f"    {arg} \\" for arg in args])
            cmd_body = f"apptainer run --nv {self.config.get('image')} \\\n{args_str}"

        elif self.config.group == "dgx":
            container_name = f"vllm-dgx-{self.name}"
            hf_token_path = self.config.get('hf_token_path', '$HOME/.config/cloudmesh/ai/keys/hugging.txt')
            api_key_path = self.config.get('api_key_path', '$HOME/.config/cloudmesh/ai/keys/server_master_key.txt')
            
            args = [
                f"--model {self.config.get('model')}",
                f"--tensor-parallel-size {self.config.get('tensor_parallel_size', 4)}",
                f"--gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.95)}",
                f"--max-model-len {self.config.get('max_model_len', 131072)}",
            ]
            
            max_batched = self.config.get('max_num_batched_tokens')
            if max_batched:
                args.append(f"--max-num-batched-tokens {max_batched}")
            
            extra_args = self.config.get('extra_args', '')
            if extra_args:
                for line in extra_args.splitlines():
                    if line.strip():
                        args.append(line.strip())
            
            if self.config.get('enable_prefix_caching', True):
                args.append("--enable-prefix-caching")
            args.append(f"--load-format {self.config.get('load_format', 'safetensors')}")
            if self.config.get('enable_auto_tool_choice', True):
                args.append("--enable-auto-tool-choice")
            args.append(f"--tool-call-parser {self.config.get('tool_call_parser', 'gemma4')}")
            
            args_str = "\n".join([f"    {arg} \\" for arg in args])
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
                f"{args_str}"
            )
        else:
            args = [
                f"--model {self.config.get('model')}",
                f"--tensor-parallel-size {self.config.get('tensor_parallel_size', 1)}",
                f"--gpu-memory-utilization {self.config.get('gpu_memory_utilization', 0.90)}",
            ]
            max_batched = self.config.get('max_num_batched_tokens')
            if max_batched:
                args.append(f"--max-num-batched-tokens {max_batched}")
            
            extra_args = self.config.get('extra_args', '')
            if extra_args:
                for line in extra_args.splitlines():
                    if line.strip():
                        args.append(line.strip())
            
            args.append(f"--port {port}")
            args_str = "\n".join([f"    {arg} \\" for arg in args])
            cmd_body = f"python3 -m vllm.entrypoints.openai.api_server \\\n{args_str}"

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
