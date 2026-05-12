class VLLMScript:
    """Generates a shell script to launch a vLLM server based on the provided configuration."""

    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path

    def generate(self):
        """Generate the bash script as a string."""
        try:
            server_cfg = self.config[self.config_path]
        except (KeyError, TypeError):
            raise ValueError(f"Configuration for '{self.config_path}' not found.")

        if not server_cfg:
            raise ValueError(f"Configuration for '{self.config_path}' is empty.")

        # 1. The 'script' field is MANDATORY. If it doesn't exist, raise an error.
        custom_script = server_cfg.get("script")
        if not custom_script:
            raise ValueError(f"Configuration at '{self.config_path}' must contain a 'script' field.")

        # 2. Replace every variable in the config within the script
        resolved_script = custom_script
        for key, value in server_cfg.items():
            if key == "script":
                continue
            
            # Replace {key} with the value
            resolved_script = resolved_script.replace(f"{{{key}}}", str(value))
        
        # Special case for {gpu} if it's not explicitly in the config but tensor_parallel_size is
        if "{gpu}" in resolved_script:
            gpu_val = server_cfg.get("tensor_parallel_size") or server_cfg.get("gpus", "1")
            resolved_script = resolved_script.replace("{gpu}", str(gpu_val))

        # 3. Construct the final shell script
        port = server_cfg.get("remote_port", 8000)
        image = server_cfg.get("image", "vllm/vllm-openai")
        
        shell = server_cfg.get("shell", "/bin/bash")
        lines = [
            f"#!{shell}",
            f"# vLLM Launch Script for {self.config_path}",
            "",
            f"export PORT={port}",
            f"export VLLM_IMAGE=\"{image}\"",
            "",
            resolved_script
        ]
        
        return "\n".join(lines)