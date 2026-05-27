#!/usr/bin/env python
import subprocess
import textwrap
import os
from tabulate import tabulate
import sys


def set_terminal_title(title):
    # Construct the ANSI escape sequence
    sys.stdout.write(f"\033]0;{title}\007")
    # Flush stdout to ensure the terminal processes it immediately
    sys.stdout.flush()


def get_hf_token():
    token_path = os.path.expanduser("~/gemma/HF_token.txt")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    return ""


def launch_model(config):
    # 1. Use .get() with sensible defaults to prevent KeyError
    port = config.get("port", 18000)
    host = config.get("host", "white")
    model_id = config.get("model_id")
    max_len = config.get("max_len", 4096)
    memory = config.get("memory", 0.90)
    extra = config.get("extra", "")
    quant = config.get("quant")
    name = config.get("name", "Unknown Model")

    set_terminal_title(f"{host.upper()} | {name}")

    # 2. Build the base command with textwrap.dedent
    docker_base = textwrap.dedent(f"""
        docker rm -f vllm-server 2>/dev/null || true;
        docker run -it --rm \\
          --name vllm-server \\
          --gpus all \\
          --ipc=host \\
          --ulimit memlock=-1 \\
          --ulimit stack=67108864 \\
          -p {port}:{port} \\
          -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \\
          -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    """).strip()

    # 3. Determine the image based on host
    image = (
        "nvcr.io/nvidia/vllm:26.03-py3"
        if host == "spark"
        else "vllm/vllm-openai:latest"
    )

    # 4. Construct command parts
    server_cmd = (
        "python3 -m vllm.entrypoints.openai.api_server" if host == "spark" else ""
    )

    # Using a list to build the command helps manage spaces and quotes cleanly
    cmd_parts = [
        docker_base,
        image,
        server_cmd,
        f"--host 0.0.0.0 --port {port}",
        f'--model "{model_id}"',
        f"--max-model-len {max_len}",
        f"--gpu-memory-utilization {memory}",
        extra,
    ]

    if quant:
        cmd_parts.append(f"--quantization {quant}")

    # Join with spaces, removing any empty strings
    cmd = " ".join(part for part in cmd_parts if part)

    # 5. Prepend the kill command
    full_cmd = (
        f"docker stop vllm-server 2>/dev/null || true; "
        f"docker rm vllm-server 2>/dev/null || true; "
        f"fuser -k {port}/tcp 2>/dev/null || true; "
        f"sleep 2; {cmd}"
    )  # Added sleep to allow port release

    # ----------------------
    print("\n" + "="*80)
    print(f"READY TO EXECUTE ON {config['host'].upper()}:")
    print("-" * 80)
    print(full_cmd)
    print("="*80 + "\n")
    # -----------------------

    print(f"\n--> Launching {name} on {host} (Port: {port})...")

    # 6. Execute via SSH
    ssh_cmd = f"ssh -t {host} 'export HF_TOKEN={get_hf_token()}; {full_cmd}'"
    subprocess.run(ssh_cmd, shell=True)


menu = [
    #    {
    #        "tested": False,
    #        "name":     "Qwen 2.5 14B",
    #        "host":     "white",
    #        "port":     18000,
    #        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    #        "quant":    "awq",
    #        "memory":   0.90,
    #        "max_len":  4096,
    #        "extra":    "--enforce-eager",
    #        "desc":     "Excellent: Coding, Agents, Reasoning"
    #    },
    #    {
    #        "tested": False,
    #        "name":     "Llama 3 8B",
    #        "host":     "white",
    #        "port":     18000,
    #        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    #        "quant":    None,
    #        "memory":   0.85,
    #        "max_len":  8192,
    #        "extra":    "",
    #        "desc":     "General Text, Low-Latency Prototyping"
    #    },
    {
        "tested": True,
        "name": "Gemma 4 Instruct",
        "host": "white",
        "port": 18000,
        "model_id": "google/gemma-4-e4b-it",
        "quant": None,
        "memory": 0.90,
        "max_len": 16384,
        "extra": "--enforce-eager",
        "desc": "Next-Gen: Advanced Reasoning & Modality",
    },
    {
        "tested": True,
        "name": "Qwen 2.5 Coder 32B",
        "host": "white",
        "port": 18000,
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        "quant": "awq",
        "memory": 0.95,
        "max_len": 12288,
        "extra": "--enforce-eager",
        "desc": "Optimized: SOTA Local Code Generation",
    },
    {
        "tested": False,
        "name": "Qwen 2.5 32B",
        "host": "white",
        "port": 18000,
        "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "quant": "awq",
        "memory": 0.95,
        "max_len": 2048,
        "extra": "--enforce-eager",
        "desc": "Strong: Logic, Math, Advanced Coding",
    },
    {
        "tested": False,
        "name": "Gemma 2 27B",
        "host": "white",
        "port": 18000,
        "model_id": "google/gemma-2-27b-it",
        "quant": None,
        "memory": 0.90,
        "max_len": 8192,
        "extra": "",
        "desc": "Balanced: Logic & Creative Writing",
    },
    None,
    {
        "tested": True,
        "name": "DeepSeek R1 70B",
        "host": "spark",
        "port": 18001,
        "model_id": "casperhansen/deepseek-r1-distill-llama-70b-awq",
        "quant": "awq",
        "memory": 0.90,
        "max_len": 32768,
        "extra": "--dtype float16 --enforce-eager",
        "desc": "Optimized: Deep Chain-of-Thought",
    },
    {
        "tested": False,
        "name": "Nemotron-3-Nano-30B-A3B",
        "host": "spark",
        "port": 18001,
        "model_id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "quant": "bf16", # Use "fp8" if you run into OOM errors
        "memory": 0.75,  # Slightly lower to leave room for the massive KV cache
        "max_len": 131072, # Start at 128k; 1M may trigger OOM on a single 3090
        "extra": "--trust-remote-code --enforce-eager",
        "desc": "NVIDIA Hybrid Mamba-MoE: High-Efficiency Agentic Reasoning"
    },
    { 
        "tested": False, 
        "name": "Qwen3 Coder 30B Instruct (FP4)", 
        "host": "spark", 
        "port": 18001, 
        "model_id": "ig1/Qwen3-Coder-30B-A3B-Instruct-NVFP4", 
        "quant": "nvfp4", 
        "memory": 0.90, 
        "max_len": 262144, 
        "extra": "--dtype float16", 
        "desc": "Optimized: Blackwell Native 4-bit Engine" 
    },
    #{
    #    "tested": False,
    #    "name": "Qwen 2.5 MoE A14B",
    #    "host": "spark",
    #    "port": 18001,
    #    "model_id": "Qwen/Qwen2.5-MoE-A14B-Instruct",
    #    "quant": None,
    #    "memory": 0.90,
    #    "max_len": 32768,
    #    "extra": "--enforce-eager",
    #    "desc": "High-Efficiency MoE: Coding & Logic",
    #},
    {
        "tested": False,
        "name": "Qwen 3.6 MoE A3B Instruct",
        "host": "spark",
        "port": 18001,
        "model_id": "Qwen/Qwen3.6-35B-A3B-Instruct",
        "quant": None,
        "memory": 0.90,
        "max_len": 32768,
        "extra": "--enforce-eager",
        "desc": "Frontier MoE: SOTA Agentic Coding",
    },
    {
        "tested": False,
        "name": "Qwen 3.6 MoE A3B",
        "host": "spark",
        "port": 18001,
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "quant": None,
        "memory": 0.90,
        "max_len": 32768,
        "extra": "--enforce-eager",
        "desc": "Frontier MoE: SOTA Agentic Coding",
    },
    {
        "tested": False,
        "name": "Gemma 4 Instruct",
        "host": "spark",
        "port": 18001,
        "model_id": "google/gemma-4-it",
        "quant": None,
        "memory": 0.90,
        "max_len": 16384,
        "extra": "",
        "desc": "Next-Gen: High-Speed Vision & Logic",
    },
    {
        "tested": False,
        "name": "Qwen3 Coder 32B Instruct",
        "host": "spark",
        "port": 18001,
        "model_id": "Qwen/Qwen3-Coder-32B-Instruct",
        "quant": "4-bit",
        "memory": 0.90,
        "max_len": 16384,
        "extra": "--tensor-parallel-size 1",
        "desc": "High-performance coding assistant optimized for Blackwell architecture",
    },
    {
        "tested": False,
        "name": "Llama 3 70B FP4 Instruct",
        "host": "spark",
        "port": 18001,
        "model_id": "unsloth/Llama-3-70B-Instruct-quantized-FP4",
        "quant": "compressed-tensors",
        "memory": 0.85,
        "max_len": 8192,
        "extra": "",
        "desc": "High-Speed: Deep Reasoning, Coding",
    },
    {
        "tested": False,
        "name": "Llama 3 70B FP8",
        "host": "spark",
        "port": 18001,
        "model_id": "neuralmagic/Meta-Llama-3-70B-FP8",
        "quant": "fp8",
        "memory": 0.90,
        "max_len": 8192,
        "extra": "",
        "desc": "Max Precision: Complex Logic & Text",
    },
    {
        "tested": False,
        "name": "Gemma 2 27B",
        "host": "spark",
        "port": 18001,
        "model_id": "google/gemma-2-27b-it",
        "quant": None,
        "memory": 0.90,
        "max_len": 8192,
        "extra": "",
        "desc": "Fast: Optimized for ARM/Blackwell",
    },
    {
        "tested": False,
        "name": "Llama 3 8B Instruct",
        "host": "spark",
        "port": 18001,
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quant": None,
        "memory": 0.50,
        "max_len": 4096,
        "extra": "",
        "desc": "Long-Context Text, Fluid Dialogue",
    },
]


def show_menu():
    table_data = []
    for i, cfg in enumerate(menu):
        if cfg is None:
            # Add a visual separator row
            table_data.append(["", "", "", "", "", "", "", "", ""])
            continue

        table_data.append(
            [
                i,
                "✅" if cfg["tested"] else "❌",
                cfg["name"],
                cfg["host"].upper(),
                cfg["port"],
                cfg["quant"] or "-",
                cfg["memory"],
                cfg["max_len"],
                cfg["desc"],
            ]
        )

    headers = ["ID", "Tested", "Name", "Host", "Port", "Quant", "Mem", "Ctx", "Desc"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    selected_config = None
    while True:
        show_menu()
        choice = input("\nEnter ID to launch (or 'q' to quit): ").strip()

        if choice.lower() == "q":
            return

        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(menu) and menu[idx] is not None:
                selected_config = menu[idx]
                break
            else:
                print("Invalid ID.")
        else:
            print("Please enter a valid number.")

    if selected_config:
        launch_model(selected_config)


if __name__ == "__main__":
    main()
