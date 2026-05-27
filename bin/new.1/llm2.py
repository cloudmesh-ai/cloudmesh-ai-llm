#!/usr/bin/env python
import subprocess
import textwrap
import sys
from pathlib import Path

import yaml
from tabulate import tabulate


def set_terminal_title(title):
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def get_hf_token():
    """Read HF token with error handling."""
    token_path = Path.home() / "gemma" / "HF_token.txt"
    try:
        if token_path.exists():
            return token_path.read_text().strip()
    except (IOError, PermissionError) as e:
        print(f"Warning: Could not read token file: {e}")
    return ""


def load_menu(config_path: Path = Path("models.yaml")) -> list:
    """Load model configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    menu = []
    for item in data.get("models", []):
        if item.get("separator"):
            menu.append(None)
        else:
            menu.append({
                "tested": item.get("tested", False),
                "name": item["name"],
                "host": item.get("host", "white"),
                "port": item.get("port", 18000),
                "model_id": item["model_id"],
                "quant": item.get("quant"),
                "memory": item.get("memory", 0.90),
                "max_len": item.get("max_len", 4096),
                "extra": item.get("extra", ""),
                "desc": item.get("desc", "")
            })
    return menu


def launch_model(config):
    port = config.get("port", 18000)
    host = config.get("host", "white")
    model_id = config.get("model_id")
    max_len = config.get("max_len", 4096)
    memory = config.get("memory", 0.90)
    extra = config.get("extra", "")
    quant = config.get("quant")
    name = config.get("name", "Unknown Model")

    set_terminal_title(f"{host.upper()} | {name}")

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

    image = (
        "nvcr.io/nvidia/vllm:26.03-py3"
        if host == "spark"
        else "vllm/vllm-openai:latest"
    )

    server_cmd = (
        "python3 -m vllm.entrypoints.openai.api_server" if host == "spark" else ""
    )

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

    cmd = " ".join(part for part in cmd_parts if part)

    full_cmd = (
        f"docker stop vllm-server 2>/dev/null || true; "
        f"docker rm vllm-server 2>/dev/null || true; "
        f"fuser -k {port}/tcp 2>/dev/null || true; "
        f"sleep 2; {cmd}"
    )

    print(f"\n--> Launching {name} on {host} (Port: {port})...")

    ssh_cmd = f"ssh -t {host} 'export HF_TOKEN={get_hf_token()}; {full_cmd}'"
    subprocess.run(ssh_cmd, shell=True)


def show_menu(menu):
    table_data = []
    for i, cfg in enumerate(menu):
        if cfg is None:
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
    # Load configuration with error handling
    try:
        menu = load_menu()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure models.yaml exists in the current directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing models.yaml: {e}")
        sys.exit(1)

    selected_config = None
    while True:
        show_menu(menu)
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