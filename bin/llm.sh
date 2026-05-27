#!/usr/bin/env bash
set -e

clear

# Load HF Token for gated models
if [ -f "$HOME/gemma/HF_token.txt" ]; then
    export HF_TOKEN=$(cat "$HOME/gemma/HF_token.txt")
else
    export HF_TOKEN=""
fi

# Output the comprehensive hardware architecture reference table with updated Port mappings
echo "======================================================================================================================================================================"
echo "                                                         CURRENT REMOTE RESOURCE ARCHITECTURE MAP                                                         "
echo "======================================================================================================================================================================"
echo "| Num | Host  | Model                     | FP   | Disk     | Speed      | Context | Capabilities / Strengths                | Model ID                                   | Port  |"
echo "|-----|-------|---------------------------|------|----------|------------|---------|-----------------------------------------|--------------------------------------------|-------|"
echo "| 1   | white | Qwen 2.5 14B Instruct     | AWQ  | ~9.5 GB  | ~50-60 t/s | 4096    | Excellent: Coding, Agents, Reasoning    | Qwen/Qwen2.5-14B-Instruct-AWQ              | 18000 |"
echo "| 2   | white | Llama 3 8B Instruct       | FP16 | ~16.0 GB | ~45-55 t/s | 8192    | General Text, Low-Latency Prototyping   | meta-llama/Meta-Llama-3-8B-Instr           | 18000 |"
echo "| 3   | white | Qwen 2.5 32B Instruct     | AWQ  | ~20.0 GB | ~30-35 t/s | 2048    | Strong: Logic, Math, Advanced Coding    | Qwen/Qwen2.5-32B-Instruct-AWQ              | 18000 |"
echo "|-----|-------|---------------------------|------|----------|------------|---------|-----------------------------------------|--------------------------------------------|-------|"
echo "| 4   | spark | Llama 3 70B Instruct      | FP4  | ~39.0 GB | ~45-55 t/s | 8192    | High-Speed: Deep Reasoning, Coding      | unsloth/Llama-3-70B-Instruct-quantized-FP4 | 18001 |"
echo "| 5   | spark | Llama 3 70B Instruct      | FP8  | ~75.0 GB | ~25-30 t/s | 8192    | Max Precision: Complex Logic & Text     | neuralmagic/Meta-Llama-3-70B-FP8           | 18001 |"
echo "| 6   | spark | Llama 3 8B Instruct       | FP16 | ~16.0 GB | ~120+ t/s  | 4096    | Long-Context Text, Fluid Dialogue       | meta-llama/Meta-Llama-3-8B-Instr           | 18001 |"
echo "|-----|-------|---------------------------|------|----------|------------|---------|-----------------------------------------|--------------------------------------------|-------|"
echo "| 10  | white | Qwen 2.5 Coder 32B Instr  | AWQ  | ~20.0 GB | ~30-35 t/s | 12288   | Optimized: SOTA Local Code Generation   | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ        | 18000 |"
echo "| 11  | spark | DeepSeek R1 Distill 70B   | AWQ  | ~42.0 GB | ~40-50 t/s | 32768   | Optimized: Deep Chain-of-Thought        | casperhansen/deepseek-r1-70b-awq           | 18001 |"
echo "|-----|-------|---------------------------|------|----------|------------|---------|-----------------------------------------|--------------------------------------------|-------|"
echo "| 12  | white | Gemma 2 27B Instruct      | FP16 | ~55.0 GB | ~60-70 t/s | 8192    | Balanced: Logic & Creative Writing      | google/gemma-2-27b-it                      | 18000 |"
echo "| 13  | spark | Gemma 2 27B Instruct      | FP16 | ~55.0 GB | ~80-90 t/s | 8192    | Fast: Optimized for ARM/Blackwell       | google/gemma-2-27b-it                      | 18001 |"
echo "| 14  | spark | Qwen 2.5 MoE A14B Instr   | BF16 | ~30.0 GB | ~100+ t/s  | 32768   | High-Efficiency MoE: Coding & Logic     | Qwen/Qwen2.5-MoE-A14B-Instruct             | 18001 |"
echo "| 15  | spark | Qwen 3.6 MoE A3B Instr    | BF16 | ~35.0 GB | ~120+ t/s  | 32768   | Frontier MoE: SOTA Agentic Coding       | Qwen/Qwen3.6-35B-A3B-Instruct              | 18001 |"
echo "| 16  | spark | Qwen 3.6 MoE A3B Instr    | BF16 | ~35.0 GB | ~120+ t/s  | 32768   | Frontier MoE: SOTA Agentic Coding       | Qwen/Qwen3.6-35B-A3B-Instruct              | 18001 |"
echo "| 17  | white | Gemma 4 Instruct          | FP16 | ~65.0 GB | ~40-50 t/s | 16384   | Next-Gen: Advanced Reasoning & Modality | google/gemma-4-it                          | 18000 |"
echo "| 18  | spark | Gemma 4 Instruct          | FP16 | ~65.0 GB | ~80-100 t/s| 16384   | Next-Gen: High-Speed Vision & Logic     | google/gemma-4-it                          | 18001 |"
echo "======================================================================================================================================================================"
echo " * Note: vLLM serves text/code engines only. Image Generation (Diffusion) is not supported here."
echo ""

read -p "Enter selection [1-6, 10-17, or 0/q to exit]: " SELECTION

case $SELECTION in
    1)
        echo "--> Launching execution cycle on remote x86 host 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"Qwen/Qwen2.5-14B-Instruct-AWQ\" \
          --quantization awq \
          --max-model-len 4096 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;
    2)
        echo "--> Launching execution cycle on remote x86 host 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"meta-llama/Meta-Llama-3-8B-Instruct\" \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.85"
        ;;
    3)
        echo "--> Launching execution cycle on remote x86 host 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"Qwen/Qwen2.5-32B-Instruct-AWQ\" \
          --quantization awq \
          --max-model-len 2048 \
          --gpu-memory-utilization 0.95 \
          --enforce-eager"
        ;;
    4)
        echo "--> Launching execution cycle on remote Blackwell ARM host 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"unsloth/Llama-3-70B-Instruct-quantized-FP4\" \
          --quantization compressed-tensors \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.85"
        ;;
    5)
        echo "--> Launching execution cycle on remote Blackwell ARM host 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"neuralmagic/Meta-Llama-3-70B-Instruct-FP8\" \
          --quantization fp8 \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.90"
        ;;
    6)
        echo "--> Launching execution cycle on remote Blackwell ARM host 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"meta-llama/Meta-Llama-3-8B-Instruct\" \
          --max-model-len 4096 \
          --gpu-memory-utilization 0.50"
        ;;
    10)
        echo "--> Launching specialized Coding Engine on remote x86 host 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"Qwen/Qwen2.5-Coder-32B-Instruct-AWQ\" \
          --quantization awq \
          --max-model-len 12288 \
          --gpu-memory-utilization 0.95 \
          --enforce-eager"
        ;;
    11)
        echo "--> Launching execution cycle on remote Blackwell ARM host 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"casperhansen/deepseek-r1-distill-llama-70b-awq\" \
          --quantization awq \
          --dtype float16 \
          --max-model-len 32768 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;
    12)
        echo "--> Launching Gemma 2 27B on 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker rm -f remote-3090-vllm 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"google/gemma-2-27b-it\" \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.90"
        ;;
    13)
        echo "--> Launching Gemma 2 27B on 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"google/gemma-2-27b-it\" \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.90"
        ;;
    14)
        echo "--> Launching Qwen 2.5 MoE A14B on 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"Qwen/Qwen2.5-MoE-A14B-Instruct\" \
          --max-model-len 32768 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;
    15)
        echo "--> Launching Qwen 3.6 MoE A3B on 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"Qwen/Qwen3.6-35B-A3B-Instruct\" \
          --max-model-len 32768 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;
    16)
        echo "--> Launching Qwen 3.6 MoE A3B on 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"Qwen/Qwen3.6-35B-A3B\" \
          --max-model-len 32768 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;
    17)
        echo "--> Launching Gemma 4 on remote x86 host 'white'..."
        ssh -t white "fuser -k 18000/tcp 2>/dev/null || true; \
          docker rm -f remote-3090-vllm 2>/dev/null || true; \
          docker run -it --rm \
          --name remote-3090-vllm \
          --gpus all \
          --ipc=host \
          -p 18000:18000 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          vllm/vllm-openai:latest \
          --host 0.0.0.0 \
          --port 18000 \
          --model \"google/gemma-4-it\" \
          --max-model-len 16384 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager"
        ;;  
    18)
        echo "--> Launching Gemma 4 on remote Blackwell ARM host 'spark'..."
        ssh -t spark "fuser -k 18001/tcp 2>/dev/null || true; \
          docker rm -f spark-vllm-interactive 2>/dev/null || true; \
          docker run -it --rm \
          --name spark-vllm-interactive \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -p 18001:18001 \
          -e \"HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\" \
          -v \"\$HOME/.cache/huggingface:/root/.cache/huggingface\" \
          nvcr.io/nvidia/vllm:26.03-py3 \
          python3 -m vllm.entrypoints.openai.api_server \
          --host 0.0.0.0 \
          --port 18001 \
          --model \"google/gemma-4-it\" \
          --max-model-len 16384 \
          --gpu-memory-utilization 0.90"
        ;;
    0|q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid selection. Exiting..."
        exit 1
        ;;
esac