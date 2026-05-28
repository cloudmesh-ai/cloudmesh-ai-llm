# Model Resource Allocation

## A100 (80GB) GPUs
### Local vLLM Models (Google Gemma)
| Model ID | Model Size | Precision | Weights (FP16) | Max Context Window | Number of A100 (80GB) GPUs | VRAM Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma-4-9b` | Gemma 2 9B | FP16 | ~18 GB | 131,072 | 1 | Fits comfortably. |
| `gemma2` | Gemma 2 27B | FP16 | ~54 GB | 131,072 | 2 | Weights + KV Cache require >80GB. |
| `gemma-4` | Gemma 4 31B | FP16 | ~62 GB | 131,072 | 4 | 4 GPUs recommended for KV cache stability. |
| `gemma-4-31b-fp8` | Gemma 4 31B | FP8 | ~31 GB | 131,072 | 1 | Optimized for VRAM; fits on 1 GPU with large KV cache. |
| `gemma-4-31b` | Gemma 4 31B | 4-bit | ~18 GB | 131,072 | 1 | Quantization frees up VRAM for KV Cache. |

### Local vLLM Models (Kimi / Moonshot)
*Estimations based on model parameter scale (assuming ~30B to ~70B variants)*

| Model ID | Model Variant | Precision | Weights | Max Context Window | Number of A100 (80GB) GPUs | VRAM Strategy / Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `kimi-30b-fp16-1gpu` | Kimi (~30B) | FP16 | ~60 GB | 32k - 64k | 1 | Tight fit; limit `max_model_len` to avoid OOM. |
| `kimi-30b-fp16-2gpu` | Kimi (~30B) | FP16 | ~60 GB | 128k+ | 2 | Recommended for high context stability. |
| `kimi-30b-4bit` | Kimi (~30B) | 4-bit | ~18 GB | 128k+ | 1 | Fits easily; maximize `max_model_len`. |
| `kimi-70b-4bit-1gpu` | Kimi (~70B) | 4-bit | ~40 GB | 32k - 64k | 1 | Fits; VRAM limited for large KV cache. |
| `kimi-70b-4bit-2gpu` | Kimi (~70B) | 4-bit | ~40 GB | 128k+ | 2 | Recommended for 4-bit high context. |
| `kimi-70b-fp16-2gpu` | Kimi (~70B) | FP16 | ~140 GB | 32k - 64k | 2 | Minimum requirement for FP16. |
| `kimi-70b-fp16-4gpu` | Kimi (~70B) | FP16 | ~140 GB | 128k+ | 4 | Recommended for FP16 stability and context. |

### Local vLLM Models (DeepSeek)
*Estimations for DeepSeek-Coder/Chat series*

| Model ID | Model Variant | Precision | Weights | Max Context Window | Number of A100 (80GB) GPUs | VRAM Strategy / Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek-33b-fp16` | DeepSeek 33B | FP16 | ~66 GB | 32k | 2 | Recommended for FP16 stability. |
| `deepseek-33b-4bit` | DeepSeek 33B | 4-bit | ~20 GB | 128k | 1 | High VRAM availability for KV Cache. |
| `deepseek-67b-fp16` | DeepSeek 67B | FP16 | ~134 GB | 32k | 4 | Required for FP16 weights + KV cache. |
| `deepseek-67b-4bit-1gpu` | DeepSeek 67B | 4-bit | ~40 GB | 32k | 1 | Tight fit; limit `max_model_len`. |
| `deepseek-67b-4bit-2gpu` | DeepSeek 67B | 4-bit | ~40 GB | 128k | 2 | Recommended for 4-bit high context. |

### Local vLLM Models (Qwen)
*Estimations for Qwen2 / Qwen2.5 series*

| Model ID | Model Variant | Precision | Weights | Max Context Window | Number of A100 (80GB) GPUs | VRAM Strategy / Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen2-7b-fp16` | Qwen2 7B | FP16 | ~14 GB | 128k | 1 | Fits comfortably. |
| `qwen2-72b-fp16` | Qwen2 72B | FP16 | ~144 GB | 32k | 4 | Recommended for FP16 stability. |
| `qwen2-72b-fp8` | Qwen2 72B | FP8 | ~72 GB | 32k - 64k | 2 | Optimized for H100/A100; 2 GPUs for cache. |
| `qwen2-72b-4bit-1gpu` | Qwen2 72B | 4-bit | ~40 GB | 32k | 1 | Tight fit; limit `max_model_len`. |
| `qwen2-72b-4bit-2gpu` | Qwen2 72B | 4-bit | ~40 GB | 128k | 2 | Recommended for 4-bit high context. |

---

## RTX 3090 (24GB) GPUs
*Consumer hardware target. Limited to 1 GPU. Heavily reliant on quantization (4-bit/AWQ/GPTQ).*

### Local vLLM Models (Google Gemma)
| Model ID | Model Size | Precision | Weights | Max Context Window | Number of RTX 3090 (24GB) GPUs | VRAM Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma-4-9b` | Gemma 2 9B | FP16 | ~18 GB | 8k - 16k | 1 | Tight fit; limit `max_model_len`. |
| `gemma-4-31b` | Gemma 4 31B | 4-bit | ~18 GB | 8k - 16k | 1 | Fits; VRAM limited for KV cache. |

### Local vLLM Models (DeepSeek / Qwen)
| Model ID | Model Variant | Precision | Weights | Max Context Window | Number of RTX 3090 (24GB) GPUs | VRAM Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen2-7b-fp16` | Qwen2 7B | FP16 | ~14 GB | 32k | 1 | Fits comfortably. |
| `deepseek-33b-4bit` | DeepSeek 33B | 4-bit | ~20 GB | 4k - 8k | 1 | Very tight fit; minimal KV cache. |

---

## NVIDIA Spark (Optimized)
*Optimized Production target (H100/A100 Cluster). Limited to 1 GPU per instance. Focused on `spark` model variants (FP8/FP4/Quantized).*

### Local vLLM Models (Gemma / Llama / Qwen Spark)
| Model ID | Model Variant | Precision | Weights | Max Context Window | Number of GPUs | VRAM Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma-2-27b-spark` | Gemma 2 27B | FP8/Quant | ~27 GB | 131k | 1 | Optimized for single-node high throughput. |
| `gemma-4-31b-spark-fp4` | Gemma 4 31B | FP4 | ~18 GB | 131k | 1 | Maximum VRAM for KV cache; highest context stability. |
| `llama-3-70b-fp8-spark` | Llama 3 70B | FP8 | ~70 GB | 32k | 1 | High-density FP8 deployment; tight VRAM. |
| `llama-3-70b-spark-fp4` | Llama 3 70B | FP4 | ~35 GB | 128k | 1 | Optimized for high context on single GPU. |
| `qwen2-72b-spark` | Qwen2 72B | FP8 | ~72 GB | 16k - 32k | 1 | Tight fit on 80GB; limit context. |
| `qwen2-72b-spark-fp4` | Qwen2 72B | FP4 | ~36 GB | 128k | 1 | Optimized for high context on single GPU. |