#!/usr/bin/env python3
"""
=============================================================================
GEMMA 4 vLLM MODEL DOWNLOADER & MANAGER
=============================================================================

Prerequisites:
  pip install huggingface_hub

How to Use:
  
  1. List all available instruction-tuned (-it) variants:
     python gemma4_manager.py --list

  2. Download a specific variant key for vLLM:
     python gemma4_manager.py --download <model_key>
     
     Example:
       python gemma4_manager.py --download 12b-it

  3. Run vLLM with the downloaded model:
     vllm serve ./models/gemma-4-12B-it
=============================================================================
"""

import argparse
import sys
from huggingface_hub import snapshot_download

# Dictionary mapping friendly identifiers to Hugging Face Model Repo IDs optimized for vLLM
GEMMA_4_IT_VARIANTS = {
    "e2b-it": {
        "repo_id": "google/gemma-4-E2B-it",
        "parameters": "5B",
        "focus": "On-device efficiency, Multimodal"
    },
    "e4b-it": {
        "repo_id": "google/gemma-4-E4B-it",
        "parameters": "8B",
        "focus": "On-device efficiency, Multimodal"
    },
    "12b-it": {
        "repo_id": "google/gemma-4-12B-it",
        "parameters": "Full 12B",
        "focus": "Unified Multimodal"
    },
    "26b-a4b-it": {
        "repo_id": "google/gemma-4-26B-A4B-it",
        "parameters": "MoE (4B active / 26B total)",
        "focus": "Mixture of Experts"
    },
    "31b-it": {
        "repo_id": "google/gemma-4-31B-it",
        "parameters": "31B",
        "focus": "Top-tier reasoning and intelligence"
    }
}

def list_models():
    print("\n" + "=" * 72)
    print(" GEMMA 4 INSTRUCTION-TUNED (-IT) vLLM VARIANTS ")
    print("=" * 72)
    
    for key, info in GEMMA_4_IT_VARIANTS.items():
        print(f" Key:        {key}")
        print(f" Repo ID:    {info['repo_id']}")
        print(f" Parameters: {info['parameters']}")
        print(f" Focus:      {info['focus']}")
        print("-" * 72)
    
    print("\n Usage Example:")
    print("   python gemma4_manager.py --download 12b-it\n")

def download_model(model_key):
    if model_key not in GEMMA_4_IT_VARIANTS:
        print(f"\nError: '{model_key}' is not a valid variant key.")
        print("Run 'python gemma4_manager.py --list' to see available options.\n")
        sys.exit(1)
        
    target = GEMMA_4_IT_VARIANTS[model_key]
    repo_id = target["repo_id"]
    local_dir = f"./models/{repo_id.split('/')[-1]}"
    
    print(f"\n[vLLM Prep] Preparing to download: {repo_id}")
    print(f"Destination folder: {local_dir}")
    
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir
        )
        print(f"\nSuccessfully downloaded for vLLM!")
        print(f"Local path: {downloaded_path}")
        print(f"\nTo run vLLM with this model, use:")
        print(f"   vllm serve {local_dir}\n")
        
    except Exception as e:
        print(f"\nDownload failed: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage and download Gemma 4 models for vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List all available Gemma 4 -it variants")
    parser.add_argument("--download", type=str, metavar="MODEL_KEY", help="Download a specific model key (e.g., 12b-it)")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.download:
        download_model(args.download)
    else:
        parser.print_help()