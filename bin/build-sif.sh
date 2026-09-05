#!/bin/bash
#SBATCH --job-name=build_vllm_2608  # Name of your build job
#SBATCH --output=build_vllm_%j.log   # Progress log file
#SBATCH --partition=bii-gpu          # Use your designated partition
#SBATCH --account=bii_dsc_community # Allocation group account
#SBATCH --nodes=1                   # Single node compilation task
#SBATCH --cpus-per-task=8           # Allocate enough cores for fast decompression
#SBATCH --mem=32G                   # Give Apptainer caching headroom in RAM
#SBATCH --time=00:45:00             # Allow up to 45 minutes to pull and build
#SBATCH --gres=gpu:a100:1

# 1. Load the cluster container module tools
module load apptainer

# 2. Redirect cache folders to scratch space to avoid disk quota OOMs in your $HOME folder
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# 3. Define the destination folder path and the source container target
OUTPUT_SIF="/scratch/thf2bn/vllm_gemma4_2608.sif"
SOURCE_IMAGE="docker://nvcr.io/nvidia/vllm:26.08-py3"

echo "=========================================================="
echo "Starting Apptainer image compile process..."
echo "Source: $SOURCE_IMAGE"
echo "Destination Target: $OUTPUT_SIF"
echo "=========================================================="

# 4. Execute the structural convert build binary command
apptainer build "$OUTPUT_SIF" "$SOURCE_IMAGE"

echo "=========================================================="
echo "Build finished! Verifying container signature output..."
ls -lh "$OUTPUT_SIF"
echo "=========================================================="
