#!/bin/bash
#SBATCH --job-name=build_vllm_sif
#SBATCH --mail-user=thf2bn@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bi_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/thf2bn/create_sif.out
#SBATCH --error=/home/thf2bn/create_sif.err

export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p $APPTAINER_TMPDIR

cd /scratch/thf2bn

apptainer build vllm_gemma4.sif docker://vllm/vllm-openai:gemma4


# squeue -u $USER
# cat /home/thf2bn/create_sif.out
# cat /home/thf2bn/create_sif.err
