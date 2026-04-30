#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.out
#SBATCH --error=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.err

# ---------------------------------------------------------------------------
# Demo 1: single-GPU LLM inference.
# Submit with:  bash slurm/submit.sh inference
# To change the model, prompts file, etc., edit the export VAR=... defaults
# in the block below.
# ---------------------------------------------------------------------------

export PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
export SIF_PATH=${SIF_PATH:-/scratch/p_oe_hpc/transformers-gpu.sif}
export HF_HOME_HOST=${HF_HOME_HOST:-/scratch/p_oe_hpc/${USER}/hf_cache}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/p_oe_hpc/${USER}/oe_hpc_ai/output}
export RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
export PROMPTS_FILE=${PROMPTS_FILE:-/workspace/src/prompts.txt}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
export SEED=${SEED:-42}

module load singularity/4.0

singularity exec \
    --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${HF_HOME_HOST}:/data/cache" \
    --bind "${RUN_DIR}:/output" \
    --env HF_HOME=/data/cache \
    "${SIF_PATH}" \
    python /workspace/src/inference.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --prompts_file "${PROMPTS_FILE}" \
        --output_dir /output \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}"
