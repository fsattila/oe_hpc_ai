#!/bin/bash
#SBATCH --job-name=agentic_sft_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.out
#SBATCH --error=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.err

# ---------------------------------------------------------------------------
# Demo 3 (multi-GPU DDP): full-parameter SFT of Qwen3-0.6B on xlam.
# Submit with:  bash slurm/submit.sh agentic_sft_ddp
# To scale to 4 GPUs, edit --gres=gpu:N and --cpus-per-task=N above and
# resubmit. NPROC=${SLURM_GPUS_ON_NODE} below auto-scales torchrun.
# ---------------------------------------------------------------------------

export PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
export SIF_PATH=${SIF_PATH:-/scratch/p_oe_hpc/transformers-gpu.sif}
export HF_HOME_HOST=${HF_HOME_HOST:-/scratch/p_oe_hpc/${USER}/hf_cache}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/p_oe_hpc/${USER}/oe_hpc_ai/output}
export RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}
export DATASET_NAME_OR_PATH=${DATASET_NAME_OR_PATH:-Salesforce/xlam-function-calling-60k}
export MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-1000}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
export BATCH_SIZE=${BATCH_SIZE:-4}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export MAX_STEPS=${MAX_STEPS:-200}
export WARMUP_STEPS=${WARMUP_STEPS:-20}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export LOGGING_STEPS=${LOGGING_STEPS:-10}
export SAVE_STEPS=${SAVE_STEPS:-200}
export SEED=${SEED:-42}

NPROC=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}

module load singularity/4.0

singularity exec \
    --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${HF_HOME_HOST}:/data/cache" \
    --bind "${RUN_DIR}:/output" \
    --env HF_HOME=/data/cache \
    "${SIF_PATH}" \
    torchrun --standalone --nproc_per_node="${NPROC}" \
        /workspace/src/agentic_sft.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset_name_or_path "${DATASET_NAME_OR_PATH}" \
        --output_dir /output \
        --max_train_samples "${MAX_TRAIN_SAMPLES}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LEARNING_RATE}" \
        --max_steps "${MAX_STEPS}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --seed "${SEED}"
