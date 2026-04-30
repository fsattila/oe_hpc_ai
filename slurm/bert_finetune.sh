#!/bin/bash
#SBATCH --job-name=bert_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.out
#SBATCH --error=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.err

# ---------------------------------------------------------------------------
# Demo 2 (single GPU): DistilBERT fine-tune on GLUE/SST-2.
# Submit with:  bash slurm/submit.sh bert_finetune
# ---------------------------------------------------------------------------

export PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
export SIF_PATH=${SIF_PATH:-/scratch/p_oe_hpc/transformers-gpu.sif}
export HF_HOME_HOST=${HF_HOME_HOST:-/scratch/p_oe_hpc/${USER}/hf_cache}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/p_oe_hpc/${USER}/oe_hpc_ai/output}
export RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-distilbert-base-uncased}
export DATASET_NAME=${DATASET_NAME:-glue}
export DATASET_CONFIG=${DATASET_CONFIG:-sst2}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}
export BATCH_SIZE=${BATCH_SIZE:-32}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
export LEARNING_RATE=${LEARNING_RATE:-5e-5}
export MAX_STEPS=${MAX_STEPS:-500}
export WARMUP_STEPS=${WARMUP_STEPS:-50}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export LOGGING_STEPS=${LOGGING_STEPS:-20}
export EVAL_STEPS=${EVAL_STEPS:-100}
export SAVE_STEPS=${SAVE_STEPS:-500}
export SEED=${SEED:-42}

module load singularity/4.0

singularity exec \
    --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${HF_HOME_HOST}:/data/cache" \
    --bind "${RUN_DIR}:/output" \
    --env HF_HOME=/data/cache \
    "${SIF_PATH}" \
    python /workspace/src/bert_finetune.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir /output \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --max_steps "${MAX_STEPS}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --logging_steps "${LOGGING_STEPS}" \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --seed "${SEED}"
