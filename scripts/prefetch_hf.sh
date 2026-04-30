#!/bin/bash
# ---------------------------------------------------------------------------
# Pre-download HuggingFace models and datasets into the per-user cache on
# /scratch. Run this ONCE on a Komondor login node before submitting any
# SLURM jobs — compute nodes have no internet access, so every weight and
# every dataset shard must already be present in HF_HOME.
#
# Usage:
#   bash scripts/prefetch_hf.sh
#
# To change which models / datasets are downloaded, edit the DEFAULT_MODELS
# and DEFAULT_DATASETS lists at the top of scripts/prefetch_hf.py.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
SIF_PATH=${SIF_PATH:-/scratch/p_oe_hpc/transformers-gpu.sif}
HF_HOME_HOST=${HF_HOME_HOST:-/scratch/p_oe_hpc/${USER}/hf_cache}

mkdir -p "${HF_HOME_HOST}"

module load singularity/4.0

singularity exec \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${HF_HOME_HOST}:/data/cache" \
    --env HF_HOME=/data/cache \
    "${SIF_PATH}" \
    python3 /workspace/scripts/prefetch_hf.py
