#!/bin/bash
# ---------------------------------------------------------------------------
# Generic submit wrapper for the SLURM scripts under slurm/.
#
# Usage:
#   bash slurm/submit.sh <job_name>          # e.g. "inference", "bert_finetune"
#
# To change the SLURM account, reservation, or per-user output root, edit the
# ACCOUNT / RESERVATION / OUTPUT_ROOT defaults in the block below. Setting
# RESERVATION="" submits the job without a reservation flag.
#
# Why a wrapper instead of a plain `sbatch slurm/<name>.sh`?
#   The sbatch scripts write logs to ${OUTPUT_ROOT}/<job_name>_<jobid>/slurm.out
#   via #SBATCH --output. SLURM opens that file BEFORE the job body runs, so
#   the parent directory must already exist. We submit the job in held state
#   (--hold), create the run dir using the assigned job ID, then release it.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <job_name>" >&2
    echo "  e.g. $0 inference" >&2
    exit 2
fi

JOB_NAME=$1

# Drop SSL/CA env vars leaked by the host shell (e.g. VS Code Remote injects
# paths under ~/.vscode-server that do not exist on compute nodes or inside
# the container). SLURM's default --export=ALL would otherwise carry them in.
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS

ACCOUNT=${ACCOUNT:-p_oe_hpc}
RESERVATION=${RESERVATION-oe_hpc}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/p_oe_hpc/${USER}/oe_hpc_ai/output}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script="${here}/${JOB_NAME}.sh"

if [[ ! -f "${script}" ]]; then
    echo "No such SLURM script: ${script}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

extra_args=( --account="${ACCOUNT}" )
if [[ -n "${RESERVATION}" ]]; then
    extra_args+=( --reservation="${RESERVATION}" )
fi

JOB_ID=$(sbatch --parsable --hold "${extra_args[@]}" "${script}")
RUN_DIR="${OUTPUT_ROOT}/${JOB_NAME}_${JOB_ID}"
mkdir -p "${RUN_DIR}"
scontrol release "${JOB_ID}"

echo "Submitted ${JOB_ID} -> ${RUN_DIR}"
