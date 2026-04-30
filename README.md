# oe_hpc_ai — AI workloads on the Komondor HPC

This repository walks through an AI use case on the Komondor HPC: how to run
typical AI workloads under SLURM, inside a Singularity container. Three
self-contained demos are included:

1. **Inference** — single-GPU text generation with a small instruction-tuned LLM.
2. **BERT fine-tune** — DistilBERT on GLUE/SST-2, single GPU and multi-GPU (DDP).
3. **Agentic SFT** — full-parameter supervised fine-tuning of Qwen3-0.6B on a
   function-calling dataset, single GPU and multi-GPU (DDP).

The focus is the **HPC side** (SLURM, sbatch, Singularity, GPU allocation,
scaling); the AI workloads are deliberately small so the demos run quickly
inside the `oe_hpc` reservation.

---

## 0. Prerequisites

You need a Komondor account with access to the `p_oe_hpc` project and to the
`oe_hpc` reservation on the `gpu` partition. The container image
`/scratch/p_oe_hpc/transformers-gpu.sif` is provided and shared across users.

The `submit.sh` wrapper passes `--account=p_oe_hpc` and `--reservation=oe_hpc`
on every job. To change either, edit the `ACCOUNT=` / `RESERVATION=` defaults
near the top of `slurm/submit.sh`.

---

## 1. Setup

Clone the repository and `cd` into it:

```bash
git clone <repo-url> oe_hpc_ai
cd oe_hpc_ai
```

Every job writes outputs to `/scratch/p_oe_hpc/$USER/oe_hpc_ai/output/<job>_<jobid>/`,
and HuggingFace assets are cached under `/scratch/p_oe_hpc/$USER/hf_cache/`.
Nothing is written to your home directory. These paths are set as defaults in
the SLURM scripts and can be overridden via `OUTPUT_ROOT` and `HF_HOME_HOST`.

---

## 2. Pre-fetch models and datasets (login node)

Compute nodes on Komondor have **no internet access**. Every model weight and
every dataset shard used by the jobs must already be present in the cache on
shared scratch. Run the prefetch script once on a login node before submitting
any jobs:

```bash
bash scripts/prefetch_hf.sh
```

This downloads, into `/scratch/p_oe_hpc/$USER/hf_cache/`:

- `Qwen/Qwen3-1.7B` — used in Demo 1 (inference)
- `Qwen/Qwen3-0.6B` — used in Demo 3 (SFT)
- `distilbert-base-uncased` — used in Demo 2 (fine-tune)
- `glue` (config `sst2`) — Demo 2 dataset
- `Salesforce/xlam-function-calling-60k` — Demo 3 dataset

If you want to download a different set of assets, edit the `DEFAULT_MODELS`
and `DEFAULT_DATASETS` lists near the top of `scripts/prefetch_hf.py`.

The script runs inside the same Singularity image used by the jobs, so the
library versions match.

---

## 3. Anatomy of an sbatch script

Open `slurm/inference.sh` in your editor and read it top-to-bottom. Every
job script in this repo follows the same template:

```bash
#SBATCH --job-name=<name>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:N            # number of GPUs on the node
#SBATCH --cpus-per-task=...
#SBATCH --mem=...
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.out
#SBATCH --error=/scratch/p_oe_hpc/%u/oe_hpc_ai/output/%x_%j/slurm.err

# Per-job paths
export PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
export SIF_PATH=${SIF_PATH:-/scratch/p_oe_hpc/transformers-gpu.sif}
export HF_HOME_HOST=${HF_HOME_HOST:-/scratch/p_oe_hpc/${USER}/hf_cache}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/p_oe_hpc/${USER}/oe_hpc_ai/output}
export RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Demo-specific knobs (each has a default, every one overridable via env var)
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-...}
# ...

module load singularity/4.0

singularity exec \
    --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${HF_HOME_HOST}:/data/cache" \
    --bind "${RUN_DIR}:/output" \
    --env HF_HOME=/data/cache \
    "${SIF_PATH}" \
    python3 /workspace/src/<demo>.py --model_name_or_path "${MODEL_NAME_OR_PATH}" ...
```

The three bind mounts establish a stable layout inside the container:

| Host path                                        | Container path | Purpose                          |
| ------------------------------------------------ | -------------- | -------------------------------- |
| repo root                                        | `/workspace`   | code (read-only effectively)     |
| `/scratch/p_oe_hpc/$USER/hf_cache`               | `/data/cache`  | HF model + dataset cache         |
| `/scratch/p_oe_hpc/$USER/oe_hpc_ai/output/<...>` | `/output`      | per-job output dir (logs, ckpts) |

`--nv` exposes the host CUDA driver to the container. `module load singularity/4.0`
brings the `singularity` binary into the path.

### About the `submit.sh` wrapper

Rather than calling `sbatch` directly, use the wrapper:

```bash
bash slurm/submit.sh <job_name>
```

The wrapper exists because SLURM opens the log file (`#SBATCH --output=...`)
**before** the job body runs, so the parent directory must already exist. The
wrapper submits the job in held state (`sbatch --hold`), creates
`${OUTPUT_ROOT}/<job>_<jobid>/`, then releases the job. It also attaches
`--account=p_oe_hpc` and `--reservation=oe_hpc` for you.

To change the account or reservation, edit the `ACCOUNT=` and `RESERVATION=`
defaults near the top of `slurm/submit.sh` (set `RESERVATION=""` to submit
without a reservation).

---

## 4. Demo 1 — single-GPU LLM inference

**What it does.** Loads `Qwen/Qwen3-1.7B`, applies its chat template to each
line of `src/prompts.txt`, runs greedy generation on a single GPU, and writes
`results.json` to the output directory.

**What to learn.** The minimal sbatch + Singularity recipe to run a GPU
workload, the bind-mount layout, and what a typical decode looks like
(tokens/sec).

**Submit:**

```bash
bash slurm/submit.sh inference
```

**Watch the queue:**

```bash
squeue -u $USER
```

**Read the logs once it starts:**

```bash
RUN_DIR=$(ls -dt /scratch/p_oe_hpc/$USER/oe_hpc_ai/output/inference_* | head -1)
tail -f "$RUN_DIR/slurm.out"
```

**Inspect the results:**

```bash
cat "$RUN_DIR/results.json" | head -40
```

You should see one entry per prompt with the model completion, token count,
and wall-clock time. The first prompt is slower than the rest because of
CUDA / kernel warmup.

**Tweaking knobs.** Every parameter (model, prompts file, max new tokens,
seed) has a default at the top of `slurm/inference.sh`, e.g.:

```bash
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
```

To experiment with a different model or longer generations, edit the default
in the sbatch file directly and resubmit. The same convention is used in
every demo's sbatch script.

---

## 5. Demo 2 — DistilBERT fine-tune on SST-2

**What it does.** Fine-tunes `distilbert-base-uncased` (67M parameters) on
the GLUE/SST-2 binary sentiment task. Plain HuggingFace `Trainer` with
`AutoModelForSequenceClassification`. Trains for `MAX_STEPS=500` steps
(~5 min on one A100) and writes final accuracy + the trained model to the
output directory.

**What to learn.** A complete fine-tuning loop on HPC: tokenisation,
`Trainer`, evaluation, checkpointing — and the same script scaling from one
GPU to many with no code changes.

### 5.1. Single GPU

```bash
bash slurm/submit.sh bert_finetune
```

Watch loss decay and accuracy rise:

```bash
RUN_DIR=$(ls -dt /scratch/p_oe_hpc/$USER/oe_hpc_ai/output/bert_finetune_* | head -1)
tail -f "$RUN_DIR/slurm.out"
```

Final accuracy is written to `$RUN_DIR/final_metrics.json`. Expect ~0.86–0.89
on the GLUE/SST-2 validation split.

### 5.2. Multi-GPU (DDP) on the same script

```bash
bash slurm/submit.sh bert_finetune_ddp
```

This is the **same Python script**. The only differences are:

- the sbatch requests `--gres=gpu:2` (and more memory / CPUs);
- the launcher is `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE` instead of
  `python3`.

`torchrun` starts one process per GPU and sets up the rendezvous. HuggingFace
`Trainer` then auto-detects `torch.distributed` and switches to DDP. Per-GPU
batch size stays the same; effective batch size doubles. This is what
"linear weak scaling" looks like in practice — you'll see roughly half the
wall-clock time for the same `MAX_STEPS`, with a comparable final accuracy.

### 5.3. Scaling to 4 GPUs

To use 4 GPUs instead of 2, edit the `#SBATCH` directives at the top of
`slurm/bert_finetune_ddp.sh`:

```bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
```

then resubmit with `bash slurm/submit.sh bert_finetune_ddp`. The
`NPROC=${SLURM_GPUS_ON_NODE}` line further down auto-scales `torchrun`, so no
other change is needed.

---

## 6. Demo 3 — agentic SFT on Qwen3-0.6B

**What it does.** Full-parameter supervised fine-tuning of `Qwen/Qwen3-0.6B`
on the first 1000 rows of `Salesforce/xlam-function-calling-60k`. Each row
contains a user query and a list of tool definitions; the assistant target
is one or more `<tool_call>{...}</tool_call>` blocks formatted with the
Qwen3 native tool-calling chat template. After training, the script runs
two sanity-check prompts and writes the model outputs to
`sanity_check.json`.

**Why this is the "agentic" demo.** The training loop itself is plain SFT —
nothing about it is special. What makes the resulting model agentic is
the **data format**: tool definitions + structured tool-call targets, all
rendered through `tokenizer.apply_chat_template(..., tools=..., enable_thinking=False)`.
That is the takeaway: agentic training is a data problem, not a training-loop
problem.

**Why full fine-tune and not LoRA.** At 0.6B parameters the model fits
comfortably even at this batch size, and dropping LoRA keeps the script as
small as possible. In a research setting you would likely use a LoRA adapter
to save VRAM and disk, especially at larger sizes — but that is orthogonal
to the agentic part.

### 6.1. Single GPU

```bash
bash slurm/submit.sh agentic_sft
```

Expect ~5–8 minutes of wall-clock time on one A100 with the default
`MAX_STEPS=200`. You'll see the training loss in the log:

```bash
RUN_DIR=$(ls -dt /scratch/p_oe_hpc/$USER/oe_hpc_ai/output/agentic_sft_* | head -1)
tail -f "$RUN_DIR/slurm.out"
```

After training, two sanity prompts are decoded and printed; their full
outputs land in `$RUN_DIR/sanity_check.json`. A successfully tuned model
emits responses that contain `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
blocks instead of free-text answers.

### 6.2. Multi-GPU (DDP)

```bash
bash slurm/submit.sh agentic_sft_ddp
```

Same script, `torchrun --nproc_per_node=N`. Same observation as for BERT:
half the wall clock, same per-GPU batch size, double the effective batch.
With this small a dataset (1000 samples) and short a run, the final loss
will be similar across 1, 2, and 4 GPU setups.

---

## 7. Where outputs live, and cleanup

Each job produces:

```
/scratch/p_oe_hpc/$USER/oe_hpc_ai/output/<job>_<jobid>/
├── slurm.out                  # captured stdout
├── slurm.err                  # captured stderr
├── results.json               # (Demo 1 only)
├── final_metrics.json         # (Demo 2 + 3)
├── sanity_check.json          # (Demo 3 only)
├── checkpoint-*/              # (training demos)
└── (model artifacts)          # *.safetensors, tokenizer files, etc.
```

The HuggingFace cache lives at `/scratch/p_oe_hpc/$USER/hf_cache/` and is
shared across all jobs. Do **not** delete it between runs — re-downloading
takes minutes.

To clean up your output directory:

```bash
rm -rf /scratch/p_oe_hpc/$USER/oe_hpc_ai/output/
```

---

## 8. Reference — file layout

```
oe_hpc_ai/
├── README.md                       # this file
├── singularity/
│   └── transformers-gpu.def        # def file: builds the .sif from the Docker image
├── scripts/
│   ├── prefetch_hf.sh              # login-node wrapper around prefetch_hf.py
│   └── prefetch_hf.py              # downloads models + datasets into HF_HOME
├── src/
│   ├── prompts.txt                 # default prompts for Demo 1
│   ├── inference.py                # Demo 1
│   ├── bert_finetune.py            # Demo 2 (single GPU + DDP — same script)
│   └── agentic_sft.py              # Demo 3 (single GPU + DDP — same script)
└── slurm/
    ├── submit.sh                   # wrapper: pre-creates run dir, attaches account+reservation
    ├── inference.sh                # Demo 1 sbatch (1 GPU)
    ├── bert_finetune.sh            # Demo 2 sbatch (1 GPU)
    ├── bert_finetune_ddp.sh        # Demo 2 sbatch (2 GPUs, scales to 4)
    ├── agentic_sft.sh              # Demo 3 sbatch (1 GPU)
    └── agentic_sft_ddp.sh          # Demo 3 sbatch (2 GPUs, scales to 4)
```

Every Python parameter has a default at the top of the corresponding sbatch
script, written as `export VAR=${VAR:-default_value}`. To change a
hyperparameter, edit the default in the sbatch script and resubmit.

---

## 9. Container image — how it was built

The `.sif` at `/scratch/p_oe_hpc/transformers-gpu.sif` was built from the
public Docker image `huggingface/transformers-all-latest-gpu` using the
definition file in `singularity/transformers-gpu.def`. To rebuild it (only
needed if the upstream image is updated), run on a login node with
`fakeroot` enabled:

```bash
module load singularity/4.0
singularity build --fakeroot \
    /scratch/p_oe_hpc/transformers-gpu.sif \
    singularity/transformers-gpu.def
```

The image already contains `torch`, `transformers`, `datasets`, `accelerate`,
`peft`, `tokenizers` and all common HF dependencies, so no `pip install` is
needed inside the jobs.
