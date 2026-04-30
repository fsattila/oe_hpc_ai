"""
prefetch_hf.py — pre-download HuggingFace models and datasets into a local cache.

Compute nodes on Komondor have no internet access, so anything used during
training/inference must already live in HF_HOME on shared scratch. Run this on
a login node (which has internet) once before submitting any SLURM jobs.

Models and datasets are taken from MODELS / DATASETS env vars (comma-separated)
or the CLI flags --models / --datasets. The cache root is taken from HF_HOME.
"""

import argparse
import os


DEFAULT_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    "distilbert-base-uncased",
]

DEFAULT_DATASETS = [
    "glue:sst2",
    "Salesforce/xlam-function-calling-60k",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download HF assets to HF_HOME.")
    p.add_argument("--models",
                   default=os.environ.get("MODELS", ",".join(DEFAULT_MODELS)),
                   help="Comma-separated model repo IDs.")
    p.add_argument("--datasets",
                   default=os.environ.get("DATASETS", ",".join(DEFAULT_DATASETS)),
                   help='Comma-separated dataset repo IDs. Use "name:config" for'
                        ' configured datasets, e.g. "glue:sst2".')
    p.add_argument("--cache_dir",
                   default=os.environ.get("HF_HOME"),
                   help="HF cache root (overrides HF_HOME for this run).")
    return p.parse_args()


def split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    print(f"[prefetch] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    for model_id in split_csv(args.models):
        print(f"[prefetch] model: {model_id}", flush=True)
        path = snapshot_download(repo_id=model_id, repo_type="model")
        print(f"[prefetch]   -> {path}", flush=True)

    for ds_spec in split_csv(args.datasets):
        if ":" in ds_spec:
            name, config = ds_spec.split(":", 1)
        else:
            name, config = ds_spec, None
        print(f"[prefetch] dataset: {name} config={config}", flush=True)
        ds = load_dataset(name, config) if config else load_dataset(name)
        print(f"[prefetch]   -> splits={list(ds.keys())}", flush=True)


if __name__ == "__main__":
    main()
