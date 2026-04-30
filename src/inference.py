"""
inference.py — Demo 1: text generation with a small instruction-tuned LLM.

Loads a HuggingFace causal LM, runs greedy generation on a list of prompts read
from a text file (one prompt per line) and writes the results to a JSON file
inside --output_dir.

Every parameter resolves as: CLI flag > env var > default. Run inside the
provided Singularity image; the SLURM wrapper sets HF_HOME so that weights
are read from the pre-populated /scratch cache (compute nodes have no
internet).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-GPU LLM inference demo")
    p.add_argument("--model_name_or_path",
                   default=os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-1.7B"))
    p.add_argument("--prompts_file",
                   default=os.environ.get("PROMPTS_FILE", "/workspace/src/prompts.txt"))
    p.add_argument("--output_dir",
                   default=os.environ.get("OUTPUT_DIR", "/output"))
    p.add_argument("--max_new_tokens", type=int,
                   default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def load_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[inference] model={args.model_name_or_path}", flush=True)
    print(f"[inference] prompts_file={args.prompts_file}", flush=True)
    print(f"[inference] output_dir={args.output_dir}", flush=True)
    print(f"[inference] cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = load_prompts(args.prompts_file)
    print(f"[inference] loaded {len(prompts)} prompts", flush=True)

    results = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)

        n_new = int(new_ids.shape[0])
        tps = n_new / dt if dt > 0 else 0.0
        print(f"[inference] prompt {i+1}/{len(prompts)} "
              f"new_tokens={n_new} time={dt:.2f}s tok/s={tps:.1f}", flush=True)

        results.append({
            "prompt": prompt,
            "completion": completion,
            "new_tokens": n_new,
            "seconds": dt,
            "tokens_per_second": tps,
        })

    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_name_or_path, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"[inference] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
