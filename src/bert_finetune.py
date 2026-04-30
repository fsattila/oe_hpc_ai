"""
bert_finetune.py — Demo 2: fine-tune DistilBERT on SST-2 sentiment classification.

Trains a sequence classifier on the GLUE/SST-2 split using a plain HuggingFace
Trainer. The same script runs:
  * single-GPU:   python bert_finetune.py
  * multi-GPU:    torchrun --nproc_per_node=N bert_finetune.py
The Trainer auto-detects torch.distributed and switches to DDP without any
code change — that is the whole point of this demo.

All parameters resolve as: CLI flag > env var > default.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DistilBERT SST-2 fine-tuning demo")
    p.add_argument("--model_name_or_path",
                   default=os.environ.get("MODEL_NAME_OR_PATH", "distilbert-base-uncased"))
    p.add_argument("--dataset_name",
                   default=os.environ.get("DATASET_NAME", "glue"))
    p.add_argument("--dataset_config",
                   default=os.environ.get("DATASET_CONFIG", "sst2"))
    p.add_argument("--output_dir",
                   default=os.environ.get("OUTPUT_DIR", "/output"))
    p.add_argument("--max_seq_len", type=int,
                   default=int(os.environ.get("MAX_SEQ_LEN", "128")))
    p.add_argument("--per_device_train_batch_size", type=int,
                   default=int(os.environ.get("BATCH_SIZE", "32")))
    p.add_argument("--per_device_eval_batch_size", type=int,
                   default=int(os.environ.get("EVAL_BATCH_SIZE", "64")))
    p.add_argument("--learning_rate", type=float,
                   default=float(os.environ.get("LEARNING_RATE", "5e-5")))
    p.add_argument("--max_steps", type=int,
                   default=int(os.environ.get("MAX_STEPS", "500")))
    p.add_argument("--warmup_steps", type=int,
                   default=int(os.environ.get("WARMUP_STEPS", "50")))
    p.add_argument("--weight_decay", type=float,
                   default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--logging_steps", type=int,
                   default=int(os.environ.get("LOGGING_STEPS", "20")))
    p.add_argument("--eval_steps", type=int,
                   default=int(os.environ.get("EVAL_STEPS", "100")))
    p.add_argument("--save_steps", type=int,
                   default=int(os.environ.get("SAVE_STEPS", "500")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"[bert] model={args.model_name_or_path}", flush=True)
        print(f"[bert] dataset={args.dataset_name}/{args.dataset_config}", flush=True)
        print(f"[bert] world_size={os.environ.get('WORLD_SIZE', '1')} "
              f"local_rank={os.environ.get('LOCAL_RANK', '0')}", flush=True)

    raw = load_dataset(args.dataset_name, args.dataset_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_seq_len,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        seed=args.seed,
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    if is_main:
        print(f"[bert] final metrics: {metrics}", flush=True)
        with open(Path(args.output_dir) / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
