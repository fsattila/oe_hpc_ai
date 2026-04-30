"""
agentic_sft.py — Demo 3: full-parameter SFT of a small Qwen3 model on a
function-calling dataset.

This is the "agentic" demo: nothing about the training loop is special — it is
the same supervised fine-tuning pipeline as a regular causal-LM SFT. What makes
the resulting model agentic is the *data format*: every training example is a
user query + a list of tool definitions, and the assistant target is one or
more <tool_call>...</tool_call> blocks rendered with the Qwen3 chat template.

Runs as:
  * single-GPU:   python agentic_sft.py
  * multi-GPU:    torchrun --nproc_per_node=N agentic_sft.py
The Trainer auto-detects torch.distributed and switches to DDP.

All parameters resolve as: CLI flag > env var > default.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic SFT demo (Qwen3 + xlam)")
    p.add_argument("--model_name_or_path",
                   default=os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-0.6B"))
    p.add_argument("--dataset_name_or_path",
                   default=os.environ.get("DATASET_NAME_OR_PATH",
                                          "Salesforce/xlam-function-calling-60k"))
    p.add_argument("--output_dir",
                   default=os.environ.get("OUTPUT_DIR", "/output"))
    p.add_argument("--max_train_samples", type=int,
                   default=int(os.environ.get("MAX_TRAIN_SAMPLES", "1000")))
    p.add_argument("--max_seq_len", type=int,
                   default=int(os.environ.get("MAX_SEQ_LEN", "1024")))
    p.add_argument("--per_device_train_batch_size", type=int,
                   default=int(os.environ.get("BATCH_SIZE", "4")))
    p.add_argument("--gradient_accumulation_steps", type=int,
                   default=int(os.environ.get("GRAD_ACCUM", "4")))
    p.add_argument("--learning_rate", type=float,
                   default=float(os.environ.get("LEARNING_RATE", "2e-5")))
    p.add_argument("--max_steps", type=int,
                   default=int(os.environ.get("MAX_STEPS", "200")))
    p.add_argument("--warmup_steps", type=int,
                   default=int(os.environ.get("WARMUP_STEPS", "20")))
    p.add_argument("--weight_decay", type=float,
                   default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--logging_steps", type=int,
                   default=int(os.environ.get("LOGGING_STEPS", "10")))
    p.add_argument("--save_steps", type=int,
                   default=int(os.environ.get("SAVE_STEPS", "200")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def xlam_tool_to_oai(tool: dict) -> dict:
    """Convert an xlam tool spec to the OpenAI/Qwen function-call schema.

    The xlam dataset signals optionality with a `default` key: parameters with
    a `default` are optional, the rest are required. Qwen's chat template
    expects a JSON-schema `type: object` wrapper with an explicit `required`
    list at the top level.
    """
    props: dict = {}
    required: list[str] = []
    for pname, p in tool.get("parameters", {}).items():
        prop: dict = {
            "type": p.get("type", "string"),
            "description": p.get("description", ""),
        }
        if "default" in p:
            prop["default"] = p["default"]
        else:
            required.append(pname)
        props[pname] = prop
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def render_sample(sample: dict, tokenizer) -> tuple[str, str]:
    """Return (prompt_text, target_text) for one xlam row using Qwen3 chat template."""
    query = sample["query"]
    tools_in = json.loads(sample["tools"])
    answers = json.loads(sample["answers"])

    tools = [xlam_tool_to_oai(t) for t in tools_in]
    messages = [{"role": "user", "content": query}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    if len(answers) == 0:
        body = "I don't have a tool that can help with this request."
    else:
        body = "\n".join(
            "<tool_call>\n"
            + json.dumps({"name": a["name"], "arguments": a["arguments"]},
                         ensure_ascii=False)
            + "\n</tool_call>"
            for a in answers
        )
    target_text = body + "<|im_end|>\n"
    return prompt_text, target_text


def build_tokenized_dataset(raw_dataset, tokenizer, max_seq_len: int):
    def tokenize(sample):
        prompt_text, target_text = render_sample(sample, tokenizer)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_seq_len]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)
    tokenized = tokenized.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    return tokenized


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"[sft] model={args.model_name_or_path}", flush=True)
        print(f"[sft] dataset={args.dataset_name_or_path}", flush=True)
        print(f"[sft] world_size={os.environ.get('WORLD_SIZE', '1')} "
              f"local_rank={os.environ.get('LOCAL_RANK', '0')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_dataset(args.dataset_name_or_path)
    train_raw = ds["train"]
    if args.max_train_samples > 0:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))

    train_ds = build_tokenized_dataset(train_raw, tokenizer, args.max_seq_len)
    if is_main:
        print(f"[sft] train rows after filtering: {len(train_ds)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)

    if is_main:
        metrics = train_result.metrics
        print(f"[sft] training done: {metrics}", flush=True)
        with open(Path(args.output_dir) / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        sanity_prompts = [
            ("What is the weather in Budapest tomorrow?",
             [{"name": "get_weather",
               "description": "Get weather forecast for a city.",
               "parameters": {"city": {"type": "string",
                                       "description": "City name."},
                              "date": {"type": "string",
                                       "description": "ISO date.",
                                       "default": "today"}}}]),
            ("Add 17 and 25, then multiply the result by 3.",
             [{"name": "add",
               "description": "Add two numbers.",
               "parameters": {"a": {"type": "number", "description": "First."},
                              "b": {"type": "number", "description": "Second."}}},
              {"name": "multiply",
               "description": "Multiply two numbers.",
               "parameters": {"a": {"type": "number", "description": "First."},
                              "b": {"type": "number", "description": "Second."}}}]),
        ]
        model.gradient_checkpointing_disable()
        model.eval()
        sanity_outputs = []
        for query, tools in sanity_prompts:
            tools_oai = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": {
                        "type": "object",
                        "properties": dict(t["parameters"]),
                        "required": [k for k, v in t["parameters"].items()
                                     if "default" not in v],
                    },
                },
            } for t in tools]
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tools=tools_oai,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_ids, skip_special_tokens=False)
            print(f"[sft][sanity] query: {query}", flush=True)
            print(f"[sft][sanity] output: {completion}", flush=True)
            sanity_outputs.append({"query": query, "output": completion})

        with open(Path(args.output_dir) / "sanity_check.json", "w") as f:
            json.dump(sanity_outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
