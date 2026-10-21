"""
Minimal SFT (LoRA) launcher for the CAEI step-mft strategy.

Bypasses LLaMA-Factory — uses transformers.Trainer + peft directly to
avoid the silent-exit issues we saw on transformers 4.55 + trl 0.24.

Reads the alpaca-format JSON produced by filter_sft_data.py, formats
each sample with the Qwen chat template, and runs LoRA fine-tuning.
"""

import os
import sys

# CUDA_VISIBLE_DEVICES must be set BEFORE `import torch`, otherwise torch
# initializes its context with all available GPUs and ignores later changes.
VISIBLE_GPUS = os.environ.get("FINETUNE_VISIBLE_GPUS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------- configuration -----------------------------
# All knobs the README / paper use live here. Edit these rather than
# passing env vars on the command line.

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_DATA_FILE = "sft_data_step_mft.json"
OUTPUT_DIR = "lora_output/caei_step_mft"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 2048
LOGGING_STEPS = 1           # small dataset — log every step
SAVE_STEPS = 200            # 71 samples / batch 8 = 9 steps total
WARMUP_RATIO = 0.1


# ----------------------------- helpers ---------------------------------


def build_dataset(tokenizer):
    """Load sft_data_step_mft.json, format with Qwen chat template."""
    with open(SFT_DATA_FILE) as f:
        raw = json.load(f)
    print(f"[finetune] Loaded {len(raw)} SFT samples from {SFT_DATA_FILE}")

    def render(sample):
        # Qwen chat-template format
        msgs = [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list(raw).map(render, remove_columns=["instruction", "input", "output"])

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    print(f"[finetune] Tokenized dataset: {len(ds)} samples")
    return ds


def main():
    if not os.path.exists(SFT_DATA_FILE):
        print(f"[finetune] ERROR: {SFT_DATA_FILE} not found in {os.getcwd()}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[finetune] Loading tokenizer + model from {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # single GPU (CUDA_VISIBLE_DEVICES=1 → cuda:0)
    )
    model.config.use_cache = False  # required for grad checkpointing

    print(f"[finetune] Injecting LoRA (r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})")
    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = build_dataset(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        bf16=True,
        report_to=[],          # no W&B
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print(f"[finetune] Starting training on cuda:0 (CUDA_VISIBLE_DEVICES={VISIBLE_GPUS})")
    trainer.train()

    print(f"[finetune] Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[finetune] Done. Adapter at {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
