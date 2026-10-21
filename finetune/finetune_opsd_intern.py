"""
OPSD-INTERN: On-Policy Self-Distillation for strategy *internalization*.

Differs from finetune_opsd.py in TWO ways:
    1. Output space: this trains the model to produce **actions** (not strategies).
       The student is shown a problem + admissible actions; on-policy generation
       produces an action; the teacher is shown the same problem + the reference
       *strategy* and its action distribution is the distillation target.

    2. Data layout: each sample corresponds to ONE step of a successful
       trajectory. The dataset is built by `filter_traj_intern_data.py`, which
       expands 71 successful trajectories into ~1500 step-level samples.

Pipeline per step (same shape as finetune_opsd.py):
    1. Load (student_prompt, teacher_prompt, target_text) via a tiny collator.
    2. Student on-policy generation on student_prompt (no_grad).
    3. student forward over (student_prompt + student_completion) → student_logits.
    4. teacher forward over (teacher_prompt + student_completion) with LoRA disabled
       → teacher_logits (this is what "the model would emit if it saw the strategy").
    5. JSD(student_logits ‖ teacher_logits) on completion positions only.

The student is pulled toward the teacher distribution, so over time the student
learns to *implicitly produce* the action distribution it would have produced
*if* it had been told the strategy — i.e. it internalizes the strategy.
"""

import os
import sys

# CUDA_VISIBLE_DEVICES must be set BEFORE `import torch` — otherwise torch
# locks onto every visible GPU and ignores later env changes.
# Honor whatever the launcher / torchrun has set; fall back to "0".
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FINETUNE_VISIBLE_GPUS", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------- configuration -----------------------------

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_DATA_FILE = os.environ.get("OPSD_INTERN_DATA_FILE", "data/sft_data_intern.json")
OUTPUT_DIR    = os.environ.get("OPSD_INTERN_OUTPUT_DIR", "lora_output/caei_opsd_intern")

LORA_RANK    = int(os.environ.get("OPSD_INTERN_LORA_RANK", "32"))
LORA_ALPHA   = int(os.environ.get("OPSD_INTERN_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("OPSD_INTERN_LORA_DROPOUT", "0.05"))

NUM_EPOCHS = int(os.environ.get("OPSD_INTERN_NUM_EPOCHS", "2"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("OPSD_INTERN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("OPSD_INTERN_GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("OPSD_INTERN_LR", "1e-4"))
WARMUP_RATIO  = float(os.environ.get("OPSD_INTERN_WARMUP", "0.05"))

# Sequence length knobs (independent of finetune_opsd.py)
MAX_PROMPT_LENGTH     = int(os.environ.get("OPSD_INTERN_MAX_PROMPT", "1024"))
MAX_COMPLETION_LENGTH = int(os.environ.get("OPSD_INTERN_MAX_COMP", "384"))
GENERATION_TEMPERATURE = float(os.environ.get("OPSD_INTERN_TEMP", "0.9"))
GENERATION_TOP_P       = float(os.environ.get("OPSD_INTERN_TOP_P", "0.95"))

JSD_BETA       = float(os.environ.get("OPSD_INTERN_JSD_BETA", "0.5"))
JSD_TEMPERATURE = float(os.environ.get("OPSD_INTERN_JSD_TEMP", "1.0"))

WARMUP_STEPS = 5


# ----------------------------- helpers ---------------------------------


def generalized_jsd_loss(student_logits, teacher_logits, labels,
                         beta=JSD_BETA, temperature=JSD_TEMPERATURE,
                         top_k=None, token_clip=None):
    """Generalized Jensen-Shannon Divergence between teacher and student
    token distributions, masked to completion-token positions."""
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature

    if top_k is not None and top_k > 0:
        _, top_k_indices = torch.topk(t_logits, k=top_k, dim=-1)
        s_logits = torch.gather(s_logits, dim=-1, index=top_k_indices)
        t_logits = torch.gather(t_logits, dim=-1, index=top_k_indices)

    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(s_log_probs, t_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(t_log_probs, s_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([s_log_probs + torch.log1p(-beta_t),
                         t_log_probs + torch.log(beta_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    mask = labels != -100
    jsd = jsd[mask]
    n = mask.sum().clamp_min(1)
    return jsd.sum() / n


class InternCollator:
    """Tiny collator: tokenize (student_prompt, teacher_prompt) into padded
    batches. Each item already comes from filter_traj_intern_data.py with
    `instruction` (student) and `teacher_prompt` fields."""

    def __init__(self, tokenizer, max_prompt_len=MAX_PROMPT_LENGTH):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_prompt_len = max_prompt_len

    def __call__(self, features):
        student_prompts = [f["instruction"] for f in features]
        teacher_prompts = [f["teacher_prompt"] for f in features]

        # Apply chat template (student + teacher both use the same template;
        # teacher's content already includes the reference strategy in plain
        # text, not as a chat role).
        def render(prompt):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )

        student_texts = [render(p) for p in student_prompts]
        teacher_texts = [render(p) for p in teacher_prompts]

        student_enc = self.tokenizer(
            student_texts, padding=True, truncation=True,
            max_length=self.max_prompt_len, return_tensors="pt",
        )
        teacher_enc = self.tokenizer(
            teacher_texts, padding=True, truncation=True,
            max_length=self.max_prompt_len, return_tensors="pt",
        )
        return {
            "student_prompts":           student_enc["input_ids"],
            "student_prompt_attention_mask": student_enc["attention_mask"],
            "student_prompt_length":     student_enc["input_ids"].shape[1],
            "teacher_prompts":           teacher_enc["input_ids"],
            "teacher_prompt_attention_mask": teacher_enc["attention_mask"],
            "teacher_prompt_length":     teacher_enc["input_ids"].shape[1],
        }


def load_dataset():
    """Load step-level trajectory data produced by filter_traj_intern_data.py."""
    with open(SFT_DATA_FILE) as f:
        raw = json.load(f)
    print(f"[opsd-intern] loaded {len(raw)} samples from {SFT_DATA_FILE}")
    return Dataset.from_list(raw)


# ----------------------------- main loop --------------------------------


def main():
    if not os.path.exists(SFT_DATA_FILE):
        print(f"[opsd-intern] ERROR: {SFT_DATA_FILE} not found")
        return 1

    # ---------- DDP / torchrun setup ----------
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = (local_rank == 0)
    if world_size > 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    device = torch.device("cuda:0")    # torchrun binds each rank to cuda:0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    print(f"[opsd-intern rank={local_rank}/{world_size}] loading tokenizer + base")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.config.use_cache = False

    print(f"[opsd-intern rank={local_rank}] injecting LoRA")
    peft_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    student_model = get_peft_model(base, peft_cfg)
    student_model = student_model.to(torch.bfloat16)
    if is_main:
        student_model.print_trainable_parameters()

    if world_size > 1:
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model,
            device_ids=[0],
            output_device=0,
            find_unused_parameters=True,
        )
        student_model._set_static_graph()
        inner_model = student_model.module
    else:
        student_model.to(device)
        inner_model = student_model   # no DDP wrapper, no .module attribute

    train_ds = load_dataset()
    collator = InternCollator(tokenizer)

    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank,
            shuffle=True, drop_last=False,
        )
    else:
        sampler = None

    def sample_batch_iter(ds):
        if sampler is not None:
            for idx in sampler:
                yield [ds[int(idx)]]
        else:
            for i in range(0, len(ds), PER_DEVICE_BATCH_SIZE):
                batch = [ds[j] for j in range(i, min(i + PER_DEVICE_BATCH_SIZE, len(ds)))]
                if not batch:
                    continue
                yield batch

    optimizer = torch.optim.AdamW(
        [p for p in inner_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.1 + 0.9 * min((step + 1) / max(WARMUP_STEPS, 1), 1.0),
    )

    student_model.train()
    losses = []
    grad_accum_count = 0
    optimizer.zero_grad()

    MAX_MICRO_PER_RANK = 200   # 1468 / world_size ≈ 367 per rank (2 GPUs) — cap at 200
    print(f"[opsd-intern] MAX_MICRO_PER_RANK={MAX_MICRO_PER_RANK}, world_size={world_size}")

    for epoch in range(NUM_EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, raw_features in enumerate(sample_batch_iter(train_ds)):
            if step >= MAX_MICRO_PER_RANK:
                break

            # 1. Tokenize prompts
            batch = collator(raw_features)
            student_prompts = batch["student_prompts"].to(device)
            teacher_prompts = batch["teacher_prompts"].to(device)
            sp_attn = batch["student_prompt_attention_mask"].to(device)
            tp_attn = batch["teacher_prompt_attention_mask"].to(device)
            sp_len = batch["student_prompt_length"]
            tp_len = batch["teacher_prompt_length"]

            # 2. Student on-policy generation
            with torch.no_grad():
                gen_out = inner_model.generate(
                    input_ids=student_prompts,
                    attention_mask=sp_attn,
                    max_new_tokens=MAX_COMPLETION_LENGTH,
                    do_sample=True,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
                completion_ids = gen_out[:, student_prompts.shape[1]:]
                comp_attn = torch.ones_like(completion_ids)
                comp_attn[completion_ids == tokenizer.pad_token_id] = 0
            torch.cuda.empty_cache()

            # 3. Build full sequences (prompt + completion), pad right.
            full_student_ids, full_student_attn, full_labels = _assemble_full(
                student_prompts, sp_attn, completion_ids, comp_attn,
                tokenizer.pad_token_id, sp_len,
            )
            full_teacher_ids, full_teacher_attn, _ = _assemble_full(
                teacher_prompts, tp_attn, completion_ids, comp_attn,
                tokenizer.pad_token_id, tp_len,
            )
            full_student_ids = full_student_ids.to(device)
            full_student_attn = full_student_attn.to(device)
            full_teacher_ids = full_teacher_ids.to(device)
            full_teacher_attn = full_teacher_attn.to(device)
            full_labels = full_labels.to(device)

            # 4. Student forward (with grad)
            student_out = student_model(
                input_ids=full_student_ids, attention_mask=full_student_attn,
                use_cache=False,
            )
            student_logits = student_out.logits[:, :-1, :]
            completion_start = sp_len - 1
            student_logits = student_logits[:, completion_start:, :].contiguous()
            shift_labels = full_labels[:, 1 + completion_start:].contiguous()
            del student_out
            torch.cuda.empty_cache()

            # 5. Teacher forward (no grad, LoRA disabled)
            with inner_model.disable_adapter():
                teacher_out = student_model(
                    input_ids=full_teacher_ids, attention_mask=full_teacher_attn,
                    use_cache=False,
                )
                teacher_logits = teacher_out.logits[:, :-1, :]
                teacher_completion_start = tp_len - 1
                teacher_logits = teacher_logits[:, teacher_completion_start:, :].contiguous()
                del teacher_out
            torch.cuda.empty_cache()

            # 6. Shape alignment (student and teacher may have different
            #    completion-region lengths due to padding).
            if teacher_logits.shape[1] != student_logits.shape[1]:
                min_len = min(teacher_logits.shape[1], student_logits.shape[1])
                student_logits = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            # 7. JSD loss
            loss = generalized_jsd_loss(
                student_logits, teacher_logits, shift_labels,
                beta=JSD_BETA, temperature=JSD_TEMPERATURE,
            ) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            losses.append(float(loss.detach()) * GRADIENT_ACCUMULATION_STEPS)
            grad_accum_count += 1

            # Free the two large logit tensors immediately — they peak at
            # ~[1, T, V] where V = 151936 for Qwen3 vocab. Holding both at
            # once is ~5GB which is what tipped 24GB into OOM.
            del student_logits, teacher_logits, shift_labels, loss
            torch.cuda.empty_cache()

            if grad_accum_count >= GRADIENT_ACCUMULATION_STEPS:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_accum_count = 0

                global_step = epoch * (MAX_MICRO_PER_RANK // GRADIENT_ACCUMULATION_STEPS) \
                              + step // GRADIENT_ACCUMULATION_STEPS
                if is_main and global_step % 1 == 0:
                    avg = sum(losses[-GRADIENT_ACCUMULATION_STEPS:]) / GRADIENT_ACCUMULATION_STEPS
                    print(
                        f"[opsd-intern rank=0] epoch={epoch} step={step} global_step={global_step} "
                        f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        # Save per-epoch snapshot (rank 0 only)
        if is_main:
            inner_model.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch+1}")
            tokenizer.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch+1}")
            print(f"[opsd-intern rank=0] saved epoch {epoch+1} to {OUTPUT_DIR}/epoch_{epoch+1}",
                  flush=True)
        if world_size > 1:
            torch.distributed.barrier()

    if is_main:
        print(f"[opsd-intern rank=0] saving final adapter to {OUTPUT_DIR}/", flush=True)
        inner_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"[opsd-intern rank=0] done.", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


# ----------------------------- helpers ---------------------------------


def _assemble_full(prompts, prompt_attn, completion_ids, completion_attn,
                   pad_token_id, prompt_len_scalar):
    """Right-pad prompt+completion to common length; labels mask prompt+pad."""
    B = prompts.shape[0]
    full_ids_list, full_attn_list, full_labels_list = [], [], []
    for i in range(B):
        p_ids = prompts[i]
        p_attn = prompt_attn[i]
        c_ids = completion_ids[i]
        c_attn = completion_attn[i]
        p_real_len = int(p_attn.sum().item())
        p_ids = p_ids[:p_real_len]
        c_ids = c_ids[: int(c_attn.sum().item())][:MAX_COMPLETION_LENGTH]
        seq = torch.cat([p_ids, c_ids], dim=0)
        attn = torch.cat([torch.ones_like(p_ids), torch.ones_like(c_ids)], dim=0)
        labels = torch.cat([torch.full_like(p_ids, -100), c_ids.clone()], dim=0)
        full_ids_list.append(seq); full_attn_list.append(attn); full_labels_list.append(labels)
    max_len = max(s.shape[0] for s in full_ids_list)
    out_ids, out_attn, out_labels = [], [], []
    for seq, attn, lab in zip(full_ids_list, full_attn_list, full_labels_list):
        pad = max_len - seq.shape[0]
        if pad > 0:
            seq = torch.cat([seq, torch.full((pad,), pad_token_id, dtype=seq.dtype)])
            attn = torch.cat([attn, torch.zeros(pad, dtype=attn.dtype)])
            lab = torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)])
        out_ids.append(seq); out_attn.append(attn); out_labels.append(lab)
    return torch.stack(out_ids), torch.stack(out_attn), torch.stack(out_labels)


if __name__ == "__main__":
    sys.exit(main())