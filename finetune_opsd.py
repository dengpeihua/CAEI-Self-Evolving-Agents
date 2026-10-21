"""
OPSD (On-Policy Self-Distillation) LoRA trainer for CAEI strategy.

Reuses:
    trainer/data_collator.py : SelfDistillationDataCollator (constructs
        student_prompt / teacher_prompt from each sample's problem/solution).

Algorithm per step (matches opsd_trainer.py at trl upstream):
    1. Build student and teacher prompts via SelfDistillationDataCollator.
    2. Student prompt → on-policy generate continuation (model.generate, no_grad).
    3. Concatenate student_generated_ids onto both student_prompt and teacher_prompt
       (so teacher forward uses the student's actual completion, not a reference).
    4. student forward → student_logits (with LoRA).
    5. teacher forward → teacher_logits (LoRA disabled — frozen base model acts
       as "expert who sees privileged info in the prompt").
    6. JSD loss on completion positions only (labels=-100 elsewhere).
    7. backward on student only.

This is the "self-distillation with privileged information" formulation:
    student:  sees (problem) + generates continuation C
    teacher:  sees (problem + reference solution) + teacher continuation C
    → teacher logits encode "what the optimal strategy would look like given
       the answer", student logits encode "what we'd produce naively".
    We pull student toward teacher (with KL forward) so student learns to
    produce teacher-style output even without seeing the solution.
"""

import os
import sys

# CUDA_VISIBLE_DEVICES must be set BEFORE `import torch` — otherwise torch
# locks onto every visible GPU and ignores later env changes.
# Honor whatever the launcher / torchrun has set; fall back to "0".
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FINETUNE_VISIBLE_GPUS", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports (data_collator lives in trainer/ next to opsd_trainer.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer"))

import json
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from contextlib import nullcontext

from data_collator import SelfDistillationDataCollator  # noqa: E402


# ----------------------------- configuration -----------------------------

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_DATA_FILE = os.environ.get("OPSD_SFT_DATA_FILE", "sft_data_step_mft.json")  # alpaca: {instruction, output}
OUTPUT_DIR = os.environ.get("OPSD_OUTPUT_DIR", "lora_output/caei_opsd")
TEACHER_LORA_DIR = "lora_output/caei_step_mft"  # optional: re-use SFT-trained LoRA as teacher

LORA_RANK = int(os.environ.get("OPSD_LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("OPSD_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("OPSD_LORA_DROPOUT", "0.05"))

NUM_EPOCHS = int(os.environ.get("OPSD_NUM_EPOCHS", "3"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("OPSD_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("OPSD_GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("OPSD_LR", "1e-4"))
WARMUP_RATIO = float(os.environ.get("OPSD_WARMUP", "0.1"))

# OPSD-specific
MAX_PROMPT_LENGTH = 768        # student/teacher prompt chunk (FSDP'd across 4 GPUs, KV cache manageable)
MAX_COMPLETION_LENGTH = 256    # on-policy generation cap
GENERATION_TEMPERATURE = 0.9
GENERATION_TOP_P = 0.95

# Forward-loss
JSD_BETA = 0.5                 # 0 = pure forward KL (SFT-like), 1 = pure reverse. 0.5 = symmetric.
JSD_TEMPERATURE = 1.0          # softmax temperature for distribution comparison
JSD_TOP_K = None               # restrict distillation to top-k teacher tokens (None = full vocab)
JSD_TOKEN_CLIP = None

WARMUP_STEPS = 5
LOG_EVERY_STEPS = 1
SAVE_EVERY_STEPS = 200

# ----------------------- data: alpaca → problem/solution -----------------


def load_dataset(tokenizer):
    """Load alpaca SFT file and re-key as {problem, solution} so the
    SelfDistillationDataCollator can read it.
    """
    with open(SFT_DATA_FILE) as f:
        raw = json.load(f)
    print(f"[opsd] Loaded {len(raw)} samples from {SFT_DATA_FILE}")

    def project(s):
        return {"problem": s["instruction"], "solution": s["output"]}

    return Dataset.from_list([project(r) for r in raw])


# ----------------------- JSD loss (copied from opsd_trainer.py) -----------


def generalized_jsd_loss(student_logits, teacher_logits, labels,
                         beta=JSD_BETA, temperature=JSD_TEMPERATURE,
                         top_k=JSD_TOP_K, token_clip=JSD_TOKEN_CLIP):
    """Generalized Jensen-Shannon Divergence between teacher and student
    token distributions, masked to completion-token positions.

    student_logits / teacher_logits: [B, T, V] (already shifted if needed).
    labels: [B, T] with -100 on prompt / padded positions.
    """
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


# ----------------------------- main loop -----------------------------


def main():
    if not os.path.exists(SFT_DATA_FILE):
        print(f"[opsd] ERROR: {SFT_DATA_FILE} not found")
        return 1

    # ---------- DDP / torchrun setup ----------
    # torchrun --nproc_per_node=N sets LOCAL_RANK=0..N-1 in each subprocess.
    # With CUDA_VISIBLE_DEVICES=0,1,2,3, the subprocess at local_rank=i only
    # sees physical GPU i as cuda:0 — but `local_rank` here is the *physical*
    # ordinal torchrun wants us to use. We don't call set_device because
    # torchrun handles device binding; instead we just bind device_map to 0
    # and rely on torchrun to put each process on the right card.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = (local_rank == 0)
    if world_size > 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    # All ranks see only their own GPU as cuda:0 (torchrun masks the rest).
    # device_map={"": 0} maps the model to "the local cuda:0".
    device = torch.device("cuda:0")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize the process group BEFORE model load (FSDP needs it during
    # the auto-wrap pass).
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    print(f"[opsd rank={local_rank}/{world_size}] Loading tokenizer + base model")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        # torchrun binds each rank to one GPU and masks the rest via
        # CUDA_VISIBLE_DEVICES, so every rank only sees cuda:0 — load there.
        device_map={"": 0},
    )
    base.config.use_cache = False

    print(f"[opsd rank={local_rank}] Injecting LoRA (r={LORA_RANK})")
    peft_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    student_model = get_peft_model(base, peft_cfg)
    # peft's LoRA layers default to fp32, but FSDP requires uniform dtype to
    # flatten parameters across ranks. Cast the whole model to bf16.
    student_model = student_model.to(torch.bfloat16)
    if is_main:
        student_model.print_trainable_parameters()

    # ---------- Distributed wrap: simple DDP (single full copy per rank) ----------
    if world_size > 1:
        # FSDP path was tried but caused dtype mismatch + shape collapse with
        # peft disable_adapter(). Reverting to DDP — the OOM fix is to shrink
        # MAX_PROMPT_LENGTH and MAX_COMPLETION_LENGTH instead (see top).
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model,
            device_ids=[0],     # torchrun already bound this rank to cuda:0
            output_device=0,
            find_unused_parameters=True,    # required for peft disable_adapter
        )
        # `disable_adapter()` re-enters forward with different param "active"
        # set, which trips DDP's autograd hook counter. _set_static_graph
        # tells DDP to reuse the same hooks across iters, fixing the
        # "marked as ready twice" error.
        student_model._set_static_graph()
    else:
        student_model.to(device)
    inner_model = student_model.module

    # ---------- Data: DistributedSampler ----------
    train_ds_full = load_dataset(tokenizer)
    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        reason_first=False,
        teacher_thinking=False,
    )
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            train_ds_full, num_replicas=world_size, rank=local_rank,
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

    # ---------- Optimizer (only on LoRA params) ----------
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
    MAX_MICRO_PER_RANK = 30   # 71 samples / world_size ranks; cap at 30 per rank for safety
    # Examples: 71/4 ≈ 18 (4 GPUs), 71/2 ≈ 35 (2 GPUs), 71/1 = 71 (single GPU)
    # The cap at 30 keeps step counts comparable across GPU counts.
    for epoch in range(NUM_EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, raw_features in enumerate(sample_batch_iter(train_ds_full)):
            if step >= MAX_MICRO_PER_RANK:
                break

            # 1. Build prompts via collator
            batch = collator(raw_features)
            student_prompts = batch["student_prompts"].to(device)
            teacher_prompts = batch["teacher_prompts"].to(device)
            sp_attn = batch["student_prompt_attention_mask"].to(device)
            tp_attn = batch["teacher_prompt_attention_mask"].to(device)
            sp_len = batch["student_prompt_length"]

            # 2. Student on-policy generation (no_grad — only need token ids)
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

            # 3. Build full sequences: prompt + completion, pad right
            full_student_ids, full_student_attn, full_labels = _assemble_full(
                student_prompts, sp_attn, completion_ids, comp_attn,
                tokenizer.pad_token_id, sp_len,
            )
            full_teacher_ids, full_teacher_attn, _ = _assemble_full(
                teacher_prompts, tp_attn, completion_ids, comp_attn,
                tokenizer.pad_token_id, batch["teacher_prompt_length"],
            )
            full_student_ids = full_student_ids.to(device)
            full_student_attn = full_student_attn.to(device)
            full_teacher_ids = full_teacher_ids.to(device)
            full_teacher_attn = full_teacher_attn.to(device)
            full_labels = full_labels.to(device)

            # 4. Student forward (with grad) — logits on completion positions only
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

            # 5. Teacher forward (no grad, LoRA disabled — frozen base acts as "expert")
            with inner_model.disable_adapter():
                teacher_out = student_model(
                    input_ids=full_teacher_ids, attention_mask=full_teacher_attn,
                    use_cache=False,
                )
                teacher_logits = teacher_out.logits[:, :-1, :]
                teacher_completion_start = batch["teacher_prompt_length"] - 1
                teacher_logits = teacher_logits[:, teacher_completion_start:, :].contiguous()
                del teacher_out
            torch.cuda.empty_cache()

            # 6. Shape alignment
            if teacher_logits.shape[1] != student_logits.shape[1]:
                min_len = min(teacher_logits.shape[1], student_logits.shape[1])
                student_logits = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            # 7. JSD loss
            loss = generalized_jsd_loss(
                student_logits, teacher_logits, shift_labels,
                beta=JSD_BETA, temperature=JSD_TEMPERATURE,
                top_k=JSD_TOP_K, token_clip=JSD_TOKEN_CLIP,
            ) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            losses.append(float(loss.detach()) * GRADIENT_ACCUMULATION_STEPS)
            grad_accum_count += 1

            if grad_accum_count >= GRADIENT_ACCUMULATION_STEPS:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_accum_count = 0

                global_step = epoch * (MAX_MICRO_PER_RANK // GRADIENT_ACCUMULATION_STEPS) \
                              + step // GRADIENT_ACCUMULATION_STEPS
                if is_main and global_step % LOG_EVERY_STEPS == 0:
                    avg = sum(losses[-GRADIENT_ACCUMULATION_STEPS:]) / GRADIENT_ACCUMULATION_STEPS
                    print(
                        f"[opsd rank=0] epoch={epoch} step={step} global_step={global_step} "
                        f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        # Save every epoch (rank 0 only)
        if is_main:
            inner_model.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch+1}")
            tokenizer.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch+1}")
            print(f"[opsd rank=0] Saved epoch {epoch+1} to {OUTPUT_DIR}/epoch_{epoch+1}", flush=True)
        if world_size > 1:
            torch.distributed.barrier()

    if is_main:
        print(f"[opsd rank=0] Saving final adapter to {OUTPUT_DIR}/", flush=True)
        inner_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"[opsd rank=0] Done.", flush=True)
    if world_size > 1:
        torch.distributed.barrier()

    if world_size > 1:
        torch.distributed.destroy_process_group()
    return 0


# ----------------------------- helpers -----------------------------


def _batch_iter(ds, batch_size):
    """Yield list-of-dicts batches from a HuggingFace Dataset."""
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        yield batch


def _assemble_full(prompts, prompt_attn, completion_ids, completion_attn,
                   pad_token_id, prompt_len_scalar):
    """Right-pad prompt+completion to common length; labels mask prompt+pad."""
    B = prompts.shape[0]
    full_ids_list = []
    full_attn_list = []
    full_labels_list = []
    for i in range(B):
        p_ids = prompts[i]            # scalar-len may differ; use prompt_attn to clip
        p_attn = prompt_attn[i]
        c_ids = completion_ids[i]
        c_attn = completion_attn[i]
        # clip padding on prompt side
        p_real_len = int(p_attn.sum().item())
        p_ids = p_ids[:p_real_len]
        c_ids = c_ids[: int(c_attn.sum().item())]
        # truncate completion to cap
        MAX_COMP_CAP = 512
        c_ids = c_ids[:MAX_COMP_CAP]

        seq = torch.cat([p_ids, c_ids], dim=0)
        attn = torch.cat([torch.ones_like(p_ids), torch.ones_like(c_ids)], dim=0)
        labels = torch.cat([torch.full_like(p_ids, -100), c_ids.clone()], dim=0)
        full_ids_list.append(seq)
        full_attn_list.append(attn)
        full_labels_list.append(labels)

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
