"""
Confirm transformers AutoModelForCausalLM gives us last-layer hidden
state cleanly with bf16.

This validates the "方案 A" architecture: load Qwen3 in the agent
process, do forward, grab the planning token.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # single free GPU

import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

print("[probe] loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print(f"[probe] tokenizer loaded in {time.time()-t0:.1f}s")

print("[probe] loading model (bf16, single GPU)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)
model.eval()
print(f"[probe] model loaded in {time.time()-t0:.1f}s")
print(f"[probe] model device: {model.device}, dtype: {model.dtype}")

# Build a chat: user=task+obs, assistant=think+action
think = (
    "Current Situation Analysis:\n"
    "- The task is to look at alarmclock under the desklamp.\n"
    "- We are at desk 1 where the alarmclock and desklamp are located.\n"
    "- The desklamp is already on.\n"
)
action = 'look at alarmclock 2 under desklamp 1'

user_text = (
    "You are an expert agent in the ALFRED environment.\n"
    "Your task is: examine the alarmclock under the desklamp.\n"
    "Your current observation is: On the desk you see an alarmclock and a desklamp.\n"
    "Reason step-by-step about what to do."
)
assistant_text = f"{think}<action>{action}</action>"

messages = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": assistant_text},
]
chat_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print(f"[probe] chat template len: {len(chat_text)} chars")

# Tokenize
inputs = tokenizer(
    chat_text,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
    add_special_tokens=False,
).to(model.device)
print(f"[probe] input_ids shape: {tuple(inputs['input_ids'].shape)} (T tokens)")
attn_mask = inputs["attention_mask"]
true_length = attn_mask.sum().item()
print(f"[probe] true_length (attention mask sum): {true_length}")

# forward
print("[probe] forward pass...")
t0 = time.time()
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
    )
print(f"[probe] forward done in {time.time()-t0:.2f}s")

# Inspect hidden_states
hs = outputs.hidden_states
print(f"[probe] hidden_states: {len(hs)} tensors (embedding + {len(hs)-1} layers)")
print(f"[probe] last layer shape: {tuple(hs[-1].shape)}")  # (1, T, 2560)

# Get planning token = last layer, last real token
last_pos = true_length - 1
planning_token = hs[-1][0, last_pos, :].float()    # [2560] bf16 -> fp32
print(f"[probe] planning_token shape: {tuple(planning_token.shape)}, dtype: {planning_token.dtype}")

# Sanity: norm, mean
import numpy as np
arr = planning_token.cpu().numpy()
print(f"[probe] planning_token stats: mean={arr.mean():.4f}, std={arr.std():.4f}, "
      f"norm={np.linalg.norm(arr):.2f}, range=[{arr.min():.2f}, {arr.max():.2f}]")

# Memory
print(f"[probe] GPU memory after: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"[probe] GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print("[probe] OK")
