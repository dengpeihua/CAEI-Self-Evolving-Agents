"""
Minimal test: can vLLM Engine mode (vllm.LLM, not api_server) expose
hidden_states for the last generated token?

We load Qwen3-4B-Instruct-2507 directly into a Python process on a single
free GPU, generate a short prompt, and inspect the returned outputs
for any "hidden_states" attribute.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # free GPU (card 1)

import torch
from vllm import LLM, SamplingParams

print("[probe] importing vllm done")
print("[probe] CUDA visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])

# load Qwen3-4B in offline Engine mode, single GPU
llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    dtype="bfloat16",
    gpu_memory_utilization=0.6,
    max_model_len=2048,
    tensor_parallel_size=1,
    enforce_eager=True,   # avoid graph capture issues; pure eager mode
    download_dir=None,
)
print("[probe] vllm.LLM loaded Qwen3-4B-Instruct-2507")

prompt = "You are at desk 1. The task is 'look at alarmclock under the desklamp'. Reason step-by-step and choose an action."
params = SamplingParams(max_tokens=64, temperature=0.0)
outputs = llm.generate([prompt], params)
print(f"[probe] got {len(outputs)} outputs")

out = outputs[0]
text = out.outputs[0].text
print(f"[probe] generated text ({len(text)} chars):")
print(repr(text[:300]))

# Inspect RequestOutput / CompletionOutput attributes for hidden_states
print("\n[probe] === inspecting outputs[0].outputs[0] attributes ===")
for attr in dir(out.outputs[0]):
    if attr.startswith("_"):
        continue
    val = getattr(out.outputs[0], attr)
    if isinstance(val, torch.Tensor):
        print(f"  .{attr}: tensor shape={tuple(val.shape)} dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        print(f"  .{attr}: list of len={len(val)}")
    else:
        v = repr(val)[:80]
        print(f"  .{attr}: {v}")

# Some vLLM versions expose hidden_states only when explicitly enabled
# during sampling; test by passing extra sampling param.
print("\n[probe] testing SamplingParams return_hidden_states / logprobs / etc.")
import inspect
sig = inspect.signature(SamplingParams.__init__)
opts = [p.name for p in sig.parameters.values() if p.name != "self"]
print("  SamplingParams options:", opts)
