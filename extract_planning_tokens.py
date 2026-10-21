"""
Extract planning tokens from each (think, action) sample.

For every sample:
- take the agent's `think` text
- build a chat template: user = the agent's full prompt, assistant = think + "<action>...</action>" placeholder
- we feed the full prompt through the model and capture the LAST hidden state of the
  LAST token that belongs to the assistant turn. By paper convention this is the
  "planning token" -- it integrates the CoT's planning information.
- save (vector, label) pairs.

Note: We can't get tokenized alignment to the original trajectory perfectly.
So we approximate: take the planning token = last hidden state of the last input token
that the model sees when generating the chat-completion for our prompt.
"""
import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_last_assistant_token_position(input_ids, assistant_token_id):
    """Find the position of the last `<|im_start|>assistant\\n` token sequence, or
    fall back to ``assistant_token_id``."""
    # ChatML pattern: <|im_start|>assistant\n
    # The last token of the input (the prediction point) is always the LAST token.
    return len(input_ids) - 1


def build_chat(tokenizer, think, action, task, obs):
    """Mimic ALFWorld agent prompt: user has task + obs + admissible actions,
    assistant has think + action. We use only the LAST training step (no full
    trajectory prefix, so the planning token we extract is purely about THIS
    step's reasoning)."""
    user_text = (
        f"You are an expert agent operating in the ALFRED Embodied Environment.\n"
        f"Your task is to: {task}\n"
        f"Your current observation is: {obs[:800]}\n"
        f"Reason step-by-step about what to do."
    )
    assistant_text = think + f"\n<action>{action}</action>"
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_file", default="thinking_samples.json")
    ap.add_argument("--output_vectors", default="planning_vectors.npy")
    ap.add_argument("--output_labels", default="planning_labels.json")
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layer_idx", type=int, default=-1,
                    help="-1 = last layer (paper convention)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()

    with open(args.samples_file) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    model.eval()

    vectors = []
    labels = []

    with torch.no_grad():
        for i, s in enumerate(samples):
            chat_text = build_chat(
                tokenizer,
                think=s["think"],
                action=s["action"],
                task=s["task"],
                obs=s.get("obs_excerpt", "") or "(no obs)",
            )
            inputs = tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_tokens,
                add_special_tokens=False,
            )
            input_ids = inputs["input_ids"].to(args.device)
            attn_mask = inputs["attention_mask"].to(args.device)
            if input_ids.shape[1] >= args.max_tokens:
                # truncated -- accept it but keep the last position
                pass

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states is a tuple of length (num_layers + 1).
            # hidden_states[0] is the embedding layer; hidden_states[layer] is the
            # output of that layer. We want the "last layer" output.
            hs = outputs.hidden_states[args.layer_idx]    # [B, T, hidden]
            # last token position
            last_pos = attn_mask.sum().item() - 1
            token_embedding = hs[0, last_pos, :].float().cpu().numpy()  # [hidden]
            vectors.append(token_embedding)
            labels.append(s["label"])

            if (i + 1) % 10 == 0:
                print(f"  processed {i+1}/{len(samples)}")

    vectors = np.stack(vectors, axis=0)
    np.save(args.output_vectors, vectors)

    with open(args.output_labels, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"vectors shape: {vectors.shape}")
    print(f"labels: {len(labels)}")
    from collections import Counter
    print(f"label distribution: {Counter(labels)}")
    print(f"saved to {args.output_vectors} and {args.output_labels}")


if __name__ == "__main__":
    main()