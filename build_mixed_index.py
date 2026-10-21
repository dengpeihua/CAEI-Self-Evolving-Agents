
import os
import hnswlib, numpy as np, pickle, json
import torch
import re
import argparse

from utils import embed_texts

dataset_name = "alfworld"
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
base_model_safe = base_model_name.replace('/', '_')
traj_file = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"

prefix = "eval_in_distribution_v2_mixed_"
index_path  = f"{retrieval_data_path}/{prefix}hnsw.index"
meta_path   = f"{retrieval_data_path}/{prefix}hnsw_meta.json"
values_path = f"{retrieval_data_path}/{prefix}values.pkl"

print(f"Building mixed (success+failure) index from {traj_file}...")
print(f"Index will be saved to {index_path}...")
with open(traj_file, 'r') as reader:
    items = json.load(reader)

pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
keys, values = [], []

success_count = 0
failure_count = 0

for item in items:
    for step in reversed(item["steps"]):
        if step["curr_prompt"] != "None":
            full_traj = step["curr_prompt"]
            observation_action_pairs = [p for p in re.findall(pattern, full_traj, flags=re.DOTALL) if p.startswith("[Observation")]
            last_action = step["curr_action"]
            action_match = re.search(action_pattern, last_action, flags=re.DOTALL)
            if action_match:
                action_text = action_match.group(1).strip()
            else:
                action_text = ""
            observation_match = re.search(observation_pattern, full_traj, flags=re.IGNORECASE | re.DOTALL)
            if observation_match:
                observation_text = observation_match.group(1).strip()
            else:
                observation_text = ""

            observation_action_pairs.append(f"[Last Observation: '{observation_text}', Last Action: '{action_text}]'")

            break

    reward = "success" if item["won"] else "failure"
    if reward == "success":
        success_count += 1
    else:
        failure_count += 1

    valid_steps = []
    for step_idx, step in enumerate(item["steps"]):
        if step["curr_action"] == "None" or step["curr_prompt"] == "None":
            continue
        valid_steps.append((step_idx, step))

    for valid_idx, (step_idx, step) in enumerate(valid_steps):
        if valid_idx % 1 != 0:
            continue
        keys.append(step["curr_prompt"])
        action_match = re.search(action_pattern, step["curr_action"], flags=re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
        else:
            action_text = ""

        start_idx = min(step_idx, len(observation_action_pairs))
        remaining_trajectory = '\n'.join(observation_action_pairs[start_idx:])

        experience_item = f"""
Past State:
{step['curr_prompt']}

Past Action:
{action_text}

Remaining Trajectory:
{remaining_trajectory}

Final Outcome:
{reward}
"""
        values.append(experience_item)

print(f"Trajectories: {success_count} success, {failure_count} failure")
print(f"Key 0: {keys[0][:300]}")
print(f"num of entries: {len(keys)}")

_emb_model = os.environ.get("EMBED_MODEL_NAME", "./Qwen3-Embedding-4B")
emb = embed_texts(keys, model=_emb_model, normalize=True)
emb = torch.tensor(emb, dtype=torch.float32)
emb = emb.cpu().contiguous().numpy().astype("float32")
d = emb.shape[1]

M, ef_construction, ef_search = 32, 200, 100

index = hnswlib.Index(space="cosine", dim=d)
index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
index.set_ef(ef_search)

index.save_index(index_path)
with open(meta_path, "w") as f:
    json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

with open(values_path, "wb") as f:
    pickle.dump(values, f)

print(f"Done! Saved {len(keys)} entries to mixed index.")
