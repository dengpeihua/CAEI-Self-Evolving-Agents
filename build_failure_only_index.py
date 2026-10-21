
import os
import hnswlib
import numpy as np
import pickle
import json
import torch
import re

from utils import embed_texts

dataset_name = "alfworld"
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
base_model_safe = base_model_name.replace('/', '_')

traj_file = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"
os.makedirs(retrieval_data_path, exist_ok=True)

prefix = "eval_in_distribution_v2_"
index_path = f"{retrieval_data_path}/{prefix}hnsw_failure_only.index"
meta_path = f"{retrieval_data_path}/{prefix}hnsw_meta_failure_only.json"
values_path = f"{retrieval_data_path}/{prefix}values_failure_only.pkl"

print(f"Building FAILURE-ONLY index from {traj_file}...")
print(f"Index will be saved to {index_path}...")

with open(traj_file, 'r') as reader:
    items = json.load(reader)

pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"

keys = []
values = []

success_count = 0
failure_count = 0

for item in items:
    won = item.get("won", False)

    if not won:
        failure_count += 1
    else:
        success_count += 1
        continue  # SKIP successful trajectories for failure-only index

    # Process this failed trajectory
    for step in reversed(item["steps"]):
        if step["curr_prompt"] != "None":
            full_traj = step["curr_prompt"]
            observation_action_pairs = [
                p for p in re.findall(pattern, full_traj, flags=re.DOTALL)
                if p.startswith("[Observation")
            ]
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

            observation_action_pairs.append(f"[Last Observation: '{observation_text}', Last Action: '{action_text}']")
            break

    reward = "failure"

    valid_steps = []
    for step_idx, step in enumerate(item["steps"]):
        if step["curr_action"] == "None" or step["curr_prompt"] == "None":
            continue
        valid_steps.append((step_idx, step))

    step_stride = 1
    for valid_idx, (step_idx, step) in enumerate(valid_steps):
        if valid_idx % step_stride != 0:
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

print(f"Trajectories: {success_count} success (skipped), {failure_count} failure (indexed)")
print(f"Total index entries: {len(values)}")

if len(values) > 0:
    print("Building embeddings...")
    _emb_model = os.environ.get("EMBED_MODEL_NAME", "./Qwen3-Embedding-4B")
    emb = embed_texts(keys, model=_emb_model, normalize=True)
    emb = torch.tensor(emb, dtype=torch.float32)
    emb = emb.cpu().contiguous().numpy().astype("float32")
    d = emb.shape[1]

    M, ef_construction, ef_search = 32, 200, 100

    print("Building HNSW index...")
    index = hnswlib.Index(space="cosine", dim=d)
    index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
    index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
    index.set_ef(ef_search)

    print("Saving index files...")
    index.save_index(index_path)
    with open(meta_path, "w") as f:
        json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

    with open(values_path, "wb") as f:
        pickle.dump(values, f)

    print(f"Done! Saved {len(values)} entries to failure-only index.")
else:
    print("No entries to index!")
