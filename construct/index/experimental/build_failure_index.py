# 从指定评测日志构建只包含失败状态经验的实验索引。
import os
import sys

# 设置模块搜索路径；当前上溯层级实际停在 construct，独立运行时需注意 utils 导入。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hnswlib, numpy as np, pickle, json
import torch
import re
import argparse

from utils import embed_texts

# 使用固定数据集、模型与评测日志路径，指定失败索引的输出文件名。
dataset_name = "alfworld"
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
base_model_safe = base_model_name.replace('/', '_')
traj_file = "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"

prefix = "eval_in_distribution_v2_"
index_path  = f"{retrieval_data_path}/{prefix}hnsw_failure_only.index"
meta_path   = f"{retrieval_data_path}/{prefix}hnsw_meta_failure_only.json"
values_path = f"{retrieval_data_path}/{prefix}values_failure_only.pkl"

# 读取评测轨迹，准备观测动作解析规则和键值列表。
print(f"Building failure-only index from {traj_file}...")
print(f"Index will be saved to {index_path}...")
with open(traj_file, 'r') as reader:
    items = json.load(reader)

pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
keys, values = [], []

# 从每条轨迹的最后有效提示恢复历史片段，补上末端观测和动作。
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

    # 只保留失败轨迹，并过滤缺少有效提示或动作的步骤。
    reward = "success" if item["won"] else "failure"
    if reward == "success":
        continue

    valid_steps = []
    for step_idx, step in enumerate(item["steps"]):
        if step["curr_action"] == "None" or step["curr_prompt"] == "None":
            continue
        valid_steps.append((step_idx, step))

    # 为每个有效步骤生成检索键与四部分失败经验，附上该状态之后的轨迹。
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

# 对失败状态提示做嵌入编码，并整理为 float32 矩阵。
print(f"Key 0: {keys[0][:300]}")
print(f"num of entries: {len(keys)}")

_emb_model = os.environ.get("EMBED_MODEL_NAME", "./Qwen3-Embedding-4B")
emb = embed_texts(keys, model=_emb_model, normalize=True)
emb = torch.tensor(emb, dtype=torch.float32)
emb = emb.cpu().contiguous().numpy().astype("float32")
d = emb.shape[1]

# 构建 HNSW 并保存索引、参数和失败经验；需预先准备输出目录及非空数据。
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

print(f"Done! Saved {len(keys)} entries to failure-only index.")
