# 构建失败经验索引，并增加轨迹计数、输出目录创建及空数据保护。
import os
import sys

# 设置模块搜索路径；当前层级停在 construct，独立运行需注意 utils 导入。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hnswlib
import numpy as np
import pickle
import json
import torch
import re

from utils import embed_texts

# 固定评测日志来源并创建输出目录，产物命名与另一失败索引脚本相同。
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

# 读取轨迹，初始化解析规则、检索键值及成功失败计数。
with open(traj_file, 'r') as reader:
    items = json.load(reader)

pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"

keys = []
values = []

success_count = 0
failure_count = 0

# 统计轨迹结果并跳过成功轨迹；缺失 won 时按失败处理。
for item in items:
    won = item.get("won", False)

    if not won:
        failure_count += 1
    else:
        success_count += 1
        continue

    # 恢复当前失败轨迹的历史观测动作片段，并补充末端片段。
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

    # 收集有效步骤；以步骤提示为键，生成包含后续轨迹与失败结果的经验。
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

# 输出筛选统计；仅在存在经验条目时调用嵌入服务并构建 HNSW。
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

    # 保存索引图、检索参数与失败经验正文。
    print("Saving index files...")
    index.save_index(index_path)
    with open(meta_path, "w") as f:
        json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

    with open(values_path, "wb") as f:
        pickle.dump(values, f)

    print(f"Done! Saved {len(values)} entries to failure-only index.")
else:
    print("No entries to index!")
