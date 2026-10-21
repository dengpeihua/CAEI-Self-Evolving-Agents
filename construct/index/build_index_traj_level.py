# 为完整轨迹建立检索索引，供轨迹级记忆基线使用。
import os
import sys

# 配置项目模块搜索路径，导入共享的嵌入接口。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hnswlib, numpy as np, pickle, json
import torch
import re
import json
import argparse
import os

from utils import embed_texts

# 解析数据集、模型和轨迹结果过滤条件，检查两个过滤开关互斥。
parser = argparse.ArgumentParser(description='Build index for dataset')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--base_model_name', type=str, required=True, help='Name of the base model')
parser.add_argument('--correct_only', action='store_true', help='Only index successful episodes')
parser.add_argument('--failure_only', action='store_true', help='Only index failed episodes')
args = parser.parse_args()
if args.correct_only and args.failure_only:
    raise ValueError("--correct_only and --failure_only are mutually exclusive")

correct_only = args.correct_only
failure_only = args.failure_only

# 定位训练日志，准备轨迹级索引及元数据、经验文件路径。
dataset_name = args.dataset_name
base_model_name = args.base_model_name
base_model_safe = base_model_name.replace('/', '_')
traj_file = f"logs/{dataset_name}_old/{base_model_safe}/traj_train.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"
os.makedirs(retrieval_data_path, exist_ok=True)
index_path = f"{retrieval_data_path}/train_traj_level_hnsw.index"
meta_path  = f"{retrieval_data_path}/train_traj_level_hnsw_meta.json"
values_path = f"{retrieval_data_path}/train_traj_level_values.pkl"
if correct_only:
    index_path = index_path.replace(".index", "_correct_only.index")
    meta_path = meta_path.replace(".json", "_correct_only.json")
    values_path = values_path.replace(".pkl", "_correct_only.pkl")
elif failure_only:
    index_path = index_path.replace(".index", "_failure_only.index")
    meta_path = meta_path.replace(".json", "_failure_only.json")
    values_path = values_path.replace(".pkl", "_failure_only.pkl")
print(f"Building index from {traj_file}...")
print(f"Index will be saved to {index_path}...")
with open(traj_file, 'r') as reader:
    items = json.load(reader)

# 筛选轨迹，以首步完整提示为检索键；从最后有效提示恢复历史并补上末端动作。
pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
keys, values = [], [] 
for item in items:
    reward = "success" if item["won"] else "failure"
    if correct_only and reward == "failure":
        continue
    if failure_only and reward == "success":
        continue
    task_instr = item["steps"][0]["curr_prompt"]
    keys.append(task_instr)
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

            # 将任务、整条轨迹及最终结果组合为一条返回经验，不再逐步骤拆分。
            experience = f"Task:\n{task_instr}\n\nAgent Trajectory:\n{observation_action_pairs}\n\nFinal Outcome: {reward}"
            values.append(experience)
            
            break

# 将轨迹级检索键编码为归一化向量，并转换为 float32 矩阵。
print("sample key", keys[10])
print("sample value", values[10])
print(f"num of entries: {len(keys)}")

emb = embed_texts(keys, model="Qwen/Qwen3-Embedding-4B", normalize=True)
emb = torch.tensor(emb, dtype=torch.float32)
emb = emb.cpu().contiguous().numpy().astype("float32")
d = emb.shape[1]

# 建立余弦距离 HNSW，用连续编号关联向量和经验正文。
M, ef_construction, ef_search = 32, 200, 100

index = hnswlib.Index(space="cosine", dim=d)
index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
index.set_ef(ef_search)

# 保存索引、参数元数据和完整轨迹经验列表。
index.save_index(index_path)
with open(meta_path, "w") as f:
    json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

with open(values_path, "wb") as f:
    pickle.dump(values, f)
