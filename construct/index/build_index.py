# 将训练轨迹拆为状态级经验，并建立用于相似状态检索的 HNSW 索引。
import os
import sys

# 配置项目模块搜索路径，以便复用根目录的嵌入服务接口。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hnswlib, numpy as np, pickle, json
import torch
import re
import json
import argparse

from utils import embed_texts

# 解析数据集、模型、采样步长和输入输出选项；校验步长及成功、失败过滤条件。
parser = argparse.ArgumentParser(description='Build index for dataset')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--base_model_name', type=str, required=True, help='Name of the base model')
parser.add_argument('--step_stride', type=int, default=1, help='Keep one entry every K steps (default: 1)')
parser.add_argument('--correct_only', action='store_true', help='Only index successful episodes')
parser.add_argument('--failure_only', action='store_true', help='Only index failed episodes')
parser.add_argument(
    '--traj_file',
    type=str,
    default=None,
    help=(
        "Override source trajectory file. Defaults to "
        "logs/{dataset_name}_old/{base_model_safe}/traj_train.json. "
        "Pass a path to e.g. a traj_eval_in_distribution file when doing "
        "self-iteration experiments."
    ),
)
parser.add_argument(
    '--index_prefix',
    type=str,
    default=None,
    help=(
        "Override the index file prefix. Defaults to 'train_'. Useful for "
        "lab experiments like 'eval_in_distribution_' so the new index "
        "doesn't overwrite the production train_* index."
    ),
)
args = parser.parse_args()
if args.step_stride < 1:
    raise ValueError(f"step_stride must be >= 1, got {args.step_stride}")
if args.correct_only and args.failure_only:
    raise ValueError("--correct_only and --failure_only are mutually exclusive")

correct_only = args.correct_only
failure_only = args.failure_only

# 确定轨迹来源及索引保存位置，用不同后缀区分成功、失败和混合记忆。
dataset_name = args.dataset_name
base_model_name = args.base_model_name
base_model_safe = base_model_name.replace('/', '_')
if args.traj_file:
    traj_file = args.traj_file
else:
    traj_file = f"logs/{dataset_name}_old/{base_model_safe}/traj_train.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"
os.makedirs(retrieval_data_path, exist_ok=True)

prefix = args.index_prefix or "train_"
index_path  = f"{retrieval_data_path}/{prefix}hnsw.index"
meta_path   = f"{retrieval_data_path}/{prefix}hnsw_meta.json"
values_path = f"{retrieval_data_path}/{prefix}values.pkl"
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

# 准备轨迹解析规则；从每条轨迹的最后有效提示恢复观测动作历史，并补入末端片段。
pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
keys, values = [], [] 
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

    # 根据整条轨迹的最终结果筛选经验来源。
    reward = "success" if item["won"] else "failure"
    if correct_only and reward == "failure":
        continue
    if failure_only and reward == "success":
        continue

    # 排除提示或动作标记为字符串 None 的步骤，保留原始步骤下标。
    valid_steps = []
    for step_idx, step in enumerate(item["steps"]):
        if step["curr_action"] == "None" or step["curr_prompt"] == "None":
            continue
        valid_steps.append((step_idx, step))

    # 按有效步骤间隔采样，以当前提示作为检索键，构造状态、动作、后续轨迹和结果四部分经验。
    # 后续轨迹按原始步骤下标切片，依赖日志片段与步骤位置的对应关系。
    for valid_idx, (step_idx, step) in enumerate(valid_steps):
        if valid_idx % args.step_stride != 0:
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

# 调用嵌入服务编码检索键，并转换为 HNSW 接收的连续 float32 向量矩阵。
print(f"Key 0: {keys[0]}")
print(f"Value 0: {values[0]}")
print(f"num of entries: {len(keys)}")

import os as _os
_emb_model = _os.environ.get("EMBED_MODEL_NAME", "./Qwen3-Embedding-4B")
emb = embed_texts(keys, model=_emb_model, normalize=True)
emb = torch.tensor(emb, dtype=torch.float32)
emb = emb.cpu().contiguous().numpy().astype("float32")
d = emb.shape[1]

# 建立余弦距离 HNSW；向量的整数编号与经验列表下标一一对应。
M, ef_construction, ef_search = 32, 200, 100

index = hnswlib.Index(space="cosine", dim=d)
index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
index.set_ef(ef_search)

# 分别保存索引图、检索参数和经验正文，供运行时加载。
index.save_index(index_path)
with open(meta_path, "w") as f:
    json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

with open(values_path, "wb") as f:
    pickle.dump(values, f)
