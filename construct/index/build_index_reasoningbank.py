# 将轨迹归纳为可复用策略，再对策略文本建立检索索引。
import os
import sys

# 配置项目模块搜索路径，复用嵌入接口。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hnswlib, numpy as np, pickle, json
import torch
import re
import json
import asyncio
import random
import argparse

from openai import AsyncOpenAI, APITimeoutError
import os
import httpx

from tqdm.asyncio import tqdm

from utils import embed_texts

# 解析数据集、模型与成功或失败轨迹的过滤条件。
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

# 确定轨迹日志和 reasoningbank 索引产物路径。
dataset_name = args.dataset_name
base_model_name = args.base_model_name
base_model_safe = base_model_name.replace('/', '_')
traj_file = f"logs/{dataset_name}_old/{base_model_safe}/traj_train.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"
os.makedirs(retrieval_data_path, exist_ok=True)
index_path = f"{retrieval_data_path}/train_reasoningbank_hnsw.index"
meta_path  = f"{retrieval_data_path}/train_reasoningbank_hnsw_meta.json"
values_path = f"{retrieval_data_path}/train_reasoningbank_values.pkl"
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

# 配置语言模型服务地址，按地址缓存客户端以复用连接。
servers = os.environ.get("VLLM_SERVERS", "127.0.0.1").split(",")
clients = {}

# 请求语言模型归纳经验；仅对指定超时异常做至多三次尝试，重试之间指数退避。
async def call_llm(prompt, model_name=os.environ.get("REASONINGBANK_BUILDER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")):
    ip_addr = random.choice(servers)
    if ip_addr not in clients:
        clients[ip_addr] = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=f"http://{ip_addr}:8001/v1",
            timeout=httpx.Timeout(600.0),
        )
    client = clients[ip_addr]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=int(os.environ.get("REASONINGBANK_BUILDER_MAX_TOKENS", "512")),
            )
            return resp.choices[0].message.content.strip()
        except (APITimeoutError, httpx.ReadTimeout) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise e

# 组织策略索引构建流程：准备轨迹提示、并发生成策略、解析并建立索引。
async def main():
    with open(traj_file, 'r') as reader:
        items = json.load(reader)

    # 遍历并筛选轨迹，恢复观测动作历史，组合任务、轨迹和最终结果。
    pattern = r'\[.*?\]\n'
    action_pattern = r"<action>(.*?)</action>"
    observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
    keys, values = [], [] 
    prompts = []
    for item in items:
        task_instr = item["steps"][0]["curr_prompt"]
        reward = "success" if item["won"] else "failure"
        if correct_only and reward == "failure":
            continue
        if failure_only and reward == "success":
            continue

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

                experience = f"Task:\n{task_instr}\n\nAgent Trajectory:\n{observation_action_pairs}\n\nFinal Outcome: {reward}"
                
                break
        
        # 要求模型解释成功或失败原因，并输出至多三个可迁移的 Markdown 记忆项。
        prompt = f"""
You will be given a task description and a corresponding trajectory that represents **how an agent attempted to accomplish the task** — the trajectory may be either **successful** or **failed**.

Your task is to extract and summarize **useful, generalizable insights** in the format of **memory items**, depending on whether the agent’s trajectory was successful or not.

---

## **Guidelines**

* If the trajectory is **successful**, summarize insights that explain **why and how the agent succeeded**.
* If the trajectory is **failed**, summarize insights that explain **why the agent failed** and **what strategies could prevent such failures in the future**.
* The goal of the summarized memory items is to help improve performance on future similar tasks by providing **generalizable lessons** (not specific to the query or domain).

---

## **Important Notes**

* First, **reflect on the trajectory’s outcome** (success or failure) and reason about **why it happened**.
* You can extract **at most 3** memory items per trajectory.
* Avoid repeating or overlapping items.
* Do **not** mention task-specific information. Focus on **general strategies, reasoning patterns, or process insights**.

---

## **Output Format**

Your output must strictly follow the Markdown format below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1–3 sentences describing the insights learned from the trajectory (for success: what worked well; for failure: what to improve or avoid)>
```

"""
        prompts.append(prompt + experience)

    # 限制同时在途的请求数量，并按请求完成顺序收集策略摘要。
    memory_items = [None] * len(prompts)
    semaphore = asyncio.Semaphore(300)

    async def call_with_limit(p):
        async with semaphore:
            return await call_llm(p)

    with tqdm(total=len(prompts), desc="Processing LLM calls") as pbar:
        tasks = [asyncio.create_task(call_with_limit(p)) for p in prompts]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            memory_items[i] = await task
            pbar.update(1)

    # 拆分模型输出并去掉编号标题；每条策略正文同时作为检索键和返回值。
    for memory_item in memory_items:
        items = re.split(r'(?=^# Memory Item \d+)', memory_item.strip(), flags=re.MULTILINE)
        for item in items:
            item = item.strip()
            if item:
                lines = item.split('\n')
                if lines and re.match(r'# Memory Item \d+', lines[0]):
                    item = '\n'.join(lines[1:]).strip()
                if item:
                    keys.append(item)
                    values.append(item)

    # 将策略文本编码为归一化的 float32 向量。
    print("sample key", keys[10])
    print("sample value", values[10])
    print(f"num of entries: {len(keys)}")

    emb = embed_texts(keys, model="./Qwen3-Embedding-4B", normalize=True)
    emb = torch.tensor(emb, dtype=torch.float32)
    emb = emb.cpu().contiguous().numpy().astype("float32")
    d = emb.shape[1]

    # 建立余弦距离 HNSW，使用经验列表下标作为检索编号。
    M, ef_construction, ef_search = 32, 200, 100

    index = hnswlib.Index(space="cosine", dim=d)
    index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
    index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
    index.set_ef(ef_search)

    # 保存策略索引、参数和策略正文。
    index.save_index(index_path)
    with open(meta_path, "w") as f:
        json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

    with open(values_path, "wb") as f:
        pickle.dump(values, f)

if __name__ == "__main__":
    asyncio.run(main())
