<h1 align="center">AdaMEM: Test-Time Adaptive Memory for Language Agents</h1>

<p align="center">
  <b>Yunxiang Zhang &nbsp;·&nbsp; Yiheng Li &nbsp;·&nbsp; Ali Payani &nbsp;·&nbsp; Lu Wang</b>
</p>

<p align="center">
  <b>ICML 2026</b>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/XXXX.XXXXX">Paper</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/yunx-z/AdaMEM">Code</a>
</p>

---

## Overview

A central challenge for language agents is utilizing past experience to adapt to dynamic test-time conditions. While recent work demonstrates the promise of agentic memory mechanisms, most systems restrict retrieval to episode initiation. Consequently, agents are forced to rely on static guidance that becomes increasingly misaligned as long-horizon tasks unfold. To address this rigidity, we propose the **Adaptive Memory Agent (AdaMEM)**, a novel framework for agent test-time adaptation. Without updating model parameters online, AdaMEM adapts agent behavior via a hybrid memory architecture: it maintains a long-term trajectory memory of raw experiences collected offline while generating dynamic short-term strategy memory on-the-fly to guide decision-making. This mechanism enables the trade-off between token efficiency and adaptability across varying inference-time compute levels. Empirically, AdaMEM significantly outperforms static memory baselines, achieving relative gains of up to 13% on ALFWorld and 11% on WebShop, with consistent leading performance extending to agentic search on HotpotQA. To further enhance this adaptation, we develop **STEP-MFT**, a Step-wise Memory Fine-Tuning technique that trains the policy to synthesize high-quality strategies from retrieved experiences, yielding additional performance gains. Our work establishes a new scaling dimension for agentic memory, supporting continuous reasoning and self-evolution post-deployment in real-world environments.

---

## Memory Types

| Name | Description |
|---|---|
| `no-memory` | Base ReAct agent with no external memory |
| `synapse` | Episode-level trajectory retrieval ([Synapse](https://openreview.net/forum?id=Pc8AU1aF5e) baseline) |
| `reasoningbank` | Episode-level strategy retrieval ([ReasoningBank](https://arxiv.org/abs/2509.25140) baseline) |
| `adamem-high` | Step-wise strategy synthesis — generates a *transient* strategy at each step the agent requests memory (**AdaMEM-HIGH**) |
| `adamem-low` | Step-wise strategy synthesis with persistent strategy and agent-controlled refresh (**AdaMEM-LOW**) |
| `adamem-max` | Generates a fresh strategy at *every* step unconditionally (**AdaMEM-MAX / ablation**) |
| `adamem-max-without-trajectory-memory` | Strategy synthesis without retrieval (no long-term memory; ablation) |
| `adamem-max-without-strategy-memory` | Raw trajectory injection at every step without synthesis (ablation) |

---

## Installation

```bash
git clone https://github.com/yunx-z/AdaMEM.git
cd AdaMEM
pip install -e .
pip install -r requirements.txt
```

Install the ALFWorld and WebShop environments by following the setup instructions in the verl-agent repo:

- **ALFWorld**: https://github.com/langfengQ/verl-agent#1-alfworld
- **WebShop**: https://github.com/langfengQ/verl-agent#2-webshop

---

## Quick Start

### 1. Build the Long-Term Memory Index

Collect successful trajectories on the training split first (see Step 3 below for how to run a training-split rollout), then build the HNSW index:

```bash
# Step-level index (used by all AdaMEM variants)
python build_index.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --correct_only

# Episode-level index (for synapse baseline)
python build_index_traj_level.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507

# Strategy index (for reasoningbank baseline)
python build_index_reasoningbank.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507
```

### 2. Run Inference

Start a vLLM server, then launch the agent:

```bash
# Start vLLM、
llm模型部署：
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-model-len 32768

CUDA_VISIBLE_DEVICES=1,3 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-model-len 32768 \
  --tensor-parallel-size 2  
  --task embedding

embedding模型部署 (注意在data目录下)：
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model ./Qwen3-Embedding-4B \
    --host 0.0.0.0 --port 8002 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.6 \
    --task embed

# AdaMEM-LOW on ALFWorld (seen split)
SPLIT=eval_in_distribution \
MEM_TYPE=adamem-low \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

# Iter 1
SPLIT=eval_in_distribution \
MEM_TYPE=adamem-low \
RETRIEVAL_INDEX_PREFIX=eval_in_distribution_ \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

SPLIT=eval_in_distribution \
MEM_TYPE=adamem_lda \
RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

SPLIT=eval_in_distribution \
MEM_TYPE=lda_delay \
RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

SPLIT=eval_in_distribution \
  MEM_TYPE=lda_reflect \
  RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

SPLIT=eval_in_distribution \
  MEM_TYPE=lda_reflect \
  RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  MODEL_CONTEXT_WINDOW_TOKENS=32768 \
  RETRIEVAL_TOPK=3 \
  RETRIEVAL_OUTPUT_TOKEN_RESERVE=2048 \
  python -m examples.prompt_agent.gpt4o_alfworld


# Iter 2
SPLIT=eval_in_distribution \
MEM_TYPE=adamem-low \
RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -u -m examples.prompt_agent.gpt4o_alfworld

# baseline (no memory)
SPLIT=eval_in_distribution \
MEM_TYPE="" \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

# AdaMEM-LOW on WebShop
MEM_TYPE=adamem-low \
MODEL_NAME=langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop \
  python -m examples.prompt_agent.gpt4o_webshop


  SPLIT=eval_in_distribution \
  MEM_TYPE=lda_delay \
  RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_mixed_ \
  CORRECT_ONLY=false \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

  选项 2: 仅使用成功轨迹

  SPLIT=eval_in_distribution \
  MEM_TYPE=lda_delay \
  RETRIEVAL_INDEX_PREFIX=eval_in_distribution_v2_ \
  CORRECT_ONLY=true \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

  ---
  轨迹分析回顾

  ┌──────────┬────────┐
  │   指标   │  数值  │
  ├──────────┼────────┤
  │ 总任务数 │ 140    │
  ├──────────┼────────┤
  │ 成功任务 │ 95     │
  ├──────────┼────────┤
  │ 失败任务 │ 45     │
  ├──────────┼────────┤
  │ 成功率   │ 67.86% │
  └──────────┴────────┘

```

Set `MEM_TYPE` to any value from the table above. Omit `MEM_TYPE` (or set it empty) for the `no-memory` baseline.

**Key environment variables**

| Variable | Default | Description |
|---|---|---|
| `SPLIT` | — | ALFWorld split: `train`, `eval_in_distribution`, `eval_out_of_distribution` |
| `MEM_TYPE` | `None` | Memory mechanism (see table above) |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Policy model served by vLLM |
| `STRATEGY_MODEL_NAME` | same as `MODEL_NAME` | Strategy synthesis model (can differ from policy for off-policy setup) |
| `RETRIEVAL_TOPK` | `1` | Number of retrieved experiences *k* |
| `OPENAI_BASE_IP_ADDR` | `127.0.0.1` | vLLM server host:port |

### 3. STEP-MFT Training

STEP-MFT trains the model to generate high-utility strategies using process-level supervision.

**Step 1 — Collect training trajectories with `adamem-high`** (generates dense per-step strategy/action pairs):

```bash
SPLIT=train \
MEM_TYPE=adamem-high \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  python -m examples.prompt_agent.gpt4o_alfworld

采集原始轨迹：
SPLIT=train MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507  python -m examples.prompt_agent.gpt4o_alfworld

python -m examples.prompt_agent.collect_trajectories
```

**Step 2 — Filter SFT data** using `filter_sft_data.py`:

```bash
# STEP-MFT: keep only strategies that changed the agent's action (recommended)
python filter_sft_data.py \
  --traj_file "logs/alfworld/*/traj_train_adamem-high*.json" \
  --mode step-mft \
  --output_file sft_data_step_mft.json

# Outcome-MFT: keep all strategies from successful trajectories (weaker baseline)
python filter_sft_data.py \
  --traj_file "logs/alfworld/*/traj_train_adamem-high*.json" \
  --mode outcome-mft \
  --output_file sft_data_outcome_mft.json
```

**Step 3 — Fine-tune with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**:

```bash
# Register sft_data_step_mft.json as a dataset in LLaMA-Factory, then:
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --finetuning_type lora \
  --lora_rank 32 \
  --dataset sft_data_step_mft \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1
```

We use NVIDIA A40/L40S GPUs (48 GB). Run inference with the fine-tuned adapter by pointing `MODEL_NAME` and `STRATEGY_MODEL_NAME` at the saved adapter path.

---


```bash 


```

评测结果：

# 仅8条成功轨迹：

2026-06-14 22:32:14,388 - Env 97: Empty strategy_response after regeneration!
2026-06-14 22:32:27,491 -   Memory retrieval (strategy reuse): 3/73 refreshed
2026-06-14 22:32:27,855 - Step 47; Dones (69/140); SR 0.4928571428571429
2026-06-14 22:39:35,451 -   Memory retrieval (strategy reuse): 4/71 refreshed
2026-06-14 22:39:35,837 - Step 48; Dones (69/140); SR 0.4928571428571429
2026-06-14 22:47:30,910 - Env 108: Empty strategy_response after regeneration!
2026-06-14 22:47:57,666 -   Memory retrieval (strategy reuse): 6/71 refreshed
2026-06-14 22:47:58,001 - Step 49; Dones (70/140); SR 0.5
2026-06-14 22:55:13,540 -   Memory retrieval (strategy reuse): 5/70 refreshed
2026-06-14 22:55:13,888 - All environments finished early!
2026-06-14 22:55:15,337 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low.json after test 0
2026-06-14 22:55:15,337 - Test 0 overall success: 0.5071
2026-06-14 22:55:15,337 -     pick_and_place                     : 0.8571 (30/35)
2026-06-14 22:55:15,338 -     pick_two_obj_and_place             : 0.6250 (15/24)
2026-06-14 22:55:15,338 -     look_at_obj_in_light               : 0.2308 (3/13)
2026-06-14 22:55:15,338 -     pick_heat_then_place_in_recep      : 0.3125 (5/16)
2026-06-14 22:55:15,338 -     pick_cool_then_place_in_recep      : 0.2400 (6/25)
2026-06-14 22:55:15,338 -     pick_clean_then_place_in_recep     : 0.4444 (12/27)
2026-06-14 22:55:15,339 - Test 0 time elapsed: 18690.52s


# 120+ 条成功轨迹(topk = 1)：

2026-06-15 22:52:15,741 - Step 46; Dones (65/140); SR 0.4642857142857143
2026-06-15 22:56:04,430 - Env 13: Empty strategy_response after regeneration!
2026-06-15 22:56:38,076 -   Memory retrieval (strategy reuse): 3/75 refreshed
2026-06-15 22:56:38,396 - Step 47; Dones (65/140); SR 0.4642857142857143
2026-06-15 23:00:05,083 -   Memory retrieval (strategy reuse): 2/75 refreshed
2026-06-15 23:00:05,434 - Step 48; Dones (65/140); SR 0.4642857142857143
2026-06-15 23:03:31,355 -   Memory retrieval (strategy reuse): 1/75 refreshed
2026-06-15 23:03:31,760 - Step 49; Dones (66/140); SR 0.4714285714285714
2026-06-15 23:06:57,481 -   Memory retrieval (strategy reuse): 1/74 refreshed
2026-06-15 23:06:57,837 - All environments finished early!
2026-06-15 23:06:59,316 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260615_201529_b76f09f5.json after test 0
2026-06-15 23:06:59,317 - Test 0 overall success: 0.4714
2026-06-15 23:06:59,317 -     pick_and_place                     : 0.8571 (30/35)
2026-06-15 23:06:59,317 -     pick_two_obj_and_place             : 0.3333 (8/24)
2026-06-15 23:06:59,317 -     look_at_obj_in_light               : 0.1538 (2/13)
2026-06-15 23:06:59,317 -     pick_heat_then_place_in_recep      : 0.6250 (10/16)
2026-06-15 23:06:59,317 -     pick_cool_then_place_in_recep      : 0.2000 (5/25)
2026-06-15 23:06:59,317 -     pick_clean_then_place_in_recep     : 0.4074 (11/27)
2026-06-15 23:06:59,317 - Test 0 time elapsed: 10289.91s

# 120+ 条成功轨迹(topk = 4)：

2026-06-16 02:21:13,374 - Step 49; Dones (77/140); SR 0.55

# 120+ 条成功轨迹(topk = 8)：

2026-06-16 18:24:56,719 - Step 46; Dones (81/140); SR 0.5785714285714286
2026-06-16 18:27:46,983 - Env 80: Empty strategy_response after regeneration!
2026-06-16 18:28:01,924 -   Memory retrieval (strategy reuse): 1/59 refreshed
2026-06-16 18:28:02,277 - Step 47; Dones (81/140); SR 0.5785714285714286
2026-06-16 18:30:54,147 -   Memory retrieval (strategy reuse): 1/59 refreshed
2026-06-16 18:30:54,501 - Step 48; Dones (81/140); SR 0.5785714285714286
2026-06-16 18:34:42,030 -   Memory retrieval (strategy reuse): 3/59 refreshed
2026-06-16 18:34:42,384 - Step 49; Dones (81/140); SR 0.5785714285714286
2026-06-16 18:38:37,093 -   Memory retrieval (strategy reuse): 4/59 refreshed
2026-06-16 18:38:37,416 - All environments finished early!
2026-06-16 18:38:38,989 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260616_155117_fd965648.json after test 0
2026-06-16 18:38:38,990 - Test 0 overall success: 0.5786
2026-06-16 18:38:38,990 -     pick_and_place                     : 0.9429 (33/35)
2026-06-16 18:38:38,990 -     pick_two_obj_and_place             : 0.3333 (8/24)
2026-06-16 18:38:38,990 -     look_at_obj_in_light               : 0.2308 (3/13)
2026-06-16 18:38:38,990 -     pick_heat_then_place_in_recep      : 0.7500 (12/16)
2026-06-16 18:38:38,990 -     pick_cool_then_place_in_recep      : 0.3600 (9/25)
2026-06-16 18:38:38,990 -     pick_clean_then_place_in_recep     : 0.5926 (16/27)
2026-06-16 18:38:38,990 - Test 0 time elapsed: 10041.29s

# 120+ 条成功轨迹(topk = 8) + 固定pitfalls：

2026-07-11 12:56:23,887 - Step 40; Dones (58/140); SR 0.4142857142857143
2026-07-11 13:01:42,793 -   Memory retrieval (strategy reuse): 8/82 refreshed
2026-07-11 13:01:43,185 - Step 41; Dones (59/140); SR 0.42142857142857143
2026-07-11 13:06:42,155 -   Memory retrieval (strategy reuse): 5/81 refreshed
2026-07-11 13:06:42,555 - Step 42; Dones (59/140); SR 0.42142857142857143

# 加入persistent skills 

2026-07-13 23:55:55,706 - Step 45; Dones (89/140); SR 0.6357142857142857
2026-07-13 23:59:15,101 -   Memory retrieval (strategy reuse): 5/51 refreshed
2026-07-13 23:59:15,423 - Step 46; Dones (89/140); SR 0.6357142857142857
2026-07-14 00:02:45,355 -   Memory retrieval (strategy reuse): 4/51 refreshed
2026-07-14 00:02:45,670 - Step 47; Dones (90/140); SR 0.6428571428571429
2026-07-14 00:05:53,698 -   Memory retrieval (strategy reuse): 3/50 refreshed
2026-07-14 00:05:54,038 - Step 48; Dones (90/140); SR 0.6428571428571429
2026-07-14 00:09:26,831 -   Memory retrieval (strategy reuse): 4/50 refreshed
2026-07-14 00:09:27,196 - Step 49; Dones (91/140); SR 0.65
2026-07-14 00:13:21,178 -   Memory retrieval (strategy reuse): 5/49 refreshed
2026-07-14 00:13:21,517 - All environments finished early!
2026-07-14 00:13:23,177 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260713_210317_c22ed8d5.json after test 0
2026-07-14 00:13:23,177 - Test 0 overall success: 0.6500
2026-07-14 00:13:23,177 -     pick_and_place                     : 1.0000 (35/35)
2026-07-14 00:13:23,177 -     pick_two_obj_and_place             : 0.2083 (5/24)
2026-07-14 00:13:23,177 -     look_at_obj_in_light               : 0.2308 (3/13)
2026-07-14 00:13:23,177 -     pick_heat_then_place_in_recep      : 0.7500 (12/16)
2026-07-14 00:13:23,177 -     pick_cool_then_place_in_recep      : 0.7200 (18/25)
2026-07-14 00:13:23,178 -     pick_clean_then_place_in_recep     : 0.6667 (18/27)
2026-07-14 00:13:23,178 - Test 0 time elapsed: 11405.89s

# 加入persistent skills （第二次迭代）       

2026-07-14 19:58:49,970 - Step 45; Dones (95/140); SR 0.6785714285714286
2026-07-14 20:01:11,452 -   Memory retrieval (strategy reuse): 0/45 refreshed
2026-07-14 20:01:11,752 - Step 46; Dones (95/140); SR 0.6785714285714286
2026-07-14 20:03:30,519 -   Memory retrieval (strategy reuse): 0/45 refreshed
2026-07-14 20:03:30,836 - Step 47; Dones (95/140); SR 0.6785714285714286
2026-07-14 20:05:47,687 -   Memory retrieval (strategy reuse): 0/45 refreshed
2026-07-14 20:05:48,040 - Step 48; Dones (95/140); SR 0.6785714285714286
2026-07-14 20:08:05,200 -   Memory retrieval (strategy reuse): 0/45 refreshed
2026-07-14 20:08:05,549 - Step 49; Dones (95/140); SR 0.6785714285714286
2026-07-14 20:10:28,731 -   Memory retrieval (strategy reuse): 0/45 refreshed
2026-07-14 20:10:29,029 - All environments finished early!
2026-07-14 20:10:30,320 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-low_20260714_175615_849396e0.json after test 0
2026-07-14 20:10:30,321 - Test 0 overall success: 0.6786
2026-07-14 20:10:30,321 -     pick_and_place                     : 1.0000 (35/35)
2026-07-14 20:10:30,321 -     pick_two_obj_and_place             : 0.1667 (4/24)
2026-07-14 20:10:30,321 -     look_at_obj_in_light               : 0.3077 (4/13)
2026-07-14 20:10:30,321 -     pick_heat_then_place_in_recep      : 0.6875 (11/16)
2026-07-14 20:10:30,321 -     pick_cool_then_place_in_recep      : 0.8000 (20/25)
2026-07-14 20:10:30,321 -     pick_clean_then_place_in_recep     : 0.7778 (21/27)
2026-07-14 20:10:30,321 - Test 0 time elapsed: 8054.59s

# 加入反思机制
2026-08-01 15:37:04,285 - Step 43; Dones (96/140); SR 0.6857142857142857
2026-08-01 15:39:06,085 -   Memory retrieval (LDA-delay reuse): 7/44 refreshed (LDA<0 and no cooldown)
2026-08-01 15:39:06,448 - Step 44; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:40:51,658 -   Memory retrieval (LDA-delay reuse): 9/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:40:51,966 - Step 45; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:42:41,573 -   Memory retrieval (LDA-delay reuse): 12/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:42:41,914 - Step 46; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:44:36,399 -   Memory retrieval (LDA-delay reuse): 4/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:44:36,739 - Step 47; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:46:08,163 -   Memory retrieval (LDA-delay reuse): 1/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:46:08,515 - Step 48; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:48:01,977 -   Memory retrieval (LDA-delay reuse): 4/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:48:02,269 - Step 49; Dones (99/140); SR 0.7071428571428572
2026-08-01 15:50:00,452 -   Memory retrieval (LDA-delay reuse): 6/41 refreshed (LDA<0 and no cooldown)
2026-08-01 15:50:00,762 - All environments finished early!
2026-08-01 15:50:02,204 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_lda_delay.json after test 0
2026-08-01 15:50:02,205 - Test 0 overall success: 0.7071
2026-08-01 15:50:02,205 -     pick_and_place                     : 1.0000 (35/35)
2026-08-01 15:50:02,205 -     pick_two_obj_and_place             : 0.4167 (10/24)
2026-08-01 15:50:02,206 -     look_at_obj_in_light               : 0.2308 (3/13)
2026-08-01 15:50:02,206 -     pick_heat_then_place_in_recep      : 0.6250 (10/16)
2026-08-01 15:50:02,206 -     pick_cool_then_place_in_recep      : 0.8400 (21/25)
2026-08-01 15:50:02,206 -     pick_clean_then_place_in_recep     : 0.7407 (20/27)
2026-08-01 15:50:02,206 - Test 0 time elapsed: 5832.99s


# 加入反思机制 （成功+失败轨迹）
2026-08-04 16:26:28,282 -   Memory retrieval (LDA-delay reuse): 4/39 refreshed (LDA<0 and no cooldown)
2026-08-04 16:26:28,607 - All environments finished early!
2026-08-04 16:26:29,536 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_lda_delay_20260804_114453_4641a904.json after test 0
2026-08-04 16:26:29,536 - Test 0 overall success: 0.7214
2026-08-04 16:26:29,536 -     pick_and_place                     : 1.0000 (35/35)
2026-08-04 16:26:29,536 -     pick_two_obj_and_place             : 0.4583 (11/24)
2026-08-04 16:26:29,536 -     look_at_obj_in_light               : 0.3077 (4/13)
2026-08-04 16:26:29,537 -     pick_heat_then_place_in_recep      : 0.6875 (11/16)
2026-08-04 16:26:29,537 -     pick_cool_then_place_in_recep      : 0.7600 (19/25)
2026-08-04 16:26:29,537 -     pick_clean_then_place_in_recep     : 0.7778 (21/27)
2026-08-04 16:26:29,537 - Test 0 time elapsed: 16884.64s


# 无记忆（baseline）

2026-06-16 11:10:50,858 - Step 44; Dones (63/140); SR 0.45
2026-06-16 11:12:31,870 - Step 45; Dones (63/140); SR 0.45
2026-06-16 11:14:13,186 - Step 46; Dones (64/140); SR 0.45714285714285713
2026-06-16 11:15:57,277 - Step 47; Dones (65/140); SR 0.4642857142857143
2026-06-16 11:17:43,472 - Step 48; Dones (65/140); SR 0.4642857142857143
2026-06-16 11:19:25,453 - Step 49; Dones (65/140); SR 0.4642857142857143
2026-06-16 11:21:12,511 - All environments finished early!
2026-06-16 11:21:12,984 - Trajectories saved to logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution.json after test 0
2026-06-16 11:21:12,985 - Test 0 overall success: 0.4643
2026-06-16 11:21:12,985 -     pick_and_place                     : 0.9429 (33/35)
2026-06-16 11:21:12,985 -     pick_two_obj_and_place             : 0.5000 (12/24)
2026-06-16 11:21:12,985 -     look_at_obj_in_light               : 0.2308 (3/13)
2026-06-16 11:21:12,985 -     pick_heat_then_place_in_recep      : 0.3125 (5/16)
2026-06-16 11:21:12,985 -     pick_cool_then_place_in_recep      : 0.2400 (6/25)
2026-06-16 11:21:12,985 -     pick_clean_then_place_in_recep     : 0.2222 (6/27)
2026-06-16 11:21:12,985 - Test 0 time elapsed: 4372.50s

● # 🎯 ReasoningBank Baseline — 各类别 SR

  ┌───────────────────────────────────┬─────────────┬────────┐
  │           Task Category           │ Win / Total │ SR (%) │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ pick_and_place_with_movable_recep │     35 / 45 │ 77.78% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ pick_heat_then_place_in_recep     │       5 / 6 │ 83.33% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ pick_clean_then_place_in_recep    │     19 / 27 │ 70.37% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ pick_cool_then_place_in_recep     │      8 / 25 │ 32.00% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ pick_two_obj_and_place            │      8 / 24 │ 33.33% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ look_at_obj_in_light              │      1 / 13 │  7.69% │
  ├───────────────────────────────────┼─────────────┼────────┤
  │ OVERALL                           │    76 / 140 │ 54.29% │
  └───────────────────────────────────┴─────────────┴────────┘




idea:

1.论文仅仅利用成功轨迹，尝试利用失败轨迹总结失败共性作为长期通用错误提示

2.TTA测试时LTM被冻结，没有利用到测试时成功轨迹，尝试将测试时成功轨迹加入LTM

3.基于置信度的自适应决策（模型输出logits熵）  效果会比agent输出的refresh好吗？
置信度衡量agent当前决策的把握程度，refresh衡量当前策略的效用

4.STM策略缓存：面对许多相似的步骤，可以复用过往策略（通过关键特征作为检索键）

5.LTM中的成功轨迹可能荣誉，尝试成功轨迹剪枝

6.课程式微调策略模型


## OPSD Training (On-Policy Self-Distillation)

`finetune_opsd.py` trains a LoRA adapter on the **successful trajectory SFT data** (`sft_data_step_mft.json`) using **on-policy self-distillation**: the same base model acts as both student and teacher — the teacher sees a privileged reference solution in its prompt, the student sees only the problem; both forward over the student's on-policy generation; a JSD loss pulls the student distribution toward the teacher's.

Single command, GPU-count-agnostic:

```bash
# 2 GPUs (default)
./launch_opsd_ngpu.sh

# 4 GPUs
NUM_GPUS=4 ./launch_opsd_ngpu.sh

# Custom GPU list, output dir, etc.
NUM_GPUS=2 GPUS="0 3" \
  OUTPUT_DIR=lora_output/my_v2 \
  SFT_DATA_FILE=sft_data_step_mft.json \
  ./launch_opsd_ngpu.sh
```

All trainer knobs are env-var configurable:

| Env var | Default | What it controls |
|---|---|---|
| `NUM_GPUS` | 2 | Number of GPU ranks |
| `GPUS` | `"0 1 2 3"` | Space-separated GPU IDs (first `NUM_GPUS` are used) |
| `OPSD_OUTPUT_DIR` | `lora_output/adamem_opsd` | Where the adapter is saved |
| `OPSD_SFT_DATA_FILE` | `sft_data_step_mft.json` | Input data (alpaca format) |
| `OPSD_NUM_EPOCHS` | 3 | Training epochs |
| `OPSD_LR` | 1e-4 | Peak learning rate |
| `OPSD_BATCH_SIZE` | 1 | Per-device micro-batch |
| `OPSD_GRAD_ACCUM` | 8 | Gradient accumulation steps |
| `OPSD_LORA_RANK` | 32 | LoRA rank |
| `OPSD_LORA_ALPHA` | 64 | LoRA alpha |
| `OPSD_WARMUP` | 0.1 | LR warmup ratio |

Output: `lora_output/adamem_opsd*/` containing LoRA adapter (≈132 MB) plus `epoch_1/`, `epoch_2/`, `epoch_3/` snapshots.

Logs: `logs/opsd_<N>gpu/rank_<i>.log` per rank.



2026-07-30 11:42:13,484 - Step 15; Dones (62/140); SR 0.44285714285714284