# CAEI：Cognitive-Aware Experience Internalization for Self-Evolving Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**CAEI** is a framework that unifies
cognitive-aware test-time adaptation with on-policy hindsight self-distillation into a
closed-loop continual learning pipeline.

> **CAEI：Cognitive-Aware Experience Internalization for Self-Evolving Agents**

CAEI operates through three integrated stages:

1. **Stage 1 — CoT-Aware Memory Construction**: Build a factual memory pool from both
   successful *and* failed trajectories, and derive a Chain-of-Thought cognitive
   orientation axis via Linear Discriminant Analysis (LDA).
2. **Stage 2 — Test-Time Adaptation**: Dynamically synthesize procedural policies and
   trigger adaptive reflection only when the agent's internal hesitation score exceeds a confidence threshold, enabling robust real-time recalibration.
3. **Stage 3 — OPSD Experience Internalization**: An On-Policy Self-Distillation (OPSD)
   mechanism progressively internalizes slow, deliberative reasoning into fast, intuitive responses, reducing inference latency while continually enhancing task performance.

---

## Repository Structure

```
CAEI/
├── agent_system/           # Core agent framework (envs, memory, rollout, reward)
├── test/
│   └── prompt_agent/       # Agent entry scripts (gpt4o_alfworld.py, collect_trajectories.py)
├── construct/              # Memory index building & LDA classifier training
│   ├── index/              #   HNSW index construction (step-level, traj-level, reasoningbank)
│   ├── lda/                #   LDA pipeline (extract → cluster → classify)
│   └── README.md           #   Detailed usage for construct/
├── trainer/                # Custom trainers & OPSD training scripts
│   ├── opsd_trainer.py
│   ├── data_collator.py
│   └── sdar_trainer/       #   SDAR / verl-based training shell scripts
├── finetune/               # SFT data filtering & fine-tuning scripts
├── analysis/               # Trajectory & LDA decision analysis tools
├── skills/                 # Skill / memory bank generation pipeline
├── scripts/                # Utility scripts (vLLM / HF probing)
├── utils.py                # Core utilities (embed_texts, RetrievalService, etc.)
└── logs/                   # Trajectory output (gitignored)
├── retrieval_data/         # Built HNSW indexes (gitignored)
├── lora_output/            # Saved LoRA checkpoints (gitignored)
```

---

## Quick Start (ALFWorld)

The following walks through the full three-stage pipeline on **ALFWorld** with
**Qwen3-4B-Instruct-2507** as the backbone model and **Qwen3-Embedding-4B** for retrieval.

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install ALFWorld (follow official instructions)
pip install alfworld
export ALFWORLD_DATA=$HOME/.cache/alfworld
```

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# vLLM policy server (served on port 8001)
OPENAI_BASE_IP_ADDR=127.0.0.1:8001

# vLLM embedding server (served on port 8002)
EMB_VLLM_SERVER=127.0.0.1
EMBED_MODEL_NAME=Qwen/Qwen3-Embedding-4B

# Model name
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
```

---

### Stage 1: CoT-Aware Memory Construction

> Build the factual memory pool and train the LDA hesitation classifier.

#### Step 1.1 — Collect training trajectories

First, roll out the base model (no memory) on the ALFWorld training split to collect
raw trajectories that will serve as the factual memory pool:

```bash
# ============================================================
# Terminal 1: Start the policy LLM server (vLLM, port 8001)
# ============================================================
# Single-GPU version:
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768

# Multi-GPU (2 GPUs, tensor-parallel) version:
# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen3-4B-Instruct-2507 \
#   --host 0.0.0.0 --port 8001 \
#   --dtype bfloat16 --max-model-len 32768 \
#   --tensor-parallel-size 2

# ============================================================
# Terminal 2: Start the embedding model server (vLLM, port 8002)
# ============================================================
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-4B \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --task embed

# ============================================================
# Terminal 3: Collect training trajectories
# ============================================================
# No-memory rollout on the ALFWorld training split
python -m test.prompt_agent.collect_trajectories
```

Trajectories are saved to `logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train_*.json`.

#### Step 1.2 — Build the HNSW memory index

Build the step-level factual memory index (used by all CAEI variants):

```bash
# Build step-level index from training trajectories (both success + failure)
python construct/index/build_index.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json
```

The index is saved to `retrieval_data/alfworld/Qwen_Qwen3-4B-Instruct-2507/`.

> **Note:** The embedding model name must match what your embedding server. By default it
> reads from `EMBED_MODEL_NAME` env var (default: `Qwen/Qwen3-Embedding-4B`).
> Set `EMB_VLLM_SERVER` if your embedding server is not on localhost.

Additional index variants (for ablations / baselines):

```bash
# Correct-only index (only successful trajectories — for caei_lda / caei_high modes)
python construct/index/build_index.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json \
  --correct_only

# Failure-only index (only failed trajectories — required for lda_reflect mode
python construct/index/build_index.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json \
  --failure_only

# Episode-level index (Synapse baseline)
python construct/index/build_index_traj_level.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json

# Strategy index (ReasoningBank baseline)
python construct/index/build_index_reasoningbank.py \
  --dataset_name alfworld \
  --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json
```

#### Step 1.3 — Train the LDA hesitation classifier

Extract confident/hesitant CoT samples from the trajectory pool, then fit the LDA
cognitive orientation axis:

```bash
# Step 1: Extract representative thinking samples (4-way taxonomy)
python construct/lda/extract_thinking_samples.py \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json \
  --output_file construct/lda/data/thinking_samples.json

# Step 2: Extract confident/hesitant binary labeled samples
python construct/lda/extract_confident_hesitant_v2.py \
  --traj_file logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_train.json \
  --output_file construct/lda/data/confident_hesitant_samples_v2.json \
  --n_per_class 200

# Step 3: Extract planning-token hidden states (requires HF model access)
python construct/lda/extract_planning_tokens.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --samples_file construct/lda/data/confident_hesitant_samples_v2.json \
  --output_vectors construct/lda/data/lda_vectors_v2.npy \
  --output_labels construct/lda/data/lda_labels_v2.json

# Step 4: Train LDA classifier + save deployable model
python construct/lda/lda_planning_classifier_v2.py \
  --vectors construct/lda/data/lda_vectors_v2.npy \
  --labels construct/lda/data/lda_labels_v2.json \
  --samples construct/lda/data/confident_hesitant_samples_v2.json
```

This produces:
- `construct/lda/data/lda_model_v2.npz` — the deployed LDA model (w, b, scaler)
- `construct/lda/data/lda_result_v2.json` — cross-validation metrics (accuracy, AUC)
- `construct/lda/data/lda_proj_v2.npy` — 1D projections for visualization

---

### Stage 2: Test-Time Adaptation (CAEI Inference)

> Run the CAEI agent with hesitation-aware adaptive reflection.

Make sure both vLLM servers (policy on port 8001, embedding on port 8002) are running.

```bash
# Full CAEI with LDA + reflection (requires both correct_only and failure_only indexes)
MEM_TYPE=lda_reflect \
SPLIT=eval_in_distribution \
python -m test.prompt_agent.gpt4o_alfworld

# Lightweight CAEI (LDA-triggered strategy refresh, no failed-trajectory reflection)
MEM_TYPE=caei_lda \
SPLIT=eval_in_distribution \
python -m test.prompt_agent.gpt4o_alfworld

# Baseline: no memory (plain ReAct)
SPLIT=eval_in_distribution \
python -m test.prompt_agent.gpt4o_alfworld
```

**Available memory types** (`MEM_TYPE`):

| `MEM_TYPE` | Description | Required indexes |
|---|---|---|
| *(unset)* | No memory (base ReAct agent) | none |
| `synapse` | Synapse baseline: full-trajectory retrieval | traj-level index |
| `reasoningbank` | ReasoningBank baseline: pre-synthesized strategy retrieval | reasoningbank index |
| `caei_lda` | CAEI — LDA hesitation score triggers adaptive reflection | correct_only index |
| `caei_lda_delay` | CAEI with cooldown (skip 5 steps after each reflection) | correct_only index |
| `lda_reflect` | **Full CAEI** — failed-trajectory retrieval during reflection | correct_only + failure_only |

> **Note for `lda_reflect` mode:** This mode retrieves SUCCESSFUL trajectories for
> initial strategy synthesis and FAILED trajectories for reflection generation. You
> must build both `--correct_only` and `--failure_only` indexes (see Step 1.2 above).

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEM_TYPE` | `None` | Memory mode (see table above) |
| `SPLIT` | `eval_in_distribution` | Dataset split: `train`, `eval_in_distribution`, `eval_out_of_distribution` |
| `TOPK` | `8` | Number of retrieved trajectories |
| `LDA_MODEL_PATH` | `construct/lda/data/lda_model_v2.npz` | Path to trained LDA model |
| `LDA_THRESHOLD` | `0.0` | Hesitation score threshold τ for reflection trigger |
| `OPSD_LDA_ENABLED` | `true` | Enable LDA scoring (set to `false` to disable) |
| `CORRECT_ONLY` | `true` | Use success-only index (set to `false` for all trajectories) |
| `RETRIEVAL_INDEX_PREFIX` | `train_` | Prefix for index file names (e.g., `train_`, `eval_`) |
| `MEMORY_MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Model name used for trajectory memory (determines index directory) |
| `NUM_ENVS` | `1` | Number of parallel environments |
| `TEST_TIMES` | `1000` | Number of evaluation episodes |
| `OPENAI_BASE_IP_ADDR` | `127.0.0.1:8001` | vLLM policy server address |
| `EMB_VLLM_SERVER` | `127.0.0.1` | vLLM embedding server hostname (uses port 8002) |
| `EMBED_MODEL_NAME` | `Qwen/Qwen3-Embedding-4B` | Embedding model name |

> **GPU memory note for LDA mode:** The LDA scorer loads a separate copy of the
> base model (via HuggingFace `transformers`) for hidden-state extraction. Make sure
> you have at least one additional GPU available, or use a smaller model for the
> encoder. If no GPU is available, the agent will gracefully degrade to CAEI-low
> (strategy-only, no LDA triggering).

Output trajectories are saved to `logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_<split>_<mem_type>_<timestamp>.json`.

---

### Stage 3: OPSD Experience Internalization

> On-Policy Self-Distillation progressively internalizes slow, deliberative
> reasoning into fast, intuitive responses — forming a closed-loop continual
> learning pipeline.

OPSD consists of **two complementary distillation objectives**, jointly optimized
to prevent exposure bias and catastrophic forgetting:

| | Action Internalization (3a) | Reflection Learning (3b) |
|---|---|---|
| **Paper** | Internalizing Slow Thinking into Intuition | Learning to Reflect Reasonably |
| Student sees | task + state only (p₁ = (x)) | task + history (p₂ = (x, τ_{<t})) |
| Teacher sees | task + strategy + reflection (h₁ = (τ⁺, z₀, f_t)) | full reflection prompt (h₂ = (f_t⁺)) |
| Target | correct action output | high-quality reflection content |
| Filter | steps where strategy/reflection changed the action (A>0) | reflections that changed the action (A(s,f_t)>0) |

---

#### Step 3.1 — Filter distillation data from CAEI trajectories

Run **both** data filtering modes on the CAEI trajectories collected in Stage 2:

```bash
# (a) Action internalization data — student sees task, teacher sees strategy+reflection
python finetune/filter_sft_data.py \
  --traj_file "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_*_caei_lda_*.json" \
  --mode opsd-action \
  --output_file finetune/data/opsd_action_data.json \
  --tokenizer_name Qwen/Qwen3-4B-Instruct-2507

# (b) Reflection learning data — student sees task+history, teacher sees reflection
python finetune/filter_sft_data.py \
  --traj_file "logs/alfworld/Qwen_Qwen3-4B-Instruct-2507/traj_*_lda_reflect_*.json" \
  --mode opsd-reflection \
  --output_file finetune/data/opsd_reflection_data.json \
  --tokenizer_name Qwen/Qwen3-4B-Instruct-2507 \
  --min_reflection_len 40
```

> **Note:** For reflection data (3b), use trajectories from `MEM_TYPE=lda_reflect`,
> which includes `reflection_prompt` and `reflection_response` fields in each step.

---

#### Step 3.2a — Train: Action Internalization

```bash
# 4-GPU DDP OPSD training (action internalization)
bash trainer/run_opsd_action_qwen3_4b_4gpu.sh
```

Key configuration:

| Variable | Default | Description |
|---|---|---|
| `OPSD_ACTION_DATA` | `finetune/data/opsd_action_data.json` | Action-internalization training data |
| `OPSD_ACTION_OUTPUT_DIR` | `lora_output/caei_opsd_action_4b` | LoRA output directory |
| `OPSD_LORA_RANK` | `32` | LoRA rank |
| `OPSD_LR` | `1e-6` | Learning rate |
| `OPSD_NUM_EPOCHS` | `150` | Training epochs |
| `OPSD_BATCH_SIZE` | `4` | Per-device batch size |

---

#### Step 3.2b — Train: Reflection Learning

```bash
# 4-GPU DDP OPSD training (reflection learning)
bash trainer/run_opsd_reflection_qwen3_4b_4gpu.sh
```

Key configuration:

| Variable | Default | Description |
|---|---|---|
| `OPSD_REFLECTION_DATA` | `finetune/data/opsd_reflection_data.json` | Reflection-learning training data |
| `OPSD_REFLECTION_OUTPUT_DIR` | `lora_output/caei_opsd_reflection_4b` | LoRA output directory |
| `OPSD_LORA_RANK` | `32` | LoRA rank |
| `OPSD_LR` | `1e-6` | Learning rate |
| `OPSD_NUM_EPOCHS` | `150` | Training epochs |
| `OPSD_BATCH_SIZE` | `4` | Per-device batch size |

> **Paper setup:** LoRA rank = 32, AdamW lr = 1×10⁻⁶, global batch size = 16,
> trained for 150 epochs on 4× A100 (40GB) GPUs with FSDP.

---

#### Step 3.3 — Deploy the distilled model (close the loop)

After OPSD training, the distilled LoRA adapter can be loaded into the vLLM server for
the next iteration. Replace the base model with the distilled model and return to
Stage 2 to continue the closed-loop continual learning cycle.

```bash
# Serve the distilled model with both action + reflection LoRA adapters
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-models \
    caei_opsd_action=lora_output/caei_opsd_action_4b,\
    caei_opsd_reflection=lora_output/caei_opsd_reflection_4b \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768
```

Then return to **Stage 2** with the distilled model for the next round of the
continual learning loop.

---

## Analysis Tools

```bash
# Analyze LDA decisions in trajectories
python analysis/analyze_lda_decisions.py \
  --traj_file logs/alfworld/.../traj_eval_in_distribution_caei_lda_*.json

# Deep analysis of LDA vs correct decisions
python analysis/analyze_lda_correct.py \
  --traj_file logs/alfworld/.../traj_eval_in_distribution_caei_lda_*.json
```

---

## License

This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.
