#!/bin/bash
#
# OPSD Stage 3b: Reflection Learning
# ==================================
# Paper Section 3.3 — Learning to Reflect Reasonably
#
# Student sees task + history (p₂ = (x, τ_{<t}))
# Teacher sees the full reflection prompt with retrieved trajectories (h₂ = (f_t⁺))
#
# The student learns to produce high-quality metacognitive reflections
# from context alone, without memory retrieval. Only reflections that
# actually changed the agent's action (action advantage A(s, f_t) > 0)
# are used — preserving precision over recall.
#
# Data: finetune/data/opsd_reflection_data.json
#       (from filter_sft_data.py --mode opsd-reflection)
#
# Usage:
#   bash trainer/run_opsd_reflection_qwen3_4b_4gpu.sh [vllm|sglang]
#
set -x
ENGINE=${1:-vllm}

# --- Model & data ---
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export OPSD_REFLECTION_DATA="finetune/data/opsd_reflection_data.json"
export OPSD_REFLECTION_OUTPUT_DIR="lora_output/caei_opsd_reflection_4b"

# --- Training hyperparameters ---
export OPSD_LORA_RANK=32
export OPSD_LORA_ALPHA=64
export OPSD_LR=1e-6
export OPSD_NUM_EPOCHS=150
export OPSD_BATCH_SIZE=4
export OPSD_GRAD_ACCUM=1

# --- Training setup ---
export TOTAL_GPUS=4
export ALFWORLD_DATA=$HOME/.cache/alfworld

cd "$(dirname "$0")/.." || exit 1

torchrun --nproc_per_node=$TOTAL_GPUS \
    finetune/finetune_opsd_reflection.py
