#!/bin/bash
#
# OPSD Stage 3a: Action Internalization
# =====================================
# Paper Section 3.3 — Internalizing Slow Thinking into Intuition
#
# Student sees only the task + state (p₁ = (x))
# Teacher sees task + state + strategy + reflection (h₁ = (τ⁺, z₀, f_t))
#
# The student learns to produce correct actions without memory retrieval
# or iterative thinking — distilling "slow deliberation" into "fast intuition".
#
# Data: finetune/data/opsd_action_data.json
#       (from filter_sft_data.py --mode opsd-action)
#
# Usage:
#   bash trainer/run_opsd_action_qwen3_4b_4gpu.sh [vllm|sglang]
#
set -x
ENGINE=${1:-vllm}

# --- Model & data ---
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export OPSD_ACTION_DATA="finetune/data/opsd_action_data.json"
export OPSD_ACTION_OUTPUT_DIR="lora_output/caei_opsd_action_4b"

# --- Training hyperparameters (paper Appendix A.3) ---
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
    finetune/finetune_opsd_action.py
