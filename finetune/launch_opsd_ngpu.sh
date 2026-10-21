#!/bin/bash
# Launch OPSD training across N GPUs (configurable). Each rank is bound to
# one GPU via CUDA_VISIBLE_DEVICES=<gpu_id>; RANK / LOCAL_RANK / WORLD_SIZE
# are passed explicitly so NCCL can rendezvous on master_addr:master_port.
#
# Usage:
#   NUM_GPUS=2 ./launch_opsd_ngpu.sh             # 2 GPUs, default dirs
#   NUM_GPUS=4 ./launch_opsd_ngpu.sh             # 4 GPUs
#   NUM_GPUS=2 LOG_SUBDIR=opsd_2g_v2 OUTPUT_DIR=lora_output/v2 ./launch_opsd_ngpu.sh
#
# All configuration is via environment variables; no edits needed in this
# file or in finetune_opsd.py. See README block at the bottom.

set -e
cd "$(dirname "$0")"

# ---------- Configuration via env (with sensible defaults) ----------
NUM_GPUS=${NUM_GPUS:-2}
GPUS=${GPUS:-"0 1 2 3"}                    # space-separated list of GPU IDs
LOG_SUBDIR=${LOG_SUBDIR:-"opsd_${NUM_GPUS}gpu"}
OUTPUT_DIR=${OUTPUT_DIR:-"lora_output/caei_opsd_${NUM_GPUS}gpu"}
MASTER_PORT=${MASTER_PORT:-29500}
SFT_DATA_FILE=${SFT_DATA_FILE:-data/sft_data_step_mft.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-finetune_opsd.py}    # or finetune_opsd_intern.py
LOG_PREFIX=${LOG_PREFIX:-"opsd"}                    # opsd | opsd_intern

# Make the GPU list work for any NUM_GPUS by slicing the first N entries.
read -ra GPUS_ARR <<< "$GPUS"
GPUS_USED=("${GPUS_ARR[@]:0:$NUM_GPUS}")

# Sanity check.
if [[ ${#GPUS_USED[@]} -ne $NUM_GPUS ]]; then
  echo "ERROR: requested NUM_GPUS=$NUM_GPUS but GPUS list has only ${#GPUS_ARR[@]} entries: $GPUS" >&2
  exit 1
fi

LOG_DIR="logs/${LOG_PREFIX}_${NUM_GPUS}gpu"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[launch] config:"
echo "  NUM_GPUS    = $NUM_GPUS"
echo "  GPUS        = ${GPUS_USED[*]}"
echo "  TRAIN_SCRIPT= $TRAIN_SCRIPT"
echo "  LOG_DIR     = $LOG_DIR"
echo "  OUTPUT_DIR  = $OUTPUT_DIR"
echo "  SFT_DATA    = $SFT_DATA_FILE"
echo "  MASTER_PORT = $MASTER_PORT"

# Clean up any stale finetune processes from a previous run.
# Only kill processes that aren't us or our parent shell.
pkill -f "finetune_opsd" 2>/dev/null || true
sleep 1

for i in "${!GPUS_USED[@]}"; do
  GPU_ID=${GPUS_USED[$i]}
  RANK=$i
  LOG_FILE="${LOG_DIR}/rank_${RANK}.log"
  : > "$LOG_FILE"
  echo "[launch] starting rank=$RANK on GPU=$GPU_ID, log=$LOG_FILE"
  ( CUDA_VISIBLE_DEVICES=$GPU_ID \
    LOCAL_RANK=$RANK \
    RANK=$RANK \
    WORLD_SIZE=$NUM_GPUS \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$MASTER_PORT \
    OPSD_OUTPUT_DIR="$OUTPUT_DIR" \
    OPSD_SFT_DATA_FILE="$SFT_DATA_FILE" \
    OPSD_INTERN_DATA_FILE="$SFT_DATA_FILE" \
    OPSD_INTERN_OUTPUT_DIR="$OUTPUT_DIR" \
    python -u "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 ) &
done

echo "[launch] all $NUM_GPUS ranks spawned. Waiting for them to finish."
wait
echo "[launch] all ranks done."

# ---------- README ----------
: <<'README'
Environment variables (all optional, defaults shown):
    NUM_GPUS=2                         # number of GPU ranks
    GPUS="0 1 2 3"                     # space-separated GPU IDs to use
    LOG_SUBDIR=opsd_${NUM_GPUS}gpu     # subdir under logs/
    OUTPUT_DIR=lora_output/caei_opsd_${NUM_GPUS}gpu
    MASTER_PORT=29500                  # NCCL rendezvous port
    SFT_DATA_FILE=data/sft_data_step_mft.json

Examples:
    # 2-GPU run on GPUs 2,3
    NUM_GPUS=2 GPUS="2 3" ./launch_opsd_ngpu.sh

    # 4-GPU run on all four cards
    NUM_GPUS=4 ./launch_opsd_ngpu.sh

    # 2-GPU run with custom output dir
    NUM_GPUS=2 OUTPUT_DIR=lora_output/my_run_v1 ./launch_opsd_ngpu.sh

    # 1-GPU run (degenerate; no NCCL)
    NUM_GPUS=1 GPUS="0" ./launch_opsd_ngpu.sh
README