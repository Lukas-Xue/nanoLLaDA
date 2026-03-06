#!/bin/bash
# nanoLLaDA: Train a masked diffusion language model from scratch.
# Designed for 4x L4 GPUs (~23GB each). Takes a few hours.
#
# Usage:
#   bash run.sh                          # basic run
#   WANDB_RUN=myrun bash run.sh          # with wandb logging

set -e
export OMP_NUM_THREADS=1
export NANOLLADA_BASE_DIR="$HOME/.cache/nanollada"
mkdir -p $NANOLLADA_BASE_DIR

# --- Python venv setup ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# --- wandb ---
WANDB_RUN=${WANDB_RUN:-dummy}

# --- Download data ---
# 8 shards for tokenizer training (~800MB), then 80 more in background
python -m nanollada.dataset -n 8
python -m nanollada.dataset -n 80 &
DATASET_PID=$!

# --- Train tokenizer ---
python -m scripts.tok_train

# --- Wait for data ---
echo "Waiting for dataset download..."
wait $DATASET_PID

# --- Pretrain ---
torchrun --standalone --nproc_per_node=4 -m scripts.train \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=16 \
    --total-batch-size=131072 \
    --lr=3e-4 \
    --eval-every=250 \
    --sample-every=500 \
    --run=$WANDB_RUN

echo "Done! Model saved to $NANOLLADA_BASE_DIR/checkpoints/d12/"
