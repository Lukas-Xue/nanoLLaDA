"""Checkpoint save/load utilities for nanoLLaDA."""

import os
import glob
import json
import logging
import torch
import torch.distributed as dist
from nanollada.common import get_base_dir, is_ddp_initialized

logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, keep_last=3):
    # Rank 0 creates directory, saves model + metadata, and cleans old checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model_data, os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
        with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        # Delete old checkpoints, keeping only the last `keep_last`
        _cleanup_old_checkpoints(checkpoint_dir, keep_last)
    # Barrier ensures directory exists before other ranks write optimizer state
    if is_ddp_initialized():
        dist.barrier()
    # Each rank saves its own optimizer shard
    if optimizer_data is not None:
        torch.save(optimizer_data, os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt"))

def _cleanup_old_checkpoints(checkpoint_dir, keep_last):
    """Delete all but the most recent `keep_last` checkpoints."""
    meta_files = sorted(glob.glob(os.path.join(checkpoint_dir, "meta_*.json")))
    if len(meta_files) <= keep_last:
        return
    for meta_path in meta_files[:-keep_last]:
        step_str = os.path.basename(meta_path).replace("meta_", "").replace(".json", "")
        for pattern in [f"model_{step_str}.pt", f"meta_{step_str}.json", f"optim_{step_str}_rank*.pt"]:
            for f in glob.glob(os.path.join(checkpoint_dir, pattern)):
                os.remove(f)
        logger.info(f"Deleted old checkpoint: step {step_str}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=True, rank=0):
    model_data = torch.load(os.path.join(checkpoint_dir, f"model_{step:06d}.pt"), map_location=device)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json")) as f:
        meta_data = json.load(f)
    optimizer_data = None
    if load_optimizer:
        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        if os.path.exists(optim_path):
            try:
                optimizer_data = torch.load(optim_path, map_location=device)
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}. Will use fresh optimizer.")
        else:
            logger.warning(f"Optimizer file not found: {optim_path}. Will use fresh optimizer.")
    return model_data, optimizer_data, meta_data
