"""Checkpoint save/load utilities for nanoLLaDA."""

import os
import json
import logging
import torch
import torch.distributed as dist
from nanollada.common import get_base_dir, is_ddp_initialized
from nanollada.model import DiffusionTransformer, DiffusionTransformerConfig

logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    # Rank 0 creates directory and saves model + metadata
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model_data, os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
        with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
    # Barrier ensures directory exists before other ranks write optimizer state
    if is_ddp_initialized():
        dist.barrier()
    # Each rank saves its own optimizer shard
    if optimizer_data is not None:
        torch.save(optimizer_data, os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt"))

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=True, rank=0):
    model_data = torch.load(os.path.join(checkpoint_dir, f"model_{step:06d}.pt"), map_location=device)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json")) as f:
        meta_data = json.load(f)
    optimizer_data = None
    if load_optimizer:
        optimizer_data = torch.load(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt"), map_location=device)
    return model_data, optimizer_data, meta_data
