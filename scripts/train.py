"""
Pretrain a masked diffusion language model (LLaDA-style).

Single GPU:   python -m scripts.train
Multi-GPU:    torchrun --nproc_per_node=4 -m scripts.train

The core idea: randomly mask tokens with ratio t ~ Uniform(0,1),
predict the masked tokens, loss = cross_entropy / mask_ratio.
This is the ELBO bound on negative log-likelihood for masked diffusion.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext

import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanollada.model import DiffusionTransformer, DiffusionTransformerConfig
from nanollada.dataloader import distributed_data_loader
from nanollada.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    autodetect_device_type, get_peak_flops, get_base_dir,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
)
from nanollada.tokenizer import get_tokenizer, get_token_bytes
from nanollada.checkpoint import save_checkpoint, load_checkpoint
from nanollada.generate import generate
from nanollada.diffusion import forward_process, compute_diffusion_loss

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pretrain masked diffusion LM")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--device-type", type=str, default="")
# Model
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=1024)
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=float, default=10.5)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=131072)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.3)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Eval / logging
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=10*131072)
parser.add_argument("--sample-every", type=int, default=500)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--resume-from-step", type=int, default=-1)
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanollada", name=args.run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
mask_id = tokenizer.get_mask_token_id()
print0(f"Vocab size: {vocab_size:,}, mask_id: {mask_id}")

# -----------------------------------------------------------------------------
# Model
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = DiffusionTransformerConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_embd=model_dim,
)
with torch.device("meta"):
    model = DiffusionTransformer(config)
model_config_kwargs = asdict(config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

# Resume
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model  # uncompiled, un-DDP'd model for checkpointing and inference
# Wrap in DDP before compile (DDP must wrap the raw module so no_sync works)
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[ddp_local_rank])
model = torch.compile(model, dynamic=True)

# Param counts
num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = orig_model.estimate_flops()
print0(f"Parameters: {num_params:,}")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Training horizon
total_batch_size = args.total_batch_size
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    # Scaling: compute-optimal tokens from param:data ratio
    num_scaling_params = num_params - orig_model.transformer.wte.weight.numel()
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    num_iterations = target_tokens // total_batch_size
print0(f"Training iterations: {num_iterations:,}")
print0(f"Total tokens: {total_batch_size * num_iterations:,}")

# -----------------------------------------------------------------------------
# Optimizer (AdamW — simple and reliable)
def get_param_groups(model):
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(get_param_groups(model), lr=args.lr, betas=(args.beta1, args.beta2), fused=True)
if resuming and optimizer_data is not None:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data
elif resuming:
    print0("WARNING: No optimizer state found, resuming with fresh optimizer")

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# LR schedule
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * args.final_lr_frac

# -----------------------------------------------------------------------------
# Dataloader
dataloader_resume = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = distributed_data_loader(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume)
input_ids, dataloader_state_dict = next(train_loader)

# Validation loader builder
def build_val_loader():
    return distributed_data_loader(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)

# -----------------------------------------------------------------------------
# Eval: compute validation loss
@torch.no_grad()
def evaluate_val_loss(model, val_loader, steps, mask_id):
    total_loss = torch.tensor(0.0, device=device)
    for i, (input_ids, _) in enumerate(val_loader):
        if i >= steps:
            break
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_id)
        logits = model(noisy_batch)
        if masked_indices.any():
            token_loss = F.cross_entropy(
                logits[masked_indices], input_ids[masked_indices], reduction='none'
            ) / p_mask[masked_indices]
            total_loss += token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    world_size = dist.get_world_size() if is_ddp_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return (total_loss / (steps * world_size)).item()

# -----------------------------------------------------------------------------
# Training loop
if not resuming:
    step = 0
    val_loss = None
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    ls = meta_data["loop_state"]
    val_loss = meta_data.get("val_loss")
    min_val_loss = ls["min_val_loss"]
    smooth_train_loss = ls["smooth_train_loss"]
    total_training_time = ls["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank: {tokens_per_fwdbwd:,}")
print0(f"Total batch size: {total_batch_size:,} => grad accum steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations

    # Eval
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_loss = evaluate_val_loss(model, build_val_loader(), max(eval_steps, 1), mask_id)
        print0(f"Step {step:05d} | Val loss: {val_loss:.6f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({"step": step, "val/loss": val_loss, "total_training_time": total_training_time})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "The planets of the solar system are:",
        ]
        for prompt_text in prompts:
            tokens = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())
            prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            out = generate(orig_model, prompt_tensor, mask_id, steps=64, gen_length=64, temperature=0.)
            decoded = tokenizer.decode(out[0, len(tokens):].tolist())
            print0(f"  Prompt: {prompt_text}")
            print0(f"  Output: {decoded}")
            print0()
        model.train()

    # Save
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), {
            "step": step, "val_loss": val_loss, "model_config": model_config_kwargs,
            "user_config": user_config, "total_batch_size": total_batch_size,
            "dataloader_state_dict": dataloader_state_dict,
            "loop_state": {"min_val_loss": min_val_loss, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},
        }, rank=ddp_rank)

    if last_step:
        break

    # --- Training step ---
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # Suppress AllReduce on all but the last micro-step (DDP optimization)
        is_last_micro = (micro_step == grad_accum_steps - 1)
        ctx = model.no_sync if (ddp and not is_last_micro) else lambda: nullcontext()
        with ctx():
            loss = compute_diffusion_loss(model, input_ids, mask_id)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
        input_ids, dataloader_state_dict = next(train_loader)
    # All-reduce train loss for accurate logging across ranks
    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

    # LR schedule
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    eta_str = ""
    if steps_done > 0:
        eta_str = f" | eta: {(total_training_time / steps_done) * (num_iterations - step) / 60:.1f}m"
    epoch = f"{dataloader_state_dict['epoch']} pq:{dataloader_state_dict['pq_idx']} rg:{dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | loss: {debiased:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch}{eta_str}")
    if step % 100 == 0:
        wandb_run.log({"step": step, "train/loss": debiased, "train/lrm": lrm, "train/tok_per_sec": tok_per_sec, "train/mfu": mfu, "total_training_time": total_training_time})

    first_step = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step:
        gc.collect(); gc.freeze(); gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"Peak memory: {get_max_memory() / 1024**2:.0f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_loss is not None:
    print0(f"Min val loss: {min_val_loss:.6f}")
wandb_run.finish()
compute_cleanup()
