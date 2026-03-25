"""
Supervised fine-tuning (SFT) for nanoLLaDA.

Finetunes a pretrained masked diffusion model on conversation data.
The key difference from pretraining: only response tokens are masked.

Usage:
    torchrun --nproc_per_node=4 -m scripts.sft
    torchrun --nproc_per_node=4 -m scripts.sft --checkpoint-dir ~/.cache/nanollada/checkpoints/d20-full --step 34000
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse
from contextlib import nullcontext

import wandb
import torch
import torch.distributed as dist

from nanollada.model import DiffusionTransformer, DiffusionTransformerConfig
from nanollada.tokenizer import get_tokenizer
from nanollada.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    autodetect_device_type, get_peak_flops, get_base_dir,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized,
)
from nanollada.checkpoint import save_checkpoint, load_checkpoint
from nanollada.generate import generate
from nanollada.sft import compute_sft_loss

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SFT for nanoLLaDA")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
# Model
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--step", type=int, default=-1)
# Training
parser.add_argument("--num-iterations", type=int, default=2000)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--total-batch-size", type=int, default=32768)
parser.add_argument("--lr", type=float, default=3e-5, help="SFT learning rate (lower than pretraining)")
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.3)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Eval
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--sample-every", type=int, default=200)
parser.add_argument("--save-every", type=int, default=500)
# Data
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--max-rows", type=int, default=-1, help="limit dataset rows (-1 = all)")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Setup
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

if device_type == "cuda":
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanollada-sft", name=args.run, config=user_config)

# Load pretrained model
base_dir = get_base_dir()
if args.checkpoint_dir is None:
    ckpt_root = os.path.join(base_dir, "checkpoints")
    dirs = sorted(os.listdir(ckpt_root))
    assert dirs, f"No checkpoints in {ckpt_root}"
    args.checkpoint_dir = os.path.join(ckpt_root, dirs[0])

if args.step == -1:
    model_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.startswith("model_") and f.endswith(".pt")])
    args.step = int(model_files[-1].split("_")[1].split(".")[0])

model_data, _, meta_data = load_checkpoint(args.checkpoint_dir, args.step, device, load_optimizer=False)
model_config_kwargs = meta_data["model_config"]
config = DiffusionTransformerConfig(**model_config_kwargs)
# Override max_seq_len if specified
if args.max_seq_len != config.sequence_len:
    print0(f"Overriding sequence_len: {config.sequence_len} -> {args.max_seq_len}")
    config.sequence_len = args.max_seq_len
    model_config_kwargs["sequence_len"] = args.max_seq_len

model = DiffusionTransformer(config)
model.load_state_dict(model_data)
del model_data
model.to(device)
model.train()
nparams = sum(p.numel() for p in model.parameters())
print0(f"Loaded pretrained model: {nparams:,} params from {args.checkpoint_dir} step {args.step}")

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
orig_model = model.module if ddp else model

tokenizer = get_tokenizer()
mask_id = tokenizer.get_mask_token_id()
bos_id = tokenizer.get_bos_token_id()
eos_id = tokenizer.get_eos_token_id()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

# -----------------------------------------------------------------------------
# SFT Data: load SmolTalk conversations, tokenize into (input_ids, prompt_lengths)

def render_conversation(messages):
    """
    Render a conversation to token ids and compute prompt length.
    Format: <|bos|> User: {msg}\nAssistant: {response}<|eos|>
    Only the last assistant response is the "answer" for SFT.
    For multi-turn, everything before the last assistant turn is "prompt".
    """
    text = ""
    last_assistant_start = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"System: {content}\n"
        elif role == "user":
            text += f"User: {content}\n"
            last_assistant_start = len(text)
        elif role == "assistant":
            last_assistant_start = len(text)
            text += f"Assistant: {content}\n"

    # Tokenize full conversation
    tokens = tokenizer.encode(text, prepend=bos_id, append=eos_id)
    # Tokenize prompt (everything before last assistant response)
    prompt_text = text[:last_assistant_start]
    prompt_tokens = tokenizer.encode(prompt_text, prepend=bos_id)
    prompt_len = len(prompt_tokens)

    # Clamp: prompt must be < total length
    if prompt_len >= len(tokens):
        prompt_len = max(len(tokens) - 1, 1)

    return tokens, prompt_len


def load_sft_mix():
    """
    Load SFT data mix following nanochat's recipe:
    - SmolTalk (general conversations)
    - MMLU x3 (teaches multiple choice answering)
    - GSM8K x4 (teaches math reasoning)
    """
    from datasets import load_dataset
    import re as _re
    import random as pyrandom

    MC_LETTERS = ('A', 'B', 'C', 'D')
    data = []

    def add_conversation(messages):
        tokens, prompt_len = render_conversation(messages)
        if len(tokens) <= args.max_seq_len and prompt_len > 0:
            data.append((tokens, prompt_len))

    # 1. SmolTalk — general conversations
    print0("Loading SmolTalk...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    if args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    for row in ds:
        if len(row["messages"]) >= 2:
            add_conversation(row["messages"])
    print0(f"  SmolTalk: {len(data)} conversations")

    # 2. MMLU x3 — multiple choice Q&A
    print0("Loading MMLU (auxiliary_train, 3 epochs)...")
    mmlu_ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    mmlu_ds = mmlu_ds.map(lambda row: row['train'], remove_columns=['train'])
    mmlu_start = len(data)
    for epoch in range(3):
        for row in mmlu_ds:
            q, choices, answer_idx = row["question"], row["choices"], row["answer"]
            query = f"Multiple Choice question: {q}\n"
            query += "".join([f"- {c}={l}\n" for l, c in zip(MC_LETTERS, choices)])
            query += "\nRespond only with the letter of the correct answer."
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": MC_LETTERS[answer_idx]},
            ]
            add_conversation(messages)
    print0(f"  MMLU: {len(data) - mmlu_start} conversations (3 epochs)")

    # 3. GSM8K x4 — math reasoning (simplified, no tool calls)
    print0("Loading GSM8K (4 epochs)...")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    gsm_start = len(data)
    for epoch in range(4):
        for row in gsm_ds:
            answer = _re.sub(r'<<[^>]+>>', '', row["answer"])
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": answer},
            ]
            add_conversation(messages)
    print0(f"  GSM8K: {len(data) - gsm_start} conversations (4 epochs)")

    # Shuffle deterministically
    pyrandom.Random(42).shuffle(data)
    print0(f"Total SFT mix: {len(data)} conversations")
    return data


train_data = load_sft_mix()
assert len(train_data) > 0, "No training data!"

# SFT dataloader: pack conversations into batches with padding
import random

def sft_data_loader(data, B, T, device, seed=42):
    """
    Yields (input_ids [B, T], prompt_lengths [B]) batches.
    Pads shorter sequences with BOS tokens (which are in the prompt region, so never masked).
    """
    rng = random.Random(seed + ddp_rank)
    indices = list(range(len(data)))
    epoch = 0
    while True:
        rng.shuffle(indices)
        epoch += 1
        for start in range(ddp_rank * B, len(indices), ddp_world_size * B):
            batch_indices = indices[start:start + B]
            if len(batch_indices) < B:
                break
            input_ids = torch.full((B, T), eos_id, dtype=torch.long)
            prompt_lengths = torch.zeros(B, dtype=torch.long)
            content_lengths = torch.zeros(B, dtype=torch.long)
            for i, idx in enumerate(batch_indices):
                tokens, plen = data[idx]
                seq_len = min(len(tokens), T)
                input_ids[i, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
                prompt_lengths[i] = plen
                content_lengths[i] = seq_len
            yield input_ids.to(device), prompt_lengths.to(device), content_lengths.to(device), epoch


train_loader = sft_data_loader(train_data, args.device_batch_size, args.max_seq_len, device)

# Grad accum
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Batch: {args.device_batch_size}x{args.max_seq_len} x {ddp_world_size} GPUs x {grad_accum_steps} accum = {args.total_batch_size:,} tokens")

num_flops_per_token = orig_model.estimate_flops()

# LR schedule
def get_lr_multiplier(it):
    warmup_iters = int(args.warmup_ratio * args.num_iterations)
    warmdown_iters = int(args.warmdown_ratio * args.num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it >= args.num_iterations - warmdown_iters:
        progress = (args.num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * args.final_lr_frac
    return 1.0

# Eval: compute SFT val loss
@torch.no_grad()
def evaluate_sft_loss(model, data, steps, device):
    """Quick SFT loss on a subset of data."""
    total_loss = torch.tensor(0.0, device=device)
    loader = sft_data_loader(data, args.device_batch_size, args.max_seq_len, device, seed=9999)
    for i in range(steps):
        input_ids, prompt_lengths, content_lengths, _ = next(loader)
        loss = compute_sft_loss(orig_model, input_ids, prompt_lengths, content_lengths, mask_id)
        total_loss += loss.detach()
    world_size = dist.get_world_size() if is_ddp_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return (total_loss / (steps * world_size)).item()


# -----------------------------------------------------------------------------
# SFT checkpoint directory
depth = config.n_layer
sft_checkpoint_dir = os.path.join(base_dir, "sft_checkpoints", f"d{depth}")

# Training loop
step = 0
smooth_train_loss = 0
total_training_time = 0
input_ids, prompt_lengths, content_lengths, epoch = next(train_loader)

print0(f"\nStarting SFT for {args.num_iterations} steps")
print0(f"LR: {args.lr}, warmup: {args.warmup_ratio}, warmdown: {args.warmdown_ratio}")

while True:
    last_step = step == args.num_iterations

    # Eval
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loss = evaluate_sft_loss(orig_model, train_data[:1000], 20, device)
        print0(f"Step {step:05d} | SFT val loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val/sft_loss": val_loss, "total_training_time": total_training_time})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "User: What is the capital of France?\nAssistant:",
            "User: Explain gravity in simple terms.\nAssistant:",
            "User: Write a haiku about the ocean.\nAssistant:",
        ]
        for prompt_text in prompts:
            tokens = tokenizer.encode(prompt_text, prepend=bos_id)
            prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            out = generate(orig_model, prompt_tensor, mask_id, steps=64, gen_length=128, temperature=0.)
            decoded = tokenizer.decode(out[0, len(tokens):].tolist())
            print0(f"  {prompt_text} {decoded}")
        print0()
        model.train()

    # Save
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(sft_checkpoint_dir, step, orig_model.state_dict(), None, {
            "step": step, "val_loss": val_loss if step > 0 else None,
            "model_config": model_config_kwargs, "user_config": user_config,
            "pretrained_from": {"dir": args.checkpoint_dir, "step": args.step},
        }, rank=ddp_rank)

    if last_step:
        break

    # --- Training step ---
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        is_last_micro = (micro_step == grad_accum_steps - 1)
        ctx = model.no_sync if (ddp and not is_last_micro) else lambda: nullcontext()
        with ctx():
            loss = compute_sft_loss(model if not ddp else model.module, input_ids, prompt_lengths, content_lengths, mask_id)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
        input_ids, prompt_lengths, content_lengths, epoch = next(train_loader)

    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

    # LR schedule
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = args.lr * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct = 100 * step / args.num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    if step > 5:
        total_training_time += dt
    print0(f"step {step:05d}/{args.num_iterations} ({pct:.1f}%) | loss: {debiased:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | epoch: {epoch}")
    if step % 50 == 0:
        wandb_run.log({"step": step, "train/loss": debiased, "train/lrm": lrm, "train/tok_per_sec": tok_per_sec, "total_training_time": total_training_time})

    step += 1
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

print0(f"Peak memory: {get_max_memory() / 1024**2:.0f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
wandb_run.finish()
compute_cleanup()
