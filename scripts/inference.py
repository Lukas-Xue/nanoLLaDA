"""
Generate text with a trained nanoLLaDA model.

Usage:
    python -m scripts.inference --prompt "The capital of France is"
    python -m scripts.inference --prompt "Once upon a time" --gen-length 128 --steps 128 --temperature 0.5
    python -m scripts.inference --prompt "Explain gravity" --cfg-scale 1.0 --block-length 32
"""

import os
import json
import argparse
import torch

from nanollada.model import DiffusionTransformer, DiffusionTransformerConfig
from nanollada.tokenizer import get_tokenizer
from nanollada.generate import generate
from nanollada.common import get_base_dir

parser = argparse.ArgumentParser(description="Generate text with nanoLLaDA")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint directory (default: auto-detect latest)")
parser.add_argument("--step", type=int, default=-1, help="checkpoint step to load (-1 = latest)")
parser.add_argument("--gen-length", type=int, default=64)
parser.add_argument("--steps", type=int, default=64, help="number of unmasking steps")
parser.add_argument("--block-length", type=int, default=None, help="semi-autoregressive block size (default: fully parallel)")
parser.add_argument("--temperature", type=float, default=0., help="sampling temperature (0 = greedy)")
parser.add_argument("--cfg-scale", type=float, default=0., help="classifier-free guidance scale (0 = disabled)")
parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# Find checkpoint
base_dir = get_base_dir()
if args.checkpoint_dir is None:
    # Auto-detect: find the first checkpoint directory
    ckpt_root = os.path.join(base_dir, "checkpoints")
    dirs = sorted(os.listdir(ckpt_root)) if os.path.exists(ckpt_root) else []
    assert dirs, f"No checkpoints found in {ckpt_root}"
    args.checkpoint_dir = os.path.join(ckpt_root, dirs[0])
    print(f"Using checkpoint dir: {args.checkpoint_dir}")

# Find step
if args.step == -1:
    model_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.startswith("model_") and f.endswith(".pt")])
    assert model_files, f"No model files in {args.checkpoint_dir}"
    args.step = int(model_files[-1].split("_")[1].split(".")[0])
    print(f"Using step: {args.step}")

# Load config from metadata
meta_path = os.path.join(args.checkpoint_dir, f"meta_{args.step:06d}.json")
with open(meta_path) as f:
    meta = json.load(f)
model_config = DiffusionTransformerConfig(**meta["model_config"])

# Load model
model = DiffusionTransformer(model_config)
model_path = os.path.join(args.checkpoint_dir, f"model_{args.step:06d}.pt")
model.load_state_dict(torch.load(model_path, map_location=args.device))
model.to(args.device).eval()
print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

# Load tokenizer
tokenizer = get_tokenizer()
mask_id = tokenizer.get_mask_token_id()

# Generate
tokens = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())
prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=args.device)

with torch.no_grad():
    out = generate(
        model, prompt_tensor, mask_id,
        steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
        temperature=args.temperature, cfg_scale=args.cfg_scale, remasking=args.remasking,
    )

generated = tokenizer.decode(out[0, len(tokens):].tolist())
print(f"\nPrompt: {args.prompt}")
print(f"Output: {generated}")
