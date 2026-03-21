"""
Evaluate a trained nanoLLaDA model.

Supports three modes (comma-separated):
  --eval val      : Diffusion validation loss
  --eval core     : CORE metric (MC accuracy on ICL tasks)
  --eval sample   : Generate samples via iterative unmasking

Default: all three.

Examples:
    # Single GPU, all evals
    python -m scripts.eval

    # Multi-GPU, CORE only
    torchrun --nproc_per_node=4 -m scripts.eval --eval core

    # Quick test with fewer MC samples and limited examples
    python -m scripts.eval --eval core --mc-num 8 --max-per-task 50
"""

import os
import csv
import json
import time
import random
import zipfile
import argparse

import torch

from nanollada.model import DiffusionTransformer, DiffusionTransformerConfig
from nanollada.tokenizer import get_tokenizer
from nanollada.generate import generate
from nanollada.eval import evaluate_val_loss, evaluate_task
from nanollada.dataloader import distributed_data_loader
from nanollada.common import (
    compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type,
)

# Download helper (inline to avoid adding dependencies)
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def download_eval_bundle(dest_dir):
    """Download and extract the CORE eval bundle."""
    import urllib.request
    from filelock import FileLock
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "eval_bundle.zip")
    lock_path = zip_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(os.path.join(dest_dir, "eval_bundle")):
            return
        print0(f"Downloading eval bundle...")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
    print0(f"Eval bundle ready at {dest_dir}/eval_bundle")


# -----------------------------------------------------------------------------
# Model loading

def load_model_from_checkpoint(checkpoint_dir, step, device):
    """Load model + config from a checkpoint directory."""
    if step == -1:
        model_files = sorted([f for f in os.listdir(checkpoint_dir)
                              if f.startswith("model_") and f.endswith(".pt")])
        assert model_files, f"No model files in {checkpoint_dir}"
        step = int(model_files[-1].split("_")[1].split(".")[0])

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    config = DiffusionTransformerConfig(**meta["model_config"])

    model = DiffusionTransformer(config)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model, config, meta, step


def find_checkpoint_dir(base_dir):
    """Auto-detect the first checkpoint directory."""
    ckpt_root = os.path.join(base_dir, "checkpoints")
    dirs = sorted(os.listdir(ckpt_root)) if os.path.exists(ckpt_root) else []
    assert dirs, f"No checkpoints found in {ckpt_root}"
    return os.path.join(ckpt_root, dirs[0])


# -----------------------------------------------------------------------------
# CORE evaluation (adapted for diffusion)

def evaluate_core(model, tokenizer, mask_id, device, mc_num=32, mc_batch=8, max_per_task=-1):
    """
    Evaluate on the CORE benchmark using ELBO-based multiple-choice scoring.
    Same task data as nanochat, but scored via diffusion likelihood instead of
    autoregressive loss.
    """
    import yaml

    base_dir = get_base_dir()
    eval_dir = os.path.join(base_dir, "eval_data")
    download_eval_bundle(eval_dir)
    bundle_dir = os.path.join(eval_dir, "eval_bundle")

    config_path = os.path.join(bundle_dir, "core.yaml")
    data_base = os.path.join(bundle_dir, "eval_data")
    meta_path = os.path.join(bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load random baselines
    random_baselines = {}
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results = {}
    centered_results = {}

    for task in config['icl_tasks']:
        label = task['label']
        task_type = task['icl_task_type']

        task_meta = {
            'task_type': task_type,
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
        }

        data_path = os.path.join(data_base, task['dataset_uri'])
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        t0 = time.time()
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, {len(data)} examples)... ", end='')

        accuracy = evaluate_task(
            model, tokenizer, data, mask_id, device, task_meta,
            mc_num=mc_num, mc_batch=mc_batch,
            max_seq_len=model.config.sequence_len,
        )

        results[label] = accuracy
        baseline = random_baselines.get(label, 25.0)
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        centered_results[label] = centered

        elapsed = time.time() - t0
        print0(f"acc: {accuracy:.4f} | centered: {centered:.4f} | {elapsed:.1f}s")

    core_metric = sum(centered_results.values()) / max(len(centered_results), 1)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Evaluate nanoLLaDA model")
    parser.add_argument('--eval', type=str, default='val,core,sample',
                        help='Comma-separated: val,core,sample')
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--step', type=int, default=-1, help='-1 = latest')
    parser.add_argument('--device-type', type=str, default='')
    # Val loss
    parser.add_argument('--device-batch-size', type=int, default=16)
    parser.add_argument('--val-steps', type=int, default=50)
    # CORE
    parser.add_argument('--mc-num', type=int, default=32, help='MC samples for likelihood')
    parser.add_argument('--mc-batch', type=int, default=8, help='MC batch size')
    parser.add_argument('--max-per-task', type=int, default=-1, help='-1 = all')
    # Sampling
    parser.add_argument('--gen-length', type=int, default=64)
    parser.add_argument('--gen-steps', type=int, default=64)
    args = parser.parse_args()

    eval_modes = set(m.strip() for m in args.eval.split(','))

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type

    # Increase NCCL timeout — MC eval is slow, default 10min can be too short
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Set longer timeout after init
    if ddp:
        import datetime
        torch.distributed.barrier()
        # Use the PG's set_timeout if available (PyTorch 2.5+)
        try:
            pg = torch.distributed.group.WORLD
            pg._set_default_timeout(datetime.timedelta(hours=1))
        except Exception:
            pass  # older PyTorch, rely on default

    # Load model
    base_dir = get_base_dir()
    ckpt_dir = args.checkpoint_dir or find_checkpoint_dir(base_dir)
    model, config, meta, step = load_model_from_checkpoint(ckpt_dir, args.step, device)
    nparams = sum(p.numel() for p in model.parameters())
    print0(f"Loaded model: {nparams:,} params, step {step}, from {ckpt_dir}")

    tokenizer = get_tokenizer()
    mask_id = tokenizer.get_mask_token_id()

    # --- Sampling ---
    if 'sample' in eval_modes and ddp_rank == 0:
        print0("\n" + "=" * 70)
        print0("Samples (iterative unmasking)")
        print0("=" * 70)
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "The planets of the solar system are:",
        ]
        for prompt_text in prompts:
            tokens = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())
            prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = generate(model, prompt_tensor, mask_id,
                               steps=args.gen_steps, gen_length=args.gen_length, temperature=0.)
            decoded = tokenizer.decode(out[0, len(tokens):].tolist())
            print0(f"  Prompt: {prompt_text}")
            print0(f"  Output: {decoded}")
            print0()

    # --- Validation loss ---
    if 'val' in eval_modes:
        print0("\n" + "=" * 70)
        print0("Validation Loss")
        print0("=" * 70)
        val_loader = distributed_data_loader(
            tokenizer, args.device_batch_size, config.sequence_len, split="val", device=device
        )
        val_loss = evaluate_val_loss(model, val_loader, args.val_steps, mask_id, device)
        print0(f"Val diffusion loss: {val_loss:.6f}")

    # --- CORE ---
    if 'core' in eval_modes:
        print0("\n" + "=" * 70)
        print0("CORE Evaluation (diffusion ELBO scoring)")
        print0("=" * 70)
        core_results = evaluate_core(
            model, tokenizer, mask_id, device,
            mc_num=args.mc_num, mc_batch=args.mc_batch,
            max_per_task=args.max_per_task,
        )

        if ddp_rank == 0:
            output_dir = os.path.join(base_dir, "eval_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"core_step{step:06d}.csv")
            with open(output_path, 'w', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    cen = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {cen:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nCORE metric: {core_results['core_metric']:.4f}")
            print0(f"Results saved to: {output_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
