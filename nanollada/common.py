"""Common utilities for nanoLLaDA."""

import os
import logging
import torch
import torch.distributed as dist
from filelock import FileLock
import urllib.request

# Compute dtype detection
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOLLADA_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOLLADA_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, using fp32)"
    return torch.float32, "auto-detected: no CUDA"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_base_dir():
    if os.environ.get("NANOLLADA_BASE_DIR"):
        d = os.environ.get("NANOLLADA_BASE_DIR")
    else:
        d = os.path.join(os.path.expanduser("~"), ".cache", "nanollada")
    os.makedirs(d, exist_ok=True)
    return d

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

def is_ddp_requested():
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized():
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def compute_init(device_type="cuda"):
    assert device_type in ["cuda", "cpu"]
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    def log(self, *a, **kw): pass
    def finish(self): pass

def get_peak_flops(device_name):
    name = device_name.lower()
    table = (
        (["h200"], 989e12), (["h100"], 989e12), (["a100"], 312e12),
        (["l40s"], 362e12), (["l4"], 121e12), (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in table:
        if all(p in name for p in patterns):
            return flops
    return float('inf')
