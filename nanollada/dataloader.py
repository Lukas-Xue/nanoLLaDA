"""Distributed dataloader for nanoLLaDA pretraining. Adapted from nanoChat."""

import torch
import pyarrow.parquet as pq
from nanollada.common import get_dist_info
from nanollada.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found. Run: python -m nanollada.dataset -n 8"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                base_idx = resume_rg_idx // ddp_world_size + 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def distributed_data_loader(tokenizer, B, T, split, device="cuda", resume_state_dict=None, buffer_size=1000):
    """
    Dataloader for masked diffusion pretraining.
    Unlike autoregressive training, we don't need (x, y) pairs with offset.
    We just need packed sequences of token ids — masking happens in the training loop.
    Yields: (input_ids [B, T], state_dict)
    """
    row_capacity = T
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size=128)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(B * T, dtype=torch.long, device=device)
    cpu_ids = cpu_buffer.view(B, T)
    input_ids = gpu_buffer.view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                # Best-fit: find largest doc that fits
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx, best_len = i, len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos+len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # Crop shortest doc to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos+remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_ids.copy_(row_buffer)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield input_ids, state_dict
