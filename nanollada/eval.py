"""
Evaluation utilities for masked diffusion models.

For diffusion LMs, we can't just do autoregressive log-likelihood.
Instead we use Monte Carlo estimation of the ELBO (same as training loss)
to score how well the model predicts masked tokens, following the LLaDA paper.

Supports three CORE task types:
  - multiple_choice: vary the answer, pick lowest ELBO
  - schema: vary the context, same continuation, pick lowest ELBO
  - language_modeling: check if greedy unmasking produces the exact continuation
"""

import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
from nanollada.diffusion import forward_process


# -----------------------------------------------------------------------------
# Monte Carlo log-likelihood estimation (ELBO)

@torch.no_grad()
def mc_loglikelihood(model, input_ids, prompt_len, mask_id, mc_num=32, mc_batch=8):
    """
    Estimate negative log-likelihood of the continuation (tokens after prompt_len)
    via Monte Carlo sampling of the ELBO, following LLaDA eval_llada.py.

    Returns: negative ELBO (lower = better fit, like NLL)
    """
    device = input_ids.device
    seq = input_ids.expand(mc_batch, -1).clone()
    L = seq.shape[1]
    target_len = L - prompt_len

    if target_len <= 0:
        return 0.0

    loss_acc = []
    for _ in range(mc_num // mc_batch):
        k = torch.randint(1, target_len + 1, (mc_batch,), device=device)
        indices = torch.arange(target_len, device=device).unsqueeze(0).expand(mc_batch, -1)
        is_mask = indices < k.unsqueeze(1)
        for i in range(mc_batch):
            is_mask[i] = is_mask[i][torch.randperm(target_len, device=device)]
        full_mask = torch.cat([
            torch.zeros(mc_batch, prompt_len, dtype=torch.bool, device=device),
            is_mask
        ], dim=1)
        noisy = torch.where(full_mask, mask_id, seq)
        p_mask = (k.float() / target_len).unsqueeze(1).expand(mc_batch, L)

        logits = model(noisy)
        loss = F.cross_entropy(logits[full_mask], seq[full_mask], reduction='none') / p_mask[full_mask]
        loss_acc.append(loss.sum().item() / mc_batch)

    return sum(loss_acc) / len(loss_acc)


# -----------------------------------------------------------------------------
# Greedy unmasking prediction (for language_modeling tasks)

@torch.no_grad()
def greedy_unmask_matches(model, input_ids, prompt_len, mask_id):
    """
    Check if greedy one-step unmasking of the continuation matches the target.
    This is the diffusion analogue of checking greedy autoregressive prediction.

    Masks all continuation tokens, runs one forward pass, checks if argmax
    at each masked position matches the original token.
    """
    device = input_ids.device
    seq = input_ids.clone()
    target = seq[0, prompt_len:].clone()
    seq[0, prompt_len:] = mask_id

    logits = model(seq)
    preds = logits[0, prompt_len:].argmax(dim=-1)
    return torch.all(preds == target).item()


# -----------------------------------------------------------------------------
# Diffusion validation loss

@torch.no_grad()
def evaluate_val_loss(model, val_loader, steps, mask_id, device):
    """Compute average diffusion loss on validation data."""
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
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return (total_loss / (steps * world_size)).item()


# -----------------------------------------------------------------------------
# Tokenization helpers

def _tokenize_and_split(tokenizer, full_text, prompt_text, max_seq_len):
    """
    Tokenize full_text, find the split point using prompt_text, handle
    tokenizer boundary issues and context window truncation.
    Returns (tokens, prompt_len) ready for mc_loglikelihood.
    """
    tokens = tokenizer.encode(full_text, prepend=tokenizer.get_bos_token_id())
    prompt_tokens = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())
    prompt_len = len(prompt_tokens)
    # Ensure at least 1 continuation token
    if prompt_len >= len(tokens):
        prompt_len = max(len(tokens) - 1, 1)
    # Truncate from the left (keep the end which has the answer)
    if len(tokens) > max_seq_len:
        excess = len(tokens) - max_seq_len
        tokens = tokens[excess:]
        prompt_len = max(prompt_len - excess, 1)
    return tokens, prompt_len


def _build_fewshot_prefix(data, idx, num_fewshot, task_type):
    """Build a few-shot prefix string from examples, excluding the current item."""
    if num_fewshot <= 0:
        return ""
    rng = random.Random(1234 + idx)
    available = [i for i in range(len(data)) if i != idx]
    fs_indices = rng.sample(available, min(num_fewshot, len(available)))
    prefix = ""
    for fi in fs_indices:
        ex = data[fi]
        if task_type == 'multiple_choice':
            prefix += f"{ex['query']} {ex['choices'][ex['gold']]}\n\n"
        elif task_type == 'schema':
            prefix += f"{ex['context_options'][ex['gold']]} {ex['continuation']}\n\n"
        elif task_type == 'language_modeling':
            prefix += f"{ex['context']} {ex['continuation']}\n\n"
    return prefix


# -----------------------------------------------------------------------------
# Per-example evaluation dispatchers

@torch.no_grad()
def evaluate_example(model, tokenizer, item, mask_id, device, task_type,
                     mc_num=32, mc_batch=8, num_fewshot=0,
                     fewshot_pool=None, idx=0, max_seq_len=1024,
                     continuation_delimiter=" "):
    """
    Evaluate a single example. Dispatches to the right logic based on task_type.
    Returns True if correct.
    """
    prefix = _build_fewshot_prefix(fewshot_pool or [], idx, num_fewshot, task_type)

    if task_type == 'multiple_choice':
        return _eval_mc(model, tokenizer, item, mask_id, device, prefix,
                        mc_num, mc_batch, max_seq_len)
    elif task_type == 'schema':
        return _eval_schema(model, tokenizer, item, mask_id, device, prefix,
                            mc_num, mc_batch, max_seq_len, continuation_delimiter)
    elif task_type == 'language_modeling':
        return _eval_lm(model, tokenizer, item, mask_id, device, prefix,
                        max_seq_len, continuation_delimiter)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _eval_mc(model, tokenizer, item, mask_id, device, prefix,
             mc_num, mc_batch, max_seq_len):
    """Multiple choice: vary the answer, pick lowest ELBO."""
    scores = []
    for choice in item['choices']:
        full = prefix + item['query'] + " " + choice
        prompt = prefix + item['query']
        tokens, prompt_len = _tokenize_and_split(tokenizer, full, prompt, max_seq_len)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        scores.append(mc_loglikelihood(input_ids=input_ids, prompt_len=prompt_len,
                                       model=model, mask_id=mask_id,
                                       mc_num=mc_num, mc_batch=mc_batch))
    return scores.index(min(scores)) == item['gold']


def _eval_schema(model, tokenizer, item, mask_id, device, prefix,
                 mc_num, mc_batch, max_seq_len, continuation_delimiter):
    """Schema: vary the context, same continuation, pick lowest ELBO."""
    scores = []
    continuation = item['continuation']
    for ctx in item['context_options']:
        full = prefix + ctx + continuation_delimiter + continuation
        prompt = prefix + ctx + continuation_delimiter
        tokens, prompt_len = _tokenize_and_split(tokenizer, full, prompt, max_seq_len)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        scores.append(mc_loglikelihood(input_ids=input_ids, prompt_len=prompt_len,
                                       model=model, mask_id=mask_id,
                                       mc_num=mc_num, mc_batch=mc_batch))
    return scores.index(min(scores)) == item['gold']


def _eval_lm(model, tokenizer, item, mask_id, device, prefix,
             max_seq_len, continuation_delimiter):
    """
    Language modeling: check if greedy unmasking produces the exact continuation.
    This is the diffusion analogue of autoregressive greedy-match.
    """
    full = prefix + item['context'] + continuation_delimiter + item['continuation']
    prompt = prefix + item['context'] + continuation_delimiter
    tokens, prompt_len = _tokenize_and_split(tokenizer, full, prompt, max_seq_len)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return greedy_unmask_matches(model, input_ids, prompt_len, mask_id)


# -----------------------------------------------------------------------------
# Task-level evaluation

@torch.no_grad()
def evaluate_task(model, tokenizer, data, mask_id, device, task_meta,
                  mc_num=32, mc_batch=8, max_seq_len=1024):
    """
    Evaluate a task across all examples, distributed across ranks.
    Returns mean accuracy.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    n = len(data)
    correct = torch.zeros(n, dtype=torch.float32, device=device)
    for idx in range(rank, n, world_size):
        is_correct = evaluate_example(
            model, tokenizer, data[idx], mask_id, device,
            task_type=task_meta['task_type'],
            mc_num=mc_num, mc_batch=mc_batch,
            num_fewshot=task_meta.get('num_fewshot', 0),
            fewshot_pool=data, idx=idx,
            max_seq_len=max_seq_len,
            continuation_delimiter=task_meta.get('continuation_delimiter', ' '),
        )
        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    return correct.mean().item()
