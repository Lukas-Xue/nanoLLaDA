"""
Iterative unmasking generation for nanoLLaDA.
Implements the reverse diffusion process: start from all [MASK], iteratively unmask.
Supports both fully parallel and semi-autoregressive (block-by-block) generation.
"""

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling for categorical distributions."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Precompute how many tokens to unmask at each step (uniform schedule)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    # Clamp steps to mask count — no point in more steps than masked tokens
    effective_steps = torch.clamp(torch.tensor(steps, device=mask_index.device), max=mask_num.min().item())
    effective_steps = max(effective_steps.item(), 1)
    base = mask_num // effective_steps
    remainder = mask_num % effective_steps
    num_transfer = torch.zeros(mask_num.size(0), effective_steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def generate(model, prompt, mask_id, steps=128, gen_length=128, block_length=None,
             temperature=0., cfg_scale=0., remasking='low_confidence'):
    """
    Generate text via iterative unmasking.

    Args:
        model: DiffusionTransformer
        prompt: tensor of shape (B, L) — the prompt token ids
        mask_id: the [MASK] token id
        steps: total number of unmasking steps across all blocks
        gen_length: number of tokens to generate
        block_length: if set, generate in blocks of this size (semi-autoregressive).
                      Must evenly divide gen_length. None = fully parallel (one block).
        temperature: Gumbel noise temperature (0 = greedy)
        cfg_scale: classifier-free guidance scale. Uses the formulation from LLaDA:
                   logits = unconditional + (1 + scale) * (conditional - unconditional)
                   so scale=0 means no guidance (reduces to conditional logits).
        remasking: 'low_confidence' or 'random'
    Returns:
        tensor of shape (B, L + gen_length) — full sequence
    """
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    steps_per_block = steps // num_blocks

    device = model.get_device()
    B = prompt.shape[0]
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = prompt.shape[1] + (block_idx + 1) * block_length

        # Compute transfer schedule for this block's masked tokens
        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)
        actual_steps = num_transfer_tokens.shape[1]

        for i in range(actual_steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            # Don't unmask tokens outside the current block (or in the prompt)
            x0_p[:, :block_start] = -float('inf')
            x0_p[:, block_end:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def generate_visual(model, prompt, mask_id, steps=128, gen_length=128, block_length=None,
                    temperature=0., cfg_scale=0., remasking='low_confidence'):
    """
    Same as generate(), but yields (step, total_steps, x) at each unmasking step
    for visualization. Use with torch.no_grad() externally.

    Yields:
        (step, total_steps, x): step index (0 = initial all-masked state),
        total number of steps, and current sequence tensor (B, L+gen_length).
    """
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    device = model.get_device()
    B = prompt.shape[0]
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    # Pre-compute total actual steps across all blocks
    total_steps = 0
    for bi in range(num_blocks):
        bs = prompt.shape[1] + bi * block_length
        be = prompt.shape[1] + (bi + 1) * block_length
        bm = (x[:, bs:be] == mask_id)
        total_steps += get_num_transfer_tokens(bm, steps_per_block).shape[1]

    global_step = 0
    yield global_step, total_steps, x.clone()

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = prompt.shape[1] + (block_idx + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)
        actual_steps = num_transfer_tokens.shape[1]

        for i in range(actual_steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            x0_p[:, :block_start] = -float('inf')
            x0_p[:, block_end:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            global_step += 1
            yield global_step, total_steps, x.clone()
