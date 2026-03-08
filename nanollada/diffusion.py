"""Masked diffusion forward process and loss computation for LLaDA."""

import torch
import torch.nn.functional as F


def forward_process(input_ids, mask_id, eps=1e-3):
    """
    Randomly mask tokens in a batch. Each sequence gets a random mask ratio.
    Position 0 (BOS) is never masked.

    Returns: (noisy_batch, masked_indices, p_mask)
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(b, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    masked_indices[:, 0] = False
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    return noisy_batch, masked_indices, p_mask


def compute_diffusion_loss(model, input_ids, mask_id):
    """
    Compute the masked diffusion training loss (ELBO).
    1% of the time, randomly truncates the sequence (LLaDA paper guideline).
    """
    if torch.rand(1).item() < 0.01:
        random_length = torch.randint(1, input_ids.shape[1] + 1, (1,)).item()
        input_ids = input_ids[:, :random_length]

    noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_id)
    logits = model(noisy_batch)

    token_loss = F.cross_entropy(
        logits[masked_indices], input_ids[masked_indices], reduction='none'
    ) / p_mask[masked_indices]
    loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    return loss
