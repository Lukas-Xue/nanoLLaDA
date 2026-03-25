"""
SFT diffusion loss for LLaDA.

The key difference from pretraining: only mask the response, never the prompt.
This teaches the model to generate responses conditioned on prompts.
See LLaDA GUIDELINES.md Appendix B.1.
"""

import torch
import torch.nn.functional as F


def compute_sft_loss(model, input_ids, prompt_lengths, content_lengths, mask_id, eps=1e-3):
    """
    Compute SFT diffusion loss. Same as pretraining loss but:
    1. Only actual response tokens are masked (prompt and padding stay visible)
    2. Loss is normalized by answer length (not sequence length)

    Args:
        input_ids: (B, L) token ids (prompt + response + EOS padding)
        prompt_lengths: (B,) number of prompt tokens per row
        content_lengths: (B,) total real content length per row (prompt + response + 1 EOS)
        mask_id: mask token id
    """
    b, l = input_ids.shape
    device = input_ids.device

    # Random mask ratio per sequence
    t = torch.rand(b, device=device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(b, l)

    # Build position mask: True only for actual response tokens (not prompt, not padding)
    positions = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
    is_response = (positions >= prompt_lengths.unsqueeze(1)) & (positions < content_lengths.unsqueeze(1))

    # Only mask within real response region
    random_mask = torch.rand((b, l), device=device) < p_mask
    masked_indices = random_mask & is_response

    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    if not masked_indices.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits = model(noisy_batch)

    # Answer length = real response tokens only (not padding)
    answer_lengths = is_response.sum(dim=1, keepdim=True).float().expand(b, l)

    token_loss = F.cross_entropy(
        logits[masked_indices], input_ids[masked_indices], reduction='none'
    ) / p_mask[masked_indices]

    # Normalize by answer length (LLaDA SFT guideline)
    loss = torch.sum(token_loss / answer_lengths[masked_indices]) / b
    return loss
