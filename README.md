# nanoLLaDA

The minimal masked diffusion language model. Train your own [LLaDA](https://arxiv.org/abs/2502.09992) from scratch with just a few files.

Inspired by Karpathy's [nano series](https://github.com/karpathy/nanochat) — making language models easy and affordable.

## What is This?

Most language models (GPT, LLaMA, etc.) generate text **left to right**, one token at a time. LLaDA does something fundamentally different: it starts with a sequence of `[MASK]` tokens and **iteratively reveals them**, like solving a crossword puzzle — filling in the tokens it's most confident about first, then using that context to figure out the rest.

This repo is a minimal, from-scratch implementation of that idea. ~500 lines of core code, trains on 4 GPUs.

## Quick Start

**One command** (requires GPUs + [uv](https://github.com/astral-sh/uv)):

```bash
bash run.sh
```

This downloads data, trains a tokenizer, and pretrains a 135M parameter diffusion model.

### Step by Step

```bash
# Setup
uv venv && uv sync --extra gpu && source .venv/bin/activate

# Download data (8 shards for tokenizer, 80 for training)
python -m nanollada.dataset -n 80

# Train tokenizer (32K vocab, ~1 min)
python -m scripts.tok_train

# Pretrain on 4 GPUs
torchrun --standalone --nproc_per_node=4 -m scripts.train

# Pretrain on 1 GPU (smaller batch)
python -m scripts.train --device-batch-size=8 --total-batch-size=32768 --num-iterations=1000

# Generate text from trained model
python -m scripts.inference --prompt "The capital of France is"
```

## How It Works

### Training: Mask and Predict

Every training step:

1. Take a sequence of tokens: `[BOS] The capital of France is Paris`
2. Pick a random mask ratio `t` (e.g. 0.6 means mask 60%)
3. Randomly mask tokens: `[BOS] The [MASK] of [MASK] [MASK] Paris`
4. Model predicts what's behind each `[MASK]`
5. Loss = cross entropy on masked positions, divided by mask ratio `t`

The division by `t` is the key mathematical insight — it makes the loss an upper bound on the negative log-likelihood (ELBO), which means this is a proper generative model, not just BERT.

```python
# The entire training algorithm in ~10 lines:
t = torch.rand(batch_size)                              # random mask ratio per sequence
p_mask = (1 - eps) * t + eps                             # scale to [0.001, 1.0]
masked = torch.rand(b, l) < p_mask[:, None]              # mask each token independently
masked[:, 0] = False                                      # never mask BOS
noisy = torch.where(masked, MASK_ID, input_ids)           # apply masks
logits = model(noisy)                                     # predict
loss = CE(logits[masked], input_ids[masked]) / p_mask[masked]  # weighted loss
loss = loss.sum() / (batch_size * seq_len)                # normalize
```

### Generation: Iterative Unmasking

1. Start with: `[BOS] The capital of France is [MASK] [MASK] [MASK] ...`
2. Run the model — it predicts a candidate for every `[MASK]`
3. Score each candidate by the model's confidence (softmax probability)
4. Unmask the most confident ones
5. Repeat — with more context revealed, remaining predictions improve
6. After N steps, all masks are filled in

```bash
# Generate with default settings (greedy, fully parallel)
python -m scripts.inference --prompt "The meaning of life is"

# More creative (temperature adds randomness)
python -m scripts.inference --prompt "Once upon a time" --temperature 0.5

# Semi-autoregressive (generate in 32-token blocks, left to right)
python -m scripts.inference --prompt "Explain gravity" --block-length 32 --gen-length 128 --steps 128

# Classifier-free guidance (amplifies prompt relevance)
python -m scripts.inference --prompt "The chemical formula of water is" --cfg-scale 1.0
```

## How is This Different from GPT?

This is the core question. Here's a precise comparison:

| | GPT (Autoregressive) | LLaDA (Masked Diffusion) |
|---|---|---|
| **Attention** | Causal — token `i` only sees tokens `0..i` | Bidirectional — every token sees every other token |
| **Training** | Predict the next token | Predict all masked tokens simultaneously |
| **Loss** | `CE(logits[i], token[i+1])` | `CE(logits[masked], token[masked]) / mask_ratio` |
| **Generation** | One token at a time, left to right | All at once, iterative refinement over N steps |
| **KV Cache** | Yes — reuse past computations | No — full recomputation each step |
| **Speed** | Fast (one forward pass per token) | Slower (N forward passes for all tokens) |

### The One Code Change

The architectural difference is literally one boolean. In GPT (nanoChat):
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # left-to-right only
```

In LLaDA (nanoLLaDA):
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # full bidirectional
```

Everything else — RoPE, RMSNorm, ReLU² MLP, the transformer architecture — is identical.

### Why Bidirectional?

In GPT, when predicting token 5, the model can only see tokens 0–4. In LLaDA, when predicting a masked token, the model sees *all* unmasked tokens — including ones that come after it. This is strictly more information, which is why masked diffusion can match autoregressive models with the same parameter count.

The tradeoff is generation speed: GPT generates one token per forward pass (with KV cache), while LLaDA needs multiple forward passes to iteratively unmask. Think of it like the difference between writing a sentence word by word vs. filling in a cloze test — the cloze approach uses more context but requires multiple passes.

## File Structure

```
nanoLLaDA/
├── nanollada/
│   ├── model.py           # The bidirectional transformer (~170 lines)
│   ├── generate.py        # Iterative unmasking generation (~110 lines)
│   ├── dataloader.py      # Distributed data loading with best-fit packing
│   ├── dataset.py         # Dataset download (ClimbMix-400B from HuggingFace)
│   ├── tokenizer.py       # BPE tokenizer (rustbpe + tiktoken)
│   ├── checkpoint.py      # Save/load checkpoints
│   └── common.py          # Shared utilities (DDP, device detection)
├── scripts/
│   ├── train.py           # Pretraining script (~340 lines)
│   ├── tok_train.py       # Tokenizer training
│   └── inference.py       # Text generation from trained model
├── run.sh                 # One-command training
├── pyproject.toml         # Dependencies
└── README.md
```

### What Each File Does

**`model.py`** — The transformer. Same architecture as GPT (RoPE, QK-norm, ReLU² MLP) but with `is_causal=False` for bidirectional attention. Takes token IDs in (including `[MASK]` tokens), returns logits over the vocabulary for every position.

**`generate.py`** — The reverse diffusion process. Starts from all `[MASK]`, iteratively unmasks by picking the most confident predictions. Supports fully parallel and semi-autoregressive (block-by-block) generation, Gumbel noise sampling, and classifier-free guidance.

**`train.py`** — The training loop. Loads data, masks tokens with random ratios, computes the weighted cross-entropy loss, updates the model with AdamW. Handles multi-GPU (DDP), gradient accumulation, LR scheduling, checkpointing, and periodic evaluation/sampling.

**`dataloader.py`** — Reads parquet files, tokenizes text on the fly, packs multiple documents into fixed-length sequences using best-fit bin packing. Each GPU rank reads different data shards.

**`tokenizer.py`** — BPE tokenizer with two special tokens: `<|bos|>` (beginning of sequence) and `<|mask|>` (the mask token for diffusion). Trained with rustbpe, inference with tiktoken.

**`inference.py`** — Load a trained checkpoint and generate text. Auto-detects the latest checkpoint.

## Configuration

Default config trains a ~135M parameter model:

| Parameter | Default | Description |
|---|---|---|
| `--depth` | 12 | Transformer layers |
| `--aspect-ratio` | 64 | Width = depth × aspect_ratio |
| `--head-dim` | 64 | Attention head dimension |
| `--max-seq-len` | 1024 | Context length |
| `--total-batch-size` | 131072 | Tokens per optimizer step |
| `--lr` | 3e-4 | AdamW learning rate |
| `--warmup-ratio` | 0.05 | LR warmup (fraction of training) |
| `--warmdown-ratio` | 0.3 | LR decay (fraction of training) |
| `--target-param-data-ratio` | 10.5 | Tokens per parameter (Chinchilla-style) |

### Scaling Up

```bash
# ~500M params on 8x H100
torchrun --standalone --nproc_per_node=8 -m scripts.train \
    --depth=24 --max-seq-len=2048 --device-batch-size=32 --total-batch-size=524288
```

The model width scales automatically: `width = depth × 64`, so depth=24 gives width=1536.

## Key Concepts for Learners

### Why Divide Loss by Mask Ratio?

When 90% of tokens are masked, the model has many predictions to make but little context — each prediction is hard. When 10% are masked, there are few predictions but lots of context — each is easy. Without the `1/t` weighting, the loss would be dominated by high mask ratios (many hard predictions). The weighting balances all mask ratios equally, and mathematically makes the loss an ELBO — a valid bound on the data likelihood.

### What is the Forward Process?

Borrowed from diffusion model terminology. The "forward process" adds noise (masks tokens). The "reverse process" removes noise (unmasks tokens). During training, we simulate the forward process and train the model to reverse it.

### What is Classifier-Free Guidance?

A trick to make generation more relevant to the prompt. Run the model twice: once with the prompt visible (conditional), once with the prompt masked (unconditional). The difference tells you "what the prompt contributes." Amplify that difference to get stronger prompt-following. Free at inference time — no special training needed.

### Why Semi-Autoregressive?

Fully parallel generation (unmask everything at once) can struggle with long sequences because distant tokens have weak dependencies. Semi-autoregressive generation splits the output into blocks (e.g. 32 tokens each) and generates them left to right. Each block is generated in parallel, but later blocks can see earlier blocks. This gives a middle ground between fully parallel (fast, less coherent) and fully autoregressive (slow, most coherent).

## Dataset

Uses [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), the same dataset as nanoChat. High-quality English web text in parquet format, downloaded on demand.

## References

- [LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) — the original paper
- [nanoChat](https://github.com/karpathy/nanochat) — the nano series autoregressive baseline this was adapted from
- [SMDM](https://github.com/ML-GSAI/SMDM) — scaling laws for masked diffusion models
