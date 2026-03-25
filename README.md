# <img src="assets/logo.png" width="500" alt="nanoLLaDA">

A minimal implementation of [LLaDA](https://arxiv.org/abs/2502.09992) — the masked diffusion language model — built for learning and experimentation.

Part of the **nano** series, inspired by Karpathy's [nanoChat](https://github.com/karpathy/nanochat). ~500 lines of core code. Trains on 4 GPUs.

![Diffusion Demo](assets/diffusion_demo.gif)

> **🚧 Early stage.** This repo covers **pretraining, generation, and evaluation**. SFT and VRPO from the LLaDA paper are not yet implemented. Contributions welcome — we're all learning together!

**New to diffusion language models?** Check out [`tutorial.ipynb`](tutorial.ipynb) for a complete walkthrough — it downloads data, trains a tokenizer, trains a small model, and generates text, all from scratch.

## The Big Idea

GPT generates text left-to-right, one token at a time. LLaDA starts with all `[MASK]` tokens and **iteratively reveals them** — filling in the most confident predictions first, then refining with more context. It's a diffusion model, but for text.

The entire architectural difference from GPT is **one line**:

```python
# GPT:    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # left-to-right
# LLaDA:  y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidirectional
```

The training loss is also different — mask random tokens, predict them, weight by `1/mask_ratio`:

```python
loss = CE(logits[masked], targets[masked]) / mask_ratio  # this makes it an ELBO
```

That's it. Everything else (RoPE, RMSNorm, ReLU² MLP) is identical to a standard transformer.

## Quick Start

```bash
bash run.sh  # downloads data, trains tokenizer, pretrains model
```

Or step by step:

```bash
uv venv && uv sync --extra gpu && source .venv/bin/activate
python -m nanollada.dataset -n 80          # download data
python -m scripts.tok_train                 # train tokenizer
torchrun --nproc_per_node=4 -m scripts.train  # pretrain
python -m scripts.inference --prompt "The capital of France is"  # generate
```

## Hardware Guide (4× NVIDIA L4, 23GB)

| Depth | Params | Batch/GPU | Memory | Throughput | Time (compute-optimal) |
|---|---|---|---|---|---|
| 4 | 20M | 32 | 9.8 GB | 127K tok/s | ~2 min |
| 12 | 135M | 16 | 11.1 GB | 143K tok/s | ~2.3 hours |
| 20 | 477M | 8 | 15.6 GB | 43K tok/s | ~4 days |
| 24 | 780M | 4 | 16.5 GB | 8K tok/s | very slow |

d12 for fast experiments, d20 for serious training.

## How It Works

**Training:** Randomly mask tokens (ratio `t ~ Uniform(0,1)`), predict them, loss = `CE / t`. The `1/t` weighting makes this an ELBO on the data likelihood — a proper generative model, not just BERT.

**Generation:** Start fully masked → run model → unmask the most confident predictions → repeat for N steps. Supports temperature sampling, semi-autoregressive blocks (`--block-length`), and classifier-free guidance (`--cfg-scale`).

## File Structure

```
nanollada/
  model.py        # Bidirectional transformer (is_causal=False)
  generate.py     # Iterative unmasking generation
  diffusion.py    # Forward process and training loss
  sft.py          # SFT diffusion loss (only masks response tokens)
  eval.py         # MC log-likelihood, CORE benchmark evaluation
  dataloader.py   # Distributed data loading
  dataset.py      # Dataset download (ClimbMix-400B)
  tokenizer.py    # BPE tokenizer with <|bos|>, <|eos|>, <|mask|>
  checkpoint.py   # Save/load with auto-cleanup
  common.py       # Shared utilities (DDP, device detection)
scripts/
  train.py        # Pretraining (DDP, grad accum, checkpointing)
  sft.py          # Supervised fine-tuning (SmolTalk + MMLU + GSM8K)
  eval.py         # Evaluate: val loss, CORE benchmark, samples
  inference.py    # Generate text from a checkpoint
  tok_train.py    # Train the tokenizer
tutorial.ipynb    # Interactive end-to-end walkthrough
```

## Evaluation

```bash
# All evals: validation loss + CORE benchmark + samples
torchrun --nproc_per_node=4 -m scripts.eval

# Quick test (fewer MC samples, limited examples)
python -m scripts.eval --eval core --mc-num 8 --max-per-task 50

# Just validation loss
python -m scripts.eval --eval val

# Just samples
python -m scripts.eval --eval sample --gen-length 128 --gen-steps 128
```

The CORE benchmark (from the [DCLM paper](https://arxiv.org/abs/2406.11794)) evaluates in-context learning across 22 tasks using three methods:
- **Multiple choice** (11 tasks): score each answer option via ELBO, pick lowest loss
- **Schema** (2 tasks): score each context option with a shared continuation via ELBO
- **Language modeling** (9 tasks): check if greedy one-shot unmasking produces the exact continuation

`--mc-num` controls accuracy vs speed: 32 is a good default, 128 matches the LLaDA paper, 8 is fine for quick sanity checks.

### CORE Results — d20-v2 Base (477M params, step 66400)

Trained on ClimbMix-400B for ~55 hours on 4× L4 GPUs (~9.2B tokens, 20× Chinchilla ratio). Val diffusion loss: 3.02. Evaluated with `--mc-num 32 --max-per-task 500`.

| Task | Type | Accuracy | Centered |
|---|---|---|---|
| bigbench_cs_algorithms | lm | 76.4% | +0.764 |
| lambada_openai | lm | 59.6% | +0.596 |
| arc_easy | mc | 55.6% | +0.408 |
| bigbench_qa_wikidata | lm | 31.0% | +0.310 |
| piqa | mc | 58.0% | +0.160 |
| bigbench_dyck_languages | lm | 17.8% | +0.178 |
| bigbench_language_identification | mc | 25.4% | +0.179 |
| copa | mc | 58.0% | +0.160 |
| commonsense_qa | mc | 32.2% | +0.153 |
| bigbench_operators | lm | 10.0% | +0.100 |
| squad | lm | 7.0% | +0.070 |
| hellaswag (0-shot) | mc | 30.2% | +0.069 |
| hellaswag (10-shot) | mc | 30.2% | +0.069 |
| winograd | schema | 53.1% | +0.062 |
| agi_eval_lsat_ar | mc | 24.8% | +0.060 |
| bigbench_repeat_copy_logic | lm | 0.0% | 0.000 |
| coqa | lm | 0.0% | 0.000 |
| winogrande | schema | 48.6% | −0.028 |
| arc_challenge | mc | 22.2% | −0.037 |
| openbook_qa | mc | 17.4% | −0.101 |
| boolq | mc | 47.2% | −0.390 |
| **CORE** | | | **0.131** |

### SFT Results — d20-v2 SFT (477M params)

Fine-tuned on SmolTalk + MMLU×3 + GSM8K×4 (~359K conversations) for 4000 steps. The SFT model follows the `User: ...\nAssistant: ...` conversation format and uses `<|eos|>` to signal end of response.

| Metric | Base | SFT (raw) | SFT (chat) |
|---|---|---|---|
| CORE | 0.131 | 0.125 | 0.067 |
| Val loss | 3.02 | — | — |
| SFT val loss | — | 0.39 | — |

SFT slightly lowers CORE scores — this is expected and matches the behavior of autoregressive models. CORE measures base in-context learning via ELBO scoring, not instruction following. SFT models should be evaluated with generation-based benchmarks (ChatCORE), which is not yet implemented for diffusion models.

For context, this is a 477M parameter model trained from scratch — not a fine-tuned LLaDA-8B. The CORE metric uses the same benchmark and centering as [nanoChat](https://github.com/karpathy/nanochat), so scores are directly comparable between autoregressive and diffusion models at the same scale.

## What's Missing

From the [LLaDA paper](https://arxiv.org/abs/2502.09992) and follow-ups:

- **ChatCORE** — generation-based evaluation for SFT models (generate answers, check correctness)
- **VRPO** — preference alignment from [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
- **Faster inference** — block diffusion, consistency distillation, caching

## References

- [LLaDA paper](https://arxiv.org/abs/2502.09992) — Large Language Diffusion Models
- [nanoChat](https://github.com/karpathy/nanochat) — the autoregressive baseline this was adapted from
- [SMDM](https://github.com/ML-GSAI/SMDM) — scaling laws for masked diffusion models
