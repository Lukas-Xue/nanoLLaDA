"""Train a BPE tokenizer for nanoLLaDA."""

import os
import time
import argparse
import torch
from nanollada.tokenizer import RustBPETokenizer
from nanollada.common import get_base_dir
from nanollada.dataset import parquets_iter_batched

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000)
parser.add_argument('--doc-cap', type=int, default=10_000)
parser.add_argument('--vocab-size', type=int, default=32768)
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}, doc_cap: {args.doc_cap:,}, vocab_size: {args.vocab_size:,}")

def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
print(f"Training time: {time.time() - t0:.2f}s")

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Sanity check
test = "Hello world! This is a test. 你好世界 🌍"
assert tokenizer.decode(tokenizer.encode(test)) == test
print("Tokenizer sanity check passed")

# Cache token bytes for BPB evaluation
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for tid in range(vocab_size):
    s = tokenizer.decode([tid])
    token_bytes.append(0 if s in special_set else len(s.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes, path)
print(f"Saved token_bytes to {path}")
print(f"Vocab size: {vocab_size}, mask_id: {tokenizer.get_mask_token_id()}")
