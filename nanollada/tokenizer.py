"""BPE Tokenizer for nanoLLaDA (rustbpe + tiktoken)."""

import os
import pickle
from functools import lru_cache

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|mask|>",  # the mask token for diffusion
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

import rustbpe
import tiktoken

class RustBPETokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe", pat_str=pattern,
            mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_mask_token_id(self):
        return self.encode_special("<|mask|>")

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None: ids.insert(0, prepend_id)
            if append is not None: ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids: row.insert(0, prepend_id)
            if append is not None:
                for row in ids: row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *a, **kw):
        return self.encode(*a, **kw)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer to {tokenizer_dir}")

# --- Convenience ---
from nanollada.common import get_base_dir

def get_tokenizer():
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    path = os.path.join(get_base_dir(), "tokenizer", "token_bytes.pt")
    assert os.path.exists(path), f"Token bytes not found at {path}. Run tok_train first."
    return torch.load(path, map_location=device)
