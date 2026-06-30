"""
Identity tokenizer for the Polysemy × Context experiment (component 2).

This is the trainer-side counterpart of the synthetic generator's ``IdentityVocab``: a
trivial whitespace tokenizer that maps each *form* symbol to exactly one token id (no BPE
merges/splits), so the controlled per-form sense-ambiguity ``H(S|W)`` survives end to end
(see docs/adr/0003-forms-are-tokens-identity-tokenizer.md). It is a drop-in replacement
for ``RustBPETokenizer`` on the train/eval data path.

Id layout (chosen so the on-disk ``vocab.json`` form->id table is used verbatim):
- ids ``0 .. |V|-1`` are the corpus forms, exactly as written in ``vocab.json``;
- the special tokens (``<|bos|>`` etc.) are appended at ``|V| .. |V|+len(SPECIAL)-1`` and
  are owned by the tokenizer, NOT by the corpus vocab.

``token_bytes`` here is **1 byte per real form, 0 per special token**. That makes the
trainer's bits-per-byte metric equal *bits per form* (= bits per token, since the corpus
is a 1:1 form<->token stream) — i.e. the "BPC" the analysis (component 3) compares against
the analytic entropy floor. The raw cross-entropy ``loss`` (nats/token) gives perplexity.

This module deliberately avoids importing ``nanochat.tokenizer`` (which pulls in
rustbpe/tiktoken/huggingface) so it stays cheap and usable in envs without those deps.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Union

# Mirror of nanochat.tokenizer.SPECIAL_TOKENS, duplicated here to avoid importing the heavy
# BPE tokenizer module. Only <|bos|> is used by base pretraining; the rest are included so
# the identity tokenizer is a faithful drop-in (e.g. for chat rendering) if ever needed.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

VOCAB_FILENAME = "vocab.json"


class IdentityTokenizer:
    """Whitespace 1-form = 1-id tokenizer with appended special tokens."""

    def __init__(self, forms: Sequence[str], special_tokens: Sequence[str] = SPECIAL_TOKENS):
        # forms occupy ids 0..|V|-1 in the given order; specials are appended after.
        self.num_forms = len(forms)
        self.special_tokens = list(special_tokens)
        self.itos: List[str] = list(forms) + list(self.special_tokens)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        assert len(self.stoi) == len(self.itos), "duplicate symbol in identity vocab"
        self.bos_token_id = self.stoi["<|bos|>"]

    # ---- constructors ----
    @classmethod
    def from_vocab_json(cls, path: str, special_tokens: Sequence[str] = SPECIAL_TOKENS) -> "IdentityTokenizer":
        """Load a form->id table written by the generator (``write_vocab``).

        The file maps form symbol -> contiguous id in ``0..|V|-1``; we restore that order
        and append the special tokens, so the corpus ids are used exactly as written.
        """
        with open(path, "r", encoding="utf-8") as f:
            form_to_id = json.load(f)
        forms = [None] * len(form_to_id)
        for sym, idx in form_to_id.items():
            assert 0 <= idx < len(form_to_id), f"vocab id {idx} out of range for |V|={len(form_to_id)}"
            assert forms[idx] is None, f"duplicate id {idx} in {path}"
            forms[idx] = sym
        assert all(s is not None for s in forms), f"vocab.json {path} ids are not contiguous 0..|V|-1"
        return cls(forms, special_tokens=special_tokens)

    @classmethod
    def from_data_dir(cls, data_dir: str, special_tokens: Sequence[str] = SPECIAL_TOKENS) -> "IdentityTokenizer":
        return cls.from_vocab_json(os.path.join(data_dir, VOCAB_FILENAME), special_tokens=special_tokens)

    # ---- nanochat tokenizer interface ----
    def get_vocab_size(self) -> int:
        return len(self.itos)

    def get_special_tokens(self):
        return set(self.special_tokens)

    def id_to_token(self, id: int) -> str:
        return self.itos[id]

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.stoi[text]

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def _encode_one(self, text: str, prepend=None, append=None) -> List[int]:
        ids: List[int] = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        ids.extend(self.stoi[t] for t in text.split())
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def encode(self, text: Union[str, List[str]], prepend=None, append=None, num_threads: int = None):
        # num_threads is accepted for interface parity with RustBPETokenizer; ignored here.
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def token_bytes(self, device: str = "cpu"):
        """Tensor (vocab_size,): 1 for each real form, 0 for special tokens.

        With this, ``evaluate_bpb``'s bytes-normalized metric equals bits-per-form, and
        special tokens (e.g. <|bos|>) are masked out of the metric (byte length 0).
        """
        import torch
        tb = torch.zeros(self.get_vocab_size(), dtype=torch.int64, device=device)
        tb[: self.num_forms] = 1  # forms count as 1 "byte" (= 1 character/token); specials stay 0
        return tb


def get_identity_tokenizer(data_dir: str) -> IdentityTokenizer:
    """Convenience loader mirroring ``nanochat.tokenizer.get_tokenizer`` for a data dir."""
    return IdentityTokenizer.from_data_dir(data_dir)


def get_identity_token_bytes(tokenizer: IdentityTokenizer, device: str = "cpu"):
    """Convenience mirror of ``nanochat.tokenizer.get_token_bytes`` for the identity path."""
    return tokenizer.token_bytes(device=device)
