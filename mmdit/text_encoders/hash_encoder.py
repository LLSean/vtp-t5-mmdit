"""A deterministic (no-download) text encoder for smoke tests.

This is *not* intended for quality. It exists to:
  - validate the full training loop without external model downloads,
  - provide stable, repeatable text conditioning signals.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class HashTextEncoderConfig:
    dim: int = 512
    max_length: int = 128
    vocab_size: int = 65536
    trainable: bool = False


def _stable_hash_to_int(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().strip())


class HashTextEncoder(nn.Module):
    """Maps text to a fixed-length token embedding sequence."""

    def __init__(self, cfg: HashTextEncoderConfig) -> None:
        super().__init__()
        if cfg.vocab_size < 1024:
            raise ValueError("vocab_size is too small for stable hashing.")
        self.cfg = cfg
        self.pad_id = 0
        self.empty_id = 1
        self.embed = nn.Embedding(int(cfg.vocab_size), int(cfg.dim))
        if not bool(cfg.trainable):
            for p in self.embed.parameters():
                p.requires_grad = False

    def encode(self, captions: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes captions.

        Args:
            captions: Iterable of caption strings.

        Returns:
            A tuple of:
              - tokens: (B, T, D)
              - mask: (B, T) bool, True for valid tokens
        """

        caps = [str(c) for c in captions]
        device = self.embed.weight.device
        t = int(self.cfg.max_length)
        ids = torch.full((len(caps), t), fill_value=self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(caps), t), dtype=torch.bool, device=device)

        for i, cap in enumerate(caps):
            toks = _tokenize(cap)
            if not toks:
                toks = ["<empty>"]
            tok_ids: list[int] = []
            for tok in toks[:t]:
                if tok == "<empty>":
                    tok_ids.append(self.empty_id)
                else:
                    tok_ids.append(2 + (_stable_hash_to_int(tok) % (int(self.cfg.vocab_size) - 2)))
            n = len(tok_ids)
            ids[i, :n] = torch.tensor(tok_ids, dtype=torch.long, device=device)
            mask[i, :n] = True

        tokens = self.embed(ids)  # (B, T, D)
        return tokens, mask


