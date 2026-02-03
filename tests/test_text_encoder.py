from __future__ import annotations

import torch

from mmdit.text_encoders.hash_encoder import HashTextEncoder, HashTextEncoderConfig


def test_hash_text_encoder_shapes_and_mask() -> None:
    enc = HashTextEncoder(HashTextEncoderConfig(dim=32, max_length=8, vocab_size=2048, trainable=False))
    tokens, mask = enc.encode(["hello world", ""])
    assert tokens.shape == (2, 8, 32)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool
    assert mask[0].sum().item() >= 2
    assert mask[1].sum().item() >= 1


def test_hash_text_encoder_deterministic() -> None:
    enc = HashTextEncoder(HashTextEncoderConfig(dim=16, max_length=8, vocab_size=2048, trainable=False))
    a1, m1 = enc.encode(["a dog"])
    a2, m2 = enc.encode(["a dog"])
    assert torch.allclose(a1, a2)
    assert torch.equal(m1, m2)


def test_hash_text_encoder_outputs_on_module_device() -> None:
    enc = HashTextEncoder(HashTextEncoderConfig(dim=16, max_length=8, vocab_size=2048, trainable=False))
    enc = enc.to(torch.device("cpu"))
    tokens, mask = enc.encode(["hello"])
    assert tokens.device == enc.embed.weight.device
    assert mask.device == enc.embed.weight.device


