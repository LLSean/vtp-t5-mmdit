"""HuggingFace text encoder (optional).

This encoder is useful once your pipeline is verified and you want real text
semantics (instead of the deterministic hash encoder).

Dependencies (optional):
  pip install transformers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class HfTextEncoderConfig:
    name_or_path: str
    max_length: int = 77
    freeze: bool = True
    use_processor: bool = False
    trust_remote_code: bool = False


class HfTextEncoder(nn.Module):
    """Wraps a HuggingFace encoder-only (or text-only) model."""

    def __init__(self, cfg: HfTextEncoderConfig, *, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        try:
            from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise RuntimeError("transformers is not installed. Install via: pip install transformers") from e

        self.processor = None
        self.tokenizer = None
        if bool(cfg.use_processor):
            # * Some multimodal models (e.g., T5Gemma 2) recommend AutoProcessor.
            # * However, processor/tokenizer wiring can be version-sensitive. For our use case
            # * (text-only conditioning), we can safely fall back to AutoTokenizer.
            try:
                self.processor = AutoProcessor.from_pretrained(
                    cfg.name_or_path, trust_remote_code=bool(cfg.trust_remote_code)
                )
            except Exception as e:
                msg = str(e)
                if "image_token_id" in msg:
                    # * Common failure mode when using Gemma processors with an incompatible tokenizer class.
                    # * Fallback: use a plain tokenizer for text-only use.
                    self.processor = None
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        cfg.name_or_path,
                        use_fast=True,
                        trust_remote_code=bool(cfg.trust_remote_code),
                    )
                else:
                    raise
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.name_or_path,
                use_fast=True,
                trust_remote_code=bool(cfg.trust_remote_code),
            )

        # * Prefer AutoModel; fall back to Seq2SeqLM for encoder-decoder models if needed.
        model = None
        try:
            model = AutoModel.from_pretrained(cfg.name_or_path, trust_remote_code=bool(cfg.trust_remote_code))
        except Exception:
            model = AutoModelForSeq2SeqLM.from_pretrained(cfg.name_or_path, trust_remote_code=bool(cfg.trust_remote_code))

        self.model = model.to(device)
        self.model.eval()

        if bool(cfg.freeze):
            for p in self.model.parameters():
                p.requires_grad = False

        # * Best-effort hidden size discovery.
        # * Some configs (e.g., T5Gemma2) store hidden_size under encoder/decoder sub-configs.
        hidden = _infer_hidden_size(self.model)
        if hidden is None:
            raise RuntimeError(
                "Could not infer text hidden size from model/config. "
                "Please report the model name and transformers version."
            )
        self.dim = int(hidden)
        self._is_encoder_decoder = bool(getattr(self.model.config, "is_encoder_decoder", False))

    @torch.no_grad()
    def encode(self, captions: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor]:
        caps = [str(c) for c in captions]
        if self.processor is not None:
            batch = self.processor(
                text=caps,
                padding="max_length",
                truncation=True,
                max_length=int(self.cfg.max_length),
                return_tensors="pt",
            )
        else:
            assert self.tokenizer is not None
            batch = self.tokenizer(
                caps,
                padding="max_length",
                truncation=True,
                max_length=int(self.cfg.max_length),
                return_tensors="pt",
            )
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids, dtype=torch.long)
        attn_mask = attn_mask.to(self.device, non_blocking=True)

        # * Autocast on CUDA to reduce memory.
        autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            if self._is_encoder_decoder:
                # * Use encoder hidden states as conditioning tokens.
                if hasattr(self.model, "get_encoder"):
                    enc = self.model.get_encoder()
                else:
                    enc = getattr(self.model, "encoder", None)
                if enc is None:
                    raise RuntimeError("Encoder-decoder model does not expose an encoder.")
                out = enc(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
                tokens = out.last_hidden_state
            else:
                out = self.model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
                tokens = out.last_hidden_state  # (B, T, D)

        mask = attn_mask.to(torch.bool)
        return tokens, mask


def _infer_hidden_size(model: nn.Module) -> Optional[int]:
    """Tries to infer the text hidden size from a HF model/config."""

    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    def _get(obj: object) -> Optional[int]:
        if obj is None:
            return None
        v = getattr(obj, "hidden_size", None)
        if v is None:
            v = getattr(obj, "d_model", None)
        return int(v) if v is not None else None

    # * Common top-level names.
    hidden = _get(cfg)
    if hidden is not None:
        return hidden

    # * Common nested names across encoder-decoder and multimodal configs.
    for path in (
        ("encoder",),
        ("decoder",),
        ("text_config",),
        ("encoder", "text_config"),
        ("decoder", "text_config"),
        ("encoder_config",),
        ("decoder_config",),
    ):
        obj = cfg
        for key in path:
            obj = getattr(obj, key, None)
            if obj is None:
                break
        hidden = _get(obj)
        if hidden is not None:
            return hidden

    # * Fallback to embedding dim.
    emb = None
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()  # type: ignore[assignment]
    if emb is not None:
        if hasattr(emb, "embedding_dim"):
            return int(getattr(emb, "embedding_dim"))
        if hasattr(emb, "weight") and getattr(emb, "weight") is not None:
            w = getattr(emb, "weight")
            if hasattr(w, "shape") and len(w.shape) >= 2:
                return int(w.shape[1])
    return None

