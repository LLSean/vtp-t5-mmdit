"""Captions a directory of images with T5Gemma 2 and writes a train_images.jsonl.

This script is designed to produce a JSONL compatible with the project's T2I
pipeline (see `data/edges2shoes_t2i/train_images.jsonl`):
  {"id":"img_000001","image_path":".../000001.png","caption":"..."}

Example:
  python3 -m mmdit.scripts.caption_images_t5gemma2 \
    --image_dir danbooru_surtr_solo_100k \
    --out_root data/danbooru_surtr_solo_100k_t2i \
    --model_name_or_path ./t5gemma-2-270m-270m \
    --length_policy cycle \
    --caption_mode single \
    --device cuda \
    --dtype auto \
    --batch_size 4 \
    --resume
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from PIL import Image
import torch
from tqdm import tqdm
torch.backends.cuda.enable_flash_sdp(True)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")


@dataclass(frozen=True)
class _CaptionSpec:
    """A caption style/length specification."""

    name: str
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class _GenerationOutput:
    """A single decoded generation output plus lightweight metadata."""

    text: str
    is_truncated: bool


def _infer_primary_model_device(model: torch.nn.Module) -> torch.device:
    """Infers a reasonable device to place model inputs on.

    This matters for models loaded with `device_map="auto"` (or other sharding)
    where inputs can otherwise remain on CPU and trigger warnings / slowdowns.
    """

    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device) and dev.type != "meta":
        return dev

    # * Accelerate-style sharding exposes `hf_device_map`.
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for v in hf_map.values():
            if isinstance(v, str) and v not in ("cpu", "disk", "meta"):
                try:
                    return torch.device(v)
                except Exception:  # pylint: disable=broad-except
                    continue

    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _move_tensors_to_device(batch: object, device: torch.device) -> object:
    """Moves all tensors in a batch-like object to a device.

    Supports:
      - Transformers BatchEncoding / BatchFeature (has `.to(device)`)
      - Plain dicts
    """

    if hasattr(batch, "to"):
        try:
            return batch.to(device)  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            pass
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


def _maybe_use_local_transformers(*, repo_root: Path, mode: str) -> None:
    """Optionally prepends the bundled `./transformers/src` to sys.path.

    Args:
        repo_root: Workspace root.
        mode: One of: "auto", "yes", "no".
    """

    m = str(mode).lower().strip()
    if m not in ("auto", "yes", "no"):
        raise ValueError("--use_local_transformers must be one of: auto, yes, no")
    if m == "no":
        return
    # * The bundled `./transformers` in this repo targets newer Python (3.10+). Avoid importing it on older runtimes
    # * unless the user explicitly requests it (and accepts the risk of import errors).
    if sys.version_info < (3, 10) and m == "auto":
        return
    tf_src = repo_root / "transformers" / "src"
    if not tf_src.is_dir():
        return
    if m == "yes" or m == "auto":
        sys.path.insert(0, str(tf_src))


def _iter_images(image_dir: Path, *, recursive: bool) -> Iterator[Path]:
    """Yields image files under a directory, sorted by path."""

    if recursive:
        paths = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    else:
        paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    for p in sorted(paths, key=lambda x: x.as_posix()):
        yield p


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _make_sample_id(rel_path: Path) -> str:
    """Creates a stable, filesystem-safe sample id from a relative image path."""

    raw = rel_path.as_posix()
    sid = _SAFE_ID_RE.sub("_", raw).strip("_")
    return sid or "img"


def _load_image_rgb(path: Path) -> Image.Image:
    """Loads an image from disk and converts to RGB (GIF uses first frame)."""

    img = Image.open(path)
    # * PIL opens GIFs as the first frame by default; ensure RGB for the processor.
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _parse_dtype(dtype: str, *, device: torch.device) -> torch.dtype:
    """Parses dtype string to a torch dtype."""

    d = str(dtype).lower().strip()
    if d == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError("--dtype must be one of: auto, bf16, fp16, fp32")


def _decode_text(processor: object, sequences: torch.Tensor) -> list[str]:
    """Best-effort decode for different processor/tokenizer APIs."""

    if hasattr(processor, "batch_decode"):
        # * Most processors expose tokenizer-like decoding.
        return list(processor.batch_decode(sequences, skip_special_tokens=True))  # type: ignore[attr-defined]
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "batch_decode"):
        return list(tok.batch_decode(sequences, skip_special_tokens=True))  # type: ignore[attr-defined]
    if hasattr(processor, "decode"):
        return [str(processor.decode(seq, skip_special_tokens=True)) for seq in sequences]  # type: ignore[attr-defined]
    raise RuntimeError("Processor does not support decode/batch_decode.")


def _read_existing_ids(jsonl_path: Path) -> set[str]:
    """Reads existing sample IDs from a JSONL, best-effort."""

    if not jsonl_path.is_file():
        return set()
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "id" in item:
                sid = str(item.get("id", "")).strip()
                if sid:
                    seen.add(sid)
    return seen


def _build_caption_specs(
    *,
    backend: str,
    language: str,
    short_max_new_tokens: int,
    medium_max_new_tokens: int,
    long_max_new_tokens: int,
    short_prompt: Optional[str],
    medium_prompt: Optional[str],
    long_prompt: Optional[str],
) -> dict[str, _CaptionSpec]:
    """Builds caption prompt templates and max token constraints."""

    b = str(backend).lower().strip()
    if b not in ("t5gemma2", "qwen3vl"):
        raise ValueError("--backend must be one of: t5gemma2, qwen3vl")

    lang = str(language).lower().strip()
    if lang not in ("en", "zh"):
        raise ValueError("--language must be one of: en, zh")

    if b == "qwen3vl":
        # * Chat-template models do NOT require `<start_of_image>` in the text.
        if lang == "zh":
            defaults = {
                "short": "请为这张图片写一个很短的caption（不超过10个词）。",
                "medium": "请用一句话描述这张图片。",
                "long": "请用2-3句话详细描述这张图片，包含主体、外观、背景和画风，避免重复。",
            }
        else:
            defaults = {
                "short": "Write a very short caption for this image (max 10 words).",
                "medium": "Describe this image in one sentence.",
                "long": (
                    "Describe this image in 2-3 complete sentences, including subject, appearance, background, and art style. "
                    "Avoid repetition."
                ),
            }
    else:
        # * T5Gemma 2 expects `<start_of_image>` in the prompt text.
        if lang == "zh":
            defaults = {
                # * Default to a continuation-style prompt (mirrors the official English example) which tends to be
                # * more reliable for pretrained captioning.
                "short": "<start_of_image> 在这张图片中，有",
                "medium": "<start_of_image> 在这张图片中，有",
                "long": "<start_of_image> 在这张图片中，有。请用2-3句话详细描述，避免重复：",
            }
        else:
            defaults = {
                # * Official-style prompt (recommended by the model docs).
                "short": "Please give this image a caption. <start_of_image>. Description:",
                "medium": "<start_of_image> in this image, there is",
                "long": (
                    "<start_of_image> in this image, there is. "
                    "Write 2-3 complete sentences describing the image in detail, without repeating yourself:"
                ),
            }

    return {
        "short": _CaptionSpec(
            name="short",
            prompt=str(short_prompt or defaults["short"]),
            max_new_tokens=int(short_max_new_tokens),
        ),
        "medium": _CaptionSpec(
            name="medium",
            prompt=str(medium_prompt or defaults["medium"]),
            max_new_tokens=int(medium_max_new_tokens),
        ),
        "long": _CaptionSpec(
            name="long",
            prompt=str(long_prompt or defaults["long"]),
            max_new_tokens=int(long_max_new_tokens),
        ),
    }


def _choose_length_name(
    *,
    policy: str,
    idx: int,
    rng: random.Random,
    weights: tuple[float, float, float],
) -> str:
    """Chooses a caption length bucket name."""

    p = str(policy).lower().strip()
    if p == "cycle":
        return ("short", "medium", "long")[idx % 3]
    if p == "random":
        w_short, w_med, w_long = weights
        total = float(w_short) + float(w_med) + float(w_long)
        if total <= 0:
            raise ValueError("--length_weights must sum to > 0 when --length_policy=random")
        r = rng.random() * total
        if r < w_short:
            return "short"
        if r < w_short + w_med:
            return "medium"
        return "long"
    if p in ("short", "medium", "long"):
        return p
    raise ValueError("--length_policy must be one of: cycle, random, short, medium, long")


def _auto_repetition_penalty(length_name: str) -> float:
    """Default repetition penalty per length bucket.

    Notes:
      - For long captions, a small penalty helps reduce looping/repetition.
      - For short captions, keep it neutral.
    """

    return 1.08 if length_name == "long" else 1.0


def _auto_no_repeat_ngram_size(length_name: str) -> int:
    """Default no-repeat ngram size per length bucket."""

    return 3 if length_name == "long" else 0


def _trim_to_last_sentence_boundary(text: str, *, language: str) -> str:
    """Trims text to the last sentence boundary, best-effort.

    This helps when the model hits `max_new_tokens` and returns a partial tail.
    """

    t = str(text).strip()
    if not t:
        return ""
    lang = str(language).lower().strip()
    if lang == "zh":
        seps = ("。", "！", "？", "；", "…")
    else:
        seps = (".", "!", "?", ";")
    last = max((t.rfind(s) for s in seps), default=-1)
    if last <= 0:
        return t
    return t[: last + 1].strip()


def _is_truncated_generation(
    seq: torch.Tensor,
    *,
    pad_token_id: Optional[int],
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> bool:
    """Heuristic: treat a sample as truncated if it didn't end with EOS and likely hit length limit."""

    ids = list(seq.detach().to("cpu").tolist())
    if pad_token_id is not None:
        # * Trim only trailing padding (do NOT remove leading pad tokens; T5-style models use pad as start token).
        while len(ids) > 1 and ids[-1] == pad_token_id:
            ids.pop()
    if not ids:
        return False
    ended_by_eos = ids[-1] in eos_token_ids
    if ended_by_eos:
        return False
    # * For encoder-decoder models, sequences usually have an initial start token + up to max_new_tokens.
    return len(ids) >= int(max_new_tokens)


@torch.no_grad()
def _generate_captions_batch(
    *,
    processor: object,
    model: torch.nn.Module,
    images: list[Image.Image],
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> list[_GenerationOutput]:
    """Runs the model on a batch of images and returns decoded captions + metadata.

    Note:
      - `Gemma3Processor` treats a flat `list[Image]` as "multiple images for a single prompt" (one sample).
        For batched captioning (one prompt per image), we must pass a nested list: `[[img1], [img2], ...]`.
    """

    # * Some processors accept scalar string for text; we always pass list for batching.
    prompts = [prompt] * len(images)
    batched_images = [[img] for img in images]
    model_inputs = processor(text=prompts, images=batched_images, return_tensors="pt", padding=True)  # type: ignore[misc]
    model_inputs = {k: v.to(device) for k, v in model_inputs.items() if torch.is_tensor(v)}
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "num_beams": int(num_beams),
    }
    if float(repetition_penalty) > 0 and float(repetition_penalty) != 1.0:
        generate_kwargs["repetition_penalty"] = float(repetition_penalty)
    if int(no_repeat_ngram_size) > 0:
        generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if bool(do_sample):
        generate_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
        if int(top_k) > 0:
            generate_kwargs["top_k"] = int(top_k)
    generation = model.generate(**model_inputs, **generate_kwargs)
    texts = _decode_text(processor, generation)

    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos is None:
        eos_ids: set[int] = set()
    elif isinstance(eos, (list, tuple, set)):
        eos_ids = {int(x) for x in eos}
    else:
        eos_ids = {int(eos)}

    pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(getattr(model, "config", None), "pad_token_id", None)
    pad_id = int(pad) if pad is not None else None

    outputs: list[_GenerationOutput] = []
    for txt, seq in zip(texts, generation):
        cap = str(txt).strip()
        truncated = bool(eos_ids) and _is_truncated_generation(
            seq, pad_token_id=pad_id, eos_token_ids=eos_ids, max_new_tokens=int(max_new_tokens)
        )
        outputs.append(_GenerationOutput(text=cap, is_truncated=truncated))
    return outputs


@torch.no_grad()
def _generate_captions_batch_qwen3vl(
    *,
    processor: object,
    model: torch.nn.Module,
    image_paths: list[Path],
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> list[_GenerationOutput]:
    """Chat-template generation (mirrors caption_test.py).

    Steps:
      - Build one "conversation" per image.
      - `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")`
      - Move inputs to the model device to avoid CPU/CUDA mismatch warnings.
      - `model.generate(**inputs, ...)`
      - Trim prompt tokens from output, then `processor.batch_decode(...)`.
    """

    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("backend=qwen3vl requires processor.apply_chat_template().")

    # * Decoder-only generation should use left padding for correct results.
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"

    conversations: list[list[dict[str, object]]] = []
    for p in image_paths:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(p)},
                        {"type": "text", "text": str(prompt)},
                    ],
                }
            ]
        )

    inputs = processor.apply_chat_template(  # type: ignore[attr-defined]
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    device = _infer_primary_model_device(model)
    inputs = _move_tensors_to_device(inputs, device)

    gen_kwargs: dict[str, object] = {"max_new_tokens": int(max_new_tokens)}
    gen_kwargs["do_sample"] = bool(do_sample)
    gen_kwargs["num_beams"] = int(num_beams)
    if float(repetition_penalty) > 0 and float(repetition_penalty) != 1.0:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    if int(no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if bool(do_sample):
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
        if int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)

    generated_ids = model.generate(**inputs, **gen_kwargs)

    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None and isinstance(inputs, dict):
        input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        raise RuntimeError("Could not access input_ids from processor outputs.")

    trimmed: list[torch.Tensor] = []
    for in_ids, out_ids in zip(input_ids, generated_ids):
        trimmed.append(out_ids[len(in_ids) :])

    if not hasattr(processor, "batch_decode"):
        raise RuntimeError("Processor does not support batch_decode().")
    texts = processor.batch_decode(  # type: ignore[attr-defined]
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos is None:
        eos_ids: set[int] = set()
    elif isinstance(eos, (list, tuple, set)):
        eos_ids = {int(x) for x in eos}
    else:
        eos_ids = {int(eos)}

    outputs: list[_GenerationOutput] = []
    for txt, ids in zip(texts, trimmed):
        cap = str(txt).strip()
        tid = ids.detach().to("cpu").tolist()
        ended_by_eos = bool(tid) and bool(eos_ids) and int(tid[-1]) in eos_ids
        is_truncated = (len(tid) >= int(max_new_tokens)) and not ended_by_eos
        outputs.append(_GenerationOutput(text=cap, is_truncated=is_truncated))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption images with T5Gemma 2 and write train_images.jsonl.")
    parser.add_argument(
        "--backend",
        type=str,
        default="t5gemma2",
        choices=["t5gemma2", "qwen3vl"],
        help="Captioning backend. qwen3vl mirrors the caption_test.py chat-template generation flow.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="danbooru_surtr_solo_100k",
        help="Directory containing images to caption.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images under --image_dir.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/danbooru_surtr_solo_100k_t2i",
        help="Output root. Writes train_images.jsonl under it.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./t5gemma-2-270m-270m",
        help=(
            "Model id or local path. "
            "t5gemma2: use a local path like ./t5gemma-2-270m-270m. "
            "qwen3vl: use a model id like Qwen/Qwen3-VL-4B-Instruct."
        ),
    )
    parser.add_argument(
        "--use_local_transformers",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help="If ./transformers/src exists, optionally prefer it for importing transformers.",
    )
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="single",
        choices=["single", "all"],
        help=(
            "single: one caption per image; all: write 3 lines per image (short/medium/long) "
            "with id suffixes (_short/_medium/_long)."
        ),
    )
    parser.add_argument(
        "--length_policy",
        type=str,
        default="cycle",
        help="When caption_mode=single: cycle | random | short | medium | long.",
    )
    parser.add_argument(
        "--length_weights",
        type=str,
        default="1,1,1",
        help="Comma-separated weights for short,medium,long when --length_policy=random (e.g. 0.2,0.6,0.2).",
    )
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--short_max_new_tokens", type=int, default=24)
    parser.add_argument("--medium_max_new_tokens", type=int, default=48)
    parser.add_argument("--long_max_new_tokens", type=int, default=128)
    parser.add_argument("--short_prompt", type=str, default=None)
    parser.add_argument("--medium_prompt", type=str, default=None)
    parser.add_argument("--long_prompt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4, help="Caption batch size (images per forward).")
    parser.add_argument("--max_images", type=int, default=0, help="If > 0, only process the first N images.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (used when --length_policy=random).")
    parser.add_argument("--resume", action="store_true", help="If output JSONL exists, skip already-written ids.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", help="auto|bf16|fp16|fp32")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for more diverse captions.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0, help="Sampling top_k (only used when --do_sample). 0 keeps model default.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=0.0,
        help="Repetition penalty. 0 means auto (recommended).",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=-1,
        help="No-repeat ngram size. -1 means auto (recommended).",
    )
    parser.add_argument(
        "--truncation_policy",
        type=str,
        default="retry",
        choices=["keep", "trim", "retry"],
        help="What to do when generation looks truncated (hit max_new_tokens without EOS).",
    )
    parser.add_argument(
        "--truncation_retry_factor",
        type=float,
        default=1.5,
        help="When truncation_policy=retry, multiply max_new_tokens by this factor for the retry.",
    )
    parser.add_argument(
        "--truncation_retry_max_new_tokens",
        type=int,
        default=256,
        help="When truncation_policy=retry, cap the retry max_new_tokens to this value.",
    )
    parser.add_argument(
        "--truncation_apply_to",
        type=str,
        default="long",
        choices=["off", "long", "all"],
        help="Which length buckets the truncation policy applies to.",
    )
    parser.add_argument(
        "--empty_caption_policy",
        type=str,
        default="fallback",
        choices=["fallback", "skip", "keep", "error"],
        help=(
            "What to do if decoded caption is empty. "
            "fallback: retry with official prompt once; skip: omit the sample; keep: write empty; error: raise."
        ),
    )
    args = parser.parse_args()

    backend = str(args.backend).lower().strip()

    repo_root = Path(__file__).resolve().parents[2]
    _maybe_use_local_transformers(repo_root=repo_root, mode=args.use_local_transformers)

    if backend == "t5gemma2":
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor as HfAutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers is not installed (or not importable). "
                "Install via: pip install transformers"
            ) from e
    elif backend == "qwen3vl":
        try:
            from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor as MsAutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "backend=qwen3vl requires modelscope. Install it in your environment (e.g. `pip install modelscope`)."
            ) from e
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"--image_dir not found: {image_dir}")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_root / "train_images.jsonl"

    if backend == "t5gemma2":
        if str(args.device).lower() == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(str(args.device).lower())
        dtype = _parse_dtype(args.dtype, device=device)

        # * Load processor/model once.
        processor = HfAutoProcessor.from_pretrained(str(args.model_name_or_path), trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_name_or_path), torch_dtype=dtype)
        model = model.to(device)
        model.eval()
    else:
        # * Mirror caption_test.py: `dtype="auto", device_map="auto"`.
        # * Note: inputs are moved to the model device inside `_generate_captions_batch_qwen3vl` to avoid CPU/CUDA mismatch.
        device = torch.device("cpu")  # unused placeholder for qwen3vl code path
        processor = MsAutoProcessor.from_pretrained(str(args.model_name_or_path))
        # * Decoder-only models: left padding is required for correct generation when padding a batch.
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "padding_side", None) != "left":
            tok.padding_side = "left"
        model = Qwen3VLForConditionalGeneration.from_pretrained(  # type: ignore[name-defined]
            str(args.model_name_or_path),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        # * Helpful one-time diagnostic: if any large blocks are on CPU/disk, generation will be very slow.
        hf_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            cpu = sum(1 for v in hf_map.values() if v == "cpu")
            disk = sum(1 for v in hf_map.values() if v == "disk")
            if cpu or disk:
                print(f"[warn] qwen3vl device_map includes offload: cpu={cpu} disk={disk} (this will be slow)")

    # * Align generation config with our CLI to avoid noisy warnings like:
    # * "The following generation flags are not valid and may be ignored: ['top_p', 'top_k']"
    # * which happens when `do_sample=False` but sampling-only params remain set.
    if backend == "t5gemma2" and hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = bool(args.do_sample)
        if not bool(args.do_sample):
            # * Reset sampling-only knobs to greedy defaults.
            model.generation_config.temperature = 1.0
            model.generation_config.top_p = 1.0
            model.generation_config.top_k = 50
        else:
            # * Optionally override with CLI when sampling.
            model.generation_config.temperature = float(args.temperature)
            model.generation_config.top_p = float(args.top_p)
            if int(args.top_k) > 0:
                model.generation_config.top_k = int(args.top_k)

    caption_specs = _build_caption_specs(
        backend=backend,
        language=args.language,
        short_max_new_tokens=args.short_max_new_tokens,
        medium_max_new_tokens=args.medium_max_new_tokens,
        long_max_new_tokens=args.long_max_new_tokens,
        short_prompt=args.short_prompt,
        medium_prompt=args.medium_prompt,
        long_prompt=args.long_prompt,
    )

    weights_parts = [p.strip() for p in str(args.length_weights).split(",") if p.strip()]
    if len(weights_parts) != 3:
        raise ValueError("--length_weights must have 3 comma-separated values: short,medium,long")
    weights = (float(weights_parts[0]), float(weights_parts[1]), float(weights_parts[2]))
    rng = random.Random(int(args.seed))

    existing = _read_existing_ids(out_jsonl) if bool(args.resume) else set()
    if existing:
        print(f"[resume] loaded {len(existing)} existing ids from {out_jsonl}")

    image_paths = list(_iter_images(image_dir, recursive=bool(args.recursive)))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    if int(args.max_images) > 0:
        image_paths = image_paths[: int(args.max_images)]

    # * We micro-batch per caption length to keep generation constraints consistent.
    queues: dict[str, list[tuple[str, Path]]] = {"short": [], "medium": [], "long": []}

    # * Open once; append if resuming, overwrite otherwise.
    mode = "a" if (bool(args.resume) and out_jsonl.is_file()) else "w"
    written = 0
    skipped = 0
    errors = 0

    def _flush(name: str, items: list[tuple[str, Path]]) -> None:
        nonlocal written, errors
        if not items:
            return
        spec = caption_specs[name]
        sids = [sid for sid, _ in items]
        paths = [p for _, p in items]
        images: list[Image.Image] = []
        kept: list[tuple[str, Path]] = []
        for sid, p in zip(sids, paths):
            if backend == "t5gemma2":
                try:
                    images.append(_load_image_rgb(p))
                    kept.append((sid, p))
                except Exception as e:  # pylint: disable=broad-except
                    errors += 1
                    print(f"[warn] failed to load image: {p} ({type(e).__name__}: {e})")
            else:
                # * backend=qwen3vl passes image paths directly to the processor.
                kept.append((sid, p))
        if not kept:
            items.clear()
            return
        if backend == "t5gemma2" and not images:
            items.clear()
            return
        rep_pen = float(args.repetition_penalty)
        if rep_pen <= 0:
            rep_pen = _auto_repetition_penalty(name)
        ngram = int(args.no_repeat_ngram_size)
        if ngram < 0:
            ngram = _auto_no_repeat_ngram_size(name)

        try:
            if backend == "qwen3vl":
                outs = _generate_captions_batch_qwen3vl(
                    processor=processor,
                    model=model,
                    image_paths=[p for _, p in kept],
                    prompt=spec.prompt,
                    max_new_tokens=spec.max_new_tokens,
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    num_beams=int(args.num_beams),
                    repetition_penalty=rep_pen,
                    no_repeat_ngram_size=ngram,
                )
            else:
                outs = _generate_captions_batch(
                    processor=processor,
                    model=model,
                    images=images,
                    prompt=spec.prompt,
                    device=device,
                    max_new_tokens=spec.max_new_tokens,
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    num_beams=int(args.num_beams),
                    repetition_penalty=rep_pen,
                    no_repeat_ngram_size=ngram,
                )
        except Exception as e:  # pylint: disable=broad-except
            errors += len(kept)
            print(f"[warn] generation failed for batch ({name}): {type(e).__name__}: {e}")
            items.clear()
            return
        assert len(outs) == len(kept)
        for (sid, p), out in zip(kept, outs):
            cap = out.text
            is_trunc = bool(out.is_truncated)
            if not cap:
                policy = str(args.empty_caption_policy).lower().strip()
                if policy == "keep":
                    pass
                elif policy == "skip":
                    errors += 1
                    continue
                elif policy == "error":
                    raise RuntimeError(f"Empty caption generated for id={sid} image_path={p}")
                elif policy == "fallback":
                    # * Retry once with the official-style prompt and a slightly larger max_new_tokens.
                    fallback_prompt = caption_specs[name].prompt
                    try:
                        if backend == "qwen3vl":
                            retry_outs = _generate_captions_batch_qwen3vl(
                                processor=processor,
                                model=model,
                                image_paths=[p],
                                prompt=fallback_prompt,
                                max_new_tokens=max(int(spec.max_new_tokens), 32),
                                do_sample=bool(args.do_sample),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k),
                                num_beams=int(args.num_beams),
                                repetition_penalty=rep_pen,
                                no_repeat_ngram_size=ngram,
                            )
                        else:
                            retry_outs = _generate_captions_batch(
                                processor=processor,
                                model=model,
                                images=[_load_image_rgb(p)],
                                prompt=fallback_prompt,
                                device=device,
                                max_new_tokens=max(int(spec.max_new_tokens), 32),
                                do_sample=bool(args.do_sample),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k),
                                num_beams=int(args.num_beams),
                                repetition_penalty=rep_pen,
                                no_repeat_ngram_size=ngram,
                            )
                        if retry_outs and retry_outs[0].text.strip():
                            cap = retry_outs[0].text.strip()
                            is_trunc = bool(retry_outs[0].is_truncated)
                        else:
                            errors += 1
                            continue
                    except Exception as e:  # pylint: disable=broad-except
                        errors += 1
                        print(f"[warn] empty-caption fallback failed for {p} ({type(e).__name__}: {e})")
                        continue
                else:
                    raise ValueError(f"Unknown empty caption policy: {policy}")

            # * Handle truncation (common for long captions).
            trunc_apply = str(args.truncation_apply_to).lower().strip()
            trunc_policy = str(args.truncation_policy).lower().strip()
            trunc_active = trunc_apply == "all" or (trunc_apply == "long" and name == "long")
            if trunc_active and is_trunc and trunc_policy != "keep":
                if trunc_policy == "trim":
                    cap = _trim_to_last_sentence_boundary(cap, language=args.language)
                elif trunc_policy == "retry":
                    # * Retry once with a higher max_new_tokens, then trim if still truncated.
                    retry_max = int(min(int(args.truncation_retry_max_new_tokens), int(spec.max_new_tokens * float(args.truncation_retry_factor))))
                    retry_max = max(retry_max, int(spec.max_new_tokens) + 16)
                    try:
                        if backend == "qwen3vl":
                            retry_outs2 = _generate_captions_batch_qwen3vl(
                                processor=processor,
                                model=model,
                                image_paths=[p],
                                prompt=spec.prompt,
                                max_new_tokens=retry_max,
                                do_sample=bool(args.do_sample),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k),
                                num_beams=int(args.num_beams),
                                repetition_penalty=rep_pen,
                                no_repeat_ngram_size=ngram,
                            )
                        else:
                            retry_outs2 = _generate_captions_batch(
                                processor=processor,
                                model=model,
                                images=[_load_image_rgb(p)],
                                prompt=spec.prompt,
                                device=device,
                                max_new_tokens=retry_max,
                                do_sample=bool(args.do_sample),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k),
                                num_beams=int(args.num_beams),
                                repetition_penalty=rep_pen,
                                no_repeat_ngram_size=ngram,
                            )
                        if retry_outs2 and retry_outs2[0].text.strip():
                            cap = retry_outs2[0].text.strip()
                            if retry_outs2[0].is_truncated:
                                cap = _trim_to_last_sentence_boundary(cap, language=args.language)
                        else:
                            cap = _trim_to_last_sentence_boundary(cap, language=args.language)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"[warn] truncation retry failed for {p} ({type(e).__name__}: {e})")
                        cap = _trim_to_last_sentence_boundary(cap, language=args.language)
                else:
                    raise ValueError(f"Unknown truncation policy: {trunc_policy}")

            rec = {"id": sid, "image_path": str(p), "caption": cap}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
        items.clear()

    with out_jsonl.open(mode, encoding="utf-8") as f:
        pbar = tqdm(image_paths, desc="caption", unit="img")
        for idx, img_path in enumerate(pbar):
            rel = img_path.relative_to(image_dir)
            base_id = _make_sample_id(rel)

            if str(args.caption_mode).lower() == "all":
                targets = ("short", "medium", "long")
            else:
                targets = (
                    _choose_length_name(policy=args.length_policy, idx=idx, rng=rng, weights=weights),
                )

            for tname in targets:
                sid = base_id if str(args.caption_mode).lower() == "single" else f"{base_id}_{tname}"
                if sid in existing:
                    skipped += 1
                    continue
                queues[tname].append((sid, img_path))
                if len(queues[tname]) >= int(args.batch_size):
                    _flush(tname, queues[tname])

            if (idx + 1) % 50 == 0:
                pbar.set_postfix({"written": written, "skipped": skipped, "errors": errors})

        for tname in ("short", "medium", "long"):
            _flush(tname, queues[tname])

    print(f"[done] wrote {written} lines")
    if skipped:
        print(f"[skip] {skipped} (resume)")
    if errors:
        print(f"[warn] {errors} errors (see logs above)")
    print(f"[jsonl] {out_jsonl}")


if __name__ == "__main__":
    main()


