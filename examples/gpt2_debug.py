"""Debug DiReCT on GPT-2 with a tiny slice of WikiText / random text.

The script has two modes:

* **Online** (default): load ``gpt2`` from the HuggingFace hub.
* **Offline / tiny** (``--tiny``): instantiate a *randomly initialised*
  GPT-2 with a very small config (2 layers, hidden size 128).  This
  keeps the example runnable in environments without network access,
  while still exercising every DiReCT code path against a real
  ``transformers`` LM.

Typical usage::

    # online
    python examples/gpt2_debug.py --model gpt2 --num-train 64 --num-val 16 --k 8

    # offline / smoke
    python examples/gpt2_debug.py --tiny --num-train 32 --num-val 8 --k 4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch
from torch import nn

# Make ``import direct`` work when running from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from direct import DiReCTConfig, select_samples  # noqa: E402


FALLBACK_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, gradient descent is a first-order optimization.",
    "Large language models are trained on massive text corpora.",
    "Photosynthesis converts light energy into chemical energy.",
    "The Eiffel Tower is located in Paris, France.",
    "Quantum mechanics describes nature at the smallest scales.",
    "Neural networks approximate arbitrary continuous functions.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "PyTorch is an open-source machine learning framework.",
    "The mitochondrion is the powerhouse of the cell.",
    "Regular exercise improves both physical and mental health.",
    "The speed of light is approximately 299792458 meters per second.",
    "Shakespeare wrote thirty-nine plays and one hundred fifty-four sonnets.",
    "Artificial intelligence has transformed many industries.",
    "The Great Wall of China stretches over thousands of kilometers.",
    "Coffee contains caffeine, a mild central nervous system stimulant.",
    "The DNA molecule has a double helix structure.",
    "Democracy is a system of government based on popular sovereignty.",
    "The Amazon rainforest produces a significant portion of Earth's oxygen.",
    "Einstein's theory of relativity revolutionized physics.",
]


def load_texts(num_train: int, num_val: int, offline: bool
               ) -> tuple[list[str], list[str]]:
    """Load candidate training and validation texts.

    Tries HuggingFace ``datasets`` first and falls back to a hard-coded
    English mini corpus if the dataset cannot be downloaded or ``offline``
    is set.

    Args:
        num_train: Number of training samples to return.
        num_val: Number of validation samples to return.
        offline: Skip the HuggingFace ``datasets`` attempt entirely.

    Returns:
        A pair ``(train_texts, val_texts)`` of Python string lists.
    """
    if not offline:
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [
                t.strip()
                for t in ds["text"]
                if t and len(t.strip()) > 40
            ][: num_train + num_val]
            if len(texts) >= num_train + num_val:
                return texts[:num_train], texts[num_train : num_train + num_val]
        except Exception as exc:                 # pragma: no cover
            print(f"[example] datasets unavailable ({exc!r}); using fallback")

    pool = FALLBACK_TEXTS * (
        (num_train + num_val) // len(FALLBACK_TEXTS) + 1
    )
    return pool[:num_train], pool[num_train : num_train + num_val]


def build_gpt2(model_name: str, tiny: bool):
    """Return ``(tokenizer, model)`` for the requested configuration.

    Args:
        model_name: HuggingFace model id used both for the tokenizer and
            (when ``tiny=False``) the pre-trained weights.
        tiny: If ``True``, ignore ``model_name`` for the weights and
            instantiate a small randomly-initialised GPT-2.  The
            tokenizer is still fetched (or its cached copy is reused).

    Returns:
        Tuple ``(tokenizer, model)`` where the model already has
        ``pad_token_id`` set for causal-LM training.
    """
    from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                              AutoModelForCausalLM)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:                     # pragma: no cover
        # Byte-level BPE fallback that requires no download.
        print(f"[example] tokenizer download failed ({exc!r}); "
              f"falling back to bytes-level tokenizer.")
        tokenizer = _ByteTokenizer()

    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    if tiny:
        vocab_size = getattr(tokenizer, "vocab_size", 256)
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=128,
            n_ctx=128,
            n_embd=128,
            n_layer=2,
            n_head=4,
        )
        model = GPT2LMHeadModel(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


class _ByteTokenizer:
    """Minimal byte-level tokenizer used only when no other is available."""

    pad_token = "\x00"
    eos_token = "\x00"
    vocab_size = 256

    def __call__(self, text: str, return_tensors: str = "pt",
                 truncation: bool = True, max_length: int = 64):
        ids = list(text.encode("utf-8"))[:max_length] or [0]
        ids_t = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": ids_t,
            "attention_mask": torch.ones_like(ids_t),
        }


def make_loss_fn(tokenizer, max_length: int = 64):
    """Return a ``loss_fn`` closure for a GPT-2 style causal LM.

    Args:
        tokenizer: The tokenizer paired with the model.
        max_length: Sequences are truncated / padded to this length.

    Returns:
        A ``(model, text_str) -> scalar loss tensor`` callable.
    """

    def _loss_fn(model: nn.Module, sample: str) -> torch.Tensor:
        device = next(model.parameters()).device
        enc = tokenizer(
            sample,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        return out.loss

    return _loss_fn


def gpt2_param_filter(name: str, param: nn.Parameter) -> bool:
    """Only compute gradients wrt. a small, informative subset.

    Restricting the parameter set to the LM head and the final block's
    LayerNorms keeps peak memory low while still exercising every branch
    of the DiReCT pipeline.

    Args:
        name: Fully-qualified parameter name from :meth:`named_parameters`.
        param: The parameter tensor (unused, kept for API symmetry).

    Returns:
        ``True`` iff the parameter should participate in the gradient.
    """
    del param
    keep_substrings = ("lm_head", "ln_f", "ln_1", "ln_2", "wte")
    return any(s in name for s in keep_substrings)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--tiny", action="store_true",
                        help="Use a small randomly-initialised GPT-2 "
                             "(useful when offline).")
    parser.add_argument("--num-train", type=int, default=32)
    parser.add_argument("--num-val", type=int, default=8)
    parser.add_argument("--k", type=int, default=8,
                        help="Number of samples to select (K in the paper).")
    parser.add_argument("--k-sketch", type=int, default=128,
                        help="Sketching dimension k.")
    parser.add_argument("--elbow-energy", type=float, default=0.945)
    parser.add_argument("--tau-perp", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-model", action="store_true",
                        help="Use *all* trainable params (slow, large mem).")
    parser.add_argument("--offline", action="store_true",
                        help="Skip HuggingFace dataset download.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[example] loading model = {args.model}"
          f"{' (tiny/random init)' if args.tiny else ''}")
    tokenizer, model = build_gpt2(args.model, tiny=args.tiny)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[example] model has {n_params:,} parameters")

    train_texts, val_texts = load_texts(
        args.num_train, args.num_val, offline=args.offline or args.tiny
    )
    print(f"[example] N_train = {len(train_texts)}, N_val = {len(val_texts)}")

    cfg = DiReCTConfig(
        K=args.k,
        k_sketch=args.k_sketch,
        elbow_energy=args.elbow_energy,
        tau_perp=args.tau_perp,
        seed=args.seed,
        device=args.device,
        param_filter=None if args.full_model else gpt2_param_filter,
        verbose=True,
    )

    loss_fn = make_loss_fn(tokenizer, max_length=args.max_length)

    idx = select_samples(
        model=model,
        train_samples=train_texts,
        val_samples=val_texts,
        loss_fn=loss_fn,
        cfg=cfg,
    )

    print("\n[example] Selected indices:", idx)
    print("[example] Selected texts:")
    for rank, i in enumerate(idx):
        preview = train_texts[i][:80].replace("\n", " ")
        print(f"  {rank:2d}. idx={i:3d}  |  {preview!r}")


if __name__ == "__main__":
    main()
