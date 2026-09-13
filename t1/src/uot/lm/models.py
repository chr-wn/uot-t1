"""Model loading with a fixed registry of specs (HF ids, family, chat handling)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    kind: str            # "base" | "instruct"
    params_b: float
    dtype: str = "bfloat16"
    no_think: bool = False  # Qwen3 instruct: disable thinking in chat template


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-4b-base": ModelSpec("qwen3-4b-base", "Qwen/Qwen3-4B-Base", "qwen3", "base", 4),
    "qwen3-8b-base": ModelSpec("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "qwen3", "base", 8),
    "qwen3-14b-base": ModelSpec("qwen3-14b-base", "Qwen/Qwen3-14B-Base", "qwen3", "base", 14),
    "olmo3-7b": ModelSpec("olmo3-7b", "allenai/Olmo-3-1025-7B", "olmo3", "base", 7),
    "olmo3-32b": ModelSpec("olmo3-32b", "allenai/Olmo-3-1125-32B", "olmo3", "base", 32),
    "gemma2-9b": ModelSpec("gemma2-9b", "google/gemma-2-9b", "gemma2", "base", 9),
    "mistral-7b": ModelSpec("mistral-7b", "mistralai/Mistral-7B-v0.3", "mistral", "base", 7),
    "qwen3-4b": ModelSpec("qwen3-4b", "Qwen/Qwen3-4B", "qwen3", "instruct", 4, no_think=True),
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "instruct", 8, no_think=True),
}


def load_model(key: str, device: str = "cuda", device_map=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODEL_SPECS[key]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    dtype = getattr(torch, spec.dtype)
    kw = dict(torch_dtype=dtype)
    if device_map is not None:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
    if device_map is None:
        model.to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return spec, tok, model


def wrap_chat(spec: ModelSpec, tok, prompt: str, instruction_suffix: str = "") -> str:
    """Render an instruct prompt with the chat template, leaving the assistant turn open."""
    msgs = [{"role": "user", "content": prompt + instruction_suffix}]
    kw = {}
    if spec.no_think:
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
