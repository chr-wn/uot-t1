"""Residual-stream caching at selected token positions and layers.

Given prompts with named character spans, computes token indices per position class using the
tokenizer's offset mapping, runs the model with output_hidden_states=True, and stores
float16 arrays [n_items, n_positions, n_layers, d] to an .npz with metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def spans_to_token_positions(tok, prompt: str, spans: dict[str, tuple[int, int]], add_bos: bool) -> tuple[list[int], dict[str, int]]:
    """Return (input_ids, {position_name: token_index}).

    Position names ending in '_last' map to the last token overlapping the span; names ending in
    '_first' map to the first; 'end' maps to the token after the span end (or the last overlapping
    token if none); 'last' maps to the final prompt token.
    """
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    shift = 0
    if add_bos and tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
        offs = [(0, 0)] + list(offs)
        shift = 1
    pos: dict[str, int] = {}
    for name, (a, b) in spans.items():
        if name == "last":
            pos[name] = len(ids) - 1
            continue
        overlapping = [i for i, (s, e) in enumerate(offs) if i >= shift and e > a and s < b]
        if not overlapping:
            raise ValueError(f"span {name}={(a, b)} has no tokens in {prompt!r}")
        if name.endswith("_first"):
            pos[name] = overlapping[0]
        elif name.endswith("_end"):
            nxt = overlapping[-1] + 1
            pos[name] = nxt if nxt < len(ids) else overlapping[-1]
        else:
            pos[name] = overlapping[-1]
    return ids, pos


@torch.no_grad()
def cache_residuals(model, tok, prompts: Sequence[str], spans: Sequence[dict[str, tuple[int, int]]], *,
                    layers: Sequence[int], position_names: Sequence[str], out_path: str | Path,
                    batch_size: int = 8, add_bos: bool = True, device: str = "cuda", meta: dict | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(prompts)
    enc = [spans_to_token_positions(tok, p, s, add_bos) for p, s in zip(prompts, spans)]
    d = model.config.hidden_size
    arr = np.zeros((n, len(position_names), len(layers), d), dtype=np.float16)
    tokpos = np.zeros((n, len(position_names)), dtype=np.int32)
    pad = tok.pad_token_id
    order = sorted(range(n), key=lambda i: -len(enc[i][0]))
    for b in range(0, n, batch_size):
        idxs = order[b:b + batch_size]
        L = max(len(enc[i][0]) for i in idxs)
        ids = torch.full((len(idxs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, i in enumerate(idxs):
            s = enc[i][0]
            ids[r, :len(s)] = torch.tensor(s)
            attn[r, :len(s)] = 1
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device), output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1; hs[l] = residual after layer l (hs[0]=embeddings)
        for r, i in enumerate(idxs):
            for pj, pname in enumerate(position_names):
                t = enc[i][1][pname]
                tokpos[i, pj] = t
                for lj, l in enumerate(layers):
                    arr[i, pj, lj] = hs[l][r, t].float().cpu().numpy().astype(np.float16)
    np.savez(out_path, resid=arr, tokpos=tokpos, layers=np.array(layers), position_names=np.array(position_names),
             meta=json.dumps(meta or {}))
    return out_path


def load_cache(path: str | Path):
    z = np.load(path, allow_pickle=False)
    return dict(resid=z["resid"], tokpos=z["tokpos"], layers=z["layers"].tolist(),
                position_names=[str(x) for x in z["position_names"]], meta=json.loads(str(z["meta"])))
