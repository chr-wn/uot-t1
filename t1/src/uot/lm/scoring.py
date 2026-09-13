"""Candidate scoring (mean per-token log-prob of a continuation) and greedy generation."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def _encode_prompt(tok, prompt: str, add_bos: bool) -> list[int]:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if add_bos and tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    return ids


def _encode_cont(tok, cont: str) -> list[int]:
    return tok(cont, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score_candidates(model, tok, prompts: Sequence[str], candidates: Sequence[Sequence[str]], *, batch_size: int = 16,
                     add_bos: bool = True, device: str = "cuda") -> list[list[dict]]:
    """For each prompt, returns per-candidate dicts: sum_logprob, mean_logprob, n_tokens, first_token_logprob.

    The candidate is tokenised separately (with its leading space) and appended to the prompt's ids,
    so the prompt/candidate boundary is identical across candidates.
    """
    seqs: list[tuple[int, int, list[int], int]] = []  # (prompt_idx, cand_idx, ids, n_prompt)
    for pi, (p, cs) in enumerate(zip(prompts, candidates)):
        pids = _encode_prompt(tok, p, add_bos)
        for ci, c in enumerate(cs):
            cids = _encode_cont(tok, c)
            if not cids:
                raise ValueError(f"empty candidate {c!r}")
            seqs.append((pi, ci, pids + cids, len(pids)))
    out: list[list[dict]] = [[None] * len(cs) for cs in candidates]
    pad = tok.pad_token_id
    # sort by length for efficient batching
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][2]))
    for b in range(0, len(order), batch_size):
        idxs = order[b:b + batch_size]
        L = max(len(seqs[i][2]) for i in idxs)
        ids = torch.full((len(idxs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, i in enumerate(idxs):
            s = seqs[i][2]
            ids[r, :len(s)] = torch.tensor(s)
            attn[r, :len(s)] = 1
        ids, attn = ids.to(device), attn.to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for r, i in enumerate(idxs):
            pi, ci, s, npf = seqs[i]
            tgt = torch.tensor(s[npf:], device=device)
            pos = torch.arange(npf - 1, len(s) - 1, device=device)
            lp = logp[r, pos, tgt]
            out[pi][ci] = dict(sum_logprob=float(lp.sum()), mean_logprob=float(lp.mean()), n_tokens=int(len(tgt)),
                               first_token_logprob=float(lp[0]))
    return out


@torch.no_grad()
def generate_greedy(model, tok, prompts: Sequence[str], *, max_new_tokens: int = 12, batch_size: int = 16,
                    add_bos: bool = True, device: str = "cuda") -> list[str]:
    outs: list[str] = [""] * len(prompts)
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for b in range(0, len(order), batch_size):
        idxs = order[b:b + batch_size]
        enc = [_encode_prompt(tok, prompts[i], add_bos) for i in idxs]
        L = max(len(e) for e in enc)
        ids = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, L - len(e):] = torch.tensor(e)
            attn[r, L - len(e):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for r, i in enumerate(idxs):
            outs[i] = tok.decode(gen[r, L:], skip_special_tokens=True)
    return outs


def normalise_unit_text(s: str) -> str:
    s = s.strip().split("\n")[0].strip()
    s = s.replace("²", "^2").replace("³", "^3").replace("⁻", "^-").replace("·", "*").replace("×", "*").replace(" * ", "*")
    s = " ".join(s.split())
    return s.lower()


def generation_matches(gen: str, acceptable: Sequence[str]) -> bool:
    """True iff the generation starts with an acceptable rendering followed by end/punctuation/space."""
    import re
    g = normalise_unit_text(gen)
    for a in acceptable:
        a2 = normalise_unit_text(a)
        if not a2:
            continue
        if g == a2 or re.match(re.escape(a2) + r"(?=$|[\s.,;:!?)])", g):
            return True
    return False
