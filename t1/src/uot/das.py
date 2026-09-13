"""Distributed interchange interventions on the residual stream (DAS and its controls).

A `SiteIntervention` holds an orthonormal basis B (k × d) for a subspace of the residual stream at
one layer.  Interchange replaces the base activation's coordinates in span(B) with the source's:

    h* = h_base + alpha * B^T B (h_src - h_base)          (alpha=1: full swap; alpha in (0,1): dose)

Three ways to obtain B:
  * learned (DAS): B = first k rows of a trainable orthogonal d×d matrix (Geiger et al. 2024);
  * random: a fixed random orthonormal k-frame (matched-rank control, Makelov et al. 2023);
  * fixed: rows from a probe / mean-difference directions (orthonormalised).

The intervention is applied with a forward hook on `model.model.layers[layer]` at chosen token
positions (per batch item).  Training optimises B (learned mode) to make the model's output match
a high-level counterfactual label over a small set of single-token answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiteIntervention(nn.Module):
    def __init__(self, d: int, k: int, mode: str = "learned", basis: torch.Tensor | None = None, seed: int = 0):
        super().__init__()
        self.d, self.k, self.mode = d, k, mode
        if mode == "learned":
            lin = nn.Linear(d, d, bias=False)
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
                lin.weight.copy_(q)
            self.rot = nn.utils.parametrizations.orthogonal(lin)
            self.register_buffer("B_fixed", torch.empty(0))
        elif mode == "random":
            g = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
            self.register_buffer("B_fixed", q.T.contiguous())  # k × d orthonormal rows
            self.rot = None
        elif mode == "full":
            # ceiling control: replace the whole residual vector at the site
            self.register_buffer("B_fixed", torch.eye(d))
            self.rot = None
            self.k = d
        elif mode == "fixed":
            assert basis is not None and basis.shape[1] == d
            q, _ = torch.linalg.qr(basis.T.float())  # orthonormalise the span
            self.register_buffer("B_fixed", q[:, : basis.shape[0]].T.contiguous())
            self.k = basis.shape[0]
            self.rot = None
        else:
            raise ValueError(mode)
        self.alpha = 1.0

    def basis(self) -> torch.Tensor:
        if self.mode == "learned":
            return self.rot.weight[: self.k]  # rows of an orthogonal matrix are orthonormal
        return self.B_fixed

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        B = self.basis().to(h_base.dtype)
        delta = h_src - h_base
        return h_base + self.alpha * (delta @ B.T) @ B

    def project(self, h: torch.Tensor) -> torch.Tensor:
        B = self.basis().to(h.dtype)
        return h @ B.T


@dataclass
class HookState:
    src: torch.Tensor | None = None      # [b, n_pos, d] source activations at the positions
    positions: torch.Tensor | None = None  # [b, n_pos] token indices
    capture: list | None = None          # if not None, append captured activations (no intervention)


def _layer_module(model, layer: int):
    return model.model.layers[layer]


class Intervener:
    """Attaches a hook at one layer; call `.run(...)` for source capture / base intervention."""

    def __init__(self, model, layer: int, site: SiteIntervention | None):
        self.model, self.layer, self.site = model, layer, site
        self.state = HookState()
        self._handle = _layer_module(model, layer).register_forward_hook(self._hook)

    def remove(self):
        self._handle.remove()

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        st = self.state
        if st.positions is None:
            return None
        b = torch.arange(h.shape[0], device=h.device)[:, None]
        if st.capture is not None:
            st.capture.append(h[b, st.positions].detach())
            return None
        if st.src is None or self.site is None:
            return None
        h = h.clone()
        h[b, st.positions] = self.site(h[b, st.positions], st.src.to(h.dtype))
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    def capture(self, input_ids, attention_mask, positions):
        self.state = HookState(positions=positions, capture=[])
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.state.capture[0]
        self.state = HookState()
        return out

    def intervene(self, input_ids, attention_mask, positions, src):
        self.state = HookState(positions=positions, src=src)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.state = HookState()
        return out


def answer_logits(logits: torch.Tensor, last_index: torch.Tensor, answer_token_ids: torch.Tensor) -> torch.Tensor:
    """logits [b, L, V], last_index [b], answer_token_ids [n_ans] -> [b, n_ans] logits at the last prompt token."""
    b = torch.arange(logits.shape[0], device=logits.device)
    return logits[b, last_index][:, answer_token_ids]


def train_das(model, tok, layer: int, k: int, batches, *, steps: int = 400, lr: float = 1e-3, seed: int = 0,
              mode: str = "learned", basis: torch.Tensor | None = None, device: str = "cuda", log_every: int = 50):
    """batches: iterable yielding dicts with keys
         base_ids, base_mask, base_pos, base_last, src_ids, src_mask, src_pos, cf_label, answer_token_ids
       cf_label: [b] index into answer_token_ids (the high-level counterfactual answer).
    Returns the trained SiteIntervention and a training log."""
    d = model.config.hidden_size
    site = SiteIntervention(d, k, mode=mode, basis=basis, seed=seed).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    iv = Intervener(model, layer, site)
    opt = torch.optim.Adam([p for p in site.parameters() if p.requires_grad], lr=lr) if mode == "learned" else None
    log = []
    it = iter(batches)
    for step in range(steps if opt is not None else 0):
        try:
            b = next(it)
        except StopIteration:
            it = iter(batches); b = next(it)
        src = iv.capture(b["src_ids"].to(device), b["src_mask"].to(device), b["src_pos"].to(device))
        out = iv.intervene(b["base_ids"].to(device), b["base_mask"].to(device), b["base_pos"].to(device), src)
        lg = answer_logits(out.logits.float(), b["base_last"].to(device), b["answer_token_ids"].to(device))
        loss = F.cross_entropy(lg, b["cf_label"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        acc = (lg.argmax(1) == b["cf_label"].to(device)).float().mean().item()
        log.append(dict(step=step, loss=loss.item(), acc=acc))
        if step % log_every == 0:
            print(f"  step {step} loss {loss.item():.3f} acc {acc:.3f}", flush=True)
    iv.remove()
    return site, log


@torch.no_grad()
def eval_iia(model, layer: int, site: SiteIntervention, batches, *, device: str = "cuda", alpha: float = 1.0, decide=None) -> dict:
    """Interchange-intervention accuracy and mean counterfactual log-odds over evaluation batches."""
    site.alpha = alpha
    iv = Intervener(model, layer, site)
    n = correct = 0
    logodds = []
    base_acc = 0
    for b in batches:
        src = iv.capture(b["src_ids"].to(device), b["src_mask"].to(device), b["src_pos"].to(device))
        out = iv.intervene(b["base_ids"].to(device), b["base_mask"].to(device), b["base_pos"].to(device), src)
        lg = answer_logits(out.logits.float(), b["base_last"].to(device), b["answer_token_ids"].to(device))
        pred = lg.argmax(1) if decide is None else decide(lg)
        lab = b["cf_label"].to(device)
        correct += (pred == lab).sum().item(); n += len(lab)
        lp = F.log_softmax(lg, 1)
        logodds += (lp[torch.arange(len(lab)), lab] - lp.logsumexp(1)).tolist()
    iv.remove()
    site.alpha = 1.0
    return dict(iia=correct / max(n, 1), n=n, mean_cf_logprob=float(sum(logodds) / max(len(logodds), 1)))
