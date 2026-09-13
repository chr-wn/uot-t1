"""Mechanics of the subspace intervention on a tiny random Qwen2 model (CPU)."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from uot.das import SiteIntervention, Intervener, answer_logits


def tiny():
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def test_capture_and_intervene_alpha():
    m = tiny(); d = 32
    ids = torch.randint(0, 64, (2, 7)); mask = torch.ones_like(ids)
    pos = torch.tensor([[3], [4]])
    site = SiteIntervention(d, 4, mode="random", seed=1)
    iv = Intervener(m, 1, site)
    src = iv.capture(ids, mask, pos)
    assert src.shape == (2, 1, d)
    base = iv.capture(ids.flip(0), mask, pos)
    # alpha=0 -> no change vs plain forward
    site.alpha = 0.0
    out0 = iv.intervene(ids.flip(0), mask, pos, src).logits
    ref = m(input_ids=ids.flip(0), attention_mask=mask).logits
    assert torch.allclose(out0, ref, atol=1e-5)
    # alpha=1 with src == base -> no change
    site.alpha = 1.0
    out1 = iv.intervene(ids.flip(0), mask, pos, base).logits
    assert torch.allclose(out1, ref, atol=1e-5)
    # projection is idempotent and rank-k
    B = site.basis(); assert torch.allclose(B @ B.T, torch.eye(4), atol=1e-5)
    iv.remove()


def test_learned_rotation_is_orthogonal_and_trainable():
    d = 16; site = SiteIntervention(d, 3, mode="learned", seed=0)
    B = site.basis(); assert torch.allclose(B @ B.T, torch.eye(3), atol=1e-4)
    h = torch.randn(5, d); s = torch.randn(5, d)
    out = site(h, s); assert out.shape == h.shape
    loss = out.pow(2).sum(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in site.parameters())


def test_answer_logits():
    lg = torch.randn(2, 5, 64); last = torch.tensor([4, 2]); ids = torch.tensor([7, 9])
    a = answer_logits(lg, last, ids); assert a.shape == (2, 2) and torch.allclose(a[0], lg[0, 4, ids])
