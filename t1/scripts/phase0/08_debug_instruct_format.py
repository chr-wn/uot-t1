"""A6 — why does Qwen3-8B (instruct) fail T3/T1 while Qwen3-4B (instruct) does not?
Prints the wrapped prompt tail, the greedy generation, and per-candidate scores for a few items."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl
from uot.lm.models import load_model, wrap_chat
from uot.lm.scoring import score_candidates, generate_greedy

key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
spec, tok, model = load_model(key)
for stim, suffix in (("data/phase0/T3_s0.jsonl", "\nAnswer with only the letter."), ("data/phase0/T1v2_s0.jsonl", "\nAnswer with only the unit.")):
    items = items_from_jsonl(stim)[:4] + items_from_jsonl(stim)[500:503]
    prompts = [wrap_chat(spec, tok, it.prompt, suffix) for it in items]
    print("=== wrapped tail:", repr(prompts[0][-120:]))
    gens = generate_greedy(model, tok, prompts, max_new_tokens=20, add_bos=False)
    cands = [[c if it.task == "T1" else c.strip() for c in it.candidates] for it in items]
    sc = score_candidates(model, tok, prompts, cands, add_bos=False)
    for it, g, s, cs in zip(items, gens, sc, cands):
        print(f"[{it.condition}] gen={g!r} | correct={cs[it.answer_index]!r} | scores={[round(x['mean_logprob'],2) for x in s]} n_tok={[x['n_tokens'] for x in s]}")
