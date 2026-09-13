"""E2.5 — H4 output-level test: implant a different dimension into input quantity 1 of a T1v2 prompt
(unit token, layer L, subspace basis or full) and measure whether the 4-way candidate ranking moves
toward the lawfully composed unit for the implanted dimension.  Because candidates are fixed strings
over the *original* lexemes, we measure: (a) drop in log-prob of the original correct candidate,
(b) rise of the candidate whose dimension matches the counterfactual composition when one exists in
the set (e.g. implanting the second slot's dimension into slot 1 makes the 'dimensionless'/ratio
candidate lawful), versus a matched random subspace and versus text rewriting."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl
from uot.tasks.p2_interchange import from_jsonl as p2_from
from uot.lm.models import load_model
from uot.cache import spans_to_token_positions
from uot.das import SiteIntervention, Intervener
from uot.dims import Dimension

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, default=8); ap.add_argument("--basis", required=True, help="npy or 'full' or 'random:<k>:<seed>'")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--tag", required=True)
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]; device = "cuda"
spec, tok, model = load_model(args.model); d = model.config.hidden_size
if args.basis == "full": site = SiteIntervention(d, d, mode="full").to(device)
elif args.basis.startswith("random:"): _, k, sd = args.basis.split(":"); site = SiteIntervention(d, int(k), mode="random", seed=int(sd)).to(device)
else: B = np.load(args.basis); site = SiteIntervention(d, B.shape[0], mode="fixed", basis=torch.tensor(B)).to(device)
# T1v2 items whose first slot is a single base unit and whose second slot has a different base dimension (speed, density-like)
items = [it for it in items_from_jsonl(root / "data/phase0/T1v2_s0.jsonl") if it.condition == "FAM-NAMED" and it.meta.get("q_spans") and len(it.meta["values"]) == 2]
items = [it for it in items if Dimension.parse(it.meta["slot_dims"][0]).distance() == 1 and Dimension.parse(it.meta["slot_dims"][1]).distance() == 1][: args.n]
# source: P2 real items whose u1 has the *second* slot's dimension (so slot1 := slot2's dimension -> composed = dimensionless for quotient relations)
p2 = p2_from(root / "data/phase2/P2_real_s0.jsonl"); by_dim = {}
for it in p2: by_dim.setdefault(it.d1, []).append(it)
iv = Intervener(model, args.layer, site); rng = np.random.default_rng(0)
def enc_t1(it):
    q = it.meta["q_spans"][0]; return spans_to_token_positions(tok, it.prompt, {"u": tuple(q["unit"]), "last": (len(it.prompt) - 1, len(it.prompt))}, add_bos=True)
def enc_p2(it): return spans_to_token_positions(tok, it.prompt, {"u": tuple(it.spans["u1"])}, add_bos=True)
def cand_scores(prompt_ids, cands):
    # mean log-prob of each candidate continuation given prompt ids (with current hook state)
    outs = []
    for c in cands:
        cids = tok(c, add_special_tokens=False)["input_ids"]; ids = torch.tensor([prompt_ids + cids]).to(device)
        lg = torch.log_softmax(model(input_ids=ids).logits.float(), -1)[0]; pos = torch.arange(len(prompt_ids) - 1, len(ids[0]) - 1); outs.append(float(lg[pos, torch.tensor(cids)].mean()))
    return outs
recs = []
with torch.no_grad():
    for it in items:
        d0, d1 = it.meta["slot_dims"]; srcs = by_dim.get(str(Dimension.parse(d1)), [])
        if not srcs: continue
        s = srcs[rng.integers(len(srcs))]; bi, bp = enc_t1(it); si, sp = enc_p2(s)
        src = iv.capture(torch.tensor([si]).to(device), torch.ones(1, len(si), dtype=torch.long).to(device), torch.tensor([[sp["u"]]]).to(device))
        clean = cand_scores(bi, it.candidates)
        iv.state = __import__("uot.das", fromlist=["HookState"]).HookState(positions=torch.tensor([[bp["u"]]]).to(device), src=src)
        # note: cand_scores runs the model with the hook active at position u of the base
        after = cand_scores(bi, it.candidates); iv.state = __import__("uot.das", fromlist=["HookState"]).HookState()
        # text rewrite: replace slot-1 unit string with the source's unit string
        a0, a1 = it.meta["q_spans"][0]["unit"]; tr = it.prompt[:a0] + s.meta["u1_str"] + it.prompt[a1:]
        tri, _ = spans_to_token_positions(tok, tr, {"last": (len(tr) - 1, len(tr))}, add_bos=True); rewrite = cand_scores(tri, it.candidates)
        # lawful target after implanting slot2's dimension into slot1: composed dim = d1^e0 * d1^e1
        rel_exps = {"speed": (1, -1), "acceleration": (1, -1), "power": (1, -1), "flow rate": (1, -1), "density": (1, -3), "pressure": (1, -2), "force": (1, 1), "energy": (1, 1), "momentum": (1, 1)}
        recs.append(dict(item=it.item_id, relation=it.meta["relation"], answer_index=it.answer_index, roles=it.candidate_roles, clean=clean, after=after, rewrite=rewrite, src_dim=s.d1))
iv.remove()
def summarise(key):
    dl = [r[key][r["answer_index"]] - r["clean"][r["answer_index"]] for r in recs]  # change in correct-candidate log-prob
    flips = [int(int(np.argmax(r[key])) != int(np.argmax(r["clean"]))) for r in recs]
    return dict(mean_dlogp_correct=float(np.mean(dl)), argmax_flip_rate=float(np.mean(flips)))
res = dict(model=args.model, layer=args.layer, basis=args.basis, n=len(recs), intervention=summarise("after"), text_rewrite=summarise("rewrite"))
print(json.dumps(res, indent=1)); out = root / "runs/E2.5" / args.model; out.mkdir(parents=True, exist_ok=True); json.dump(dict(res, records=recs), open(out / f"composition_{args.tag}.json", "w"))
