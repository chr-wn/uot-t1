"""E2.1/E2.2 — DAS alignment of a rival variable at one site, with controls.

--variable alg|heur  chooses the counterfactual label set; --mode learned|random|probe chooses the subspace.
Evaluates IIA on: all eval examples, disagree subset, same-dimension-swap subset, held-out-lexeme subset,
invented-unit examples (alg labels only), plus base-prompt accuracy before/after (value preservation is a
separate script).  Writes runs/E2.1/<model>/das_<tag>.json."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p2_interchange import from_jsonl
from uot.lm.models import load_model
from uot.cache import spans_to_token_positions
from uot.das import SiteIntervention, train_das, eval_iia, Intervener, answer_logits

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--position", default="u1")
ap.add_argument("--variable", default="alg"); ap.add_argument("--mode", default="learned"); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--set-seed", type=int, default=0); ap.add_argument("--tag", default=None)
ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--probe-basis", default=None, help="npy [k,d] rows for mode=probe")
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
torch.manual_seed(args.seed)
spec, tok, model = load_model(args.model)
device = "cuda"
real = {it.item_id: it for it in from_jsonl(root / f"data/phase2/P2_real_s{args.set_seed}.jsonl")}
inv = {it.item_id: it for it in from_jsonl(root / f"data/phase2/P2_inv_s{args.set_seed}.jsonl")}
EX = json.load(open(root / f"data/phase2/{args.model}_interchange_s{args.set_seed}.json"))
yes_id = tok(" yes", add_special_tokens=False)["input_ids"][0]; no_id = tok(" no", add_special_tokens=False)["input_ids"][0]
ans_ids = torch.tensor([yes_id, no_id])
enc_cache = {}
def enc(it):
    if it.item_id not in enc_cache:
        ids, pos = spans_to_token_positions(tok, it.prompt, {k: tuple(v) for k, v in it.spans.items()}, add_bos=True)
        enc_cache[it.item_id] = (ids, pos)
    return enc_cache[it.item_id]
def make_batches(examples, items_map, label_key, batch, shuffle, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(examples))
    if shuffle: rng.shuffle(idx)
    out = []
    for b in range(0, len(idx), batch):
        chunk = [examples[i] for i in idx[b:b + batch]]
        bi = [enc(items_map[e["base"]]) for e in chunk]; si = [enc(items_map[e["src"]]) for e in chunk]
        L = max(max(len(x[0]) for x in bi), max(len(x[0]) for x in si))
        def pad(seqs):
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, (s, _) in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); m[r, :len(s)] = 1
            return ids, m
        bids, bm = pad(bi); sids, sm = pad(si)
        out.append(dict(base_ids=bids, base_mask=bm, base_pos=torch.tensor([[x[1][args.position]] for x in bi]), base_last=torch.tensor([x[1]["last"] for x in bi]),
                        src_ids=sids, src_mask=sm, src_pos=torch.tensor([[x[1][args.position]] for x in si]),
                        cf_label=torch.tensor([0 if e[label_key] == 1 else 1 for e in chunk]),  # index into [yes, no]
                        answer_token_ids=ans_ids))
    return out
lab = "y_alg" if args.variable == "alg" else "y_heur"
train_ex = [e for e in EX["real"] if e["split"] == "train" and e[lab] is not None]
eval_ex = [e for e in EX["real"] if e["split"] == "eval" and e[lab] is not None]
train_b = make_batches(train_ex, real, lab, args.batch, True, args.seed)
t0 = time.time()
basis = torch.tensor(np.load(args.probe_basis)) if args.mode == "probe" else None
site, log = train_das(model, tok, args.layer, args.rank, train_b, steps=args.steps if args.mode == "learned" else 0, lr=args.lr, seed=args.seed,
                      mode=("fixed" if args.mode == "probe" else args.mode), basis=basis, device=device)
res = dict(model=args.model, layer=args.layer, position=args.position, variable=args.variable, mode=args.mode, rank=args.rank, steps=args.steps,
           train_time=time.time() - t0, final_train_acc=float(np.mean([l["acc"] for l in log[-20:]])) if log else None)
def ev(examples, items_map, label_key, name, alpha=1.0):
    if not examples: return
    r = eval_iia(model, args.layer, site, make_batches(examples, items_map, label_key, args.batch, False), device=device, alpha=alpha)
    res[name] = r; print(f"  {name:28s} IIA={r['iia']:.3f} n={r['n']}", flush=True)
ev(eval_ex, real, lab, "iia_all")
ev([e for e in eval_ex if e.get("agree") == 0], real, lab, "iia_disagree")
ev([e for e in eval_ex if e["same_dim_swap"] == 1], real, lab, "iia_same_dim_swap")
ev([e for e in eval_ex if e["heldout_lex"] == 1], real, lab, "iia_heldout_lex")
ev([e for e in eval_ex if e["y_alg"] != e["y_base"]], real, lab, "iia_output_changing")
# invented units: alg labels only
ev([e for e in EX["inv"]], inv, "y_alg", "iia_invented_alg")
# the *other* model's labels on the same subspace (does the subspace better fit alg or heur?)
other = "y_heur" if lab == "y_alg" else "y_alg"
ev([e for e in eval_ex if e[other] is not None], real, other, f"iia_other_labels({other})")
for a in (0.25, 0.5, 0.75):
    ev(eval_ex, real, lab, f"iia_all_alpha{a}", alpha=a)
out = root / "runs/E2.1" / args.model; out.mkdir(parents=True, exist_ok=True)
tag = args.tag or f"{args.variable}_{args.mode}_L{args.layer}_{args.position}_k{args.rank}_s{args.seed}"
json.dump(res, open(out / f"das_{tag}.json", "w"), indent=1)
print("wrote", out / f"das_{tag}.json")
