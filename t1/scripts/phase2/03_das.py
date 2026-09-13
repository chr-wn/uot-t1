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
ap.add_argument("--calibrate", action="store_true", help="bias-corrected readout: yes iff (yes-no logit) > median over clean base prompts (for models with a constant yes/no bias)")
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


CAL_THR = {"thr": 0.0}


def decide(a):
    """a: [b,2] logits over (yes,no) -> 0/1 predictions, optionally bias-corrected."""
    m = a[:, 0] - a[:, 1]
    return (m <= CAL_THR["thr"]).long() if args.calibrate else a.argmax(1)


@torch.no_grad()
def base_predictions(items_map, ids_list):
    """Model's own yes/no prediction (0=yes, 1=no) per base item, no intervention."""
    preds = {}
    for b in range(0, len(ids_list), args.batch):
        chunk = ids_list[b:b + args.batch]; encs = [enc(items_map[i]) for i in chunk]
        L = max(len(x[0]) for x in encs)
        ids = torch.full((len(encs), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(encs), L), dtype=torch.long)
        for r, (sq, _) in enumerate(encs): ids[r, :len(sq)] = torch.tensor(sq); m[r, :len(sq)] = 1
        lg = model(input_ids=ids.to(device), attention_mask=m.to(device)).logits.float()
        a = answer_logits(lg, torch.tensor([x[1]["last"] for x in encs]).to(device), ans_ids.to(device))
        if args.calibrate and "margins" in CAL_THR: CAL_THR["margins"] += (a[:, 0] - a[:, 1]).tolist()
        for i, p in zip(chunk, decide(a).tolist()): preds[i] = p
    return preds
train_ex = [e for e in EX["real"] if e["split"] == "train" and e[lab] is not None]
eval_ex = [e for e in EX["real"] if e["split"] == "eval" and e[lab] is not None]
train_b = make_batches(train_ex, real, lab, args.batch, True, args.seed)
t0 = time.time()
basis = torch.tensor(np.load(args.probe_basis)) if args.mode == "probe" else None
site, log = train_das(model, tok, args.layer, args.rank, train_b, steps=args.steps if args.mode == "learned" else 0, lr=args.lr, seed=args.seed,
                      mode=("fixed" if args.mode == "probe" else args.mode), basis=basis, device=device)
if args.mode == "full":
    args.rank = model.config.hidden_size
res = dict(model=args.model, layer=args.layer, position=args.position, variable=args.variable, mode=args.mode, rank=args.rank, steps=args.steps, calibrated=bool(args.calibrate),
           train_time=time.time() - t0, final_train_acc=float(np.mean([l["acc"] for l in log[-20:]])) if log else None)
if args.calibrate:
    CAL_THR["margins"] = []; base_predictions(real, sorted({e["base"] for e in EX["real"]}))
    CAL_THR["thr"] = float(np.median(CAL_THR["margins"])); del CAL_THR["margins"]; print("calibrated threshold", CAL_THR["thr"], flush=True)
bp_real = base_predictions(real, sorted({e["base"] for e in EX["real"]})); bp_inv = base_predictions(inv, sorted({e["base"] for e in EX["inv"]}))
res["base_acc_real"] = float(np.mean([bp_real[i] == (0 if real[i].same_dim else 1) for i in bp_real]))
res["base_yes_rate_real"] = float(np.mean([bp_real[i] == 0 for i in bp_real]))


per_example = []


@torch.no_grad()
def predictions(examples, items_map, label_key, alpha=1.0):
    """Per-example intervened prediction (0=yes,1=no) for the record."""
    site.alpha = alpha; ivx = Intervener(model, args.layer, site); preds = []
    for b in make_batches(examples, items_map, label_key, args.batch, False):
        src = ivx.capture(b["src_ids"].to(device), b["src_mask"].to(device), b["src_pos"].to(device))
        out = ivx.intervene(b["base_ids"].to(device), b["base_mask"].to(device), b["base_pos"].to(device), src)
        preds += decide(answer_logits(out.logits.float(), b["base_last"].to(device), b["answer_token_ids"].to(device))).tolist()
    ivx.remove(); site.alpha = 1.0
    return preds


def ev(examples, items_map, label_key, name, alpha=1.0):
    """Balanced IIA: mean of IIA on output-changing (cf label != model's base prediction) and
    output-preserving examples; also the raw IIA and the flip rate on changing examples."""
    if not examples: return
    bp = bp_real if items_map is real else bp_inv
    if name == "iia_all" and alpha == 1.0:
        for e, p in zip(examples, predictions(examples, items_map, label_key)):
            b, s_ = items_map[e["base"]], items_map[e["src"]]
            per_example.append(dict(base=e["base"], src=e["src"], d1_base=b.d1, d1_src=s_.d1, d2=b.d2, u1_base=b.u1, u1_src=s_.u1, op=b.op,
                                    y_alg=e["y_alg"], y_heur=e.get("y_heur"), base_pred=int(bp[e["base"]]), pred=int(p)))
    chg = [e for e in examples if (0 if e[label_key] == 1 else 1) != bp[e["base"]]]
    prs = [e for e in examples if (0 if e[label_key] == 1 else 1) == bp[e["base"]]]
    out = {}
    for sub, nm in ((chg, "changing"), (prs, "preserving")):
        if sub:
            r = eval_iia(model, args.layer, site, make_batches(sub, items_map, label_key, args.batch, False), device=device, alpha=alpha, decide=decide)
            out[nm] = dict(iia=r["iia"], n=r["n"])
    out["balanced_iia"] = float(np.mean([out[k]["iia"] for k in ("changing", "preserving") if k in out]))
    res[name] = out
    print(f"  {name:28s} balanced IIA={out['balanced_iia']:.3f}  changing={out.get('changing', {}).get('iia', float('nan')):.3f} (n={out.get('changing', {}).get('n', 0)})  preserving={out.get('preserving', {}).get('iia', float('nan')):.3f} (n={out.get('preserving', {}).get('n', 0)})", flush=True)
ev(eval_ex, real, lab, "iia_all")
ev([e for e in eval_ex if e.get("agree") == 0], real, lab, "iia_disagree")
ev([e for e in eval_ex if e["same_dim_swap"] == 1], real, lab, "iia_same_dim_swap")
ev([e for e in eval_ex if e["heldout_lex"] == 1], real, lab, "iia_heldout_lex")
# invented units: alg labels only
ev([e for e in EX["inv"]], inv, "y_alg", "iia_invented_alg")
# the *other* model's labels on the same subspace (does the subspace better fit alg or heur?)
other = "y_heur" if lab == "y_alg" else "y_alg"
ev([e for e in eval_ex if e[other] is not None], real, other, f"iia_other_labels({other})")
for a in (0.25, 0.5, 0.75):
    ev(eval_ex, real, lab, f"iia_all_alpha{a}", alpha=a)
out = root / "runs/E2.1" / args.model; out.mkdir(parents=True, exist_ok=True)
tag = args.tag or f"{args.variable}_{args.mode}_L{args.layer}_{args.position}_k{args.rank}_s{args.seed}"
res["per_example"] = per_example
json.dump(res, open(out / f"das_{tag}.json", "w"), indent=1)
if args.mode != "full":
    np.save(out / f"basis_{tag}.npy", site.basis().detach().cpu().numpy().astype(np.float32))
print("wrote", out / f"das_{tag}.json")
