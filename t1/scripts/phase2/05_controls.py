"""E2.2 — controls for a saved DAS/probe/random subspace at (layer, position):
  natural-projection: do natural activations, projected on the subspace, separate D1 classes? (LOO-lexeme
                      logistic accuracy on the k-dim projection vs. on k random directions)
  value-preservation: a ridge probe (last token -> log v1) trained on clean base prompts, applied after the
                      D1 interchange; report R² before/after.
  binding:            a D2 logistic probe at the u2 token trained on clean prompts, applied after the D1 swap.
  noising:            implant a *mismatching* source (different dimension) into a clean base whose answer is
                      yes (same-dim pairs): fraction of answers that flip to no (vs random subspace).
  text-rewrite:       the same counterfactuals produced by editing the unit string in the prompt (behavioural
                      reference for every interchange effect).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p2_interchange import from_jsonl
from uot.lm.models import load_model
from uot.cache import spans_to_token_positions
from uot.das import SiteIntervention, Intervener, answer_logits
from uot.probes import ALPHAS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--position", default="u1")
ap.add_argument("--basis", required=True, help="npy [k,d] (learned/probe) or 'random:<k>:<seed>'"); ap.add_argument("--tag", required=True)
ap.add_argument("--set-seed", type=int, default=0); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--probe-layer-offset", type=int, default=6, help="value/binding probes read block outputs at layer+offset (after the swap has propagated)")
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]; device = "cuda"
spec, tok, model = load_model(args.model); d = model.config.hidden_size
real = {it.item_id: it for it in from_jsonl(root / f"data/phase2/P2_real_s{args.set_seed}.jsonl")}
EX = json.load(open(root / f"data/phase2/{args.model}_interchange_s{args.set_seed}.json"))
yes_id = tok(" yes", add_special_tokens=False)["input_ids"][0]; no_id = tok(" no", add_special_tokens=False)["input_ids"][0]; ans_ids = torch.tensor([yes_id, no_id])
if args.basis.startswith("random:"):
    _, k, sd = args.basis.split(":"); site = SiteIntervention(d, int(k), mode="random", seed=int(sd)).to(device); B = site.basis().cpu().numpy()
else:
    B = np.load(args.basis); site = SiteIntervention(d, B.shape[0], mode="fixed", basis=torch.tensor(B)).to(device)
k = B.shape[0]
enc = {}
def E(it):
    if it.item_id not in enc: enc[it.item_id] = spans_to_token_positions(tok, it.prompt, {kk: tuple(v) for kk, v in it.spans.items()}, add_bos=True)
    return enc[it.item_id]
def batches(items, batch):
    for b in range(0, len(items), batch):
        ch = items[b:b + batch]; es = [E(it) for it in ch]; L = max(len(x[0]) for x in es)
        ids = torch.full((len(es), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(es), L), dtype=torch.long)
        for r, (sq, _) in enumerate(es): ids[r, :len(sq)] = torch.tensor(sq); m[r, :len(sq)] = 1
        yield ch, es, ids.to(device), m.to(device)
iv = Intervener(model, args.layer, site)
LP = min(args.layer + args.probe_layer_offset, model.config.num_hidden_layers - 1)
ivp = Intervener(model, LP, None)  # capture-only hook at the probe layer (block output)
# ---- clean activations: u1 at the intervention layer (natural projection); u2/last at the probe layer
items = [real[i] for i in sorted(real)]
acts = {"u1": [], "u2": [], "last": []}; logits_clean = []
with torch.no_grad():
    for ch, es, ids, m in batches(items, args.batch):
        acts["u1"].append(iv.capture(ids, m, torch.tensor([[x[1]["u1"]] for x in es]).to(device))[:, 0].float().cpu().numpy())
    for p in ("u2", "last"):
        for ch, es, ids, m in batches(items, args.batch):
            acts[p].append(ivp.capture(ids, m, torch.tensor([[x[1][p]] for x in es]).to(device))[:, 0].float().cpu().numpy())
    for ch, es, ids, m in batches(items, args.batch):
        lg = model(input_ids=ids, attention_mask=m).logits.float(); logits_clean.append(answer_logits(lg, torch.tensor([x[1]["last"] for x in es]).to(device), ans_ids.to(device)).cpu().numpy())
acts = {p: np.concatenate(v) for p, v in acts.items()}; logits_clean = np.concatenate(logits_clean)
d1 = np.array([it.d1 for it in items]); d2 = np.array([it.d2 for it in items]); u1 = np.array([it.u1 for it in items]); v1 = np.log(np.array([it.meta["v1"] for it in items], dtype=float))
res = dict(model=args.model, layer=args.layer, position=args.position, tag=args.tag, k=k, probe_layer=LP)
# ---- natural projection: LOO-lexeme logistic on the k-dim projection vs k random dims
def loo_acc(Z, y, groups):
    pred = np.empty_like(y, dtype=object)
    for g in sorted(set(groups)):
        te = groups == g; tr = ~te
        if len(set(y[tr])) < 2: pred[te] = y[tr][0]; continue
        sc = StandardScaler().fit(Z[tr]); clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Z[tr]), y[tr]); pred[te] = clf.predict(sc.transform(Z[te]))
    return float((pred == y).mean())
X1 = acts[args.position]
proj = X1 @ B.T; rnd = X1 @ np.linalg.qr(np.random.default_rng(1).standard_normal((d, k)))[0]
res["natural_projection_loo_acc"] = loo_acc(proj, d1, u1); res["random_projection_loo_acc"] = loo_acc(rnd, d1, u1)
res["d1_majority"] = float(max(np.mean(d1 == c) for c in set(d1)))
print(f"natural projection LOO-lexeme D1 acc: subspace={res['natural_projection_loo_acc']:.3f} random={res['random_projection_loo_acc']:.3f} majority={res['d1_majority']:.3f}", flush=True)
# ---- probes on clean activations: value (last -> log v1), D2 (u2 -> d2)
sc_v = StandardScaler().fit(acts["last"]); vprobe = RidgeCV(alphas=ALPHAS).fit(sc_v.transform(acts["last"]), v1)
sc_2 = StandardScaler().fit(acts["u2"]); d2probe = LogisticRegression(C=0.5, max_iter=2000).fit(sc_2.transform(acts["u2"]), d2)
# ---- intervened passes on eval examples: capture last/u2 activations after the D1 swap
ex = [e for e in EX["real"] if e["split"] == "eval"]
idx = {it.item_id: i for i, it in enumerate(items)}
after_last, after_u2, after_logits, flips = [], [], [], []
with torch.no_grad():
    for b in range(0, len(ex), args.batch):
        ch = ex[b:b + args.batch]; bi = [E(real[e["base"]]) for e in ch]; si = [E(real[e["src"]]) for e in ch]
        L = max(max(len(x[0]) for x in bi), max(len(x[0]) for x in si))
        def pad(seqs):
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, (sq, _) in enumerate(seqs): ids[r, :len(sq)] = torch.tensor(sq); m[r, :len(sq)] = 1
            return ids.to(device), m.to(device)
        bids, bm = pad(bi); sids, sm = pad(si)
        src = iv.capture(sids, sm, torch.tensor([[x[1][args.position]] for x in si]).to(device))
        HS = __import__("uot.das", fromlist=["HookState"]).HookState
        # intervene at (layer, u1) and capture, in the same pass, the probe-layer block outputs at last and u2
        for p, store in (("last", after_last), ("u2", after_u2)):
            iv.state = HS(positions=torch.tensor([[x[1][args.position]] for x in bi]).to(device), src=src)
            ivp.state = HS(positions=torch.tensor([[x[1][p]] for x in bi]).to(device), capture=[])
            out = model(input_ids=bids, attention_mask=bm)
            store += [a[0].float().cpu().numpy() for a in ivp.state.capture[0]]
            iv.state = HS(); ivp.state = HS()
        after_logits.append(answer_logits(out.logits.float(), torch.tensor([x[1]["last"] for x in bi]).to(device), ans_ids.to(device)).cpu().numpy())
after_last = np.array(after_last); after_u2 = np.array(after_u2); after_logits = np.concatenate(after_logits)
bidx = np.array([idx[e["base"]] for e in ex])
# value preservation: R² of the value probe on intervened vs clean activations of the same bases
def r2(p, y): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res["value_r2_clean"] = r2(vprobe.predict(sc_v.transform(acts["last"][bidx])), v1[bidx]); res["value_r2_after_swap"] = r2(vprobe.predict(sc_v.transform(after_last)), v1[bidx])
# binding: D2 readout at u2 after the D1 swap
res["d2_acc_clean"] = float((d2probe.predict(sc_2.transform(acts["u2"][bidx])) == d2[bidx]).mean()); res["d2_acc_after_swap"] = float((d2probe.predict(sc_2.transform(after_u2)) == d2[bidx]).mean())
# noising: bases with same-dim (clean answer yes) receiving a different-dim source -> flip to no?
clean_pred = logits_clean[bidx].argmax(1); after_pred = after_logits.argmax(1)
same = np.array([real[e["base"]].same_dim for e in ex]); mism = np.array([real[e["src"]].d1 != real[e["base"]].d2 for e in ex])
sel = same & mism & (clean_pred == 0)
res["noising_flip_rate"] = float((after_pred[sel] == 1).mean()) if sel.any() else None; res["noising_n"] = int(sel.sum())
# text rewrite baseline: replace u1 string by the source's u1 string in the base prompt, score yes/no
tr_prompts, tr_labels = [], []
for e in ex:
    b, s = real[e["base"]], real[e["src"]]
    a0, a1 = b.spans["u1"]; tr_prompts.append(b.prompt[:a0] + s.meta["u1_str"] + b.prompt[a1:]); tr_labels.append(0 if e["y_alg"] == 1 else 1)
from uot.lm.scoring import score_candidates
sc_tr = score_candidates(model, tok, tr_prompts, [[" yes", " no"]] * len(tr_prompts), batch_size=args.batch)
tr_pred = np.array([0 if s_[0]["mean_logprob"] > s_[1]["mean_logprob"] else 1 for s_ in sc_tr]); tr_labels = np.array(tr_labels)
chg = tr_labels != clean_pred; res["text_rewrite_iia_changing"] = float((tr_pred[chg] == tr_labels[chg]).mean()); res["text_rewrite_iia_preserving"] = float((tr_pred[~chg] == tr_labels[~chg]).mean())
res["subspace_iia_changing_alg"] = float((after_pred[chg] == tr_labels[chg]).mean()); res["subspace_iia_preserving_alg"] = float((after_pred[~chg] == tr_labels[~chg]).mean())
print(json.dumps({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in res.items()}, indent=0), flush=True)
out = root / "runs/E2.2" / args.model; out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out / f"controls_{args.tag}.json", "w"), indent=1)
iv.remove(); ivp.remove()
