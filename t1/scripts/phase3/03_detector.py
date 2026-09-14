"""E3.2b — dimension-inconsistency detector on generated solutions (clean vs corrupted final unit).
Detectors: (probe) Phase-1 categorical logistic probe on the residual at the final unit token
(hidden_states[layer], P1b cache convention) → P(expected dimension); (selfcheck) append
"Is the unit of the final answer consistent with the quantity asked for? Answer yes or no.\nAnswer:" and use
the calibrated yes–no margin; (pint) parse the final unit with the dimension parser and compare.
Reports AUROC for each on FAM-NAMED and INV-LEX subsets."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl
from uot.cache import load_cache, spans_to_token_positions
from uot.lm.models import load_model
from uot.lm.scoring import score_candidates
from uot.parse_units import parse_unit_expression, invented_units_from_prompt
from uot.dims import Dimension

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3-4b-base"); ap.add_argument("--layer", type=int, default=12); ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args(); root = Path(__file__).resolve().parents[2]; device = "cuda"
# train the categorical probe on P1b (semantic set) at hidden_states[layer], unit position
C = load_cache(root / "cache" / args.model / "P1b_s0.npz"); items = from_jsonl(root / "data/phase1/P1b_s0.jsonl")
V = np.array([it.vector for it in items]); inlat = (np.abs(V[:, 3:]).sum(1) == 0)
sel = np.array([it.condition in ("REAL-BASE", "REAL-NAMED") and it.family in ("neutral", "revealing") for it in items]) & inlat
X = C["resid"][sel, C["position_names"].index("unit"), C["layers"].index(args.layer)].astype(np.float32); Y = [str(Dimension(dict(zip(("L", "M", "T"), v[:3])))) for v in V[sel]]
mu = X.mean(0); pca = PCA(n_components=256, random_state=0).fit(X - mu); sc = StandardScaler().fit(pca.transform(X - mu))
clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(pca.transform(X - mu)), Y); classes = list(clf.classes_)
recs = [json.loads(l) for l in open(root / f"data/phase3/cot_{args.model}.jsonl")]
(root / "runs/E3.2").mkdir(parents=True, exist_ok=True)
spec, tok, model = load_model(args.model)
def probe_scores(texts, spans, expected):
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), args.batch):
            ch = list(zip(texts[b:b + args.batch], spans[b:b + args.batch], expected[b:b + args.batch])); encs = [spans_to_token_positions(tok, t, {"u": tuple(s)}, add_bos=True) for t, s, _ in ch]
            L = max(len(e[0]) for e in encs); ids = torch.full((len(encs), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(encs), L), dtype=torch.long)
            for r, (sq, _) in enumerate(encs): ids[r, :len(sq)] = torch.tensor(sq); m[r, :len(sq)] = 1
            hs = model(input_ids=ids.to(device), attention_mask=m.to(device), output_hidden_states=True).hidden_states[args.layer]
            A = np.stack([hs[r, e[1]["u"]].float().cpu().numpy() for r, e in enumerate(encs)])
            P = clf.predict_proba(sc.transform(pca.transform(A - mu)))
            for (t, s, ex), p in zip(ch, P): out.append(float(p[classes.index(ex)]) if ex in classes else float("nan"))
    return out
texts, spans, exp, labels, conds, pint_ok = [], [], [], [], [], []
for r in recs:
    ex = str(Dimension.parse(r["expected_dim"]))
    for ver, lab in (("clean", 1), ("corrupted", 0)):
        texts.append(r[ver]["text"]); spans.append(r[ver]["unit_span"]); exp.append(ex); labels.append(lab); conds.append(r["condition"])
        ustr = r[ver]["text"][r[ver]["unit_span"][0]: r[ver]["unit_span"][1]]; d = parse_unit_expression(ustr, invented_units_from_prompt(r[ver]["text"]))
        pint_ok.append(1.0 if (d is not None and d == Dimension.parse(r["expected_dim"])) else (0.0 if d is not None else 0.5))
ps = probe_scores(texts, spans, exp)
sc_prompts = [t + "\nIs the unit of the final answer consistent with the quantity asked for? Answer yes or no.\nAnswer:" for t in texts]
sc_ = score_candidates(model, tok, sc_prompts, [[" yes", " no"]] * len(sc_prompts), batch_size=args.batch); selfm = [s[0]["mean_logprob"] - s[1]["mean_logprob"] for s in sc_]
labels = np.array(labels); conds = np.array(conds); ps = np.array(ps); selfm = np.array(selfm); pint_ok = np.array(pint_ok)
# two labelings: (A) clean-vs-corrupted (artefact-prone: the corrupted unit is always a generic single word);
# (B) parser ground truth: consistent iff the stated final unit's dimension == expected (Pint/parser is the oracle
#     by construction; the question is how close the probe and the self-check get to it), over parseable items.
res = {}
parsed = pint_ok != 0.5; truth = (pint_ok == 1.0).astype(int)
for cond in ("FAM-NAMED", "INV-LEX"):
    m = (conds == cond) & ~np.isnan(ps)
    if m.sum() < 10 or len(set(labels[m])) < 2: continue
    r = dict(n=int(m.sum()), A_probe=float(roc_auc_score(labels[m], ps[m])), A_selfcheck=float(roc_auc_score(labels[m], selfm[m])), A_pint=float(roc_auc_score(labels[m], pint_ok[m])))
    mb = m & parsed
    if mb.sum() >= 10 and len(set(truth[mb])) == 2:
        r.update(n_parsed=int(mb.sum()), B_probe=float(roc_auc_score(truth[mb], ps[mb])), B_selfcheck=float(roc_auc_score(truth[mb], selfm[mb])), B_consistent_rate=float(truth[mb].mean()))
    res[cond] = r; print(cond, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
np.save(out_dir_items := root / "runs/E3.2" / f"items_{args.model}_L{args.layer}.npy", np.stack([labels, ps, selfm, pint_ok], 1))
out = root / "runs/E3.2"; json.dump(res, open(out / f"detector_{args.model}_L{args.layer}.json", "w"), indent=1)
