"""Per-point / per-operation IIA from per-example records (graded test): balanced IIA relative to the
model's base prediction, split by the implanted dimension (base L/M/T vs derived) and by operation."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[2]
files = sys.argv[1:] or glob.glob(str(root / "runs/E2.1/qwen3-4b-base/das_*_s[123].json"))
BASE = {"L", "M", "T"}
def bal(recs, key):
    chg = [r for r in recs if (0 if r[key] == 1 else 1) != r["base_pred"]]; prs = [r for r in recs if (0 if r[key] == 1 else 1) == r["base_pred"]]
    a = np.mean([r["pred"] == (0 if r[key] == 1 else 1) for r in chg]) if chg else np.nan; b = np.mean([r["pred"] == (0 if r[key] == 1 else 1) for r in prs]) if prs else np.nan
    return np.nanmean([a, b]), len(chg), len(prs)
for f in sorted(files):
    d = json.load(open(f)); recs = d.get("per_example", [])
    if not recs: continue
    key = "y_alg" if d["variable"] == "alg" else "y_heur"; recs = [r for r in recs if r.get(key) is not None]
    print(Path(f).stem, f"n={len(recs)}")
    for name, grp in (("implanted base dim", [r for r in recs if r["d1_src"] in BASE]), ("implanted derived dim", [r for r in recs if r["d1_src"] not in BASE]),
                      ("base→base (both base)", [r for r in recs if r["d1_src"] in BASE and r["d2"] in BASE]), ("derived→derived", [r for r in recs if r["d1_src"] not in BASE and r["d2"] not in BASE])):
        v, nc, npr = bal(grp, key); print(f"   {name:24s} balanced IIA={v:.3f} (changing n={nc}, preserving n={npr})")
    for op in sorted({r["op"] for r in recs}):
        v, nc, npr = bal([r for r in recs if r["op"] == op], key); print(f"   op={op:8s} balanced IIA={v:.3f} (n={nc}/{npr})")
    byd = defaultdict(list)
    for r in recs: byd[r["d1_src"]].append(r)
    print("   per implanted dimension:", {k: round(bal(v, key)[0], 2) for k, v in sorted(byd.items())})
