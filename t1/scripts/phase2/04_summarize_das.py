"""Summarise DAS json results into one table (balanced IIA per subset)."""
import glob, json, sys
from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parents[2]
rows = []
for f in glob.glob(str(root / "runs/E2.1/*/das_*.json")):
    d = json.load(open(f)); r = dict(model=d["model"], layer=d["layer"], pos=d["position"], var=d["variable"], mode=d["mode"], rank=d["rank"],
                                     train_acc=d.get("final_train_acc"), base_acc=d.get("base_acc_real"), base_yes=d.get("base_yes_rate_real"))
    for k, v in d.items():
        if k.startswith("iia_") and isinstance(v, dict) and "balanced_iia" in v:
            r[k.replace("iia_", "")] = round(v["balanced_iia"], 3)
    rows.append(r)
df = pd.DataFrame(rows)
if len(df):
    df = df.sort_values(["model", "layer", "pos", "rank", "var", "mode"])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = [c for c in ["model", "layer", "pos", "rank", "var", "mode", "train_acc", "base_acc", "all", "disagree", "same_dim_swap", "heldout_lex", "invented_alg"] if c in df.columns] + [c for c in df.columns if c.startswith("other_labels")]
    print(df[cols].to_string(index=False))
    df.to_csv(root / "runs/E2.1/summary.csv", index=False)
