"""E1.4 — cache residuals on T1v2 prompts at the pre-answer position (`last` = the output number's
last token) and at each input quantity's unit token (`u0_unit`, `u1_unit`), for probe transfer.
Writes cache/<model>/T1v2_s<seed>.npz with the same layout as the mention caches."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl
from uot.lm.models import load_model
from uot.cache import cache_residuals
from uot.quantity import fmt_value

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--stimuli", nargs="+", required=True)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "cache"))
ap.add_argument("--layers", default=None)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--device-map", default=None)
args = ap.parse_args()

spec, tok, model = load_model(args.model, device=args.device, device_map=args.device_map)
nl = model.config.num_hidden_layers
layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(set(list(range(0, nl + 1, 2)) + [nl]))
for sp in args.stimuli:
    items = items_from_jsonl(sp)
    outp = Path(args.out) / args.model / (Path(sp).stem + ".npz")
    if outp.exists():
        print("exists", outp); continue
    prompts, spans, keep = [], [], []
    for it in items:
        p = it.prompt
        sp_d = {"last": (len(p) - 1, len(p))}
        ok = True
        # locate each input quantity "<value> <unit rendering>" in the prompt
        vals = it.meta["values"]
        pos = 0
        for k, v in enumerate(vals):
            m = re.search(rf"(?<![\w.]){re.escape(fmt_value(v))}\s+", p[pos:])
            if not m:
                ok = False; break
            start = pos + m.end()
            # unit rendering ends before the next space-delimited stop word / punctuation; use the
            # slot rendering length via the candidate-style rendering stored in meta when possible
            m2 = re.match(r"[^,;.]+?(?=(,|;|\.|\s+(in|over|by|to|through|on|with|and|has|is|gives|equals|of|at|for|so|the|its|carrying|carries|acts|pushes|does|from|multiplied|divided|equals)\b))", p[start:])
            if not m2:
                ok = False; break
            end = start + len(m2.group(0).rstrip())
            sp_d[f"u{k}_unit"] = (start, end)
            pos = end
        if ok and len(vals) == 1:
            sp_d["u1_unit"] = sp_d["u0_unit"]
        if ok:
            prompts.append(p); spans.append(sp_d); keep.append(it.item_id)
    print(f"{sp}: {len(keep)}/{len(items)} items with located spans", flush=True)
    cache_residuals(model, tok, prompts, spans, layers=layers, position_names=["u0_unit", "u1_unit", "last"],
                    out_path=outp, batch_size=args.batch, device=args.device,
                    meta=dict(model=args.model, stimuli=str(sp), item_ids=keep))
    print("wrote", outp)
