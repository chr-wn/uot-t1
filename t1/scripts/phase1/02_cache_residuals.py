"""Cache residual streams at the Phase-1 position classes for one model and one stimulus file.

Output: cache/<model>/<stimfile>.npz  (float16 [n, n_pos, n_layers, d]).
Layers default to every other layer plus the last.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks.p1_mentions import from_jsonl  # noqa: E402
from uot.lm.models import load_model  # noqa: E402
from uot.cache import cache_residuals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--stimuli", nargs="+", required=True)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "cache"))
ap.add_argument("--layers", default=None, help="comma list; default every 2nd layer + last")
ap.add_argument("--positions", default="value,unit,mention_end,anaphor,last")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--device-map", default=None)
args = ap.parse_args()

spec, tok, model = load_model(args.model, device=args.device, device_map=args.device_map)
nl = model.config.num_hidden_layers
layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(set(list(range(0, nl + 1, 2)) + [nl]))
positions = args.positions.split(",")
for sp in args.stimuli:
    items = from_jsonl(sp)
    outp = Path(args.out) / args.model / (Path(sp).stem + ".npz")
    if outp.exists():
        print("exists", outp); continue
    prompts = [it.prompt for it in items]
    spans = [{k: tuple(v) for k, v in it.spans.items()} for it in items]
    cache_residuals(model, tok, prompts, spans, layers=layers, position_names=positions, out_path=outp,
                    batch_size=args.batch, device=args.device,
                    meta=dict(model=args.model, stimuli=str(sp), item_ids=[it.item_id for it in items]))
    print("wrote", outp, "layers", layers)
