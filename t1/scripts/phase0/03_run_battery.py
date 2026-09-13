"""E0.2/E0.3 — run the behavioural battery for one model over stimulus files.

Usage: python 03_run_battery.py --model qwen3-4b-base --stimuli data/phase0/T1_s0.jsonl [...] --out runs/E0.3
Writes <out>/<model>/<stimfile>.results.jsonl with per-item candidate scores and (optionally) generations.
Resumable: skips stimulus files whose results exist.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uot.tasks import items_from_jsonl  # noqa: E402
from uot.lm.models import load_model, wrap_chat  # noqa: E402
from uot.lm.scoring import score_candidates, generate_greedy, generation_matches  # noqa: E402

INSTRUCT_SUFFIX = {
    "T1": "\nAnswer with only the unit.",
    "T2": "", "T4": "",  # prompts already say 'Answer yes or no.'
    "T3": "\nAnswer with only the letter.",
    "T5": "\nAnswer with only the letter.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--stimuli", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--generate", action="store_true", help="also greedy-generate (T1 only)")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--device-map", default=None, help="e.g. 'auto' for multi-GPU")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

out_dir = Path(args.out) / args.model
out_dir.mkdir(parents=True, exist_ok=True)
todo = [p for p in args.stimuli if not (out_dir / (Path(p).stem + ".results.jsonl")).exists()]
if not todo:
    print("nothing to do"); sys.exit(0)

spec, tok, model = load_model(args.model, device=args.device, device_map=args.device_map)
device = args.device
t0 = time.time()
for sp in todo:
    items = items_from_jsonl(sp)
    if args.limit:
        items = items[: args.limit]
    task = items[0].task
    if spec.kind == "instruct":
        prompts = [wrap_chat(spec, tok, it.prompt, INSTRUCT_SUFFIX.get(task, "")) for it in items]
        # after the chat template the assistant turn is open; candidates keep their leading space for
        # T1 (unit) and use bare tokens for yes/no/letters, both variants scored
        cands = [[c if task == "T1" else c.strip() for c in it.candidates] for it in items]
        add_bos = False
    else:
        prompts = [it.prompt for it in items]
        cands = [it.candidates for it in items]
        add_bos = True
    scores = score_candidates(model, tok, prompts, cands, batch_size=args.batch, add_bos=add_bos, device=device)
    gens = None
    if args.generate and task == "T1":
        gens = generate_greedy(model, tok, prompts, max_new_tokens=12, batch_size=args.batch, add_bos=add_bos, device=device)
    with open(out_dir / (Path(sp).stem + ".results.jsonl"), "w") as f:
        for i, it in enumerate(items):
            mean = [s["mean_logprob"] for s in scores[i]]
            summ = [s["sum_logprob"] for s in scores[i]]
            rec = dict(item_id=it.item_id, task=it.task, condition=it.condition, template_id=it.template_id,
                       answer_index=it.answer_index, candidate_roles=it.candidate_roles, scores=scores[i],
                       pred_mean=int(max(range(len(mean)), key=mean.__getitem__)),
                       pred_sum=int(max(range(len(summ)), key=summ.__getitem__)),
                       lattice_distance=it.lattice_distance, named_point=it.named_point,
                       answer_dimension=it.answer_dimension, meta=it.meta)
            rec["correct_mean"] = rec["pred_mean"] == it.answer_index
            rec["correct_sum"] = rec["pred_sum"] == it.answer_index
            if gens is not None:
                rec["generation"] = gens[i]
                rec["correct_gen"] = generation_matches(gens[i], it.unit_strings)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    acc = sum(1 for i, it in enumerate(items) if max(range(4 if task in ("T1", "T3") else len(it.candidates)),
              key=[s["mean_logprob"] for s in scores[i]].__getitem__) == it.answer_index) / len(items)
    print(f"{args.model} {Path(sp).name}: n={len(items)} acc_mean={acc:.3f} elapsed={time.time() - t0:.0f}s", flush=True)
