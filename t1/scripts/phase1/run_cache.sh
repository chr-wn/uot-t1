#!/bin/bash
# Cache residuals for Phase-1 mention stimuli. Usage: run_cache.sh <gpu> model [model...]
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  echo "=== cache $m $(date)"
  DM=""; [ "$m" = "olmo3-32b" ] && DM="--device-map auto"
  $PY scripts/phase1/02_cache_residuals.py --model $m --stimuli data/phase1/${STIM:-P1}_s0.jsonl data/phase1/${STIM:-P1}_s1.jsonl --batch 16 $DM
done
echo "=== done cache $(date)"
