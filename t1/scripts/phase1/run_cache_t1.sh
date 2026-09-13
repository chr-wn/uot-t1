#!/bin/bash
# Cache T1v2 (+ arith/int) prompts at value/unit/pre-answer positions. Usage: run_cache_t1.sh <gpu> model...
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  echo "=== cache-t1 $m $(date)"
  $PY scripts/phase1/04_cache_t1_positions.py --model $m --stimuli data/phase0/T1v2_s0.jsonl data/phase0/T1arith_s0.jsonl --batch 16
done
echo "=== done cache-t1 $(date)"
