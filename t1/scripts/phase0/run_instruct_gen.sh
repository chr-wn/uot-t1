#!/bin/bash
# Instruct models re-run with generation readout for all tasks (A6). Usage: run_instruct_gen.sh <gpu> model...
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
STIM="data/phase0/T1v2_s0.jsonl data/phase0/T1v2_s1.jsonl data/phase0/T2_s0.jsonl data/phase0/T3_s0.jsonl data/phase0/T4_s0.jsonl data/phase0/T5_s0.jsonl"
for m in "$@"; do
  echo "=== instruct-gen $m $(date)"
  $PY scripts/phase0/03_run_battery.py --model $m --stimuli $STIM --out runs/E0.3i --batch 16 --generate
done
echo "=== done instruct-gen $(date)"
