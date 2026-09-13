#!/bin/bash
# Prior-only controls (E0.3b) for a list of models on one GPU. Usage: run_E0.3b_queue.sh <gpu> model1 model2 ...
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  echo "=== E0.3b $m $(date)"
  $PY scripts/phase0/03_run_battery.py --model $m --stimuli data/phase0/controls/T1_s0.noctx.jsonl data/phase0/controls/T1_s0.nounits.jsonl data/phase0/controls/T1_s1.noctx.jsonl data/phase0/controls/T1_s1.nounits.jsonl --out runs/E0.3b --batch 16
done
echo "=== done E0.3b queue $(date)"
