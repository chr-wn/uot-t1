#!/bin/bash
# T1 v2 (prefix-free candidates) + v2 prior controls for a list of models on one GPU.
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  echo "=== T1v2 $m $(date)"
  DM=""; [ "$m" = "olmo3-32b" ] && DM="--device-map auto"
  $PY scripts/phase0/03_run_battery.py --model $m --stimuli data/phase0/T1v2_s0.jsonl data/phase0/T1v2_s1.jsonl --out runs/E0.3 --batch 16 --generate $DM
  $PY scripts/phase0/03_run_battery.py --model $m --stimuli data/phase0/controls/T1v2_s0.noctx.jsonl data/phase0/controls/T1v2_s0.nounits.jsonl data/phase0/controls/T1v2_s1.noctx.jsonl data/phase0/controls/T1v2_s1.nounits.jsonl --out runs/E0.3b --batch 16 $DM
done
echo "=== done T1v2 queue $(date)"
