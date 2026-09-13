#!/bin/bash
# E1.0b: definition-by-example (arith) and integer-output free-generation cells. Usage: run_E1.0b.sh <gpu> model...
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  echo "=== E1.0b $m $(date)"
  DM=""; [ "$m" = "olmo3-32b" ] && DM="--device-map auto"
  $PY scripts/phase0/03_run_battery.py --model $m --stimuli data/phase0/T1arith_s0.jsonl data/phase0/T1int_s0.jsonl --out runs/E1.0b --batch 16 --generate $DM
done
echo "=== done E1.0b $(date)"
