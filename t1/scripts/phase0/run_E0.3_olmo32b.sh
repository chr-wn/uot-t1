#!/bin/bash
# E0.3 for Olmo-3-32B on two A6000s (bf16 ~64 GB) with device_map=auto. Usage: run_E0.3_olmo32b.sh <gpuA,gpuB>
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$1
echo "=== olmo3-32b on GPUs $1 $(date)"
STIM="data/phase0/T1_s0.jsonl data/phase0/T1_s1.jsonl data/phase0/T2_s0.jsonl data/phase0/T3_s0.jsonl data/phase0/T4_s0.jsonl data/phase0/T5_s0.jsonl"
$PY scripts/phase0/03_run_battery.py --model olmo3-32b --stimuli $STIM --out runs/E0.3 --batch 8 --generate --device-map auto
$PY scripts/phase0/03_run_battery.py --model olmo3-32b --stimuli data/phase0/controls/T1_s0.noctx.jsonl data/phase0/controls/T1_s0.nounits.jsonl --out runs/E0.3b --batch 8 --device-map auto
echo "=== done olmo3-32b $(date)"
