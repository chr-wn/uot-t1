#!/bin/bash
# E0.3 full battery on GPU. Usage: run_E0.3_gpu.sh <model-key> [gpu-index]
# Picks an idle GPU (<1 GiB used) if none given; refuses to run on a busy one.
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
MODEL=$1
GPU=${2:-$(bash scripts/gpu_pick.sh | cut -d, -f1)}
if [ -z "$GPU" ]; then echo "no idle GPU"; exit 1; fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
if [ "$USED" -gt 1024 ]; then echo "GPU $GPU busy ($USED MiB)"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "=== $MODEL on GPU $GPU $(date)"
STIM="data/phase0/T1_s0.jsonl data/phase0/T1_s1.jsonl data/phase0/T2_s0.jsonl data/phase0/T3_s0.jsonl data/phase0/T4_s0.jsonl data/phase0/T5_s0.jsonl"
$PY scripts/phase0/03_run_battery.py --model $MODEL --stimuli $STIM --out runs/E0.3 --batch 16 --generate
echo "=== done $MODEL $(date)"
