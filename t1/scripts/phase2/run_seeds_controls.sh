#!/bin/bash
# 3-seed reruns at the headline site (u1 L8, rank 64) with per-example records, then the control battery.
GPU=$1; M=${2:-qwen3-4b-base}; L=${3:-8}
cd /data/rbg/users/charlie/interp/uot/t1; source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python; export CUDA_VISIBLE_DEVICES=$GPU
for S in 1 2 3; do for V in alg heur; do
  $PY scripts/phase2/03_das.py --model $M --layer $L --position u1 --variable $V --mode learned --rank 64 --steps 1000 --lr 5e-3 --seed $S --tag ${V}_learned_L${L}_u1_k64_s$S 2>&1 | grep -E "iia_all|iia_disagree|Traceback"
done; $PY scripts/phase2/03_das.py --model $M --layer $L --position u1 --variable alg --mode random --rank 64 --seed $S --tag alg_random_L${L}_u1_k64_s$S 2>&1 | grep -E "iia_all|Traceback"; done
for S in 1 2 3; do for V in alg heur; do
  $PY scripts/phase2/05_controls.py --model $M --layer $L --position u1 --basis runs/E2.1/$M/basis_${V}_learned_L${L}_u1_k64_s$S.npy --tag ${V}_learned_L${L}_s$S 2>&1 | grep -E "natural|value|Traceback|Error" | head -3
done; $PY scripts/phase2/05_controls.py --model $M --layer $L --position u1 --basis random:64:$S --tag random64_L${L}_s$S 2>&1 | grep -E "natural|Traceback|Error" | head -2; done
echo "=== done seeds+controls $(date)"
