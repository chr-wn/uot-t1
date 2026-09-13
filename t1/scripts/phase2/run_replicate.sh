#!/bin/bash
# E2.4: pair table + interchange set + ceiling + rank-64 DAS (alg/heur/random) for other models on one GPU.
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1; source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python; export CUDA_VISIBLE_DEVICES=$GPU
for M in "$@"; do
  echo "=== replicate $M $(date)"
  $PY scripts/phase2/01_pair_table.py --model $M 2>&1 | grep -E "median|Traceback"
  $PY scripts/phase2/02_build_interchange.py --model $M --seed 0 2>&1 | tail -1
  for L in 4 8; do for V in alg heur; do
    $PY scripts/phase2/03_das.py --model $M --layer $L --position u1 --variable $V --mode full --rank 0 --tag ${V}_full_L${L}_u1 2>&1 | grep -E "iia_all|Traceback"
    $PY scripts/phase2/03_das.py --model $M --layer $L --position u1 --variable $V --mode learned --rank 64 --steps 1000 --lr 5e-3 2>&1 | grep -E "iia_all|iia_disagree|iia_heldout|iia_invented|Traceback"
  done
  $PY scripts/phase2/03_das.py --model $M --layer $L --position u1 --variable alg --mode random --rank 64 2>&1 | grep -E "iia_all|Traceback"
  done
done
echo "=== done replicate $(date)"
