#!/bin/bash
# Full-residual patching ceiling at each site (no training). Usage: run_das_ceiling.sh <gpu> <model> "<layers>" "<positions>"
GPU=$1; M=$2; LAYERS=$3; POSS=$4
cd /data/rbg/users/charlie/interp/uot/t1; source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python; export CUDA_VISIBLE_DEVICES=$GPU
for L in $LAYERS; do for P in $POSS; do for V in alg heur; do
  $PY scripts/phase2/03_das.py --model $M --layer $L --position $P --variable $V --mode full --rank 0 --tag ${V}_full_L${L}_${P} 2>&1 | grep -E "IIA|wrote|Traceback|Error"
done; done; done
echo "=== done ceiling gpu$GPU $(date)"
