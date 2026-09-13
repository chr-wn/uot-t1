#!/bin/bash
# DAS sweep on one GPU. Usage: run_das_sweep.sh <gpu> <model> "<layers>" "<positions>" "<ranks>" <steps> <lr>
GPU=$1; M=$2; LAYERS=$3; POSS=$4; RANKS=$5; STEPS=${6:-1000}; LR=${7:-5e-3}
cd /data/rbg/users/charlie/interp/uot/t1
source /data/rbg/users/charlie/.config/ml-env.sh
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
for L in $LAYERS; do for P in $POSS; do for K in $RANKS; do
  for V in alg heur; do
    $PY scripts/phase2/03_das.py --model $M --layer $L --position $P --variable $V --mode learned --rank $K --steps $STEPS --lr $LR 2>&1 | grep -E "IIA|wrote|Traceback|Error"
  done
  $PY scripts/phase2/03_das.py --model $M --layer $L --position $P --variable alg --mode random --rank $K 2>&1 | grep -E "IIA|wrote|Traceback|Error"
done; done; done
echo "=== done DAS sweep gpu$GPU $(date)"
