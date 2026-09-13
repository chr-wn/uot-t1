#!/bin/bash
# Probe sweep parallelised over positions (one process per position). Usage: run_probes_par.sh model seed [tag] [conditions] [families]
M=$1; SEED=${2:-0}; TAG=${3:-main}; CONDS=${4:-REAL-BASE,REAL-NAMED,REAL-LATTICE}; FAMS=${5:-neutral,revealing}
cd /data/rbg/users/charlie/interp/uot/t1
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export OMP_NUM_THREADS=8
OUT=runs/E1.1/$M; mkdir -p $OUT
for POS in value unit mention_end anaphor last; do
  $PY scripts/phase1/03_probe.py --cache cache/$M/P1_s$SEED.npz --stimuli data/phase1/P1_s$SEED.jsonl --out $OUT \
     --tag ${TAG}_s${SEED}_$POS --positions $POS --conditions $CONDS --families $FAMS --layers ${LAYERS:-0,4,8,12,16,20,24,28,32,36} > $OUT/log_${TAG}_s${SEED}_$POS.txt 2>&1 &
done
wait
echo "=== done probes-par $M s$SEED $TAG $(date)"
