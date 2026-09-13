#!/bin/bash
# E1.1/E1.2 probe sweeps on CPU (rosetta18). Usage: run_probes.sh model [seed]
M=$1; SEED=${2:-0}
cd /data/rbg/users/charlie/interp/uot/t1
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export OMP_NUM_THREADS=16
OUT=runs/E1.1/$M
mkdir -p $OUT
# main: real lattice + base + named, neutral + revealing families, L/M/T axes
$PY scripts/phase1/03_probe.py --cache cache/$M/P1_s$SEED.npz --stimuli data/phase1/P1_s$SEED.jsonl --out $OUT --tag main_s$SEED \
   --conditions REAL-BASE,REAL-NAMED,REAL-LATTICE --families neutral,revealing
# neutral-only (dimension carried by the unit lexeme alone)
$PY scripts/phase1/03_probe.py --cache cache/$M/P1_s$SEED.npz --stimuli data/phase1/P1_s$SEED.jsonl --out $OUT --tag neutral_s$SEED \
   --conditions REAL-BASE,REAL-NAMED,REAL-LATTICE --families neutral
echo "=== done probes $M s$SEED $(date)"
