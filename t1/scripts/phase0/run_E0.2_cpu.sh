#!/bin/bash
# E0.2 debug battery for Qwen3-4B-Base on CPU (GPUs on rosetta4 are occupied by other users).
cd /data/rbg/users/charlie/interp/uot/t1
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export OMP_NUM_THREADS=16
for f in T1_s0 T2_s0 T4_s0 T3_s0 T5_s0; do
  $PY scripts/phase0/03_run_battery.py --model qwen3-4b-base --stimuli data/phase0/$f.jsonl --out runs/E0.2 --device cpu --batch 8 $( [ $f = T1_s0 ] && echo --generate )
done
