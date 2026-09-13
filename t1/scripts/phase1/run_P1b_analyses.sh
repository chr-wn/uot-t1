#!/bin/bash
# LOO-lexeme (E1.2c) + semantic geometry (E1.3) on the P1b caches for a list of models, both seeds. CPU.
cd /data/rbg/users/charlie/interp/uot/t1
PY=/data/rbg/users/charlie/interp/uot/t1/env/bin/python
export OMP_NUM_THREADS=12
for m in "$@"; do for s in 0 1; do
  echo "=== $m s$s $(date)"
  $PY scripts/phase1/09_named_loo.py --cache cache/$m/P1b_s$s.npz --stimuli data/phase1/P1b_s$s.jsonl --out runs/E1.2c --sites unit:8,unit:16,unit:28,mention_end:12,mention_end:20,anaphor:12,anaphor:20,last:20,last:-1 2>&1 | grep -v Warning
  $PY scripts/phase1/06_geometry.py --cache cache/$m/P1b_s$s.npz --stimuli data/phase1/P1b_s$s.jsonl --out runs/E1.3 --conditions REAL-BASE,REAL-NAMED --tag semb --sites unit:16,mention_end:12,anaphor:12,last:-1 2>&1 | grep -v Warning | cut -c1-200
done; done
echo "=== done P1b analyses $(date)"
