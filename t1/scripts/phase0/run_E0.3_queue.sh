#!/bin/bash
# Sequential queue of models on one GPU (rosetta18). Usage: run_E0.3_queue.sh <gpu> model1 model2 ...
GPU=$1; shift
cd /data/rbg/users/charlie/interp/uot/t1
for m in "$@"; do bash scripts/phase0/run_E0.3_gpu.sh $m $GPU; done
