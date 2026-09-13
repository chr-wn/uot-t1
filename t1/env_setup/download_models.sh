#!/bin/bash
# Pre-download model weights into $HF_HOME (set by ~/.config/ml-env.sh). Order: smallest/most-used first.
source /data/rbg/users/charlie/.config/ml-env.sh
source /data/rbg/users/charlie/miniconda3/etc/profile.d/conda.sh
conda activate /data/rbg/users/charlie/interp/uot/t1/env
for m in Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base allenai/Olmo-3-1025-7B google/gemma-2-9b Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B-Base allenai/Olmo-3-1125-32B; do
  echo "=== $m $(date)"
  hf download "$m" --exclude "*.gguf" "original/*" "*.pth" 2>&1 | tail -2
done
echo "=== done $(date)"
