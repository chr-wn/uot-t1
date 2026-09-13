#!/bin/bash
# Print indices of GPUs on this node with < 1 GiB used, respecting the shared-cluster rule.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1024{print $1}' | tr '\n' ',' | sed 's/,$//'
