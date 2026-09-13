#!/bin/bash
# Creates the uot/t1 conda env. torch must be cu124 (driver 550 on rosetta4).
set -e
ENV=/data/rbg/users/charlie/interp/uot/t1/env
conda create -y -p $ENV python=3.11
source /data/rbg/users/charlie/miniconda3/etc/profile.d/conda.sh
conda activate $ENV
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.51,<5" accelerate safetensors sentencepiece protobuf
pip install nnsight pyvene pint numpy scipy scikit-learn pandas matplotlib seaborn pytest pyyaml tqdm einops jupyter ipykernel statsmodels requests
pip install huggingface_hub[cli]
python -c "import torch;print('cuda',torch.cuda.is_available(), torch.__version__)"
python -c "import transformers, nnsight, pyvene, pint; print(transformers.__version__, nnsight.__version__, pyvene.__version__, pint.__version__)"
