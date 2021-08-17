#!/usr/bin/env bash

# torch_scatter

CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda.replace('.', ''))")
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
pip3 install torch_scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# torchsearchsorted

git clone https://github.com/aliutkus/torchsearchsorted /tmp/torchsearchsorted/
cd /tmp/torchsearchsorted && python3 setup.py bdist_wheel -d .
pip3 install $(ls *.whl) && rm -rf /tmp/torchsearchsorted/
