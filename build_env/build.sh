# chdir to the directory of this script
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
conda env create -f ./env.yaml
source activate swift
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.10.2
pip install -r ./requirements.txt
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl  
pip install -e ../ms-swift/