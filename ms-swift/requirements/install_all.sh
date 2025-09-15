# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install "sglang[all]<0.5" -U
pip install "vllm>=0.5.1" "transformers<4.56" "trl<0.21" -U
pip install "lmdeploy>=0.5" -U
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git
pip install timm -U
pip install "deepspeed" -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases


pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# xformers 0.0.32.post1 requires torch==2.8.0, but you have torch 2.7.1+cu126 which is incompatible.
# vllm 0.10.2 requires outlines_core==0.2.11, but you have outlines-core 0.1.26 which is incompatible.
# vllm 0.10.2 requires torch==2.8.0, but you have torch 2.7.1+cu126 which is incompatible.
# vllm 0.10.2 requires torchaudio==2.8.0, but you have torchaudio 2.7.1+cu126 which is incompatible.
# vllm 0.10.2 requires torchvision==0.23.0, but you have torchvision 0.22.1+cu126 which is incompatible.
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# vllm 0.10.2 requires outlines_core==0.2.11, but you have outlines-core 0.1.26 which is incompatible.
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# outlines 0.1.11 requires outlines_core==0.1.26, but you have outlines-core 0.2.11 which is incompatible.