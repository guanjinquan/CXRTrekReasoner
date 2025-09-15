CUDA_VISIBLE_DEVICES=7 swift infer \
    --model /remote-home/xiaowu/Guanjq/MTXray/checkpoints/Qwen/Qwen2-VL-7B-Instruct \
    --stream true \
    --infer_backend vllm  \
    --max_new_tokens 2048