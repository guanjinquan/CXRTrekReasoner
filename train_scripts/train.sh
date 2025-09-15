# Exp: https://github.com/modelscope/ms-swift/pull/5307#issuecomment-3219803922
# Before running this script, please run the following `swift rollout` script first
# This script is a example for multi-turn training with dynamic num of rollout outputs
# which means a trajectory of multi turn rollout is split into multiple data
#       see details in thinking_tips_scheduler
# NOTE: for same trajectory, the reward is supported to be the same,
#       here we use the last turn data of each trajectory to compute accuracy reward
#       see details in thinking_tips reward function

export HF_ENDPOINT="https://hf-mirror.com"

# CUDA_VISIBLE_DEVICES=2 \
# swift rollout \
#     --model /remote-home/xiaowu/Guanjq/MTXray/checkpoints/google/medgemma-4b-it \
#     --vllm_use_async_engine true \
#     --external_plugins /remote-home/xiaowu/Guanjq/CXRTrekReasoner/ms-swift/myplugins/infer_scheduler.py \
#     --multi_turn_scheduler cxrtrek_scheduler_nothink_onlyrewardstage8 \
#     --vllm_max_model_len 65536 \
#     --vllm_gpu_memory_utilization 0.8 \
#     --max_turns 200

CUDA_VISIBLE_DEVICES=4,5 NPROC_PER_NODE=2 swift rlhf \
    --rlhf_type grpo \
    --model /remote-home/xiaowu/Guanjq/MTXray/checkpoints/google/medgemma-4b-it \
    --train_type lora \
    --external_plugins /remote-home/xiaowu/Guanjq/CXRTrekReasoner/ms-swift/myplugins/reward_models.py \
    --reward_funcs cxrtrek_stage8_bertscore \
    --loss_scale all \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --dataset /remote-home/xiaowu/Guanjq/CXRTrekReasoner/dataset/train_swift_demo.json \
    --split_dataset_ratio 0 \
    --max_completion_length 16384 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 2 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 2 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions false \
    --log_entropy true \
    --importance_sampling_level sequence \
    --top_entropy_quantile 0.2 \
    --num_iterations 1 