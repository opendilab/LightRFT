GROUP_METHOD=normal
N_SAMPLES=8
EPISODE=3
WARMUP=0.03
TBS=64
RBS=128
KL=0.001
LR=1e-6
MAX_LENGTH=4096
limit_mm_image_per_prompt=1  # multi-modal model
ENGINE_TP=1  # vLLM/SGLang, for 7b base model
export IGNORE_EOS=0

#############################  kwargs  ##########################

DATA_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/dataset/kg_rule_image_text_resize8192/train_val"

# --- Reward Models ---
# A JSON-formatted string specifying paths to different pretrained reward models.
# The training script uses multiple reward models for different aspects (e.g., safety, value).
REWARD_PRETRAIN_PATHS='{"safety":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/safe_orm/","value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/","knowledge":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","normal":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","general":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/"}'
# ====================================The following is only for debug====================================
ENGINE_TP=1  # vLLM/SGLang, for 7b base model
PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/base_model_after_sft_20250828" # 在sft上训练后的qwen-vl 7b


current_time=$(date +"%m%d%H%M")
LOG_BASE=log

mkdir -p $LOG_BASE

# This env may help to reduce memory usage
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN

# TODO: only for debug in 1 node

export MLP_WORKER_NUM=1
# export MLP_WORKER_GPU=6
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
# export MLP_WORKER_0_PORT=20090
export MLP_WORKER_0_PORT=20091

# export MLP_WORKER_0_HOST=10.102.207.104
export MLP_WORKER_0_HOST=localhost

###############################  volcengine env  #####################
export MASTER_ADDR=$MLP_WORKER_0_HOST
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT
###############################  volcengine env  #####################

SAVE_MODEL_NAME=lightrlhf-len_${MAX_LENGTH-}tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-rm-colocate-kg-0714
mkdir -p results/$SAVE_MODEL_NAME

set -x
export WANDB_MODE="offline" # TODO

# --- Weights & Biases (W&B) Logging ---
# It's recommended to set this as an environment variable rather than hardcoding.
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608" # Replace with your key

WANDB_PROJECT="QwenVL-7B-MultiORM-GRPO-SVKG-0908"
WANDB_RUN_NAME="QwenVL-7B-MultiORM-SVKG-grpo-${current_time}"

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/safework_t1/train_colocate.py \
   --pretrain ${PRETRAIN_PATH} \
   --fsdp \
   --use_kl_loss \
   --rm_use_engine \
   --mixed_mm_data \
   --reward_pretrain ${REWARD_PRETRAIN_PATHS} \
   --save_path results/$SAVE_MODEL_NAME \
   --ckpt_path results/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len $MAX_LENGTH \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --init_kl_coef $KL \
   --kl_estimator k3 \
   --prompt_data $DATA_PATH \
   --input_key prompt \
   --images_key images \
   --reference_key chosen \
   --apply_chat_template \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --max_ckpt_num 3 \
   --engine_mem_util 0.4 \
   --engine_tp_size $ENGINE_TP \
   --enable_engine_sleep \
   --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, like this: <think> reasoning process here </think> final thought and answer here.' \
   --l2 1.0e-2 \
   --freeze_prefix \
   --adam_offload \
   --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
   --use_wandb "${WANDB_API_KEY}" \
   --wandb_project "${WANDB_PROJECT}" \
   --wandb_run_name "${WANDB_RUN_NAME}" \
   2>&1 | tee "/mnt/shared-storage-user/puyuan/code/LightRLHF/rft_logs/20250911/7b_kg_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"



# bash /mnt/shared-storage-user/puyuan/code/LightRLHF/examples/openrlhf_v/run_grpo_rm_colocate_kg_H.sh > /mnt/shared-storage-user/puyuan/code/LightRLHF/rft_logs/20250903/7b_kg_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log 2>&1
# bash /mnt/shared-storage-user/puyuan/code/LightRLHF/examples/openrlhf_v/run_grpo_rm_colocate_kg_H.sh 
