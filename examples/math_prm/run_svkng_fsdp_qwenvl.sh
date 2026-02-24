#############################  kwargs  ##########################

NAME="svkng-1230-cpgd"
GROUP_METHOD=normal
N_SAMPLES=8
EPISODE=3
WARMUP=0.03
RBS=128
TBS=128
KL=0.001
LR=1e-6

MAX_LENGTH=8192
limit_mm_image_per_prompt=1  # multi-modal model

export IGNORE_EOS=0

#############################  kwargs  ##########################

DATA_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/dataset/svkg_rule_image_text_resize8192_20250717_v2/train_val"

# --- Reward Models ---
# A JSON-formatted string specifying paths to different pretrained reward models.
# The training script uses multiple reward models for different aspects (e.g., safety, value).
REWARD_PRETRAIN_PATHS='{"safety":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/safe_orm/","value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/","knowledge":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","normal":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","general":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/"}'

# ====================================The following is only for debug====================================
# TODO: ONLY FOR DENUG
# REWARD_PRETRAIN_PATHS='{}'
# REWARD_PRETRAIN_PATHS='{"value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/"}'
# REWARD_PRETRAIN_PATHS='{"safety":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/safe_orm/","value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/"}'

ENGINE_TP=1  # vLLM/SGLang, for 7b base model
PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/base_model_after_kg_20250905" # 在kg上训练后的qwen-vl 7b

# PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/ckpt_20251212_pyoy_step160_hf"

current_time=$(date +"%m%d%H%M")
LOG_BASE=log

mkdir -p $LOG_BASE

# This env may help to reduce memory usage
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN

# TODO: only for debug in 1 node
export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_PORT=20090
export MLP_WORKER_0_HOST=localhost

###############################  volcengine env  #####################
export MASTER_ADDR=$MLP_WORKER_0_HOST
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT
###############################  volcengine env  #####################

SAVE_MODEL_NAME=LightRFT-len_${MAX_LENGTH-}tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-rm-colocate-svkg-20251205

mkdir -p results/$NAME/$SAVE_MODEL_NAME
# Create log directory
mkdir -p rft_logs/${NAME}

set -x

export WANDB_MODE="offline"

# --- Weights & Biases (W&B) Logging ---
# It's recommended to set this as an environment variable rather than hardcoding.
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608" # Replace with your key

WANDB_PROJECT="QwenVL-7B-MultiORM-GRPO-SVKG"
WANDB_RUN_NAME="QwenVL-7B-MultiORM-SVKG-grpo-${current_time}"

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/safework_t1_runnable/train_colocate.py \
   --pretrain ${PRETRAIN_PATH} \
   --use_cpg_loss \
   --loss_agg_mode seq-mean-token-mean \
   --save_trajectories \
   --use_kl_loss \
   --num_trajectories_to_save 16 \
   --print_replay_buffer_stats \
   --fsdp \
   --rm_use_engine \
   --mixed_mm_data \
   --reward_pretrain ${REWARD_PRETRAIN_PATHS} \
   --save_path results/$NAME/$SAVE_MODEL_NAME \
   --ckpt_path results/$NAME/$SAVE_MODEL_NAME \
   --micro_train_batch_size 4 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 4 \
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
   --save_steps 20 \
   --max_ckpt_num 1 \
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
   2>&1 | tee "/mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/rft_logs/${NAME}/${NAME}_7b_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"


# bash /mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/examples/math_prm/run_svkng_fsdp_qwenvl.sh > /mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/rft_logs/${NAME}/${NAME}_7b_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log 2>&1
