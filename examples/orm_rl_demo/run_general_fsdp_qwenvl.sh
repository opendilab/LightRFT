#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NAME="${NAME:-orm-rl-demo-general-geo3k}"
N_SAMPLES=8
EPISODE="${EPISODE:-20}"
WARMUP=0.03
RBS=128
TBS=128
KL=0.001
LR=1e-6

PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=2048
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-700}"
limit_mm_image_per_prompt=1
ENGINE_TP=1

export IGNORE_EOS=0

# Reuse the existing cluster-ready path style already referenced in this repo.
DATA_PATH="/mnt/shared-storage-user/puyuan/data/geo3k"
PRETRAIN_PATH="${PRETRAIN_PATH:-/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct}"
if [ -z "${REWARD_PRETRAIN_PATHS:-}" ]; then
  REWARD_PRETRAIN_PATHS='{"general":"/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct"}'
fi
LABEL_OVERRIDE="${LABEL_OVERRIDE:-geo3k_general}"
USE_RM_ENGINE="${USE_RM_ENGINE:-1}"
export ORM_RL_DEMO_GEO3K_FORMAT_WEIGHT="${ORM_RL_DEMO_GEO3K_FORMAT_WEIGHT:-0.1}"
export ORM_RL_DEMO_GEO3K_MODEL_WEIGHT="${ORM_RL_DEMO_GEO3K_MODEL_WEIGHT:-0.2}"
export ORM_RL_DEMO_GEO3K_ACCURACY_WEIGHT="${ORM_RL_DEMO_GEO3K_ACCURACY_WEIGHT:-0.7}"
export ORM_RL_DEMO_RM_ENGINE_TP="${ORM_RL_DEMO_RM_ENGINE_TP:-1}"
export ORM_RL_DEMO_RM_ENGINE_MEM_UTIL="${ORM_RL_DEMO_RM_ENGINE_MEM_UTIL:-0.15}"
export ORM_RL_DEMO_RM_ENGINE_BACKEND="${ORM_RL_DEMO_RM_ENGINE_BACKEND:-vllm}"
export ORM_RL_DEMO_RM_ENGINE_MAX_MODEL_LEN="${ORM_RL_DEMO_RM_ENGINE_MAX_MODEL_LEN:-4096}"

current_time=$(date +"%m%d%H%M")

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p log
mkdir -p wandb

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN

export MLP_WORKER_NUM=1
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-2}"
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_PORT=20090
export MLP_WORKER_0_HOST=localhost

export MASTER_ADDR=$MLP_WORKER_0_HOST
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT

SAVE_MODEL_NAME="LightRFT-geo3k-general-orm-len_${PROMPT_MAX_LEN}_${GENERATE_MAX_LEN}-tbs_${TBS}-rbs_${RBS}-sample_${N_SAMPLES}-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-lr_${LR}"

mkdir -p "results/${NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${NAME}"

set -x

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-968275bc822c87ac741ecce2f06cdfb54dbc1608}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

WANDB_PROJECT="${WANDB_PROJECT:-ORM-RL-Demo-QwenVL-7B-Geo3K}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ORM-RL-Demo-Geo3K-General-${current_time}}"
WANDB_ORG="${WANDB_ORG:-}"
ENGINE_MEM_UTIL="${ENGINE_MEM_UTIL:-0.4}"

rm_use_engine_args=()
if [ "${USE_RM_ENGINE}" = "1" ]; then
  rm_use_engine_args+=(--rm_use_engine)
fi

wandb_org_args=()
if [ -n "${WANDB_ORG}" ]; then
  wandb_org_args+=(--wandb_org "${WANDB_ORG}")
fi

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR "${SCRIPT_DIR}/train_colocate.py" \
   --pretrain "${PRETRAIN_PATH}" \
   --loss_agg_mode seq-mean-token-mean \
   --save_trajectories \
   --num_trajectories_to_save 16 \
   --print_replay_buffer_stats \
   --fsdp \
   --use_kl_loss \
   "${rm_use_engine_args[@]}" \
   --mixed_mm_data \
   --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
   --save_path "results/${NAME}/${SAVE_MODEL_NAME}" \
   --ckpt_path "results/${NAME}/${SAVE_MODEL_NAME}" \
   --micro_train_batch_size 4 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt ${N_SAMPLES} \
   --prompt_max_len ${PROMPT_MAX_LEN} \
   --generate_max_len ${GENERATE_MAX_LEN} \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate ${LR} \
   --init_kl_coef ${KL} \
   --kl_estimator k3 \
   --prompt_data "${DATA_PATH}" \
   --input_key prompt \
   --images_key images \
   --label_key label \
   --label_override "${LABEL_OVERRIDE}" \
   --eval_steps 20 \
   --eval_split "${EVAL_SPLIT}" \
   --max_eval_samples "${MAX_EVAL_SAMPLES}" \
   --apply_chat_template \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --max_ckpt_num 1 \
   --engine_mem_util "${ENGINE_MEM_UTIL}" \
   --engine_tp_size ${ENGINE_TP} \
   --enable_engine_sleep \
   --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, and the final answer should be put in \boxed{}, like this: <think> reasoning process here </think> final thought and \boxed{answer} here.' \
   --l2 1.0e-2 \
   --freeze_prefix \
   --adam_offload \
   --limit_mm_image_per_prompt ${limit_mm_image_per_prompt} \
   --use_wandb "${WANDB_API_KEY}" \
   "${wandb_org_args[@]}" \
   --wandb_project "${WANDB_PROJECT}" \
   --wandb_run_name "${WANDB_RUN_NAME}" \
   2>&1 | tee "rft_logs/${NAME}/${NAME}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
