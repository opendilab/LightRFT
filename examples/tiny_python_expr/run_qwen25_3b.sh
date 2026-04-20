#!/bin/bash

set -euo pipefail
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NAME="${NAME:-tiny-python-expr}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/puyuan/model/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/generated}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRIPT_DIR}/artifacts}"
RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACT_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/rft_logs}"

TRAIN_SIZE="${TRAIN_SIZE:-128}"
TEST_SIZE="${TEST_SIZE:-32}"
SEED="${SEED:-42}"

N_SAMPLES="${N_SAMPLES:-4}"
EPISODE="${EPISODE:-3}"
RBS="${RBS:-16}"
TBS="${TBS:-16}"
MICRO_TRAIN_BS="${MICRO_TRAIN_BS:-1}"
MICRO_ROLLOUT_BS="${MICRO_ROLLOUT_BS:-1}"
KL="${KL:-0.001}"
LR="${LR:-1e-6}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-256}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-128}"
ENGINE_TYPE="${ENGINE_TYPE:-sglang}"
ENGINE_TP="${ENGINE_TP:-1}"
ENGINE_MEM_UTIL="${ENGINE_MEM_UTIL:-0.55}"

export IGNORE_EOS=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-2}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-localhost}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-20190}"

export MASTER_ADDR="${MLP_WORKER_0_HOST}"
export NNODES="${MLP_WORKER_NUM}"
export NODE_RANK="${MLP_ROLE_INDEX}"
export GPUS_PER_NODE="${MLP_WORKER_GPU}"
export MASTER_PORT="${MLP_WORKER_0_PORT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${ARTIFACT_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-tiny-python-expr}"
export WANDB_ORG="${WANDB_ORG:-}"

mkdir -p "${WANDB_DIR}"
mkdir -p "${RESULTS_ROOT}/${NAME}"
mkdir -p "${LOG_ROOT}/${NAME}"

fix_permissions() {
    chmod -R a+rwX "${WANDB_DIR}" "${RESULTS_ROOT}/${NAME}" "${LOG_ROOT}/${NAME}" 2>/dev/null || true
}

trap fix_permissions EXIT

python3 "${SCRIPT_DIR}/build_dataset.py" \
    --output_dir "${DATA_DIR}" \
    --train_size "${TRAIN_SIZE}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}"

current_time="$(date +"%Y%m%d_%H%M%S")"
SAVE_MODEL_NAME="LightRFT-python-expr-len_${PROMPT_MAX_LEN}_${GENERATE_MAX_LEN}-tbs_${TBS}-rbs_${RBS}-sample_${N_SAMPLES}-ep_${EPISODE}-lr_${LR}-${current_time}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-tiny-python-expr-${current_time}}"

wandb_args=()
if [ -n "${LIGHTRFT_WANDB_API_KEY:-${WANDB_API_KEY:-}}" ]; then
  WANDB_KEY_VALUE="${LIGHTRFT_WANDB_API_KEY:-${WANDB_API_KEY:-}}"
  wandb_args+=(--use_wandb "${WANDB_KEY_VALUE}")
  wandb_args+=(--wandb_project "${WANDB_PROJECT}")
  wandb_args+=(--wandb_run_name "${WANDB_RUN_NAME}")
  if [ -n "${WANDB_ORG}" ]; then
    wandb_args+=(--wandb_org "${WANDB_ORG}")
  fi
fi

set -x

torchrun \
    --nnodes "${NNODES}" \
    --nproc-per-node "${GPUS_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master-port "${MASTER_PORT}" \
    --master-addr "${MASTER_ADDR}" \
    "${SCRIPT_DIR}/train_colocate.py" \
    --pretrain "${MODEL_PATH}" \
    --save_path "${RESULTS_ROOT}/${NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "${RESULTS_ROOT}/${NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size "${MICRO_TRAIN_BS}" \
    --train_batch_size "${TBS}" \
    --micro_rollout_batch_size "${MICRO_ROLLOUT_BS}" \
    --rollout_batch_size "${RBS}" \
    --num_episodes "${EPISODE}" \
    --n_samples_per_prompt "${N_SAMPLES}" \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --generate_max_len "${GENERATE_MAX_LEN}" \
    --actor_learning_rate "${LR}" \
    --init_kl_coef "${KL}" \
    --prompt_data "${DATA_DIR}" \
    --engine_type "${ENGINE_TYPE}" \
    --engine_mem_util "${ENGINE_MEM_UTIL}" \
    --engine_tp_size "${ENGINE_TP}" \
    "${wandb_args[@]}" \
    2>&1 | tee "${LOG_ROOT}/${NAME}/${NAME}_node${NODE_RANK}_${current_time}.log"
