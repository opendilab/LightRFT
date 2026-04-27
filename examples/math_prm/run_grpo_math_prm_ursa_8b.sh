#!/bin/bash
#
# LightRFT GRPO Training Script – URSA-8B with URSA-8B-RM (Math PRM)
#
# Trains URSA-8B (multimodal math VLM) with URSA-8B-RM as the Process Reward Model.
# This is the URSA-MATH Stage 3 launcher migrated into LightRFT and aligned to the
# current Phase 6 "Stage 3 reproduction script" checkpoint.
#
# Key features:
#   - Actor: URSA-8B (hybrid vision tower + Qwen2.5-Math-Instruct)
#   - Reward: URSA-8B-RM (process reward model for step-level scoring)
#   - Algorithm: Phase 4 GRPO with PS-GRPO reward via math_psgrpo label
#   - Dataset: converted MMathCoT-1M Stage 3 manifest
#   - Runtime baseline: /data/LightRFT/Dockerfile
#
# Important baseline rule:
#   Keep the pip packages and installation order from /data/LightRFT/Dockerfile
#   unchanged unless you are explicitly doing environment migration work.
#
# Step-scoring protocol (see MathPRMReward in reward_models.py):
#   1. The actor generates a chain-of-thought response.
#   2. The response is formatted with "Step N:" headings and "†Answer:" prefix.
#   3. Each step boundary is marked with Cyrillic ' и' (U+0438) token.
#   4. A single forward pass through URSA-8B-RM yields per-step probabilities.
#   5. In Phase 4, MathPRMReward maps step scores + correctness to PS-GRPO reward.
#

################################################################################
#                         Part 1: User Configuration                           #
# Update paths and keys to match your environment before running.              #
################################################################################

# --- Actor (policy) model ---
# URSA-8B: A multimodal math VLM with hybrid vision tower (SAM-B + SigLIP-L) + Qwen2.5-Math-Instruct
# This is the output from URSA-MATH stage1 training.
PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/home/ubuntu/URSA-MATH/checkpoints/URSA-8B}"
# Example HuggingFace name (verify the exact repo name before use):
# PATH_TO_YOUR_BASE_MODEL="AI-MO/URSA-8B"

# --- Reward model ---
# URSA-8B-RM: a step-level Process Reward Model for mathematical reasoning.
# Set to your local copy or a HuggingFace model name.
PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B}"
# Example HuggingFace name (verify the exact repo name before use):
# PATH_TO_URSA_RM="AI-MO/URSA-8B-RM"

# --- Dataset ---
# Default: converted full-data Stage 3 manifest.
# The original paper uses a one-time filtered ~15K RL subset; the exact subset
# is not present locally yet, so the launcher keeps the converted manifest path
# and caps training with MAX_SAMPLES to stay close to the reported Stage 3 scale.
# Dataset format:
#   "prompt"  : the math question (string, may include images)
#   "images"  : list of image paths (optional, for multimodal problems)
#   "label"   : "math_psgrpo" → triggers Phase 4 PS-GRPO reward
#               "math_prm"  →  Phase 3 baseline PRM-only reward
#               "math_prm_combined" → PRM + rule-based accuracy
#   "reference": ground-truth answer string (optional, for rule-based component)
# See examples/data_preprocess/ for preprocessing helpers.
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl}"
EXPECTED_REWARD_LABEL="${EXPECTED_REWARD_LABEL:-math_psgrpo}"
DOCKER_BASELINE="${DOCKER_BASELINE:-/data/LightRFT/Dockerfile}"

# --- Experiment metadata ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-stage3-psgrpo}"

# --- W&B ---
# To avoid touching any system-level wandb login state, this script supports a
# run-scoped API key via LIGHTRFT_WANDB_API_KEY. When provided, it is exported
# only for the current process tree and never written into the traced torchrun
# command line.
LIGHTRFT_WANDB_API_KEY="${LIGHTRFT_WANDB_API_KEY:-}"
WANDB_KEY_SOURCE="disabled"
if [[ -n "${LIGHTRFT_WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY="${LIGHTRFT_WANDB_API_KEY}"
    WANDB_KEY_SOURCE="LIGHTRFT_WANDB_API_KEY"
else
    export WANDB_API_KEY="${WANDB_API_KEY:-}"
    if [[ -n "${WANDB_API_KEY}" ]]; then
        WANDB_KEY_SOURCE="WANDB_API_KEY"
    fi
fi
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-URSA8B-Stage3}"
export WANDB_ORG="${WANDB_ORG:-}"


################################################################################
#                       Part 2: Training Hyperparameters                       #
################################################################################

# --- GRPO (Phase 4: reward = PS-GRPO over PRM step scores + correctness) ---
# Defaults below follow the explicit Stage 3 settings documented in the local
# URSA-MATH repo where possible.
N_SAMPLES="${N_SAMPLES:-8}"           # URSA-MATH repo: responses per prompt.
EPISODE="${EPISODE:-10}"              # URSA-MATH repo: Stage 3 training episodes.
WARMUP="${WARMUP:-0.03}"              # URSA-MATH repo: LR warmup ratio.

# --- Batch sizes ---
RBS="${RBS:-128}"                     # URSA-MATH repo: rollout batch size.
TBS="${TBS:-128}"                     # URSA-MATH repo: global train batch size.
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"

# --- Optimisation ---
KL="${KL:-0.001}"                     # URSA-MATH repo: KL coefficient.
KL_TARGET="${KL_TARGET:-}"            # If set, enables AdaptiveKLController with this target.
KL_HORIZON="${KL_HORIZON:-10000}"     # Horizon for adaptive KL annealing.
LR="${LR:-1e-6}"                      # URSA-MATH repo: actor learning rate.
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"   # URSA-MATH repo: prompt length.
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-3072}" # URSA-MATH repo: generation length.
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-15360}"   # Proxy for the paper's filtered ~15K RL set.
SAVE_STEPS="${SAVE_STEPS:-20}"
MAX_CKPT_NUM="${MAX_CKPT_NUM:-2}"
NUM_TRAJECTORIES_TO_SAVE="${NUM_TRAJECTORIES_TO_SAVE:-16}"

# --- Multi-modal Settings ---
limit_mm_image_per_prompt="${limit_mm_image_per_prompt:-10}"  # Max number of images per prompt.


################################################################################
#                    Part 3: Distributed Training Setup                        #
################################################################################

export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"               # Number of nodes.
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"               # GPUs per node.
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"               # Rank of this node.
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-localhost}"  # Master node IP.
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-20092}"        # Master node port.

export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# vLLM/SGLang tensor-parallelism for the *actor* inference engine.
# URSA-8B (8B params + vision towers) requires TP for efficient inference.
# URSA-8B-RM (8B params) runs on a single GPU; this controls the actor engine.
ENGINE_TYPE="${ENGINE_TYPE:-hf}"
HF_SEPARATE_ROLLOUT_ACTOR="${HF_SEPARATE_ROLLOUT_ACTOR:-1}"
HF_SEPARATE_ROLLOUT_KEEP_ON_GPU="${HF_SEPARATE_ROLLOUT_KEEP_ON_GPU:-1}"
if [[ "${ENGINE_TYPE}" == "hf" ]]; then
    ENGINE_TP="${ENGINE_TP:-1}"
    LOCAL_HF_GENERATE_MAX_BATCH_SIZE="${LOCAL_HF_GENERATE_MAX_BATCH_SIZE:-4}"
    LOCAL_HF_MAX_NEW_TOKENS="${LOCAL_HF_MAX_NEW_TOKENS:-512}"
else
    ENGINE_TP="${ENGINE_TP:-2}"
    LOCAL_HF_GENERATE_MAX_BATCH_SIZE="${LOCAL_HF_GENERATE_MAX_BATCH_SIZE:-0}"
    LOCAL_HF_MAX_NEW_TOKENS="${LOCAL_HF_MAX_NEW_TOKENS:-0}"
fi
PATH_TO_YOUR_EVAL_DATASET="${PATH_TO_YOUR_EVAL_DATASET:-}"
EVAL_SPLIT="${EVAL_SPLIT:-}"
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"
EVAL_HOLDOUT_SIZE="${EVAL_HOLDOUT_SIZE:-500}"
EVAL_HOLDOUT_SEED="${EVAL_HOLDOUT_SEED:-42}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-0}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.0}"
EVAL_NO_REPEAT_NGRAM_SIZE="${EVAL_NO_REPEAT_NGRAM_SIZE:-0}"
USE_URSA_ENGINE_WRAPPER="${USE_URSA_ENGINE_WRAPPER:-1}"
URSA_ENGINE_CHECKPOINT_DIR="${URSA_ENGINE_CHECKPOINT_DIR:-/data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with \"Step N:\" (e.g. \"Step 1:\", \"Step 2:\") on its own line. After all steps, output exactly one final answer line prefixed with \"†Answer:\" (e.g. \"†Answer: 42\"). Stop immediately after the \"†Answer:\" line and do not output any extra text, repeated answer markers, or additional steps.}"
ENABLE_PROFILE="${ENABLE_PROFILE:-0}"


################################################################################
#                      Part 4: Execution and Logging                           #
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export IGNORE_EOS=0
if [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != "YOUR_WANDB_API_KEY" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
WANDB_HEARTBEAT_INTERVAL_SECS="${WANDB_HEARTBEAT_INTERVAL_SECS:-60}"
export PATH_TO_YOUR_BASE_MODEL
export PATH_TO_URSA_RM
export PATH_TO_YOUR_MATH_DATASET
export EXPECTED_REWARD_LABEL
export DOCKER_BASELINE
export N_SAMPLES
export EPISODE
export RBS
export TBS
export MICRO_TRAIN_BATCH_SIZE
export MICRO_ROLLOUT_BATCH_SIZE
export MLP_WORKER_NUM
export MLP_WORKER_GPU
export TEMPERATURE
export KL
export LR
export PROMPT_MAX_LEN
export GENERATE_MAX_LEN
export MAX_SAMPLES
export ENGINE_TYPE
export LOCAL_HF_GENERATE_MAX_BATCH_SIZE
export LOCAL_HF_MAX_NEW_TOKENS
export HF_SEPARATE_ROLLOUT_KEEP_ON_GPU
export NUM_TRAJECTORIES_TO_SAVE
export WANDB_HEARTBEAT_INTERVAL_SECS

python - <<'PY'
import json
import os
from pathlib import Path

dataset_path = Path(os.environ["PATH_TO_YOUR_MATH_DATASET"])
expected_label = os.environ["EXPECTED_REWARD_LABEL"]
base_model_path = Path(os.environ["PATH_TO_YOUR_BASE_MODEL"])
rm_model_path = Path(os.environ["PATH_TO_URSA_RM"])
docker_baseline = Path(os.environ["DOCKER_BASELINE"])
if not dataset_path.exists():
    raise SystemExit(f"[run_grpo_math_prm_ursa_8b.sh] Dataset not found: {dataset_path}")
for path_label, path_value in (
    ("base model", base_model_path),
    ("reward model", rm_model_path),
):
    if str(path_value).startswith("/") and not path_value.exists():
        raise SystemExit(
            f"[run_grpo_math_prm_ursa_8b.sh] {path_label} path not found: {path_value}"
        )
if not docker_baseline.exists():
    raise SystemExit(
        "[run_grpo_math_prm_ursa_8b.sh] Frozen runtime baseline not found: "
        f"{docker_baseline}"
    )

seen = set()
with dataset_path.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 128:
            break
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        seen.add(record.get("label"))

if seen != {expected_label}:
    raise SystemExit(
        "[run_grpo_math_prm_ursa_8b.sh] Expected dataset label "
        f"{expected_label!r}, but sampled labels were {sorted(seen)!r}. "
        "Rebuild the manifest with examples/math_prm/tools/prepare_ursa_stage3_manifest.py "
        "or override EXPECTED_REWARD_LABEL if you intentionally want another reward path."
    )
print(
    "[run_grpo_math_prm_ursa_8b.sh] Dataset label check passed: "
    f"{expected_label!r} from {dataset_path}"
)

world_size = int(os.environ["MLP_WORKER_NUM"]) * int(os.environ["MLP_WORKER_GPU"])
micro_train_batch_size = int(os.environ["MICRO_TRAIN_BATCH_SIZE"])
train_batch_size = int(os.environ["TBS"])
if train_batch_size % (micro_train_batch_size * world_size) != 0:
    raise SystemExit(
        "[run_grpo_math_prm_ursa_8b.sh] train batch size is not divisible by "
        "(micro_train_batch_size * world_size): "
        f"{train_batch_size} % ({micro_train_batch_size} * {world_size}) != 0"
    )
grad_accum = train_batch_size // (micro_train_batch_size * world_size)

ursa_stage3_targets = {
    "num_episodes": ("EPISODE", "10"),
    "n_samples_per_prompt": ("N_SAMPLES", "8"),
    "temperature": ("TEMPERATURE", "1.0"),
    "init_kl_coef": ("KL", "0.001"),
    "actor_learning_rate": ("LR", "1e-6"),
    "prompt_max_len": ("PROMPT_MAX_LEN", "1024"),
    "generate_max_len": ("GENERATE_MAX_LEN", "3072"),
    "rollout_batch_size": ("RBS", "128"),
    "train_batch_size": ("TBS", "128"),
    "micro_rollout_batch_size": ("MICRO_ROLLOUT_BATCH_SIZE", "4"),
    "micro_train_batch_size": ("MICRO_TRAIN_BATCH_SIZE", "4"),
    "max_samples_proxy": ("MAX_SAMPLES", "15360"),
}
alignment_summary = []
for name, (env_key, expected_value) in ursa_stage3_targets.items():
    current_value = os.environ[env_key]
    status = "aligned" if current_value == expected_value else f"override({current_value})"
    alignment_summary.append(f"{name}={status}")

print(
    "[run_grpo_math_prm_ursa_8b.sh] URSA Stage 3 preflight: "
    f"engine_type={os.environ['ENGINE_TYPE']}, "
    f"local_hf_max_new_tokens={os.environ['LOCAL_HF_MAX_NEW_TOKENS']}, "
    f"hf_separate_rollout_keep_on_gpu={os.environ['HF_SEPARATE_ROLLOUT_KEEP_ON_GPU']}, "
    f"world_size={world_size}, "
    f"train_batch_size={train_batch_size}, "
    f"micro_train_batch_size={micro_train_batch_size}, "
    f"gradient_accumulation={grad_accum}"
)
print(
    "[run_grpo_math_prm_ursa_8b.sh] URSA Stage 3 default snapshot: "
    + ", ".join(alignment_summary)
)
print(
    "[run_grpo_math_prm_ursa_8b.sh] Frozen runtime baseline: "
    f"{docker_baseline}"
)
PY

# JSON config passed to --reward_pretrain.
# Format: '{"<type>": "<path>"}' where <type> must match a RewardModelType value.
# URSA-8B-RM is a text-only HF model → engine mode NOT recommended for PRM
# (requires logit access).  The builder in reward_models_utils.py ignores
# use_engine for math_prm/math_psgrpo and loads via HF directly.
REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

KL_TARGET_ARGS=()
if [[ -n "${KL_TARGET}" ]]; then
    KL_TARGET_ARGS=(--kl_target "${KL_TARGET}")
fi

WANDB_ARGS=()
WANDB_ENABLE_REASON="disabled"
WANDB_USE_WANDB_ARG=""
if [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != "YOUR_WANDB_API_KEY" ]]; then
    WANDB_ENABLE_REASON="${WANDB_KEY_SOURCE}"
    WANDB_USE_WANDB_ARG="__env__"
elif python - <<'PY' >/dev/null 2>&1
import wandb
raise SystemExit(0 if bool(wandb.api.api_key) else 1)
PY
then
    WANDB_ENABLE_REASON="existing_wandb_login"
    WANDB_USE_WANDB_ARG="__existing_login__"
fi

if [[ -n "${WANDB_USE_WANDB_ARG}" ]]; then
    WANDB_ARGS=(
        --use_wandb "${WANDB_USE_WANDB_ARG}"
        --wandb_project "${WANDB_PROJECT}"
        --wandb_run_name "${WANDB_RUN_NAME}"
    )
    if [[ -n "${WANDB_ORG}" ]]; then
        WANDB_ARGS+=(
            --wandb_org "${WANDB_ORG}"
        )
    fi
    echo "[run_grpo_math_prm_ursa_8b.sh] WANDB enabled for this run via ${WANDB_ENABLE_REASON}."
else
    echo "[run_grpo_math_prm_ursa_8b.sh] WANDB disabled for this run."
fi

HF_ROLLOUT_ARGS=()
if [[ "${ENGINE_TYPE}" == "hf" && "${HF_SEPARATE_ROLLOUT_ACTOR}" == "1" ]]; then
    HF_ROLLOUT_ARGS=(
        --hf_separate_rollout_actor
    )
    if [[ "${HF_SEPARATE_ROLLOUT_KEEP_ON_GPU}" == "1" ]]; then
        HF_ROLLOUT_ARGS+=(
            --hf_separate_rollout_keep_on_gpu
        )
    fi
    echo "[run_grpo_math_prm_ursa_8b.sh] Separate local HF rollout actor enabled."
fi

PROFILE_ARGS=()
if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    PROFILE_ARGS=(
        --enable_profile
    )
    echo "[run_grpo_math_prm_ursa_8b.sh] Step profiling enabled."
fi

EVAL_ARGS=()
if [[ "${EVAL_MAX_SAMPLES}" -gt 0 ]]; then
    EVAL_ARGS=(
        --eval_steps "${EVAL_STEPS}"
        --max_eval_samples "${EVAL_MAX_SAMPLES}"
        --eval_holdout_size "${EVAL_HOLDOUT_SIZE}"
        --eval_holdout_seed "${EVAL_HOLDOUT_SEED}"
        --eval_n_samples_per_prompt "${EVAL_N_SAMPLES}"
        --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}"
        --eval_temperature "${EVAL_TEMPERATURE}"
        --eval_top_p "${EVAL_TOP_P}"
        --eval_top_k "${EVAL_TOP_K}"
        --eval_repetition_penalty "${EVAL_REPETITION_PENALTY}"
        --eval_no_repeat_ngram_size "${EVAL_NO_REPEAT_NGRAM_SIZE}"
    )
    if [[ "${EVAL_DO_SAMPLE}" == "1" ]]; then
        EVAL_ARGS+=(
            --eval_do_sample
        )
    fi

    if [[ -n "${PATH_TO_YOUR_EVAL_DATASET}" ]]; then
        EVAL_ARGS+=(
            --eval_data "${PATH_TO_YOUR_EVAL_DATASET}"
        )
        echo "[run_grpo_math_prm_ursa_8b.sh] Runtime eval uses explicit eval_data: ${PATH_TO_YOUR_EVAL_DATASET}"
    elif [[ -n "${EVAL_SPLIT}" ]]; then
        EVAL_ARGS+=(
            --eval_split "${EVAL_SPLIT}"
        )
        echo "[run_grpo_math_prm_ursa_8b.sh] Runtime eval uses split '${EVAL_SPLIT}'."
    elif [[ "${EVAL_HOLDOUT_SIZE}" -gt 0 ]]; then
        echo "[run_grpo_math_prm_ursa_8b.sh] Runtime eval uses a deterministic held-out subset from prompt_data (size=${EVAL_HOLDOUT_SIZE}, seed=${EVAL_HOLDOUT_SEED}) to mirror the paper's fixed in-domain eval protocol."
    else
        echo "[run_grpo_math_prm_ursa_8b.sh] Runtime eval disabled because no eval_data/eval_split/heldout subset is configured."
    fi
else
    echo "[run_grpo_math_prm_ursa_8b.sh] Runtime eval disabled because EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES}."
fi

if [[ "${ENGINE_TYPE}" != "hf" && "${USE_URSA_ENGINE_WRAPPER}" == "1" && -d "${PATH_TO_YOUR_BASE_MODEL}" ]]; then
    echo "[run_grpo_math_prm_ursa_8b.sh] Preparing URSA engine wrapper checkpoint at ${URSA_ENGINE_CHECKPOINT_DIR}"
    PATH_TO_YOUR_BASE_MODEL="$(
        python examples/math_prm/tools/prepare_ursa_engine_checkpoint.py \
            --source-model-path "${PATH_TO_YOUR_BASE_MODEL}" \
            --output-path "${URSA_ENGINE_CHECKPOINT_DIR}"
    )"
    echo "[run_grpo_math_prm_ursa_8b.sh] Using wrapped URSA checkpoint: ${PATH_TO_YOUR_BASE_MODEL}"
fi

set -x


################################################################################
#                         Part 5: Main Training Command                        #
################################################################################

python -m torch.distributed.run \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/math_prm/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --mixed_mm_data \
    --save_trajectories \
    --num_trajectories_to_save ${NUM_TRAJECTORIES_TO_SAVE} \
    --print_replay_buffer_stats \
    --loss_agg_mode "seq-mean-token-mean" \
    --fsdp \
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH_SIZE} \
    --rollout_batch_size ${RBS} \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY \
    --no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --use_kl_loss \
    --init_kl_coef $KL \
    --kl_estimator "k3" \
    "${KL_TARGET_ARGS[@]}" \
    --prompt_data "${PATH_TO_YOUR_MATH_DATASET}" \
    --max_samples ${MAX_SAMPLES} \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --apply_chat_template \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps ${SAVE_STEPS} \
    --max_ckpt_num ${MAX_CKPT_NUM} \
    --engine_type "${ENGINE_TYPE}" \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --local_hf_generate_max_batch_size ${LOCAL_HF_GENERATE_MAX_BATCH_SIZE} \
    --local_hf_max_new_tokens ${LOCAL_HF_MAX_NEW_TOKENS} \
    --enable_engine_sleep \
    "${HF_ROLLOUT_ARGS[@]}" \
    --system_prompt "${SYSTEM_PROMPT}" \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
    "${EVAL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# This script migrates URSA-MATH stage3 training to LightRFT framework.       #
#                                                                              #
# Step 1: Prepare URSA-8B model                                                #
#   - Download or train URSA-8B (stage1 output from URSA-MATH)                #
#   - Model structure: Hybrid vision tower (SAM-B + SigLIP-L) + Qwen2.5-Math  #
#   - Set PATH_TO_YOUR_BASE_MODEL to the model directory                       #
#                                                                              #
# Step 2: Prepare URSA-8B-RM reward model                                      #
#   - Download or train URSA-8B-RM (stage2 output from URSA-MATH)             #
#   - This is a UrsaForTokenClassification model for step-level scoring       #
#   - Set PATH_TO_URSA_RM to the model directory                               #
#                                                                              #
# Step 3: Prepare MMathCoT-1M stage3 dataset                                   #
#   - For the current machine, the default path points to the converted full   #
#     Phase 1 manifest under /data/LightRFT/tmp/ursa_stage3/                  #
#   - Dataset format (JSON/JSONL):                                             #
#     {                                                                        #
#       "prompt": "math question text",                                       #
#       "images": ["path/to/image1.jpg", ...],  # optional                    #
#       "label": "math_psgrpo",                 # default Phase 4+ path       #
#       "reference": "ground truth answer"      # optional                    #
#     }                                                                        #
#   - Set PATH_TO_YOUR_MATH_DATASET to the dataset directory                  #
#                                                                              #
# Step 4: Configure training hyperparameters (Part 2)                          #
#   - Current default path is Phase 4+: reward label = math_psgrpo            #
#   - Phase 3 baseline remains available only when you intentionally provide   #
#     a math_prm-labeled manifest and override EXPECTED_REWARD_LABEL          #
#   - You can override all key hyperparameters and paths via environment vars #
#   - Current launcher defaults follow the explicit Stage 3 values documented  #
#     in the local URSA-MATH repo:                                             #
#       EPISODE=10, N_SAMPLES=8, RBS=128, TBS=128,                            #
#       MICRO_TRAIN_BATCH_SIZE=4, MICRO_ROLLOUT_BATCH_SIZE=4,                 #
#       KL=0.001, LR=1e-6, PROMPT_MAX_LEN=1024, GENERATE_MAX_LEN=3072         #
#   - On the current 8-GPU machine this batch is realized as:                  #
#       micro_train_batch_size=4 x world_size=8 x grad_accum=4 = 128          #
#   - Paper-scale data curation uses one-time filtering from 20K candidates    #
#     down to ~15K RL samples. Because that exact subset is not yet present    #
#     locally, the launcher keeps the converted manifest path but defaults     #
#     MAX_SAMPLES to 15360 as a scale proxy.                                   #
#   - Current deliberate differences vs original paper runtime:                #
#       local hardware is 8x A100 instead of the paper's default 32x H100      #
#       rollout uses the local HF engine path under the frozen Docker baseline #
#                                                                              #
# Step 5: Run training                                                         #
#   bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh                       #
#   - For the Phase 3 baseline smoke path, use                                #
#       bash examples/math_prm/tools/run_phase3_smoke.sh                       #
#     which exports a math_prm-labeled manifest and time-boxed settings.      #
#   - For data/resource smoke checks before RL training, you can reuse:        #
#       python /home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py  #
#       python /home/ubuntu/URSA-MATH/examples/validate_dataset_entrypoints.py \
#           --policy-model /home/ubuntu/URSA-MATH/checkpoints/URSA-8B          \
#           --prm-model /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B          #
#                                                                              #
# Key differences from URSA-MATH original implementation:                      #
#   - Uses LightRFT's FSDP/DeepSpeed training infrastructure                  #
#   - Integrates with vLLM/SGLang-compatible rollout engines                  #
#   - Co-locates reward model with actor for memory efficiency                #
#   - All URSA model code is self-contained in examples/math_prm/ursa_model/  #
#                                                                              #
# Response format (enforced by system_prompt):                                 #
#   Step 1: <reasoning>                                                        #
#   Step 2: <reasoning>                                                        #
#   ...                                                                        #
#   †Answer: <final answer>                                                    #
#                                                                              #
# URSA-8B-RM scoring protocol (Phase 3 baseline):                              #
#   - Scans for "Step N:" headings in the response                            #
#   - Inserts Cyrillic ' и' (U+0438) marker at each step boundary            #
#   - Single forward pass yields per-step probabilities                       #
#   - Minimum step score used as final sequence reward                        #
#                                                                              #
# Ablations / variants:                                                        #
#   - label="math_prm": PRM-only reward (Phase 3 baseline)                    #
#   - label="math_prm_combined": PRM + rule-based accuracy ablation           #
#   - Adjust aggregation in reward_models.py MathPRMReward:                   #
#       "min"  – most conservative (default, PS-GRPO)                         #
#       "avg"  – softer, less sensitive to single bad step                    #
#       "last" – only final step score (similar to ORM)                       #
#                                                                              #
################################################################################
