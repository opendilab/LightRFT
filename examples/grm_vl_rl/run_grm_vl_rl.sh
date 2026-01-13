# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY

#############################  kwargs  ##########################

# --- GRPO Settings ---
N_SAMPLES=8
EPISODE=2
WARMUP=0.03
 
RBS=32   # rollout_batch_size
TBS=64   # train_batch_size

# --- Learning and Model Settings ---
KL=0.001
LR=1e-6
PROMPT_MAX_LEN=12288      # Max length of the input prompt.
GENERATE_MAX_LEN=4096     # Max length of the generated response.

# --- Multi-modal Settings ---
limit_mm_image_per_prompt=2
limit_mm_video_per_prompt=2

# --- Datasets ---
data_files=(
   # I2V
   "rapidata-i2v:path/to/rapidata/image-to-video/train.parquet"

   # T2V
   # "rapidata-t2v:path/to/rapidata/text-2-video/train.parquet"
   # "videogen-rewardbench:path/to/VideoGen-RewardBench/videogen-rewardbench.csv"

   # T2I
   # "hpdv3:path/to/HPDv3/train.json"
)
DATA_PATH=$(printf '%s,' "${data_files[@]}" | sed 's/,$//')

# -- Pre-processing Settings ---
video_fps=2.0
max_pixels=282240    # 360*28*28

# --- System Prompt ---
# Path to the YAML file containing system prompts for different tasks.
# Or you can directly specify the system prompt string here if uniform for all tasks.
SYSTEM_PROMPT_PATH="examples/grm_vl_rl/system_prompts.yaml"

# --- Base Model Path ---
# Path to the base model. Can be a Hugging Face model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
# or a local directory containing the model files.
PRETRAIN_PATH="path/to/your/pretrained/model"

# --- vLLM/SGLang Engine Settings ---
ENGINE_TP=1    # Tensor parallelism size for the inference engine. Adjust based on your model and GPU setup.

# --- PyTorch Distributed Environment Variables ---
export MASTER_ADDR=localhost  # Address of the master node, alternatively set to 127.0.0.1
export NNODES=1               # Number of nodes
export NODE_RANK=0            # Rank of this node
export GPUS_PER_NODE=8        # Number of GPUs per node
export MASTER_PORT=20091      # Port for communication

# --- Generate dynamic names and paths ---
EXPERIMENT_NAME="LightRFT-GRM-VL-GRPO"
current_time=$(date +"%m%d%H%M")
RUN_NAME="${EXPERIMENT_NAME}--dataset-base_model-${current_time}"
SAVE_MODEL_NAME=${RUN_NAME}-${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-lr_${LR}
LOG_BASE=log

# --- Create directories for logs and checkpoints ---
mkdir -p $LOG_BASE
mkdir -p results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}

# --- System and Environment Optimizations ---
export IGNORE_EOS=0
export WANDB_MODE="offline"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN

# --- Logging ---
export WANDB_API_KEY="your_wandb_key"
WANDB_PROJECT=${EXPERIMENT_NAME}
WANDB_RUN_NAME="${RUN_NAME}-${current_time}"

# --- Set execution verbosity ---
set -x

# ---  Main Training Command --- 
torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/grm_vl_rl/train_colocate.py \
   --pretrain ${PRETRAIN_PATH} \
   --save_trajectories \
   --fsdp \
   --use_kl_loss \
   --mixed_mm_data \
   --save_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
   --ckpt_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
   --micro_train_batch_size 8 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len $PROMPT_MAX_LEN \
   --generate_max_len $GENERATE_MAX_LEN \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --init_kl_coef $KL \
   --kl_estimator k3 \
   --prompt_data $DATA_PATH \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --max_ckpt_num 5 \
   --engine_mem_util 0.8 \
   --engine_tp_size $ENGINE_TP \
   --enable_engine_sleep \
   --system_prompt_path "$SYSTEM_PROMPT_PATH" \
   --l2 1.0e-2 \
   --freeze_prefix \
   --adam_offload \
   --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
   --limit_mm_video_per_prompt $limit_mm_video_per_prompt \
   --use_wandb "${WANDB_API_KEY}" \
   --wandb_project "${WANDB_PROJECT}" \
   --wandb_run_name "${WANDB_RUN_NAME}" \
   --fps $video_fps \
   --max_pixels $max_pixels \
   2>&1 | tee "log/lightrft_grm_vl_rl_${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"