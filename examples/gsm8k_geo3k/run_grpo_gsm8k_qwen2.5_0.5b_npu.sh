#!/bin/bash

################################################################################
#                    Huawei NPU Training Configuration Script                   #
# This script has been modified to support Huawei Ascend NPU devices.         #
# Key changes:                                                                 #
# - NPU-specific environment variables                                         #
# - HCCL backend for distributed communication                                 #
# - torch_npu library configuration                                            #
# - Adjusted inference engine settings for NPU compatibility                   #
################################################################################

# --- 环境初始化 ---

# 1. 初始化 Conda 的 Shell 功能
eval "$(conda shell.bash hook)"

# 2. 激活您的目标环境 (需要安装 torch_npu)
conda activate /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312

# 将您的项目根目录添加到 PYTHONPATH
export PYTHONPATH=/mnt/shared-storage-user/puyuan/code/LightRFT:$PYTHONPATH


################################################################################
#                       NPU Environment Configuration                          #
################################################################################

# --- NPU设备配置 ---
# 设置NPU可见设备 (类似于CUDA_VISIBLE_DEVICES)
# 如果需要限制使用特定NPU,取消下面的注释并设置
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- NPU库路径配置 ---
# 根据您的Ascend安装路径进行调整
# 典型的Ascend安装路径: /usr/local/Ascend
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend}

# 添加CANN库路径
if [ -d "${ASCEND_HOME_PATH}/latest" ]; then
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/latest/lib64/plugin/opskernel:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/latest/lib64/plugin/nnengine:$LD_LIBRARY_PATH
    export ASCEND_TOOLKIT_PATH=${ASCEND_HOME_PATH}/latest
    export ASCEND_AICPU_PATH=${ASCEND_HOME_PATH}/latest
fi

# 添加torch_npu相关路径 (如果torch_npu是通过pip安装的)
TORCH_NPU_PATH=$(python3 -c "import torch_npu; print(torch_npu.__path__[0])" 2>/dev/null)
if [ ! -z "$TORCH_NPU_PATH" ]; then
    export LD_LIBRARY_PATH=${TORCH_NPU_PATH}/lib:$LD_LIBRARY_PATH
fi

# --- NPU日志和调试配置 ---
# 设置NPU日志级别 (0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR)
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

# 设置NPU算子行为 (类似于CUDA的一些配置)
export COMBINED_ENABLE=1  # 使能算子融合优化
export TASK_QUEUE_ENABLE=1  # 使能任务队列优化

# --- HCCL配置 (华为集合通信库,相当于NVIDIA的NCCL) ---
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
# 如果遇到通信问题,可以启用详细日志
# export HCCL_DEBUG=1

# 验证NPU是否可用
echo "=== Checking NPU Environment ==="
python3 << 'EOF'
import torch
try:
    import torch_npu
    print(f"✓ torch_npu imported successfully")
    print(f"✓ NPU available: {torch.npu.is_available()}")
    if torch.npu.is_available():
        print(f"✓ NPU count: {torch.npu.device_count()}")
        for i in range(torch.npu.device_count()):
            print(f"  - NPU {i}: {torch.npu.get_device_name(i)}")
    else:
        print("✗ No NPU devices detected!")
        exit(1)
except ImportError as e:
    print(f"✗ ERROR: torch_npu not installed: {e}")
    print("Please install torch_npu: pip install torch_npu")
    exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: NPU environment check failed!"
    echo "Please ensure:"
    echo "  1. CANN toolkit is installed (typically at /usr/local/Ascend)"
    echo "  2. torch_npu is installed: pip install torch_npu"
    echo "  3. NPU devices are properly configured and drivers loaded"
    echo "  4. Run 'npu-smi info' to check NPU status"
    exit 1
fi
echo "=== NPU Environment Check Passed ==="
echo ""


################################################################################
#                           Part 1: User Configuration                         #
# Please update the following paths and settings to match your environment.    #
################################################################################

# --- Model and Dataset Paths ---
# Path to the base model. Can be a Hugging Face model name or a local directory.
PATH_TO_YOUR_BASE_MODEL="/data/puyuan/LightRFT/model/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775"

# Path to the preprocessed GSM8K dataset.
PATH_TO_YOUR_GSM8K_DATASET="/data/puyuan/LightRFT/data/gsm8k"

# --- Experiment and Logging ---
# A descriptive name for your experiment.
EXPERIMENT_NAME="lightrft-gsm8k-grpo-npu-$(date +%Y%m%d)"

# Your Weights & Biases API key.
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_PROJECT="LightRFT-GSM8K-NPU-Experiments"


################################################################################
#                       Part 2: Training Hyperparameters                       #
################################################################################

# --- GRPO Settings ---
GROUP_METHOD="normal"
N_SAMPLES=8              # Number of samples per prompt for GRPO (must be > 1).
EPISODE=30               # Total number of training episodes.
WARMUP=0.03              # Learning rate warmup ratio.

# --- Batch Size Configuration ---
# 注意:NPU的批处理大小可能需要根据NPU内存调整
RBS=128                  # Rollout Batch Size.
TBS=128                  # Train Batch Size.

# --- Learning and Model Settings ---
KL=0.01                  # KL divergence coefficient.
LR=1e-6                  # Actor learning rate.
MAX_LENGTH=3072          # Max sequence length (prompt + generation).
PROMPT_MAX_LEN=1024      # Max length of the input prompt.
GENERATE_MAX_LEN=2048    # Max length of the generated response.

# --- Evaluation Settings ---
EVAL_SPLIT="test"        # Dataset split for evaluation.
MAX_EVAL_SAMPLES=1319    # Set to 1319 for a full evaluation on the GSM8K test set.


################################################################################
#                    Part 3: Distributed Training Setup                        #
#          NPU版本使用HCCL作为通信后端,而不是NCCL                               #
################################################################################

# --- Single-Node Distributed Setup ---
export MLP_WORKER_NUM=1                 # Number of nodes.
export MLP_WORKER_GPU=8                 # Number of NPUs per node (保持变量名为GPU以兼容代码).
export MLP_ROLE_INDEX=0                 # Rank of the current node.
export MLP_WORKER_0_HOST="localhost"    # IP address of the master node (node 0).
export MLP_WORKER_0_PORT=20090          # Port for the master node.

# --- PyTorch Distributed Environment Variables ---
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# --- 重要:设置使用HCCL后端 ---
# 对于NPU,我们使用HCCL而不是NCCL
# 这个环境变量会被LightRFT代码识别并使用正确的后端
export ACCELERATOR_TYPE="npu"  # 标识使用NPU而不是GPU

# --- vLLM/SGLang Engine Settings ---
# 注意:vLLM和SGLang主要为GPU设计,在NPU上可能不完全支持
# 如果遇到推理引擎问题,可能需要:
# 1. 禁用推理引擎并使用纯PyTorch推理(需要代码修改)
# 2. 等待vLLM的NPU支持版本
# 3. 使用华为提供的推理加速方案(如MindIE)
ENGINE_TP=2  # Tensor parallelism size for the inference engine.


################################################################################
#                      Part 4: Execution and Logging                           #
################################################################################

# --- Generate dynamic names and paths ---
current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

# --- Create directories for logs and checkpoints ---
mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"


# --- System and Environment Optimizations ---
# NPU版本:不使用NCCL相关的优化
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # NCCL specific, 不适用于NPU
# export NCCL_DEBUG="WARN"  # NCCL specific, 不适用于NPU

export IGNORE_EOS=0
# export WANDB_MODE="offline" # Set to "online" for real-time W&B logging.

# --- Set execution verbosity ---
set -x


################################################################################
#                         Part 5: Main Training Command                        #
################################################################################

# 注意:以下命令可能需要进一步调整以完全支持NPU
# 主要问题点:
# 1. vLLM引擎可能不支持NPU,可能需要禁用或替换
# 2. 代码中的CUDA API调用需要替换为NPU API
# 3. 通信后端需要使用HCCL而不是NCCL

echo "=== Starting GRPO Training on NPU ==="
echo "Model: ${PATH_TO_YOUR_BASE_MODEL}"
echo "Dataset: ${PATH_TO_YOUR_GSM8K_DATASET}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "NPU Count: ${GPUS_PER_NODE}"
echo ""

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --save_trajectories \
    --advantage_estimator "group_norm" \
    --fsdp \
    --use_kl_loss \
    --flash_attn \
    --engine_type vllm \
    --enable_engine_sleep \
    --rm_use_engine \
    --reward_pretrain "{}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size 4 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size ${RBS} \
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
    --kl_estimator "k3" \
    --prompt_data "${PATH_TO_YOUR_GSM8K_DATASET}" \
    --input_key "prompt" \
    --label_key "label" \
    --eval_steps 20 \
    --eval_split "${EVAL_SPLIT}" \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --apply_chat_template \
    --gradient_checkpointing \
    --save_steps 20 \
    --max_ckpt_num 3 \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, the final answer MUST BE put in \\boxed{}, like this: <think> reasoning process here </think> final thought and \\boxed{answer} here.' \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --text_only \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                        NPU兼容性说明和后续步骤                                #
#                                                                              #
# 这个脚本提供了基本的NPU环境配置,但要完全支持NPU训练,还需要:                    #
#                                                                              #
# 1. 代码修改:                                                                  #
#    - 修改 lightrft/strategy/strategy_base.py 中的设备API                     #
#    - 修改 lightrft/strategy/utils/distributed_util.py 中的通信后端            #
#    - 适配vLLM推理引擎或使用替代方案                                            #
#                                                                              #
# 2. 依赖安装:                                                                  #
#    pip install torch_npu  # 华为NPU的PyTorch扩展                             #
#                                                                              #
# 3. 推理引擎:                                                                  #
#    vLLM目前主要支持GPU。对于NPU,可以考虑:                                      #
#    - 使用华为MindIE推理引擎                                                    #
#    - 等待vLLM的NPU支持                                                        #
#    - 修改代码以使用纯PyTorch推理                                               #
#                                                                              #
# 4. 性能优化:                                                                  #
#    - 根据NPU特性调整批处理大小                                                 #
#    - 使用NPU特定的优化选项                                                     #
#    - 调整HCCL参数以优化通信性能                                                #
#                                                                              #
# 详细的代码修改补丁请参考同目录下的NPU_COMPATIBILITY_PATCHES.md文件              #
################################################################################
