cd ../../../
pip install -e . --no-deps
cd -

export http_proxy=100.68.170.107:3128
export https_proxy=100.68.170.107:3128

# This env may help to reduce memory usage
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

############################### volcengine env #####################

export MASTER_ADDR=$MLP_WORKER_0_HOST

export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT

############################### volcengine env #####################

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR test_actorvl_fused_linear_logprob.py 2>&1 | tee test_fused_linear_logprob.log
