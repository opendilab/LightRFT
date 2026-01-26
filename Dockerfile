# 使用 NVIDIA 官方提供的 PyTorch 25.01 镜像 (包含 PyTorch 2.5+, CUDA 12.8)
# 这是目前 CUDA 集群环境下最稳定、性能最好的基础镜像
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 安装 LightRFT 视觉任务和视频处理所需的系统库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .


RUN pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install ninja
RUN pip install deepspeed==0.18.3 --no-binary deepspeed --no-cache-dir --force-reinstall
RUN pip install vllm==0.13.0  --no-cache-dir --force-reinstall



# 1. 先安装基础依赖
# RUN pip install --no-cache-dir -r requirements.txt

# 复制整个仓库代码
COPY . .

# 以开发模式安装 LightRFT
# RUN pip install -e .

RUN pip install --no-deps .

RUN \
    pip install datasets \
    pip install librosa \
    pip install peft \
    pip install tensorboard \
    pip install decord \
    pip install easydict matplotlib \
    pip install wandb \
    pip install mathruler \
    pip install pylatexenc

RUN aria2c -x 16 -s 16 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
RUN pip install  flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

RUN pip install sglang==0.5.6.post2

# 集群环境性能优化环境变量
ENV NCCL_DEBUG=INFO
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 默认启动命令
CMD ["/bin/bash"]
