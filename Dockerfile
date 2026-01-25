# 使用 NVIDIA 官方提供的 CUDA 12.8 开发镜像作为基础
# 这确保了底层驱动和编译环境的兼容性
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 设置环境变量，避免交互式安装时的提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装 Python 3.12 和必要的基础工具
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 将 python3.12 设置为默认 python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --set python /usr/bin/python3.12

# 升级 pip
RUN python -m pip install --upgrade pip

# 安装 PyTorch 2.9.1 (针对 CUDA 12.8)
# 注意：在 2026 年，请根据当时的官方指令调整 index-url
RUN pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装 flash-attn (由于编译较慢，建议优先尝试官方预编译包)
# 如果预编译包不可用，此步骤会尝试从源码编译
RUN pip install flash-attn==2.8.3 --no-build-isolation

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt 以利用 Docker 缓存
COPY requirements.txt .

# 安装其他依赖
RUN pip install -r requirements.txt

# 复制整个仓库代码
COPY . .

# 以开发模式安装 LightRFT
RUN pip install -e .

# 设置环境变量以优化 CUDA 集群下的性能
ENV NCCL_DEBUG=INFO
ENV PYTHONUNBUFFERED=1

# 默认启动命令（可以根据实际需要修改）
CMD ["/bin/bash"]
