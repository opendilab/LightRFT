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

# 1. 先安装基础依赖
# 2. 安装 flash-attn (官方镜像通常已带，但指定版本安装更稳妥)
# 3. 安装 vllm 和 sglang (LightRFT 的核心推理引擎)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation \
    && pip install --no-cache-dir vllm>=0.13.3 sglang>=0.5.6.post2

# 复制整个仓库代码
COPY . .

# 以开发模式安装 LightRFT
RUN pip install -e .

# 集群环境性能优化环境变量
ENV NCCL_DEBUG=INFO
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 默认启动命令
CMD ["/bin/bash"]
