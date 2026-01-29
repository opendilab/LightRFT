# Use NVIDIA official PyTorch 25.01 image (includes PyTorch 2.5+, CUDA 12.8)
# This is the most stable and performant base image for CUDA cluster environments
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set environment variables for non-interactive installation and optimization
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    NCCL_DEBUG=INFO \
    PYTHONPATH=/app

# Install all system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    aria2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install PyTorch packages - CRITICAL: Order must not be changed due to environment sensitivity
RUN pip install --no-cache-dir torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ninja for compilation support
RUN pip install --no-cache-dir ninja

# Install DeepSpeed with specific build configuration - CRITICAL: Order sensitive
RUN pip install deepspeed==0.18.3 --no-binary deepspeed --no-cache-dir --force-reinstall

# Install vLLM - CRITICAL: Order sensitive
RUN pip install vllm==0.13.0 --no-cache-dir --force-reinstall

# Copy application code
COPY . .

# Install LightRFT package without dependencies to avoid conflicts
RUN pip install --no-deps .

# Install additional Python dependencies - CRITICAL: Order must be maintained
RUN pip install --no-cache-dir datasets && \
    pip install --no-cache-dir librosa && \
    pip install --no-cache-dir peft && \
    pip install --no-cache-dir tensorboard && \
    pip install --no-cache-dir decord && \
    pip install --no-cache-dir easydict matplotlib && \
    pip install --no-cache-dir wandb && \
    pip install --no-cache-dir mathruler && \
    pip install --no-cache-dir pylatexenc

# Download and install Flash Attention wheel - CRITICAL: Must be after PyTorch installation
RUN aria2c -x 16 -s 16 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    && pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    && rm -f flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Install SGLang - CRITICAL: Must be last in the installation sequence
RUN pip install --no-cache-dir sglang==0.5.6.post2

# Default command
CMD ["/bin/bash"]

