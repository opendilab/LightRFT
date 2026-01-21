import os
import sys
import shutil

# 定义目标文件路径 (基于你的报错堆栈)
VLLM_PATH = "/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/lib/python3.12/site-packages/vllm"
TARGET_FILE = os.path.join(VLLM_PATH, "vllm_flash_attn/flash_attn_interface.py")


def patch_vllm():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 错误: 找不到目标文件: {TARGET_FILE}")
        return

    print(f"🔍 找到目标文件: {TARGET_FILE}")

    # 1. 备份原文件
    backup_file = TARGET_FILE + ".bak"
    if not os.path.exists(backup_file):
        shutil.copy(TARGET_FILE, backup_file)
        print(f"✅ 已创建备份: {backup_file}")
    else:
        print("ℹ️ 备份已存在，跳过备份")

    # 2. 读取原文件
    with open(TARGET_FILE, 'r') as f:
        content = f.read()

    # 3. 检查是否已经修复
    if "import flash_attn.flash_attn_interface as external_fa" in content:
        print("✅ 文件看起来已经被修复过了。")
        return

    # 4. 准备补丁代码
    # 我们不仅要替换 import，还要替换具体的函数调用
    # 最简单的方法是重写整个文件，让它直接作为外部 flash_attn 的代理

    patch_code = """
# PATCHED BY USER to fix CUDA 12.8 compatibility
# Redirects vLLM internal calls to the external (working) flash_attn library
import torch
import sys
from typing import Optional, Union

try:
    import flash_attn.flash_attn_interface as external_fa
    print("[vLLM Patch] Successfully linked to external flash_attn", file=sys.stderr)
except ImportError:
    raise ImportError("External flash_attn not installed! Please install it first.")

# Proxy functions
def flash_attn_varlen_func(*args, **kwargs):
    return external_fa.flash_attn_varlen_func(*args, **kwargs)

def flash_attn_func(*args, **kwargs):
    return external_fa.flash_attn_func(*args, **kwargs)

def flash_attn_kvpacked_func(*args, **kwargs):
    return external_fa.flash_attn_kvpacked_func(*args, **kwargs)

def flash_attn_qkvpacked_func(*args, **kwargs):
    return external_fa.flash_attn_qkvpacked_func(*args, **kwargs)

def flash_attn_varlen_kvpacked_func(*args, **kwargs):
    return external_fa.flash_attn_varlen_kvpacked_func(*args, **kwargs)

def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
    return external_fa.flash_attn_varlen_qkvpacked_func(*args, **kwargs)

def flash_attn_with_kvcache(*args, **kwargs):
    return external_fa.flash_attn_with_kvcache(*args, **kwargs)
"""

    # 5. 写入补丁
    print("🛠️ 正在应用补丁...")
    with open(TARGET_FILE, 'w') as f:
        f.write(patch_code)

    print("✅ 补丁应用成功！vLLM 现在将使用外部安装的 flash_attn。")


if __name__ == "__main__":
    patch_vllm()
