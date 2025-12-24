from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LightRFT",
    version="0.1.0",
    author="LightRFT Team",
    author_email="opendilab@pjlab.org.cn",
    description="LightRFT: Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opendilab/LightRFT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "transformers==4.51.3",
        "vllm==0.7.3",
        "sglang==0.4.5",
        "deepspeed>=0.15.0",
        "accelerate",
        "datasets",
        "wandb",
        "peft",
        "easydict",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "pylint",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "eval": [
            "latex2sympy2",
            "timeout_decorator",
            "word2number",
        ],
    },
    keywords=[
        "reinforcement learning",
        "RLVR",
        "RLHF",
        "large language models",
        "vision-language models",
        "reward models",
        "omni-modal",
        "multi-modal",
        "PPO",
        "GRPO",
        "deep learning",
    ],
    project_urls={
        "Source": "https://github.com/opendilab/LightRFT",
        "Bug Reports": "https://github.com/opendilab/LightRFT/issues",
    },
)
