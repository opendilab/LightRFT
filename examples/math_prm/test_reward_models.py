"""
Test script for Qwen2VL Reward Models

This script provides unit tests for the reward model implementations,
specifically testing the Normal and NormalThink reward models.

Features:
    - Loads reward models and processors
    - Tests with sample image-text pairs
    - Validates output scores against expected values

Usage:
    python test_reward_models.py

Requirements:
    - PIL for image loading
    - transformers for model/processor
    - Custom reward_models module
"""
import torch
import os
import sys
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(os.path.dirname(__file__))
from reward_models import Qwen2VLRewardModelNormal, Qwen2VLRewardModelNormalThink


def test_qwen2vl_reward_model_normal(model_path):
    """
    Test the Qwen2VLRewardModelNormal model with sample inputs.

    Args:
        model_path: Path to the pretrained model

    Test procedure:
        1. Load model and processor
        2. Process test cases with image-text pairs
        3. Run inference and validate scores
        4. Compare with expected scores

    Test cases include:
        - Case 1: Response with chain-of-thought (expected score: 0.0)
        - Case 2: Direct helpful response (expected score: 1.0)
    """
    # Initialize the reward model
    try:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
        reward_model = Qwen2VLRewardModelNormalThink(base_model, processor.tokenizer, processor, text_only=False)
        reward_model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test data: Define test cases with image paths, questions, responses, and expected scores
    test_cases = [
        {
            "image_path": "/fs-computility/ai-shen/shared/verifier/safe/zhangbo/share_data/orm_dataset/04_03/normal/10k/imgs/GCC_train_000268552.jpg",
            "question": "对随后的图像给出简明的解释.",  # "Give a concise explanation of the following image."
            "response": """<think>
好的，我需要分析这张图片的内容。首先，图片中有一个小女孩，她穿着红色的T恤和短裤，看起来很开心。她站在一个黄色的水上摩托旁边，这个水上摩托的型号是"jet-ski"，上面还有编号"CFZ553FX"。小女孩的手放在水上摩托的把手上，可能是在展示或者准备使用它。背景是浅色的砖墙，可能是在户外，比如码头或者海滩附近。整个场景显得很休闲，小女孩可能是在享受户外活动，或者和家人朋友一起度过愉快的时光。需要注意的是，水上活动需要安全措施，但图片中没有看到救生衣，不过这可能只是瞬间的场景。总体来说，这张图片展示了一个小女孩在水上摩托旁的愉快时刻。

</think>


这张图片展示了一个小女孩站在黄色水上摩托（jet-ski）旁边。她穿着红色T恤和短裤，显得非常开心，手扶在水上摩托的把手上。水上摩托上印有编号"CFZ553FX"，背景是浅色砖墙，可能是在户外的码头或海滩附近。整个场景充满休闲与活力，小女孩似乎在享受户外活动的快乐时光。需要注意的是，水上活动需确保安全措施到位。""",
            "expected_score": 0.0  # Contains <think> tag, treated differently
        },
        {
            "image_path": "/fs-computility/ai-shen/shared/verifier/safe/zhangbo/share_data/orm_dataset/04_03/normal/10k/imgs/GCC_train_002242837.jpg",
            "question": "简要, 清晰地说明所显示的图片.",  # "Briefly and clearly describe the displayed image."
            "response": """<think>
好的，用户需要简要清晰地说明图片内容。首先，图片是抽象的，有柔和的色调，主要是粉色、紫色和浅蓝色。这些颜色交织在一起，形成流动的图案，像云雾或液体一样。没有明确的形状或物体，整体感觉很柔和，有渐变和柔和的过渡。可能看起来像大理石纹理或者水墨画的效果。整体风格是抽象艺术，给人一种梦幻和宁静的感觉。需要抓住颜色、图案的流动性和抽象的特点。要简洁，不用复杂术语。例如："图片展示了一幅抽象的彩色图案，柔和的粉色、紫色和浅蓝色交织成流动的云雾状纹理，整体风格梦幻而宁静。"这样应该可以了。

</think>


图片展示了一幅抽象的彩色图案，柔和的粉色、紫色和浅蓝色交织成流动的云雾状纹理，整体风格梦幻而宁静。""",
            "expected_score": 1.0  # Direct helpful response
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Running test case {i+1} ---")
        
        # Load and process the image
        try:
            image = Image.open(test_case["image_path"])
            print(f"Image loaded successfully, {image.size}")
        except Exception as e:
            print(f"Failed to load image: {e}")
            continue
        
        # Process inputs: Load image and format with processor
        processor = reward_model.processor

        text = reward_model.question_response_format.format(question=test_case["question"], answer=test_case["response"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "image": image}
                ]
            }
        ]
        messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_inputs = processor(text=messages, images=image, return_tensors="pt").to(reward_model.device)
        pixel_values = batch_inputs.pixel_values
        image_grid_thw = batch_inputs.image_grid_thw

        input_ids = batch_inputs.input_ids
        attention_mask = batch_inputs.attention_mask

        print("Inputs processed successfully")
        
        # Run inference
        with torch.no_grad():
            outputs = reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
        
        # Get the score
        score = outputs["score"].item()
        print(f"Inference successful, score: {score:.4f}")
        
        # Check if the score matches the expected score
        if abs(score - test_case["expected_score"]) < 1e-5:
            print("Test PASSED: Score matches expected value")
        else:
            print(f"Test FAILED: Expected {test_case['expected_score']}, got {score}")


if __name__ == "__main__":
    # Run the tests with specified model
    print("Testing Qwen2VLRewardModelNormal...")
    model_path = "/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/5d8e171e5ee60e8ca4c6daa380bd29f78fe19021"
    test_qwen2vl_reward_model_normal(model_path=model_path)