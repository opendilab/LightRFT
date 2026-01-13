#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
import torch


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer"""
    print(f"Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer


def test_generation(model, tokenizer, prompts: List[str], max_new_tokens: int = 128):
    """Test text generation"""
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n=== Test Case {i + 1} ===")
        print(f"Input: {prompt}")

        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {generated_text}")

        results.append({"input": prompt, "output": generated_text})

    return results


def test_model_info(model, tokenizer):
    """Test model basic information"""
    print("=== Model Information ===")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Test special tokens
    print(f"Pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} ({tokenizer.bos_token_id})")


def run_benchmark_test(model, tokenizer, num_samples: int = 5):
    """Run benchmark test"""
    print(f"\n=== Running Benchmark Test ({num_samples} samples) ===")

    # Test prompts
    test_prompts = [
        "请解释一下量子计算的基本原理。", "写一首关于春天的诗。", "Python中如何实现快速排序算法？", "人工智能的未来发展会如何影响人类社会？", "请总结一下《红楼梦》的主要情节。",
        "解释一下机器学习中的过拟合现象。", "如何在Linux系统中查看CPU使用率？", "写一段描述海滩日出的优美文字。"
    ]

    # Randomly select test samples
    import random
    selected_prompts = random.sample(test_prompts, min(num_samples, len(test_prompts)))

    results = test_generation(model, tokenizer, selected_prompts, max_new_tokens=256)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test converted HuggingFace model")
    parser.add_argument("--model_path", type=str, help="Path to the converted HuggingFace model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--test_prompts",
        type=str,
        nargs="+",
        default=["写一个Python函数来计算斐波那契数列。", "请解释一下深度学习的基本概念。"],
        help="Test prompts"
    )
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark test")
    parser.add_argument("--benchmark_samples", type=int, default=5, help="Number of benchmark samples")
    parser.add_argument("--output_file", type=str, default="test_results.json", help="Output file for results")

    args = parser.parse_args()

    try:
        # Check if the model path exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path not found: {args.model_path}")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

        # Test model basic information
        test_model_info(model, tokenizer)

        all_results = {}

        # Basic generation test
        print("\n=== Basic Generation Test ===")
        basic_results = test_generation(model, tokenizer, args.test_prompts, args.max_new_tokens)
        all_results["basic_test"] = basic_results

        # Benchmark test
        if args.benchmark:
            benchmark_results = run_benchmark_test(model, tokenizer, args.benchmark_samples)
            all_results["benchmark_test"] = benchmark_results

        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output_file}")

        print("\n=== Test Completed Successfully ===")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
