#!/usr/bin/env python3
"""
Mathematical Reasoning Benchmarks Evaluation Script

This script evaluates trained models on mathematical reasoning benchmarks:
- Math500: 500 challenging math problems
- AIME 2024/2025: Competition math problems
- GPQA Diamond: Graduate-level STEM questions

Features:
    - Support for multiple benchmarks in a single run
    - Batch inference with vLLM or standard HuggingFace generation
    - Automatic answer extraction and evaluation
    - Detailed results logging with per-example analysis
    - Resume capability from checkpoint

Usage:
    # Single benchmark
    python eval_math_benchmarks.py \\
        --model_path <path_to_model> \\
        --benchmarks math500 \\
        --data_root /path/to/eval_data

    # Multiple benchmarks
    python eval_math_benchmarks.py \\
        --model_path <path_to_model> \\
        --benchmarks math500 aime2024 gpqa_diamond \\
        --data_root /path/to/eval_data \\
        --output_dir ./results

    # With vLLM for faster inference
    python eval_math_benchmarks.py \\
        --model_path <path_to_model> \\
        --benchmarks math500 \\
        --data_root /path/to/eval_data \\
        --use_vllm \\
        --vllm_tensor_parallel_size 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from lightrft.datasets import load_math_reasoning_benchmark
from lightrft.evaluation import extract_answer, evaluate_predictions
from lightrft.strategy import FakeStrategy


class MathBenchmarkEvaluator:
    """
    Evaluator for mathematical reasoning benchmarks.
    
    Supports both standard HuggingFace generation and vLLM for faster inference.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_vllm: bool = False,
        vllm_tensor_parallel_size: int = 1,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.95,
        batch_size: int = 1,
    ):
        """
        Initialize the evaluator.
        
        :param model_path: Path to the model checkpoint
        :param device: Device to run inference on
        :param use_vllm: Whether to use vLLM for inference
        :param vllm_tensor_parallel_size: Tensor parallel size for vLLM
        :param max_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature (0 for greedy)
        :param top_p: Top-p sampling parameter
        :param batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.device = device
        self.use_vllm = use_vllm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        
        print(f"Loading model from {model_path}...")
        
        if use_vllm:
            # Import vLLM
            try:
                from vllm import LLM, SamplingParams
                self.vllm_model = LLM(
                    model=model_path,
                    tensor_parallel_size=vllm_tensor_parallel_size,
                    trust_remote_code=True,
                    dtype="bfloat16",
                )
                self.sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"✓ vLLM model loaded with tensor_parallel_size={vllm_tensor_parallel_size}")
            except ImportError:
                raise ImportError("vLLM not installed. Please install with: pip install vllm")
        else:
            # Load tokenizer/processor
            try:
                # Try vision-language model first
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                )
                self.tokenizer = self.processor.tokenizer
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device,
                )
                self.is_vl_model = True
                print(f"✓ Vision-Language model loaded")
            except:
                # Fallback to text-only model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device,
                )
                self.is_vl_model = False
                print(f"✓ Text-only model loaded")
            
            self.model.eval()
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        :param prompts: List of prompt strings
        :return: List of generated responses
        """
        if self.use_vllm:
            outputs = self.vllm_model.generate(prompts, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = []
            for prompt in prompts:
                # Tokenize
                if self.is_vl_model:
                    inputs = self.processor(
                        text=[prompt],
                        images=None,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=self.top_p,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                # Decode
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                responses.append(response)
        
        return responses
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        data_root: str,
    ) -> Dict[str, Any]:
        """
        Evaluate on a single benchmark.
        
        :param benchmark_name: Name of the benchmark
        :param data_root: Root directory containing benchmark data
        :return: Evaluation results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on {benchmark_name}")
        print(f"{'='*70}\n")
        
        # Load dataset
        strategy = FakeStrategy()
        dataset = load_math_reasoning_benchmark(
            benchmark_name=benchmark_name,
            data_root=data_root,
            tokenizer=self.tokenizer,
            strategy=strategy,
        )
        
        print(f"Loaded {len(dataset)} examples from {benchmark_name}")
        
        # Generate responses
        all_prompts = []
        all_labels = []
        all_choices = []
        all_raw_data = []
        
        for i in range(len(dataset)):
            prompt, label, choices, raw_data = dataset[i]
            all_prompts.append(prompt)
            all_labels.append(label)
            all_choices.append(choices)
            all_raw_data.append(raw_data)
        
        # Batch inference
        all_responses = []
        for i in tqdm(range(0, len(all_prompts), self.batch_size), desc="Generating responses"):
            batch_prompts = all_prompts[i:i + self.batch_size]
            batch_responses = self.generate_batch(batch_prompts)
            all_responses.extend(batch_responses)
        
        # Extract answers from responses
        predictions = []
        is_multiple_choice_list = []
        
        for response, choices in zip(all_responses, all_choices):
            is_mc = choices is not None
            predicted_answer = extract_answer(response, is_multiple_choice=is_mc)
            predictions.append(predicted_answer)
            is_multiple_choice_list.append(is_mc)
        
        # Evaluate
        eval_results = evaluate_predictions(
            predictions=predictions,
            ground_truths=all_labels,
            is_multiple_choice=is_multiple_choice_list,
        )
        
        # Add detailed results
        detailed_results = []
        for i, (prompt, response, pred, label, result_info) in enumerate(
            zip(all_prompts, all_responses, predictions, all_labels, eval_results["results"])
        ):
            detailed_results.append({
                "index": i,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "response": response,
                "predicted_answer": pred,
                "ground_truth": label,
                "is_correct": result_info["is_correct"],
                "is_multiple_choice": result_info["is_multiple_choice"],
            })
        
        return {
            "benchmark": benchmark_name,
            "total": eval_results["total"],
            "correct": eval_results["correct"],
            "accuracy": eval_results["accuracy"],
            "detailed_results": detailed_results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on mathematical reasoning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["math500"],
        choices=["math500", "aime2024", "aime2025", "gpqa_diamond"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./eval_data",
        help="Root directory containing benchmark data files",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    
    # vLLM arguments
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster inference",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = MathBenchmarkEvaluator(
        model_path=args.model_path,
        device=args.device,
        use_vllm=args.use_vllm,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    
    # Evaluate on each benchmark
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for benchmark in args.benchmarks:
        try:
            results = evaluator.evaluate_benchmark(
                benchmark_name=benchmark,
                data_root=args.data_root,
            )
            all_results[benchmark] = results
            
            # Print summary
            print(f"\n{benchmark} Results:")
            print(f"  Total: {results['total']}")
            print(f"  Correct: {results['correct']}")
            print(f"  Accuracy: {results['accuracy']:.2%}")
            
            # Save individual benchmark results
            benchmark_output_file = output_dir / f"{benchmark}_{timestamp}.json"
            with open(benchmark_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Results saved to: {benchmark_output_file}")
            
        except Exception as e:
            print(f"Error evaluating {benchmark}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    summary_file = output_dir / f"summary_{timestamp}.json"
    summary = {
        "model_path": args.model_path,
        "timestamp": timestamp,
        "benchmarks": {
            name: {
                "total": results["total"],
                "correct": results["correct"],
                "accuracy": results["accuracy"],
            }
            for name, results in all_results.items()
        },
        "generation_config": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
        },
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_file}")
    
    # Print final summary
    print("\nOverall Results:")
    for benchmark, results in all_results.items():
        print(f"  {benchmark}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")


if __name__ == "__main__":
    main()

