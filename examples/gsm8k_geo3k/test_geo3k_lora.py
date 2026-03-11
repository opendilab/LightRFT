#!/usr/bin/env python3
"""
LoRA Evaluation Script for Geo3K Dataset

This script evaluates a LoRA-fine-tuned vision-language model (e.g. Qwen2.5-VL) 
on mathematical reasoning benchmarks (e.g. Geo3K). It automatically merges the LoRA adapter 
into the base weights and runs high-throughput inference using vLLM engine.

Key Features:
    - LoRA Merging: Combines parameter-efficient adapters with base vision-language models.
    - Fast Inference: Utilizes vLLM for high-throughput batch generation.
    - Consistency: Adopts the exact data pipeline and prompt mappings as training.
    - Rule-Based Evaluation: Computes format correctness and mathematical accuracy rewards.

Execution Flow:
    1. LoRA Merging: Loads the base model, merges the given LoRA adapter, and saves the unified model to `<output_dir>/merged_model` (skipped if this directory already exists).
    2. Data Loading: Prepares the Geo3K test dataset applying chat templates using LightRFT's `PromptDatasetVL`.
    3. vLLM Initialization: Loads the fully-merged weights into the vLLM engine for high-throughput batch generation.
    4. Generation: Generates responses for all prompts in the evaluation split.
    5. Scoring: Evaluates the generated answers against ground truth using `geo3k_accuracy_reward_fn` and `geo3k_format_reward_fn`.
    6. Reporting: Averages the rewards and saves the full prediction results to `<output_dir>/eval_results.json`.

Usage:
    python test_lora_geo3k.py \
        --base_model /path/to/base_model \
        --lora_path /path/to/lora \
        --eval_data /path/to/data \
        --output_dir ./eval_output
"""

import os
import argparse
import json
import torch
import sys
from tqdm import tqdm
from transformers import AutoModelForVision2Seq
from peft import PeftModel
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lightrft.datasets import PromptDatasetVL
from lightrft.utils import blending_datasets, get_tokenizer_processor_vl
from examples.gsm8k_geo3k.reward_models_utils import geo3k_accuracy_reward_fn, geo3k_format_reward_fn


# ============================================================================
# Configuration and Utilities
# ============================================================================

class MockStrategy:
    """
    Mock Strategy to avoid deepspeed/sglang imports when just running inference.
    """
    def __init__(self, args):
        self.args = args

    def print(self, msg):
        print(msg)

    def is_rank_0(self):
        return True

class MockArgs:
    """
    Mock arguments class to simulate training arguments for data blending.
    
    This guarantees compatibility with LightRFT's dataset loading utilities
    which expect a parsed arguments object containing specific configuration keys.
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.input_key = "prompt"
        self.images_key = "images"
        self.reference_key = "ground_truth"
        self.label_key = "label"
        self.apply_chat_template = True
        self.system_prompt = 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, the final answer MUST BE put in \\boxed{}, like this: <think> reasoning process here </think> final thought and \\boxed{answer} here.'
        for k, v in kwargs.items():
            setattr(self, k, v)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    
    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Merge LoRA and test on Geo3K dataset")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (e.g. Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights (e.g. /path/to/global_step5_lora)")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to Geo3K data directory")
    parser.add_argument("--eval_split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="./geo3k_lora_eval_results", help="Directory to save merged model and evaluation results")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ============================================================================
# Merging and Evaluation Functions
# ============================================================================

def merge_lora_weights(base_model_path: str, lora_path: str, save_dir: str) -> str:
    """
    Merge LoRA weights into the base model and save to disk.
    
    This function loads the base model and LoRA adapter, merges them, and saves
    the resulting full weights along with the tokenizer and processor. This ensures
    that vLLM can load a unified model directly for high-throughput inference.
    
    :param base_model_path: Path to the underlying base model
    :type base_model_path: str
    :param lora_path: Path to the LoRA adapter weights
    :type lora_path: str
    :param save_dir: Directory to save the merged model
    :type save_dir: str
    :return: Path to the directory containing the merged model
    :rtype: str
    """
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("Merging adapter...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {save_dir}...")
    model.save_pretrained(save_dir, safe_serialization=True)
    
    tokenizer, processor = get_tokenizer_processor_vl(base_model_path, model, "left", use_fast=True)
    tokenizer.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    print("Merged model saved successfully.")
    return save_dir


def evaluate_model(model_path: str, args: argparse.Namespace) -> None:
    """
    Evaluate the merged model using vLLM on the specified dataset.
    
    Loads configuration and data identically to the training pipeline,
    generates responses using vLLM, and calculates accuracy and format rewards 
    using the predefined reward functions.
    
    :param model_path: Path to the merged model weights
    :type model_path: str
    :param args: Parsed command-line arguments with evaluation settings
    :type args: argparse.Namespace
    """
    print(f"Loading tokenizer and processor from {model_path}...")
    mock_args = MockArgs(seed=args.seed)
    # create a dynamic object that PromptDatasetVL expects for strategy.args
    mock_strategy = type('MockStrategyParams', (), {'args': mock_args, 'print': print, 'is_rank_0': lambda self: True})()
    
    tokenizer, processor = get_tokenizer_processor_vl(model_path, None, "left", use_fast=True)

    print(f"Loading evaluation data from {args.eval_data}, split='{args.eval_split}'...")
    eval_data = blending_datasets(
        args.eval_data, "1.0", mock_strategy, args.seed, return_eval=False,
        train_split=args.eval_split
    )
    
    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    eval_dataset = PromptDatasetVL(
        dataset=eval_data,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.prompt_max_len,
        strategy=mock_strategy,
        input_template=None
    )
    
    print(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
    
    print("Initializing vLLM engine...")
    engine = LLM(
        model=model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 10}
    )
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_token_ids = [tokenizer.eos_token_id]
    if im_end_id is not None:
        stop_token_ids.append(im_end_id)
        
    sampling_params = SamplingParams(
        temperature=0.0,        # Greedy decoding for evaluation
        top_p=1.0,              # No nucleus sampling, consider all tokens
        max_tokens=2048,
        stop_token_ids=stop_token_ids
    )

    vllm_inputs = []
    refs = []
    
    for i in range(len(eval_dataset)):
        prompt, images, reference, label = eval_dataset[i]
        
        inp = {"prompt": prompt}
        if images and len(images) > 0:
            inp["multi_modal_data"] = {"image": images}
            
        vllm_inputs.append(inp)
        
        refs.append(reference)

    print("Running vLLM inference generation...")
    outputs = engine.generate(vllm_inputs, sampling_params)
    
    results = []
    total_acc = 0
    total_fmt = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = refs[i]
        
        if isinstance(gt, list) and len(gt) > 0:
            gt = gt[0]
            
        acc_reward = geo3k_accuracy_reward_fn(generated_text, str(gt))
        fmt_reward = geo3k_format_reward_fn(generated_text)
        
        total_acc += acc_reward
        total_fmt += fmt_reward
        
        results.append({
            "prompt": vllm_inputs[i]["prompt"],
            "generated": generated_text,
            "ground_truth": gt,
            "accuracy": acc_reward,
            "format": fmt_reward,
        })
        
    avg_acc = total_acc / len(outputs) if len(outputs) > 0 else 0
    avg_fmt = total_fmt / len(outputs) if len(outputs) > 0 else 0
    
    print(f"\n{'='*40}")
    print(f"--- Final Evaluation Results ---")
    print(f"Total Evaluated Samples: {len(outputs)}")
    print(f"Average Accuracy Reward: {avg_acc:.4f} ({(avg_acc*100):.2f}%)")
    print(f"Average Format Correctness: {avg_fmt:.4f} ({(avg_fmt*100):.2f}%)")
    print(f"{'='*40}\n")
    
    output_json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Full results saved to {output_json_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model_dir = os.path.join(args.output_dir, "merged_model")
    
    if not os.path.exists(merged_model_dir):
        print(f"Starting LoRA merge process...")
        merge_lora_weights(args.base_model, args.lora_path, merged_model_dir)
    else:
        print(f"Merged model path '{merged_model_dir}' already exists. Skipping merging step...")
        
    evaluate_model(merged_model_dir, args)
