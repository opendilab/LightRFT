"""
Rejection Sampling Inference Script

This script performs inference on a dataset using a trained GRM model,
filters out correctly predicted samples, and generates training data
with CoT reasoning for rejection sampling training.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from typing import List, Dict
from loguru import logger
from torch.utils.data import DataLoader

from lightrft.models import GenerativeRewardModelVL
from transformers import AutoProcessor
from lightrft.datasets import GRMDataset, extract_answer


TASK_INSTRUCTION_COT = """Given a caption and two images generated based on this caption, please analyze in detail the two provided images. 
Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), 
aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), 
and any other factors you deem relevant. For each evaluation dimension, 
provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each image by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. 
Then, in the <answer> tag, output exactly one of the following strings: 'Image 1 is better' or 'Image 2 is better' based on the total scores. 
No additional text is allowed in the <answer> section.
Example output format:
<think>
Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...
Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...
Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...
[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...
Total score:
Image 1: 9+8+8+6=31
Image 2: 7+8+5+8=28
</think>
<answer>Image 1 is better</answer>
Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.
Your task is provided as follows:
Text Caption: {prompt}
"""


@torch.no_grad()
def inference_and_filter(
    model_path: str,
    data_path: List[str],
    output_path: str,
    config: dict = None,
    batch_size: int = 32,
    max_new_tokens: int = 2048,
    use_cot: bool = True,
):
    """
    Perform inference on dataset and filter correctly predicted samples.
    
    Args:
        model_path: Path to the trained GRM model
        data_path: List of dataset paths in format "source:path"
        output_path: Path to save filtered samples
        config: Configuration dict for dataset
        batch_size: Batch size for inference
        max_new_tokens: Maximum tokens to generate
        use_cot: Whether to use CoT instruction (for generating reasoning)
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Load Model
    # Note: Qwen2.5-VL doesn't support dtype parameter, so we disable flash attention
    # and handle dtype conversion manually in the model class
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        
        # Use DataParallel if multiple GPUs are available
        if num_gpus > 1:
            device = "cuda"
            use_data_parallel = True
            logger.info(f"Using DataParallel with {num_gpus} GPUs")
        else:
            device = f"cuda:{torch.cuda.current_device()}"
            use_data_parallel = False
    else:
        device = "cpu"
        use_data_parallel = False
    
    model = GenerativeRewardModelVL(
        model_path,
        bf16=True,
        lora_rank=0,
        lora_alpha=0,
        target_modules=None,
        ds_config=None,
        device_map=None,  # We'll move to device manually
        use_flash_attention_2=False,  # Disable to avoid dtype issues with Qwen2.5-VL
    )
    logger.info(f"Model loaded successfully from {model_path}.")
    
    # Move model to device
    model.model = model.model.to(device)
    
    # Use DataParallel for multi-GPU inference
    if use_data_parallel:
        model.model = torch.nn.DataParallel(model.model)
        logger.info("Model wrapped with DataParallel")
    
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # Load Dataset
    dataset = GRMDataset(
        data_path,
        tokenizer=processor.tokenizer,
        strategy=None,
        processor=processor,
        max_length=8192,
        config=config,
        is_training=False,
    )

    # Reduce batch size if it's too large to avoid OOM
    # For Qwen2.5-VL with images, smaller batch size is recommended
    effective_batch_size = min(batch_size, 4)  # Limit to 4 for safety
    if batch_size > effective_batch_size:
        logger.warning(f"Reducing batch size from {batch_size} to {effective_batch_size} to avoid OOM")
    
    data_loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,  # Disable pin_memory to save memory
        collate_fn=dataset.collate_fn,
        num_workers=2,  # Reduce workers to save memory
    )

    logger.info(f"Starting inference with CoT: {use_cot}, batch_size: {effective_batch_size}")
    
    correct_samples = []
    total_samples = 0
    correct_count = 0
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        try:
            ids, mask, pixel_values, image_grid_thws, pixel_values_videos, video_grid_thws, labels, extras = batch
            
            # Ensure device is a string for .to() method
            # For DataParallel, use "cuda" to let it handle device placement
            if use_data_parallel:
                device_str = "cuda"
            else:
                device_str = str(device) if isinstance(device, torch.device) else device
            
            ids = ids.squeeze(1).to(device_str, non_blocking=False)
            mask = mask.squeeze(1).to(device_str, non_blocking=False)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device_str, non_blocking=False)
                image_grid_thws = image_grid_thws.to(device_str, non_blocking=False)
            
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(device_str, non_blocking=False)
                video_grid_thws = video_grid_thws.to(device_str, non_blocking=False)

            # Generate with unified max_new_tokens
            # Use torch.cuda.amp for mixed precision to save memory
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # Handle DataParallel wrapper
                model_to_use = model.model.module if isinstance(model.model, torch.nn.DataParallel) else model.model
                gen_ids = model_to_use.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thws,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thws,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            
            # Move to CPU and clear GPU memory immediately
            ids_cpu = ids.cpu()
            gen_ids = gen_ids.cpu()
            
            # Decode (gen_ids is already on CPU)
            gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(ids_cpu, gen_ids)]
            gen_texts = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # Clear GPU memory immediately
            del ids, mask, pixel_values, image_grid_thws, gen_ids, gen_ids_trimmed, ids_cpu
            if pixel_values_videos is not None:
                del pixel_values_videos, video_grid_thws
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Evaluate and filter
            for i, (gen_text, extra) in enumerate(zip(gen_texts, extras)):
                total_samples += 1
                predicted_answer = extract_answer(gen_text)
                gt_preference = extra['preference']  # A or B
                
                # Mapping logic: 
                # In HPDv3GRMHandler, preference "A" means Image 1 (first shown) is preferred
                # preference "B" means Image 2 (second shown) is preferred
                # But the handler randomly swaps images, so we need to check the actual mapping
                # The handler stores: preferred_path (path1) and rejected_path (path2)
                # When preference is "A", image0 (which could be preferred or rejected) is shown as Image 1
                # When preference is "B", image1 (which could be preferred or rejected) is shown as Image 1
                
                # Since the handler randomly assigns, we check based on the stored preference
                # If gt_preference is "A", it means Image 1 (first shown) is better
                # If gt_preference is "B", it means Image 2 (second shown) is better
                is_correct = False
                if gt_preference == "A" and predicted_answer == "Image 1 is better":
                    is_correct = True
                elif gt_preference == "B" and predicted_answer == "Image 2 is better":
                    is_correct = True
                
                if is_correct:
                    correct_count += 1
                    # Prepare sample for rejection sampling training
                    sample = {
                        "prompt": extra['prompt'],
                        "path1": extra['preferred_path'],
                        "path2": extra['rejected_path'],
                        "preference": gt_preference,
                        "generated_text": gen_text,
                        "predicted_answer": predicted_answer,
                    }
                    
                    # If we want to use the generated CoT reasoning, extract it
                    if use_cot:
                        # Extract reasoning from generated text
                        # Try both <think> and <think> tags (in case of different formats)
                        reasoning_match = None
                        import re
                        # Try <think> first (standard format)
                        if "<think>" in gen_text:
                            reasoning_pattern = r"<think>(.*?)</think>"
                            reasoning_match = re.search(reasoning_pattern, gen_text, re.DOTALL)
                        # Try <think> as fallback
                        elif "<think>" in gen_text:
                            reasoning_pattern = r"<think>(.*?)</think>"
                            reasoning_match = re.search(reasoning_pattern, gen_text, re.DOTALL)
                        
                        if reasoning_match:
                            reasoning = reasoning_match.group(1).strip()
                            sample["reasoning"] = reasoning
                        else:
                            # If no reasoning found, we'll use the full generated text (excluding answer)
                            # or generate it during training data preparation
                            # Remove answer part to get reasoning
                            answer_part = f"<answer>{predicted_answer}</answer>" if predicted_answer else ""
                            reasoning_candidate = gen_text.replace(answer_part, "").strip()
                            sample["reasoning"] = reasoning_candidate if reasoning_candidate else None
                    
                    correct_samples.append(sample)
                
                if total_samples % 100 == 0:
                    logger.info(f"Processed {total_samples} samples, {correct_count} correct ({correct_count/total_samples*100:.2f}%)")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM error at batch {batch_idx}. Please restart with --batch_size 1")
            # Clear cache
            torch.cuda.empty_cache()
            raise

    # Summary
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    logger.info(f"Inference completed. Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_samples})")
    logger.info(f"Filtered {len(correct_samples)} correct samples for rejection sampling training")

    # Save filtered samples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(correct_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(correct_samples)} correct samples to {output_path}")
    
    # Save statistics
    stats_path = output_path.replace('.json', '_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset paths: {data_path}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Correct samples: {correct_count}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Filtered samples for training: {len(correct_samples)}\n")
    
    return correct_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rejection Sampling Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained GRM model")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path in format 'source:path'")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save filtered samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--use_cot", action="store_true", default=True, help="Use CoT instruction for reasoning")
    parser.add_argument("--task_instruction", type=str, default=TASK_INSTRUCTION_COT, help="Task instruction template")
    
    args = parser.parse_args()
    
    # Parse data path
    data_paths = [args.data_path] if isinstance(args.data_path, str) else args.data_path.split(",")
    
    config = {
        "task_instruction": args.task_instruction,
        "name": "rejection_sampling_inference",
    }
    
    inference_and_filter(
        model_path=args.model_path,
        data_path=data_paths,
        output_path=args.output_path,
        config=config,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_cot=args.use_cot,
    )

