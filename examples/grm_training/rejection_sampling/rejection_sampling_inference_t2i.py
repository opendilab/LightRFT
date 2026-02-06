"""
Rejection Sampling Inference Script

This script performs inference on a dataset using a trained GRM model,
filters out correctly predicted samples, and generates training data
with CoT reasoning for rejection sampling training.
"""

import os
import json
import argparse
import re
from tqdm import tqdm
from typing import List, Dict
from loguru import logger
from torch.utils.data import DataLoader

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from lightrft.datasets import RFTDatasetVL, extract_answer
from lightrft.datasets.hpdv3 import HPDv3GRMHandler


def extract_response(text: str, media_type: str = "Image") -> str:
    """
    Extract the preference from the generated text.

    It first tries to extract the content from ``<answer>`` tags using :func:`extract_answer`.
    If no tags are found, it performs a heuristic search for key phrases (e.g., "Image 1 is better")
    at the end of the text.

    :param text: The generated text from the model
    :type text: str
    :param media_type: The type of media being evaluated ("Image", "Video", or "Audio"), defaults to "Image"
    :type media_type: str, optional

    :return: The extracted preference string (e.g., "Image 1 is better") or None if not found
    :rtype: str
    """
    # 1. Try extracting from <answer> tags
    ans = extract_answer(text)
    if ans:
        return ans

    # 2. Heuristic search if no tags found
    text_lower = text.lower()
    media_lower = media_type.lower()
    
    key_1 = f"{media_lower} 1 is better"
    key_2 = f"{media_lower} 2 is better"
    key_equal = f"both {media_lower}s are equally good"
    
    idx_1 = text_lower.rfind(key_1)
    idx_2 = text_lower.rfind(key_2)
    idx_equal = text_lower.rfind(key_equal)
    
    candidates = {}
    if idx_1 != -1: 
        candidates[f"{media_type} 1 is better"] = idx_1
    if idx_2 != -1: 
        candidates[f"{media_type} 2 is better"] = idx_2
    if idx_equal != -1: 
        candidates[f"Both {media_lower}s are equally good"] = idx_equal
    
    if not candidates:
        return None
        
    # Return the one that appears last in the text
    return max(candidates, key=candidates.get)


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




def inference_and_filter(
    model_path: str,
    data_path: List[str],
    output_path: str,
    config: dict = None,
    batch_size: int = 32,
    max_new_tokens: int = 2048,
    use_cot: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    """
    Perform inference on dataset and filter correctly predicted samples.
    
    :param model_path: Path to the trained GRM model
    :type model_path: str
    :param data_path: List of dataset paths in format "source:path"
    :type data_path: List[str]
    :param output_path: Path to save filtered samples
    :type output_path: str
    :param config: Configuration dict for dataset
    :type config: dict, optional
    :param batch_size: Batch size for inference
    :type batch_size: int
    :param max_new_tokens: Maximum tokens to generate
    :type max_new_tokens: int
    :param use_cot: Whether to use CoT instruction (for generating reasoning)
    :type use_cot: bool
    :param tensor_parallel_size: Number of GPUs for tensor parallelism
    :type tensor_parallel_size: int
    :param gpu_memory_utilization: GPU memory utilization ratio
    :type gpu_memory_utilization: float
    :return: List of correctly predicted samples with their generated text and reasoning
    :rtype: List[Dict]
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={
            "image": 2,
            "video": 2
        },
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,  # For deterministic output
        max_tokens=max_new_tokens,
    )
    
    logger.info(f"Model loaded successfully from {model_path}.")
    
    # Load Processor and Tokenizer for Dataset
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load Dataset
    dataset = RFTDatasetVL(
        data_path,
        processor=processor,
        tokenizer=tokenizer,
        strategy=None,
        max_length=8192,
        config=config,
        is_train=False,
    )
    
    # Fix handler mapping: RFTDatasetVL uses HPDv3PairHandler which returns 3 values,
    # but we need HPDv3GRMHandler which returns 2 values for compatibility
    for source in dataset.handlers.keys():
        if source == "hpdv3":
            dataset.handlers[source] = HPDv3GRMHandler()
            logger.info(f"Replaced handler for {source} with HPDv3GRMHandler for compatibility")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    
    logger.info(f"Starting inference with CoT: {use_cot}, batch_size: {batch_size}")
    
    correct_samples = []
    total_samples = 0
    correct_count = 0
    parse_failures = 0
    
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        try:
            input_texts, image_inputs_list, video_inputs_list, extras, _ = batch
            
            # Prepare inputs for vLLM
            inputs = []
            for i in range(len(input_texts)):
                prompt = input_texts[i]
                image_inputs = image_inputs_list[i]
                video_inputs = video_inputs_list[i]
                
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": mm_data
                })
            
            # Generate with vLLM
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            
            # Decode
            gen_texts = [output.outputs[0].text for output in outputs]
            
            # Evaluate and filter
            batch_correct = 0
            batch_total = len(gen_texts)
            for i, (gen_text, extra) in enumerate(zip(gen_texts, extras)):
                total_samples += 1
                predicted_answer = extract_response(gen_text, media_type="Image")
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
                if predicted_answer is None:
                    parse_failures += 1
                    logger.warning(f"Could not extract answer from generated text: {gen_text[:200]}...")
                elif gt_preference == "A" and predicted_answer == "Image 1 is better":
                    is_correct = True
                elif gt_preference == "B" and predicted_answer == "Image 2 is better":
                    is_correct = True
                
                if is_correct:
                    correct_count += 1
                    batch_correct += 1
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
            
            # Output real-time accuracy after each batch
            current_accuracy = correct_count / total_samples if total_samples > 0 else 0.0
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0.0
            parse_failure_rate = parse_failures / total_samples if total_samples > 0 else 0.0
            logger.info(
                f"Batch {batch_idx + 1} | "
                f"Batch Acc: {batch_accuracy*100:.2f}% ({batch_correct}/{batch_total}) | "
                f"Overall Acc: {current_accuracy*100:.2f}% ({correct_count}/{total_samples}) | "
                f"Parse Fail: {parse_failure_rate*100:.2f}% ({parse_failures}/{total_samples}) | "
                f"Filtered: {len(correct_samples)}"
            )
            
        except Exception as e:
            logger.error(f"Error at batch {batch_idx}: {e}")
            raise
    
    # Summary
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    failure_rate = parse_failures / total_samples if total_samples > 0 else 0.0
    logger.info(f"Inference completed. Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_samples})")
    logger.info(f"Parse Failure Rate: {failure_rate*100:.2f}% ({parse_failures}/{total_samples})")
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
        f.write(f"Parse failures: {parse_failures}\n")
        f.write(f"Parse Failure Rate: {failure_rate*100:.2f}%\n")
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
    
    # vLLM arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization ratio")
    
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
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

