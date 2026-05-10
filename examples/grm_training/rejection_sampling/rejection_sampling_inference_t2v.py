"""
Rejection Sampling Inference Script for Text-to-Video (T2V)

This script performs inference on a dataset using a trained GRM model,
filters out correctly predicted samples, and generates training data
with CoT reasoning for rejection sampling training.

For Rapidata-T2V, we compute gt_preference based on the sum of three dimensions:
Alignment + Coherence + Preference
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
from lightrft.datasets import extract_answer, RFTDatasetVL

# Import qwen_vl_utils for processing vision info
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    try:
        from keye_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Neither qwen_vl_utils nor keye_vl_utils is available")


TASK_INSTRUCTION_COT_T2V = """Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. 
Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. 
For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each video by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings:
'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.
Example output format:
<think>
1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...
2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...
3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...
...
[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...
Total score:
Video 1: 9+8+7+6=30
Video 2: 7+6+5+8=26
</think>
<answer>Video 1 is better</answer>

Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.
Your task is provided as follows:
Text Caption: **{prompt}**
"""


class GRMPromptDatasetVLT2V:
    """
    Dataset wrapper for vLLM inference that returns prompts and video paths
    instead of tokenized inputs. Adapted for T2V with RFTDatasetVL.
    """
    def __init__(
        self,
        dataset_paths: List[str],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        strategy=None,
        max_length: int = 8192,
        config: Dict = None,
        is_training: bool = False,
    ):
        self.base_dataset = RFTDatasetVL(
            dataset_paths,
            processor=processor,
            tokenizer=tokenizer,
            strategy=strategy,
            max_length=max_length,
            config=config,
            is_train=is_training,
        )
        self.processor = processor
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset.data[idx]
        source = item["source"]
        handler = self.base_dataset.handlers[source]
        
        # Get media info (paths)
        media_info = handler.get_media_info(item)
        
        # Load media content (needed for parse_item)
        loaded_content = self.base_dataset.media_content_loader(media_info)
        if loaded_content is None:
            raise RuntimeError(f"Failed to load media content: {media_info}")
        
        # Parse item to get messages (returns messages0, messages1, other for PairHandler)
        messages0, messages1, other = handler.parse_item(item, loaded_content, self.base_dataset.config)
        
        # Combine messages0 and messages1 to show both videos in the same conversation
        # Similar to HPDv3GRMHandler format: system prompt + Video 1 + Video 2
        messages = []
        
        # Add system prompt (from messages0)
        if len(messages0) > 0 and messages0[0].get("role") == "system":
            messages.append(messages0[0])
        
        # Add Video 1 with label
        if len(messages0) > 1 and messages0[1].get("role") == "user":
            video1_content = messages0[1]["content"]
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "**Video 1:**"
                    },
                    video1_content[0] if isinstance(video1_content, list) and len(video1_content) > 0 else video1_content
                ]
            })
        
        # Add Video 2 with label (from messages1)
        if len(messages1) > 1 and messages1[1].get("role") == "user":
            video2_content = messages1[1]["content"]
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "**Video 2:**"
                    },
                    video2_content[0] if isinstance(video2_content, list) and len(video2_content) > 0 else video2_content
                ]
            })
        
        # Get prompt text (exclude the last assistant message for inference)
        messages_for_prompt = messages[:-1] if len(messages) > 0 and messages[-1].get("role") == "assistant" else messages
        prompt_text = self.processor.apply_chat_template(
            messages_for_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Extract video information from messages using process_vision_info
        # This is the same way test_grm_vl_vllm.py does it
        # process_vision_info returns (image_inputs, video_inputs, video_kwargs)
        # but we only need image_inputs and video_inputs for vLLM
        image_inputs, video_inputs, _ = process_vision_info(
            messages_for_prompt,
            return_video_kwargs=True,
        )
        
        # Store original item for accessing raw scores
        other['_raw_item'] = item
        
        return prompt_text, image_inputs, video_inputs, other
    
    def collate_fn(self, batch):
        input_texts = []
        image_inputs_list = []
        video_inputs_list = []
        extras = []
        
        for prompt_text, image_inputs, video_inputs, other in batch:
            input_texts.append(prompt_text)
            image_inputs_list.append(image_inputs if image_inputs else None)
            video_inputs_list.append(video_inputs if video_inputs else None)
            extras.append(other)
        
        return input_texts, image_inputs_list, video_inputs_list, extras


def safe_get_score(item: Dict, key: str, default: float = 0.0) -> float:
    """
    Safely get score value from item, handling None values.
    
    :param item: Dictionary containing score values
    :param key: Key to look up in the dictionary
    :param default: Default value to use if key is missing or value is None
    :return: Float score value
    """
    value = item.get(key, default)
    return default if value is None else float(value)


def compute_total_score(item: Dict, video_num: int) -> float:
    """
    Compute total score for a video based on three dimensions.
    
    :param item: Dictionary containing score values
    :param video_num: Video number (1 or 2)
    :return: Total score (Alignment + Coherence + Preference)
    """
    alignment = safe_get_score(item, f"weighted_results{video_num}_Alignment", 0.0)
    coherence = safe_get_score(item, f"weighted_results{video_num}_Coherence", 0.0)
    preference = safe_get_score(item, f"weighted_results{video_num}_Preference", 0.0)
    return alignment + coherence + preference


def compute_gt_preference_from_scores(item: Dict) -> str:
    """
    Compute ground truth preference based on sum of three dimensions:
    Alignment + Coherence + Preference
    
    Returns "A" if video1 has higher total score, "B" if video2 has higher total score.
    """
    total_score1 = compute_total_score(item, 1)
    total_score2 = compute_total_score(item, 2)
    
    if total_score1 > total_score2:
        return "A"  # Video 1 is better
    elif total_score1 < total_score2:
        return "B"  # Video 2 is better
    else:
        return "C"  # Equal (shouldn't happen often, but handle it)


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
    video_fps: float = 2.0,
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
    :param video_fps: FPS for video processing
    :type video_fps: float
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
            "image": 0,
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
    dataset = GRMPromptDatasetVLT2V(
        data_path,
        processor=processor,
        tokenizer=tokenizer,
        strategy=None,
        max_length=8192,
        config=config,
        is_training=False,
    )
    
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
    
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        try:
            input_texts, image_inputs_list, video_inputs_list, extras = batch
            
            # Prepare inputs for vLLM (same format as test_grm_vl_vllm.py)
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
            for i, (gen_text, extra) in enumerate(zip(gen_texts, extras)):
                total_samples += 1
                predicted_answer = extract_answer(gen_text)
                
                # Get raw item to compute gt_preference from scores
                raw_item = extra.get('_raw_item', {})
                gt_preference = compute_gt_preference_from_scores(raw_item)
                
                # Mapping logic: 
                # "A" means Video 1 is better
                # "B" means Video 2 is better
                is_correct = False
                if gt_preference == "A" and predicted_answer == "Video 1 is better":
                    is_correct = True
                elif gt_preference == "B" and predicted_answer == "Video 2 is better":
                    is_correct = True
                elif gt_preference == "C":
                    # Handle tie case (should be rare)
                    logger.warning(f"Tie detected in sample {total_samples}, skipping")
                    continue
                
                if is_correct:
                    correct_count += 1
                    # Get video paths from raw item
                    data_root = raw_item.get('data_root', '')
                    video1_path = os.path.join(data_root, "videos", raw_item.get('file_name1', ''))
                    video2_path = os.path.join(data_root, "videos", raw_item.get('file_name2', ''))
                    
                    # Prepare sample for rejection sampling training
                    sample = {
                        "prompt": raw_item.get('prompt', ''),
                        "path1": video1_path,
                        "path2": video2_path,
                        "preference": gt_preference,
                        "generated_text": gen_text,
                        "predicted_answer": predicted_answer,
                        "score1_total": compute_total_score(raw_item, 1),
                        "score2_total": compute_total_score(raw_item, 2),
                    }
                    
                    # If we want to use the generated CoT reasoning, extract it
                    if use_cot:
                        # Extract reasoning from generated text
                        reasoning_match = None
                        # Try <think> tag first
                        if "<think>" in gen_text:
                            reasoning_pattern = r"<think>(.*?)</think>"
                            reasoning_match = re.search(reasoning_pattern, gen_text, re.DOTALL)
                        # Try <think> as fallback (in case model uses different format)
                        elif "<think>" in gen_text:
                            reasoning_pattern = r"<think>(.*?)</think>"
                            reasoning_match = re.search(reasoning_pattern, gen_text, re.DOTALL)
                        
                        if reasoning_match:
                            reasoning = reasoning_match.group(1).strip()
                            sample["reasoning"] = reasoning
                        else:
                            # If no reasoning found, use the full generated text (excluding answer)
                            answer_part = f"<answer>{predicted_answer}</answer>" if predicted_answer else ""
                            reasoning_candidate = gen_text.replace(answer_part, "").strip()
                            sample["reasoning"] = reasoning_candidate if reasoning_candidate else None
                    
                    correct_samples.append(sample)
                
                if total_samples % 100 == 0:
                    logger.info(f"Processed {total_samples} samples, {correct_count} correct ({correct_count/total_samples*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="Rejection Sampling Inference for T2V")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained GRM model")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path(s) in format 'source:path' (comma-separated for multiple)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save filtered samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--use_cot", action="store_true", default=True, help="Use CoT instruction for reasoning")
    parser.add_argument("--task_instruction", type=str, default=TASK_INSTRUCTION_COT_T2V, help="Task instruction template")
    
    # vLLM arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization ratio")
    parser.add_argument("--video_fps", type=float, default=2.0, help="FPS for video processing")
    
    args = parser.parse_args()
    
    # Parse data path
    data_paths = args.data_path.split(",") if isinstance(args.data_path, str) else args.data_path
    
    config = {
        "task_instruction": args.task_instruction,
        "name": "rejection_sampling_inference_t2v",
        "video_fps": args.video_fps,
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
        video_fps=args.video_fps,
    )

