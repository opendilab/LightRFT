"""
Convert filtered samples to rejection sampling training data format for T2V.

This script converts the filtered correct samples into the format required
for rejection sampling training, similar to imagegen-cot-reward dataset but for videos.
"""

import os
import json
import argparse
from typing import List, Dict
from loguru import logger


def convert_to_rejection_sampling_format(
    filtered_samples_path: str,
    output_path: str,
    task_instruction_template: str = None,
    video_fps: float = 2.0,
):
    """
    Convert filtered samples to rejection sampling training format for T2V.
    
    :param filtered_samples_path: Path to filtered samples JSON file
    :type filtered_samples_path: str
    :param output_path: Path to save converted training data
    :type output_path: str
    :param task_instruction_template: Template for task instruction
    :type task_instruction_template: str, optional
    :param video_fps: FPS for video processing
    :type video_fps: float
    :return: List of training data items in imagegen-cot-reward format (for videos)
    :rtype: List[Dict]
    """
    logger.info(f"Loading filtered samples from {filtered_samples_path}")
    
    with open(filtered_samples_path, 'r', encoding='utf-8') as f:
        filtered_samples = json.load(f)
    
    logger.info(f"Loaded {len(filtered_samples)} filtered samples")
    
    # Default task instruction template for T2V
    if task_instruction_template is None:
        task_instruction_template = """Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. 
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
Text Caption: {prompt}"""
    
    training_data = []
    
    for idx, sample in enumerate(filtered_samples):
        prompt = sample['prompt']
        path1 = sample['path1']
        path2 = sample['path2']
        preference = sample['preference']
        generated_text = sample.get('generated_text', '')
        reasoning = sample.get('reasoning', '')
        
        # Determine which video is better based on preference
        # preference "A" means Video 1 is better
        # preference "B" means Video 2 is better
        # For training data, we always use: Video 1 = preferred, Video 2 = rejected
        # This ensures consistency
        answer = "Video 1 is better" if preference == "A" else "Video 2 is better"
        video1_path = path1 if preference == "A" else path2  # preferred
        video2_path = path2 if preference == "A" else path1  # rejected
        
        # Build the response with CoT reasoning
        if reasoning:
            # Use the extracted reasoning from inference
            # Clean up the reasoning text
            reasoning_clean = reasoning.strip()
            response = f"<think>\n{reasoning_clean}\n</think>\n<answer>{answer}</answer>"
        else:
            # If no reasoning was extracted, create a placeholder
            response = f"<think>\nBased on the evaluation of semantic consistency, temporal coherence, and authenticity, I will compare the two videos.\n</think>\n<answer>{answer}</answer>"
        
        # Build conversations format
        task_instruction = task_instruction_template.format(prompt=prompt)
        
        # Create training data item in imagegen-cot-reward format (but for videos)
        # We use "images" field name to be compatible with ImageGenCoTRewardHandler
        # but store video paths - the handler will need to be modified to support videos
        # For now, we store relative paths from data_root
        training_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": task_instruction
                },
                {
                    "from": "gpt",
                    "value": response
                }
            ],
            "images": [
                video1_path if os.path.isabs(video1_path) else video1_path,
                video2_path if os.path.isabs(video2_path) else video2_path,
            ],
            "video_fps": video_fps,  # Store FPS for video processing
        }
        
        training_data.append(training_item)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Converted {idx + 1}/{len(filtered_samples)} samples")
    
    # Save training data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(training_data)} training samples to {output_path}")
    
    return training_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert filtered samples to rejection sampling training format for T2V")
    parser.add_argument("--filtered_samples_path", type=str, required=True, 
                       help="Path to filtered samples JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save converted training data")
    parser.add_argument("--task_instruction", type=str, default=None,
                       help="Task instruction template (optional)")
    parser.add_argument("--video_fps", type=float, default=2.0,
                       help="FPS for video processing")
    
    args = parser.parse_args()
    
    convert_to_rejection_sampling_format(
        filtered_samples_path=args.filtered_samples_path,
        output_path=args.output_path,
        task_instruction_template=args.task_instruction,
        video_fps=args.video_fps,
    )

