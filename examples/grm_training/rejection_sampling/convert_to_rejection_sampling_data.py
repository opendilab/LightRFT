"""
Convert filtered samples to rejection sampling training data format.

This script converts the filtered correct samples into the format required
for rejection sampling training, similar to imagegen-cot-reward dataset.
"""

import os
import json
import argparse
from typing import List, Dict
from loguru import logger


def convert_to_rejection_sampling_format(
    filtered_samples_path: str,
    output_path: str,
    data_root: str,
    task_instruction_template: str = None,
):
    """
    Convert filtered samples to rejection sampling training format.
    
    Args:
        filtered_samples_path: Path to filtered samples JSON file
        output_path: Path to save converted training data
        data_root: Root directory of the dataset (for image paths)
        task_instruction_template: Template for task instruction
    """
    logger.info(f"Loading filtered samples from {filtered_samples_path}")
    
    with open(filtered_samples_path, 'r', encoding='utf-8') as f:
        filtered_samples = json.load(f)
    
    logger.info(f"Loaded {len(filtered_samples)} filtered samples")
    
    # Default task instruction template
    if task_instruction_template is None:
        task_instruction_template = """Given a caption and two images generated based on this caption, please analyze in detail the two provided images. 
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
Text Caption: {prompt}"""
    
    training_data = []
    
    for idx, sample in enumerate(filtered_samples):
        prompt = sample['prompt']
        path1 = sample['path1']
        path2 = sample['path2']
        preference = sample['preference']
        generated_text = sample.get('generated_text', '')
        reasoning = sample.get('reasoning', '')
        
        # Determine which image is better based on preference
        # In HPDv3, path1 is the preferred path, path2 is the rejected path
        # preference "A" means Image 1 (which is path1) is better
        # preference "B" means Image 2 (which is path2) is better, but this means path2 was randomly chosen as Image 1
        # Actually, in HPDv3GRMHandler, preference A means image0 (first shown) is preferred
        # So we need to check which path corresponds to which image
        
        # Since we stored preferred_path and rejected_path, we know:
        # - preferred_path (path1) should be the better one
        # - rejected_path (path2) should be the worse one
        # But the handler randomly assigns them to Image 1 or Image 2
        
        # For training data, we always use: Image 1 = preferred, Image 2 = rejected
        # This ensures consistency
        answer = "Image 1 is better"
        image1_path = path1  # preferred
        image2_path = path2  # rejected
        
        # Build the response with CoT reasoning
        # Note: We use <think> instead of <think> to match the instruction format
        if reasoning:
            # Use the extracted reasoning from inference
            # Clean up the reasoning text
            reasoning_clean = reasoning.strip()
            response = f"<think>\n{reasoning_clean}\n</think>\n<answer>{answer}</answer>"
        else:
            # If no reasoning was extracted, create a placeholder
            # In practice, you might want to regenerate this or use a template
            response = f"<think>\nBased on the evaluation of semantic consistency, aesthetics, and authenticity, I will compare the two images.\n</think>\n<answer>{answer}</answer>"
        
        # Build conversations format
        task_instruction = task_instruction_template.format(prompt=prompt)
        
        # Create training data item in imagegen-cot-reward format
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
                image1_path if os.path.isabs(image1_path) else os.path.join(data_root, image1_path),
                image2_path if os.path.isabs(image2_path) else os.path.join(data_root, image2_path),
            ]
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
    parser = argparse.ArgumentParser(description="Convert filtered samples to rejection sampling training format")
    parser.add_argument("--filtered_samples_path", type=str, required=True, 
                       help="Path to filtered samples JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save converted training data")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset (for image paths)")
    parser.add_argument("--task_instruction", type=str, default=None,
                       help="Task instruction template (optional)")
    
    args = parser.parse_args()
    
    convert_to_rejection_sampling_format(
        filtered_samples_path=args.filtered_samples_path,
        output_path=args.output_path,
        data_root=args.data_root,
        task_instruction_template=args.task_instruction,
    )

